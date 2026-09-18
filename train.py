"""训练器模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:  # plotting is optional for model/test execution
    plt = None
    sns = None
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve
from tqdm import tqdm
import logging
import traceback

from config import Config
from device_manager import DeviceManager
from metrics import precision_at_k
from artifacts import checkpoint_config_metadata

logger = logging.getLogger(__name__)


def _checkpoint_model_config(config, kg_data=None):
    """Merge runtime dimensions with the full serialized config snapshot.

    Lowercase keys stay compatible with infer.py's dimension checks while
    the uppercase snapshot restores every training setting at load time
    through artifacts.apply_model_config.
    """
    dimensions = {
        'hidden_dim': config.HIDDEN_DIM,
        'embedding_dim': config.EMBEDDING_DIM,
        'omics_feature_dim': int(getattr(config, 'OMICS_FEATURE_DIM', 262)),
        'kg_feature_dim': int(getattr(config, 'KG_FEATURE_DIM', 768)),
    }
    if kg_data is not None:
        dimensions['node_types'] = list(kg_data.node_types)
        dimensions['edge_types'] = list(kg_data.edge_types)
    metadata = checkpoint_config_metadata(config)
    merged = dict(dimensions)
    merged.update(metadata['model_config'])
    return merged, metadata['model_config_sha256']


class MoEObjective(nn.Module):
    """Composite 4.0-MoE objective with independently supervised experts."""

    def __init__(self, config, pos_weight=None):
        super().__init__()
        self.config = config
        self.register_buffer('pos_weight', pos_weight)
        # Use a private CPU generator for sampled objectives.  Sampling from
        # the global CUDA RNG made otherwise identical runs depend on kernel
        # scheduling and on unrelated random operations in the model.
        self._sampling_generator = torch.Generator(device='cpu')
        self._sampling_generator.manual_seed(
            int(getattr(config, 'RANDOM_SEED', 42))
        )

    def _bce(self, logits, labels, sample_weight=None):
        losses = F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=self.pos_weight, reduction='none'
        )
        if sample_weight is None:
            return losses.mean()
        sample_weight = sample_weight.to(losses).reshape_as(losses)
        return (losses * sample_weight).sum() / sample_weight.sum().clamp_min(1e-12)

    def _ranking_loss(self, logits, labels):
        positive = logits[labels > 0.5]
        negative = logits[labels <= 0.5]
        if positive.numel() == 0 or negative.numel() == 0:
            return logits.new_zeros(())
        sample_count = min(
            positive.numel(), negative.numel(),
            int(getattr(self.config, 'RANKING_MAX_PAIRS', 1024)),
        )
        positive_indices = torch.randperm(
            positive.numel(), generator=self._sampling_generator
        )[:sample_count].to(logits.device)
        negative_indices = torch.randperm(
            negative.numel(), generator=self._sampling_generator
        )[:sample_count].to(logits.device)
        positive = positive[positive_indices]
        negative = negative[negative_indices]
        margin = float(getattr(self.config, 'RANKING_MARGIN', 0.2))
        return F.softplus(negative - positive + margin).mean()

    def forward(self, final_logits, labels, details):
        final_bce = self._bce(final_logits, labels)
        sl_bce = self._bce(details['sl_expert_logit'], labels)
        reliability = details.get('reliability')
        kg_sample_weight = None
        if reliability is not None and reliability.ndim == 2 and reliability.size(1) >= 3:
            novelty = reliability[:, 0].detach()
            kg_coverage = reliability[:, 2].detach()
            kg_sample_weight = 1.0 + float(
                getattr(self.config, 'KG_NOVEL_EXPERT_BOOST', 0.0)
            ) * novelty * kg_coverage
        gate_target = _routing_gate_target(details, labels, self.config, reliability)
        gate_routing = final_logits.new_zeros(())
        if gate_target is not None:
            gate_routing = -(
                gate_target
                * details['gate_weights'].clamp_min(1e-12).log()
            ).sum(dim=1).mean()
        kg_bce = self._bce(
            details['kg_expert_logit'], labels, kg_sample_weight
        )
        ranking = self._ranking_loss(final_logits, labels)
        symmetry = details.get('symmetry_loss', final_logits.new_zeros(()))
        mean_gate = details['gate_weights'].mean(dim=0)
        gate_balance = torch.square(mean_gate - 0.5).mean()

        total = (
            float(self.config.LOSS_FINAL_WEIGHT) * final_bce
            + float(self.config.LOSS_SL_EXPERT_WEIGHT) * sl_bce
            + float(self.config.LOSS_KG_EXPERT_WEIGHT) * kg_bce
            + float(self.config.LOSS_RANK_WEIGHT) * ranking
            + float(self.config.LOSS_SYMMETRY_WEIGHT) * symmetry
            + float(self.config.LOSS_GATE_BALANCE_WEIGHT) * gate_balance
            + float(getattr(self.config, 'LOSS_GATE_ROUTING_WEIGHT', 0.0))
            * gate_routing
        )
        return total, {
            'final_bce': final_bce.detach(),
            'sl_expert_bce': sl_bce.detach(),
            'kg_expert_bce': kg_bce.detach(),
            'ranking': ranking.detach(),
            'symmetry': symmetry.detach(),
            'gate_balance': gate_balance.detach(),
            'gate_routing': gate_routing.detach(),
        }


def _coverage_aware_gate_target(reliability, config):
    """Build a soft routing target from train-visible modality coverage."""
    if reliability.ndim != 2 or reliability.size(1) < 4:
        raise ValueError("reliability must have shape [number_of_edges, >=4]")
    novelty = reliability[:, 0].clamp(0.0, 1.0)
    sl_coverage = reliability[:, 1].clamp(0.0, 1.0)
    kg_coverage = reliability[:, 2].clamp(0.0, 1.0)
    strength = float(getattr(config, 'GATE_KG_PRIOR_STRENGTH', 0.0))
    kg_target = 0.5 + strength * novelty * kg_coverage * (1.0 - sl_coverage)
    kg_target = kg_target.clamp(
        float(getattr(config, 'GATE_TARGET_MIN', 0.05)),
        float(getattr(config, 'GATE_TARGET_MAX', 0.95)),
    )
    return torch.stack([1.0 - kg_target, kg_target], dim=1)


def _routing_gate_target(details, labels, config, reliability=None):
    """Per-sample soft routing target for the expert gate.

    'distill' mode routes each sample toward whichever expert currently has the
    lower per-sample loss (detached), so every sample supervises the gate.
    'coverage' mode is the legacy coverage prior, which is a constant 0.5 on
    non-augmented training data and cannot teach per-sample routing.
    """
    mode = str(getattr(config, 'GATE_ROUTING_MODE', 'coverage')).lower()
    if mode == 'distill':
        sl_logit = details.get('sl_expert_logit')
        kg_logit = details.get('kg_expert_logit')
        if sl_logit is None or kg_logit is None:
            return None
        temperature = max(
            float(getattr(config, 'GATE_DISTILL_TEMPERATURE', 0.25)), 1e-6
        )
        with torch.no_grad():
            sl_loss = F.binary_cross_entropy_with_logits(
                sl_logit.detach(), labels, reduction='none'
            )
            kg_loss = F.binary_cross_entropy_with_logits(
                kg_logit.detach(), labels, reduction='none'
            )
            kg_target = torch.sigmoid((sl_loss - kg_loss) / temperature)
        kg_target = kg_target.clamp(
            float(getattr(config, 'GATE_TARGET_MIN', 0.05)),
            float(getattr(config, 'GATE_TARGET_MAX', 0.95)),
        )
        return torch.stack([1.0 - kg_target, kg_target], dim=1)
    if reliability is None:
        return None
    return _coverage_aware_gate_target(reliability.detach(), config)


def _sample_inductive_node_mask(data, config, generator=None):
    """Sample pseudo-new genes strictly from the current training graph."""
    probability = float(getattr(
        config, 'INDUCTIVE_AUGMENTATION_PROBABILITY', 0.0
    ))
    fraction = float(getattr(
        config, 'INDUCTIVE_AUGMENTATION_NODE_FRACTION', 0.0
    ))
    if data.num_nodes < 2 or probability <= 0.0 or fraction <= 0.0:
        return None
    if generator is None:
        generator = torch.Generator(device='cpu')
        generator.manual_seed(int(getattr(config, 'RANDOM_SEED', 42)))
    if torch.rand((), generator=generator).item() >= probability:
        return None

    candidates = torch.unique(data.edge_index.reshape(-1))
    if candidates.numel() < 2:
        return None
    mask_count = int(round(candidates.numel() * fraction))
    mask_count = max(1, min(mask_count, candidates.numel() - 1))
    selected = candidates[
        torch.randperm(
            candidates.numel(), generator=generator
        )[:mask_count].to(candidates.device)
    ]
    mask = torch.zeros(
        data.num_nodes, dtype=torch.bool, device=data.edge_index.device
    )
    mask[selected] = True
    return mask


def _validation_selection_score(config, auc, aupr, f1):
    weights = np.asarray([
        float(getattr(config, 'MODEL_SELECTION_AUC_WEIGHT', 1.0)),
        float(getattr(config, 'MODEL_SELECTION_AUPR_WEIGHT', 0.0)),
        float(getattr(config, 'MODEL_SELECTION_F1_WEIGHT', 0.0)),
    ], dtype=float)
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Model-selection weights must be non-negative and non-zero")
    values = np.asarray([auc, aupr, f1], dtype=float)
    return float(np.dot(weights / weights.sum(), values))


def _build_moe_objective(config, labels, device):
    positive_count = (labels > 0.5).sum()
    negative_count = (labels <= 0.5).sum()
    pos_weight = None
    if positive_count > 0 and negative_count > 0:
        pos_weight = (negative_count / positive_count).reshape(1).to(device)
    return MoEObjective(config, pos_weight=pos_weight).to(device)


def _balanced_training_cancer_index(data, generator=None):
    """Prevent cancer-label availability from becoming a supervised shortcut."""
    cancer_index = getattr(data, 'cancer_index', None)
    if cancer_index is None or cancer_index.numel() != data.y.numel():
        return None
    balanced = cancer_index.clone()
    known = cancer_index > 1
    labels = data.y > 0.5
    class_masks = [~labels, labels]
    rates = []
    for class_mask in class_masks:
        class_count = int(class_mask.sum().item())
        rates.append(
            float((known & class_mask).sum().item()) / class_count
            if class_count else 0.0
        )
    target_rate = min(rates)
    balanced[~known] = 1
    for class_mask in class_masks:
        candidates = torch.where(known & class_mask)[0]
        keep_count = int(class_mask.sum().item() * target_rate)
        if keep_count < candidates.numel():
            if generator is None:
                generator = torch.Generator(device='cpu')
                generator.manual_seed(torch.initial_seed())
            order = torch.randperm(
                candidates.numel(), generator=generator
            ).to(candidates.device)
            balanced[candidates[order[keep_count:]]] = 1
    return balanced


def _mean_or_nan(values):
    if values.numel() == 0:
        return float('nan')
    return float(values.float().mean().item())


def _pearson_or_nan(left, right):
    left = left.float().reshape(-1)
    right = right.float().reshape(-1)
    if left.numel() < 2 or right.numel() != left.numel():
        return float('nan')
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    if denominator <= 0:
        return float('nan')
    return float((left * right).sum().div(denominator).item())


def _expert_metrics_or_nan(logits, labels):
    if logits is None or logits.numel() != labels.numel():
        return float('nan'), float('nan')
    y_true = labels.detach().float().cpu().numpy()
    if np.unique(y_true).size < 2:
        return float('nan'), float('nan')
    y_score = torch.sigmoid(logits.detach().float()).cpu().numpy()
    return (
        float(roc_auc_score(y_true, y_score)),
        float(average_precision_score(y_true, y_score)),
    )


def _summarize_gate_details(details, labels, prefix, config=None):
    """Reduce per-edge MoE routing tensors to one CSV-friendly row."""
    gate = details['gate_weights'].detach().float()
    labels = labels.detach().float().reshape(-1)
    if gate.ndim != 2 or gate.size(1) != 2 or gate.size(0) != labels.numel():
        raise ValueError(
            "gate_weights must have shape [number_of_labels, 2], got "
            f"{tuple(gate.shape)} for {labels.numel()} labels"
        )

    sl_gate = gate[:, 0]
    kg_gate = gate[:, 1]
    entropy = -(gate * gate.clamp_min(1e-12).log()).sum(dim=1)
    mean_gate = gate.mean(dim=0)
    summary = {
        f'{prefix}_n_edges': int(labels.numel()),
        f'{prefix}_sl_gate_mean': _mean_or_nan(sl_gate),
        f'{prefix}_kg_gate_mean': _mean_or_nan(kg_gate),
        f'{prefix}_sl_gate_std': float(sl_gate.std(unbiased=False).item()),
        f'{prefix}_kg_gate_std': float(kg_gate.std(unbiased=False).item()),
        f'{prefix}_sl_gate_min': float(sl_gate.min().item()),
        f'{prefix}_sl_gate_max': float(sl_gate.max().item()),
        f'{prefix}_kg_gate_min': float(kg_gate.min().item()),
        f'{prefix}_kg_gate_max': float(kg_gate.max().item()),
        f'{prefix}_gate_entropy_mean': _mean_or_nan(entropy),
        f'{prefix}_gate_balance': float(
            torch.square(mean_gate - 0.5).mean().item()
        ),
    }

    for group_name, mask in (
        ('negative', labels <= 0.5),
        ('positive', labels > 0.5),
    ):
        summary[f'{prefix}_{group_name}_n'] = int(mask.sum().item())
        summary[f'{prefix}_{group_name}_sl_gate_mean'] = _mean_or_nan(sl_gate[mask])
        summary[f'{prefix}_{group_name}_kg_gate_mean'] = _mean_or_nan(kg_gate[mask])

    reliability = details.get('reliability')
    reliability_names = (
        'new_fraction', 'sl_coverage', 'kg_coverage', 'omics_coverage',
        'kg_direct_edge', 'kg_common_neighbors',
        'kg_degree_mean', 'kg_degree_diff',
        'sl_degree_mean', 'sl_degree_diff',
    )
    if reliability is not None:
        reliability = reliability.detach().float()
    if (
        reliability is not None
        and reliability.ndim == 2
        and reliability.size(0) == labels.numel()
        and reliability.size(1) >= 4
    ):
        for column, name in enumerate(reliability_names[:reliability.size(1)]):
            values = reliability[:, column]
            summary[f'{prefix}_{name}_mean'] = _mean_or_nan(values)
            summary[f'{prefix}_kg_gate_{name}_corr'] = _pearson_or_nan(
                kg_gate, values
            )

        new_fraction = reliability[:, 0]
        for suffix, value in (('0', 0.0), ('05', 0.5), ('1', 1.0)):
            mask = torch.isclose(
                new_fraction, new_fraction.new_tensor(value), atol=1e-6
            )
            summary[f'{prefix}_new_{suffix}_n'] = int(mask.sum().item())
            summary[f'{prefix}_new_{suffix}_sl_gate_mean'] = _mean_or_nan(
                sl_gate[mask]
            )
            summary[f'{prefix}_new_{suffix}_kg_gate_mean'] = _mean_or_nan(
                kg_gate[mask]
            )

        if config is not None:
            routing_target = _routing_gate_target(
                details, labels, config, reliability
            )
            if routing_target is not None:
                target_kg = routing_target[:, 1]
                summary[f'{prefix}_kg_gate_target_mean'] = _mean_or_nan(target_kg)
                summary[f'{prefix}_kg_gate_target_mae'] = _mean_or_nan(
                    torch.abs(kg_gate - target_kg)
                )

    for expert_name, detail_key in (
        ('sl_expert', 'sl_expert_logit'),
        ('kg_expert', 'kg_expert_logit'),
    ):
        auc, aupr = _expert_metrics_or_nan(details.get(detail_key), labels)
        summary[f'{prefix}_{expert_name}_auc'] = auc
        summary[f'{prefix}_{expert_name}_aupr'] = aupr

    sl_logit = details.get('sl_expert_logit')
    kg_logit = details.get('kg_expert_logit')
    if (
        sl_logit is not None
        and kg_logit is not None
        and sl_logit.numel() == labels.numel()
        and kg_logit.numel() == labels.numel()
    ):
        avg_auc, avg_aupr = _expert_metrics_or_nan(
            (sl_logit.detach() + kg_logit.detach()) * 0.5, labels
        )
        summary[f'{prefix}_avg_fusion_auc'] = avg_auc
        summary[f'{prefix}_avg_fusion_aupr'] = avg_aupr
    return summary

# ==================== 6. 训练器（使用设备管理器，支持训练验证划分） ====================
class Trainer:
    def __init__(self, model, train_data, val_data, kg_data, gene_mapping, processor=None, config=None, device_manager=None):
        if config is None:
            self.config = Config()
        else:
            self.config = config

        # 设备管理器
        self.device_manager = device_manager or DeviceManager()

        logger.info(f"训练器使用设备管理器，目标设备: {self.device_manager.target_device}")

        # 移动模型到设备
        self.model = self.device_manager.move_model(model)

        # 移动训练和验证数据到设备
        self.train_data = self.device_manager.move_data(train_data)
        self.val_data = self.device_manager.move_data(val_data)

        # 移动知识图谱数据到设备
        self.kg_data = self.device_manager.move_hetero_data(kg_data)

        self.gene_mapping = gene_mapping
        self.sampling_generator = torch.Generator(device='cpu')
        self.sampling_generator.manual_seed(
            int(getattr(self.config, 'RANDOM_SEED', 42))
        )

        # Materialize all data-dependent KG modules before any optimizer is created.
        self.kg_schema = None
        if hasattr(self.model, 'prepare_for_kg'):
            self.kg_schema = self.model.prepare_for_kg(self.kg_data)
            self.model = self.device_manager.move_model(self.model)

        self.processor = processor
        self.cancer_vocabulary_state = None
        if processor is not None and hasattr(processor, 'cancer_vocabulary'):
            self.cancer_vocabulary_state = processor.cancer_vocabulary.state_dict()

        if hasattr(self.model, 'set_train_sl_context'):
            ordered_gene_ids = [
                gene_id for gene_id, _ in
                sorted(gene_mapping.items(), key=lambda item: item[1])
            ]
            self.model.set_train_sl_context(self.train_data, ordered_gene_ids)

        if hasattr(self.train_data, 'cancer_index'):
            known = self.train_data.cancer_index > 1
            labels = self.train_data.y > 0.5
            coverage = {}
            for name, mask in (('negative', ~labels), ('positive', labels)):
                count = int(mask.sum().item())
                coverage[name] = (
                    float((known & mask).sum().item()) / count if count else 0.0
                )
            logger.info(
                "Cancer annotation coverage before balanced masking: %s",
                coverage,
            )

        # 记录训练历史
        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []
        self.val_auprcs = []
        self.learning_rates = []
        self.last_validation_metrics = {
            'f1': 0.0,
            'precision_at_10': 0.0,
            'threshold': 0.5,
        }
        self.last_validation_gate_stats = {}
        self.gate_epoch_metrics = []

        # 详细训练历史
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'val_auprc': [],
            'val_f1': [],
            'val_precision_at_10': [],
            'val_selection_score': [],
            'learning_rate': []
        }

        # ID映射器（稍后从processor设置）
        self.id_mapper = None
        self.idx_to_gene_id = None

        # 创建输出目录
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)

    def set_id_mapper(self, id_mapper, idx_to_gene_id):
        """设置ID映射器（从processor传递过来）"""
        self.id_mapper = id_mapper
        self.idx_to_gene_id = idx_to_gene_id

    def _record_gate_epoch_metrics(self, epoch, train_stats, validation_metrics):
        row = {
            'epoch': int(epoch),
            'completed_epochs': int(epoch) + 1,
        }
        row.update(train_stats)
        row.update(self.last_validation_gate_stats)
        row.update(validation_metrics)
        self.gate_epoch_metrics.append(row)
        self.save_gate_epoch_metrics()
        logger.info(
            "Gate monitor epoch %d - train SL/KG %.4f/%.4f, "
            "val SL/KG %.4f/%.4f, val entropy %.4f",
            epoch,
            row.get('train_sl_gate_mean', float('nan')),
            row.get('train_kg_gate_mean', float('nan')),
            row.get('val_sl_gate_mean', float('nan')),
            row.get('val_kg_gate_mean', float('nan')),
            row.get('val_gate_entropy_mean', float('nan')),
        )

    def save_gate_epoch_metrics(self):
        if not self.gate_epoch_metrics:
            return
        gate_path = os.path.join(
            self.config.OUTPUT_DIR, 'gate_epoch_metrics.csv'
        )
        pd.DataFrame(self.gate_epoch_metrics).to_csv(gate_path, index=False)

    def train(self):
        """训练模型"""
        # 确保所有数据在正确设备上
        self.model = self.device_manager.move_model(self.model)
        self.train_data = self.device_manager.move_data(self.train_data)
        self.val_data = self.device_manager.move_data(self.val_data)
        self.kg_data = self.device_manager.move_hetero_data(self.kg_data)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
            betas=(0.9, 0.999)
        )

        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5,
            patience=int(getattr(self.config, 'LR_SCHEDULER_PATIENCE', 2))
        )

        objective = _build_moe_objective(
            self.config, self.train_data.y, self.device_manager.target_device
        )

        # 早停机制初始化
        best_auc = 0
        best_selection_score = float('-inf')
        best_epoch = 0
        patience_counter = 0

        logger.info("开始训练...")

        # 使用tqdm进度条
        pbar = tqdm(range(self.config.EPOCHS), desc="Training")

        for epoch in pbar:
            self.model.train()
            optimizer.zero_grad()

            # 前向传播（训练数据）
            _, edge_pred, _, expert_details = self.model(
                self.train_data, self.kg_data, self.gene_mapping,
                mode='train', return_details=True,
                cancer_index=_balanced_training_cancer_index(
                    self.train_data, generator=self.sampling_generator
                ),
            )

            # 检查edge_pred中是否有NaN或Inf
            if self.config.CHECK_NAN_INF and (torch.isnan(edge_pred).any() or torch.isinf(edge_pred).any()):
                logger.warning(f"Epoch {epoch}: edge_pred contains NaN or Inf, skipping this epoch")
                self.device_manager.clear_cache()
                continue

            # 计算训练损失
            train_loss, loss_components = objective(
                edge_pred, self.train_data.y, expert_details
            )

            if torch.isnan(train_loss) or torch.isinf(train_loss):
                logger.warning(f"Epoch {epoch}: 无效的损失值，跳过")
                self.device_manager.clear_cache()
                continue

            # 反向传播
            train_loss.backward()

            # 梯度裁剪
            if self.config.USE_GRADIENT_CLIPPING:
                if self.config.GRADIENT_CLIP_VALUE > 0:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), self.config.GRADIENT_CLIP_VALUE)
                if self.config.GRADIENT_CLIP_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP_NORM)

            optimizer.step()

            self.train_losses.append(train_loss.item())
            current_lr = optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)

            monitor_interval = max(
                1, int(getattr(self.config, 'GATE_MONITOR_INTERVAL', 10))
            )
            monitor_epoch = (
                epoch % monitor_interval == 0
                or epoch == self.config.EPOCHS - 1
            )

            # 验证
            if monitor_epoch:
                try:
                    train_gate_stats = _summarize_gate_details(
                        expert_details, self.train_data.y, 'train', self.config
                    )
                    # 曲线渲染放到训练结束后的最终评估，避免每个验证点重复渲染。
                    val_loss, auc, auprc = self.evaluate(plot_curves=False)
                    self.val_losses.append(val_loss)
                    self.val_aucs.append(auc)
                    self.val_auprcs.append(auprc)

                    # 记录详细历史
                    self.train_history['epoch'].append(epoch)
                    self.train_history['train_loss'].append(train_loss.item())
                    self.train_history['val_loss'].append(val_loss)
                    self.train_history['val_auc'].append(auc)
                    self.train_history['val_auprc'].append(auprc)
                    self.train_history['val_f1'].append(
                        self.last_validation_metrics.get('f1', 0.0)
                    )
                    self.train_history['val_precision_at_10'].append(
                        self.last_validation_metrics.get('precision_at_10', 0.0)
                    )
                    selection_score = _validation_selection_score(
                        self.config, auc, auprc,
                        self.last_validation_metrics.get('f1', 0.0),
                    )
                    self.train_history['val_selection_score'].append(
                        selection_score
                    )
                    self.train_history['learning_rate'].append(current_lr)

                    self._record_gate_epoch_metrics(epoch, train_gate_stats, {
                        'val_loss': val_loss,
                        'val_auc': auc,
                        'val_aupr': auprc,
                        'val_f1': self.last_validation_metrics.get('f1', 0.0),
                        'val_precision_at_10': self.last_validation_metrics.get(
                            'precision_at_10', 0.0
                        ),
                        'val_selection_score': selection_score,
                    })

                    # 更新学习率
                    scheduler.step(selection_score)

                    # 更新进度条描述
                    pbar.set_description(
                        f"Epoch {epoch}: Train Loss={train_loss.item():.4f}, Val Loss={val_loss:.4f}, "
                        f"AUC={auc:.4f}, AUPR={auprc:.4f}, "
                        f"P@10={self.last_validation_metrics.get('precision_at_10', 0.0):.4f}")

                    # 早停机制
                    if selection_score > best_selection_score:
                        best_selection_score = selection_score
                        best_auc = auc
                        best_epoch = epoch
                        patience_counter = 0

                        # 保存最佳模型
                        merged_model_config, model_config_sha256 = (
                            _checkpoint_model_config(self.config, self.kg_data)
                        )
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'train_loss': train_loss.item(),
                            'val_loss': val_loss,
                            'auc': auc,
                            'selection_score': selection_score,
                            'auprc': auprc,
                            'aupr': auprc,
                            'f1': self.last_validation_metrics.get('f1', 0.0),
                            'precision_at_10': self.last_validation_metrics.get(
                                'precision_at_10', 0.0
                            ),
                            'train_history': self.train_history,
                            'model_config': merged_model_config,
                            'model_config_sha256': model_config_sha256,
                            'kg_schema': self.kg_schema,
                            'architecture_version': '4.0-MoE',
                            'expert_architecture': 'sl_omics_plus_hgt',
                            'pair_decoder': 'symmetric_sum_absdiff_product',
                            'cancer_vocabulary': self.cancer_vocabulary_state,
                            'train_gene_ids': list(getattr(self.model, 'train_gene_ids', [])),
                            'train_positive_pairs': list(getattr(self.model, 'train_positive_pairs', [])),
                            'loss_components': {
                                key: float(value.cpu())
                                for key, value in loss_components.items()
                            },
                        }, os.path.join(self.config.OUTPUT_DIR, 'best_model.pth'))
                    else:
                        patience_counter += 1

                    if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                        logger.info(
                            f"早停在epoch {epoch}, 最佳选择分数: "
                            f"{best_selection_score:.4f}, 对应AUC: {best_auc:.4f}"
                        )
                        break

                except Exception as e:
                    logger.error(f"Epoch {epoch}: 验证失败 - {e}")
                    continue
            else:
                # 非验证epoch，只更新损失到进度条
                pbar.set_description(f"Epoch {epoch}: Train Loss={train_loss.item():.4f}")

            # 定期清理内存
            if epoch % self.config.MEMORY_CLEANUP_FREQUENCY == 0:
                self.device_manager.clear_cache()

            # 定期绘制训练进度
            if epoch % 50 == 0 and epoch > 0 and len(self.val_aucs) > 0:
                try:
                    self.plot_training_progress(
                        os.path.join(self.config.OUTPUT_DIR, f'training_progress_epoch_{epoch}.pdf')
                    )
                except Exception as e:
                    logger.warning(f"绘制训练进度图失败: {e}")

        # 加载最佳模型
        best_model_path = os.path.join(self.config.OUTPUT_DIR, 'best_model.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device_manager.target_device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if hasattr(self.model, 'restore_train_sl_context'):
                self.model.restore_train_sl_context(
                    checkpoint.get('train_gene_ids', []),
                    checkpoint.get('train_positive_pairs', []),
                )
            logger.info(
                f"加载最佳模型 (Epoch {checkpoint['epoch']}), "
                f"选择分数: {checkpoint.get('selection_score', checkpoint['auc']):.4f}, "
                f"AUC: {checkpoint['auc']:.4f}"
            )

            # 保存注意力权重
            self.model.eval()
            with torch.no_grad():
                # 调用模型，设置 return_attention=True，mode='eval' 确保不改变模型行为
                result = self.model(
                    self.train_data, self.kg_data, self.gene_mapping,
                    mode='eval', return_attention=True
                )

                # 根据返回值解析注意力字典
                # 当 return_attention=True 时，模型应返回 (fused_embeddings, edge_pred, sl_connectivity, attention_dict)
                if isinstance(result, tuple) and len(result) == 2:
                    (fused_embeddings, edge_pred, sl_connectivity), attention_dict = result
                else:
                    logger.warning("模型未返回预期的2个返回值，请检查 return_attention 实现。")
                    attention_dict = None

                if attention_dict is not None:
                    # 将注意力字典中的所有张量移至 CPU 再保存
                    cpu_attention = {}
                    for key, value in attention_dict.items():
                        if isinstance(value, torch.Tensor):
                            cpu_attention[key] = value.cpu()
                        elif isinstance(value, tuple) and len(value) == 2:  # 处理 (edge_index, attn_weights) 元组
                            cpu_attention[key] = (value[0].cpu(), value[1].cpu())
                        else:
                            cpu_attention[key] = value  # 非张量原样保留
                    save_path = os.path.join(self.config.OUTPUT_DIR, 'attention_weights.pt')
                    torch.save(cpu_attention, save_path)
                    logger.info(f"注意力权重已保存至 {save_path}")
                else:
                    logger.warning("未获取到注意力字典，跳过保存。")

        # 最终训练过程可视化
        try:
            self.plot_training_progress(
                os.path.join(self.config.OUTPUT_DIR, 'final_training_progress.pdf')
            )
        except Exception as e:
            logger.warning(f"绘制最终训练进度图失败: {e}")

        # 保存训练历史
        self.save_training_history()

        # 训练集评估（检查过拟合）
        self.evaluate_on_train()

        # 只在非交叉验证时保存预处理模型
        # 检查输出目录是否包含'fold'，如果包含则说明是交叉验证模式
        output_dir_str = str(self.config.OUTPUT_DIR)
        if 'fold' not in output_dir_str and 'cv' not in output_dir_str.lower():
            # 单次训练模式，保存预处理模型
            if self.processor is not None:
                try:
                    # 验证processor的scaler是否已拟合
                    if hasattr(self.processor.scaler, 'n_features_in_'):
                        logger.info(f"处理器标准化器特征维度: {self.processor.scaler.n_features_in_}")
                        logger.info(f"处理器is_scaler_fitted: {self.processor.is_scaler_fitted}")
                    else:
                        logger.warning("处理器标准化器未正确拟合")

                    # 保存标准化器和PCA
                    success = self.processor.save_scaler_and_pca(self.config.OUTPUT_DIR)
                    if success:
                        logger.info("预处理模型和基因映射已保存，可用于未来推理")
                    else:
                        logger.error("保存预处理模型失败！")

                    # 保存基因映射
                    self.processor.save_gene_mapping(self.config.OUTPUT_DIR)

                except Exception as e:
                    logger.warning(f"保存预处理模型失败: {e}")
                    import traceback
                    logger.warning(traceback.format_exc())
        else:
            logger.info("交叉验证模式，跳过保存预处理模型")

        return self.train_losses, self.val_losses, self.val_aucs, self.val_auprcs


    def evaluate(self, plot_curves=True):
        """评估模型性能（验证集）"""
        self.model.eval()
        self.last_validation_gate_stats = {}
        with torch.no_grad():
            _, edge_pred, _, expert_details = self.model(
                self.val_data, self.kg_data, self.gene_mapping,
                mode='eval', return_details=True,
            )
            self.last_validation_gate_stats = _summarize_gate_details(
                expert_details, self.val_data.y, 'val', self.config
            )

            y_true = self.val_data.y.cpu().numpy()
            y_pred = torch.sigmoid(edge_pred).cpu().numpy()

            try:
                # 计算验证损失
                criterion = nn.BCEWithLogitsLoss()
                val_loss = criterion(edge_pred, self.val_data.y).item()

                # 计算评估指标
                auc = roc_auc_score(y_true, y_pred)
                auprc = average_precision_score(y_true, y_pred)
                precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)
                f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
                best_index = int(np.argmax(f1_scores[:-1])) if thresholds.size else 0
                threshold = thresholds[best_index] if thresholds.size else 0.5
                f1 = f1_score(
                    y_true, (y_pred > threshold).astype(int), zero_division=0
                )
                p_at_10 = precision_at_k(y_true, y_pred, k=10)
                self.last_validation_metrics = {
                    'f1': f1,
                    'precision_at_10': p_at_10,
                    'threshold': float(threshold),
                }
                logger.info(
                    "验证集性能 - AUC: %.4f, AUPR: %.4f, F1: %.4f, Precision@10: %.4f",
                    auc, auprc, f1, p_at_10,
                )

                # 绘制评估曲线
                if plot_curves and len(y_true) > 0 and len(np.unique(y_true)) > 1:
                    try:
                        self.plot_roc_pr_curves(y_true, y_pred,
                                                os.path.join(self.config.OUTPUT_DIR, 'val_roc_pr_curves.pdf'))
                        self.plot_confusion_matrix(y_true, y_pred,
                                                   save_path=os.path.join(self.config.OUTPUT_DIR,
                                                                          'val_confusion_matrix.pdf'))
                    except Exception as e:
                        logger.warning(f"绘制评估曲线失败: {e}")

                return val_loss, auc, auprc
            except Exception as e:
                logger.error(f"评估指标计算错误: {e}")
                return 1.0, 0.5, 0.5

    def evaluate_on_train(self):
        """评估模型在训练集上的性能（用于检查过拟合）"""
        self.model.eval()
        with torch.no_grad():
            _, edge_pred, _ = self.model(self.train_data, self.kg_data, self.gene_mapping, mode='eval')

            y_true = self.train_data.y.cpu().numpy()
            y_pred = torch.sigmoid(edge_pred).cpu().numpy()

            try:
                # 计算训练集损失
                criterion = nn.BCEWithLogitsLoss()
                train_loss = criterion(edge_pred, self.train_data.y).item()

                # 计算评估指标
                train_auc = roc_auc_score(y_true, y_pred)
                train_auprc = average_precision_score(y_true, y_pred)

                logger.info(f"训练集性能 - Loss: {train_loss:.4f}, AUC: {train_auc:.4f}, AUPRC: {train_auprc:.4f}")

                # 绘制训练集评估曲线
                try:
                    self.plot_roc_pr_curves(y_true, y_pred,
                                            os.path.join(self.config.OUTPUT_DIR, 'train_roc_pr_curves.pdf'),
                                            title_prefix="Train")
                    self.plot_confusion_matrix(y_true, y_pred,
                                               save_path=os.path.join(self.config.OUTPUT_DIR,
                                                                      'train_confusion_matrix.pdf'),
                                               title_prefix="Train")
                except Exception as e:
                    logger.warning(f"绘制训练集评估曲线失败: {e}")

                return train_loss, train_auc, train_auprc
            except Exception as e:
                logger.error(f"训练集评估指标计算错误: {e}")
                return 1.0, 0.5, 0.5

    def plot_training_progress(self, save_path=None):
        """绘制训练过程图表（更新为显示训练和验证损失）"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.flatten()

            # 1. 训练损失曲线
            if self.train_losses:
                epochs_train = list(range(len(self.train_losses)))
                axes[0].plot(epochs_train, self.train_losses, 'b-', linewidth=2, alpha=0.7, label='Train Loss')
                axes[0].set_xlabel('Epoch')
                axes[0].set_ylabel('Loss')
                axes[0].set_title('Training Loss')
                axes[0].grid(True, alpha=0.3)
                axes[0].legend()

            # 2. 验证损失曲线
            if self.val_losses and self.train_history['epoch']:
                val_epochs = self.train_history['epoch']
                axes[1].plot(val_epochs, self.val_losses, 'r-', linewidth=2, label='Val Loss')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('Loss')
                axes[1].set_title('Validation Loss')
                axes[1].grid(True, alpha=0.3)
                axes[1].legend()

            # 3. 训练和验证损失对比
            if self.train_losses and self.val_losses and self.train_history['epoch']:
                val_epochs = self.train_history['epoch']
                # 对训练损失进行采样以匹配验证epoch
                train_loss_sampled = [self.train_losses[epoch] for epoch in val_epochs if
                                      epoch < len(self.train_losses)]
                if len(train_loss_sampled) == len(val_epochs):
                    axes[2].plot(val_epochs, train_loss_sampled, 'b-', linewidth=2, label='Train Loss')
                    axes[2].plot(val_epochs, self.val_losses, 'r-', linewidth=2, label='Val Loss')
                    axes[2].set_xlabel('Epoch')
                    axes[2].set_ylabel('Loss')
                    axes[2].set_title('Train vs Validation Loss')
                    axes[2].grid(True, alpha=0.3)
                    axes[2].legend()

            # 4. 验证AUC曲线
            if self.val_aucs and self.train_history['epoch']:
                val_epochs = self.train_history['epoch']
                axes[3].plot(val_epochs, self.val_aucs, 'g-', linewidth=2, label='AUC')
                axes[3].set_xlabel('Epoch')
                axes[3].set_ylabel('AUC Score')
                axes[3].set_title('Validation AUC')
                axes[3].grid(True, alpha=0.3)
                axes[3].set_ylim(0.5, 1.0)
                axes[3].legend()

            # 5. 验证AUPRC曲线
            if self.val_auprcs and self.train_history['epoch']:
                val_epochs = self.train_history['epoch']
                axes[4].plot(val_epochs, self.val_auprcs, 'purple', linewidth=2, label='AUPRC')
                axes[4].set_xlabel('Epoch')
                axes[4].set_ylabel('AUPRC Score')
                axes[4].set_title('Validation AUPRC')
                axes[4].grid(True, alpha=0.3)
                axes[4].set_ylim(0.5, 1.0)
                axes[4].legend()

            # 6. 学习率曲线
            if self.learning_rates:
                epochs_lr = list(range(len(self.learning_rates)))
                axes[5].plot(epochs_lr, self.learning_rates, 'orange', linewidth=2)
                axes[5].set_xlabel('Epoch')
                axes[5].set_ylabel('Learning Rate')
                axes[5].set_title('Learning Rate Schedule')
                axes[5].grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"训练过程图表已保存至: {save_path}")

            plt.close()
        except Exception as e:
            logger.warning(f"绘制训练进度图失败: {e}")

    def plot_confusion_matrix(self, y_true, y_pred, threshold=0.5, save_path=None, title_prefix=""):
        """绘制混淆矩阵"""
        try:
            y_pred_binary = (y_pred > threshold).astype(int)
            cm = confusion_matrix(y_true, y_pred_binary)

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Predicted Negative', 'Predicted Positive'],
                        yticklabels=['Actual Negative', 'Actual Positive'])
            title = f'Confusion Matrix (Threshold={threshold})'
            if title_prefix:
                title = f"{title_prefix} {title}"
            plt.title(title)
            plt.ylabel('Actual')
            plt.xlabel('Predicted')

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')

            plt.close()
        except Exception as e:
            logger.warning(f"绘制混淆矩阵失败: {e}")

    def plot_roc_pr_curves(self, y_true, y_pred, save_path=None, title_prefix=""):
        """绘制ROC和PR曲线"""
        try:
            from sklearn.metrics import roc_curve, precision_recall_curve

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # ROC曲线
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            auc_score = roc_auc_score(y_true, y_pred)
            ax1.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc_score:.4f}')
            ax1.plot([0, 1], [0, 1], 'r--', linewidth=1)
            ax1.set_xlabel('False Positive Rate')
            ax1.set_ylabel('True Positive Rate')
            title1 = 'ROC Curve'
            if title_prefix:
                title1 = f"{title_prefix} {title1}"
            ax1.set_title(title1)
            ax1.legend(loc='lower right')
            ax1.grid(True, alpha=0.3)

            # PR曲线
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            auprc_score = average_precision_score(y_true, y_pred)
            ax2.plot(recall, precision, 'g-', linewidth=2, label=f'AUPRC = {auprc_score:.4f}')
            ax2.set_xlabel('Recall')
            ax2.set_ylabel('Precision')
            title2 = 'Precision-Recall Curve'
            if title_prefix:
                title2 = f"{title_prefix} {title2}"
            ax2.set_title(title2)
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')

            plt.close()
        except Exception as e:
            logger.warning(f"绘制ROC/PR曲线失败: {e}")

    def save_training_history(self):
        """保存训练历史到CSV文件"""
        history_df = pd.DataFrame(self.train_history)
        history_path = os.path.join(self.config.OUTPUT_DIR, 'training_history.csv')
        history_df.to_csv(history_path, index=False)
        logger.info(f"训练历史已保存至: {history_path}")

        # 打印训练摘要
        logger.info("\n" + "=" * 60)
        logger.info("训练摘要:")
        logger.info(f"总训练轮次: {len(self.train_losses)}")
        if len(self.train_losses) > 0:
            logger.info(f"初始训练损失: {self.train_losses[0]:.4f}")
            logger.info(f"最终训练损失: {self.train_losses[-1]:.4f}")
        if len(self.val_losses) > 0:
            logger.info(f"初始验证损失: {self.val_losses[0]:.4f}")
            logger.info(f"最终验证损失: {self.val_losses[-1]:.4f}")
        if len(self.val_aucs) > 0:
            logger.info(f"最佳验证AUC: {max(self.val_aucs):.4f}")
            logger.info(f"最佳验证AUPRC: {max(self.val_auprcs):.4f}")
        logger.info("=" * 60)

    def save_predictions(self, train_data_df, val_data_df):
        """保存训练集和验证集的预测结果"""
        try:
            self.model.eval()
            with torch.no_grad():
                # 训练集预测
                _, train_pred, _ = self.model(self.train_data, self.kg_data, self.gene_mapping, mode='eval')
                train_pred_np = torch.sigmoid(train_pred).cpu().numpy()

                # 验证集预测
                _, val_pred, _ = self.model(self.val_data, self.kg_data, self.gene_mapping, mode='eval')
                val_pred_np = torch.sigmoid(val_pred).cpu().numpy()

            # 创建训练集预测结果DataFrame
            train_results = train_data_df.copy()
            train_results['prediction'] = train_pred_np
            train_results['set'] = 'train'

            # 创建验证集预测结果DataFrame
            val_results = val_data_df.copy()
            val_results['prediction'] = val_pred_np
            val_results['set'] = 'val'

            # 合并结果
            all_results = pd.concat([train_results, val_results], ignore_index=True)

            # 保存结果
            results_path = os.path.join(self.config.OUTPUT_DIR, 'predictions_with_split.csv')
            all_results.to_csv(results_path, index=False)

            logger.info(f"预测结果已保存至: {results_path}")

            # 分析预测结果
            self.analyze_predictions(all_results)

        except Exception as e:
            logger.error(f"保存预测结果失败: {e}")

    def analyze_predictions(self, results_df):
        """分析预测结果"""
        try:
            # 按数据集分开
            train_results = results_df[results_df['set'] == 'train']
            val_results = results_df[results_df['set'] == 'val']

            # 计算最佳阈值（基于验证集）
            y_true_val = val_results['label'].values
            y_pred_val = val_results['prediction'].values

            # 找到最佳F1分数的阈值
            precisions, recalls, thresholds = precision_recall_curve(y_true_val, y_pred_val)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

            logger.info(f"最佳分类阈值: {best_threshold:.4f}")
            logger.info(f"最佳F1分数: {f1_scores[best_idx]:.4f}")

            # 应用阈值到两个数据集
            for set_name, df in [('训练集', train_results), ('验证集', val_results)]:
                y_true = df['label'].values
                y_pred = df['prediction'].values
                y_pred_binary = (y_pred > best_threshold).astype(int)

                # 计算指标
                accuracy = accuracy_score(y_true, y_pred_binary)
                precision = precision_score(y_true, y_pred_binary, zero_division=0)
                recall = recall_score(y_true, y_pred_binary, zero_division=0)
                f1 = f1_score(y_true, y_pred_binary, zero_division=0)

                logger.info(f"{set_name}指标 (阈值={best_threshold:.4f}):")
                logger.info(f"  准确率: {accuracy:.4f}")
                logger.info(f"  精确率: {precision:.4f}")
                logger.info(f"  召回率: {recall:.4f}")
                logger.info(f"  F1分数: {f1:.4f}")

                # 保存每个数据集的详细预测
                set_results = df.copy()
                set_results['prediction_binary'] = y_pred_binary
                set_results_path = os.path.join(self.config.OUTPUT_DIR, f'{set_name.lower()}_detailed_predictions.csv')
                set_results.to_csv(set_results_path, index=False)

        except Exception as e:
            logger.error(f"分析预测结果失败: {e}")


class InductiveTrainer(Trainer):
    """支持归纳式学习的训练器（修复梯度问题版本）"""

    def __init__(self, model, train_data, val_data, kg_data, gene_mapping,
                 train_gene_ids=None, all_gene_ids=None, val_gene_mapping=None,
                 val_gene_ids=None, val_kg_data=None, schema_kg_data=None, processor=None,
                 config=None, device_manager=None):
        super().__init__(model, train_data, val_data, kg_data, gene_mapping, processor, config, device_manager)

        # 专用验证KG只在no_grad前向中被消费（每GATE_MONITOR_INTERVAL轮验证一次
        # 以及训练后的校准），平时驻留CPU；编码器会在前向时按需把张量传上计算
        # 设备，因此只改变常驻位置，不改变任何计算。共享训练KG时保持原引用。
        if val_kg_data is not None:
            self.val_kg_data = val_kg_data.to('cpu')
        else:
            self.val_kg_data = self.kg_data
        schema_contexts = []
        if val_kg_data is not None:
            schema_contexts.append(self.val_kg_data)
        schema_contexts.extend(list(schema_kg_data or []))

        for schema_context in schema_contexts:
            if not hasattr(self.model, 'prepare_for_kg'):
                break
            context_schema = self.model.prepare_for_kg(schema_context)
            merged_node_dims = dict((self.kg_schema or {}).get('node_input_dims', {}))
            for node_type, input_dim in context_schema.get('node_input_dims', {}).items():
                previous_dim = merged_node_dims.get(node_type)
                if previous_dim is not None and previous_dim != input_dim:
                    raise ValueError(
                        f"KG node dimension changed for {node_type}: "
                        f"{previous_dim} vs {input_dim}"
                    )
                merged_node_dims[node_type] = input_dim
            merged_edge_types = list((self.kg_schema or {}).get('edge_types', []))
            for edge_type in context_schema.get('edge_types', []):
                if edge_type not in merged_edge_types:
                    merged_edge_types.append(edge_type)
            self.kg_schema = {
                'node_input_dims': merged_node_dims,
                'edge_types': merged_edge_types
            }
        self.model = self.device_manager.move_model(self.model)

        # Training and validation have separate node index spaces in strict mode.
        self.train_gene_ids = list(train_gene_ids) if train_gene_ids else []
        self.train_gene_ids_set = set(self.train_gene_ids) if self.train_gene_ids else None
        self.val_gene_ids = list(val_gene_ids or all_gene_ids or self.train_gene_ids)
        self.val_gene_mapping = dict(val_gene_mapping or gene_mapping)
        # Kept as an alias for plotting/checkpoint compatibility.
        self.all_gene_ids = self.val_gene_ids

        if len(self.train_gene_ids) != self.train_data.num_nodes:
            raise ValueError(
                f"Training gene IDs ({len(self.train_gene_ids)}) do not match "
                f"training nodes ({self.train_data.num_nodes})"
            )
        if len(self.val_gene_ids) != self.val_data.num_nodes:
            raise ValueError(
                f"Validation gene IDs ({len(self.val_gene_ids)}) do not match "
                f"validation nodes ({self.val_data.num_nodes})"
            )

        # 创建训练集基因掩码
        if self.all_gene_ids is not None and self.train_gene_ids_set is not None:
            self.train_gene_mask = torch.tensor(
                [gene_id in self.train_gene_ids_set for gene_id in self.all_gene_ids],
                device=self.device_manager.target_device,
                dtype=torch.bool
            )
            logger.info(f"训练集基因掩码: {self.train_gene_mask.sum().item()}/{len(self.all_gene_ids)}")
        else:
            self.train_gene_mask = None

        # 存储详细评估指标 - 初始化为空列表
        self.detailed_metrics = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'val_auprc': [],
            'val_precision_at_10': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'val_selection_score': [],
            'learning_rate': []
        }

        # 存储最佳阈值的指标
        self.best_threshold_metrics = {
            'threshold': 0.5,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'precision_at_10': 0.0,
        }

        # 确保val_losses, val_aucs, val_auprcs被正确初始化
        self.val_losses = []
        self.val_aucs = []
        self.val_auprcs = []


    def train(self):
        """训练模型"""
        # 确保所有数据在正确设备上
        self.model = self.device_manager.move_model(self.model)
        self.train_data = self.device_manager.move_data(self.train_data)
        self.val_data = self.device_manager.move_data(self.val_data)
        self.kg_data = self.device_manager.move_hetero_data(self.kg_data)
        self.val_kg_data = self.device_manager.move_hetero_data(self.val_kg_data)

        # Anomaly detection is substantially slower than ordinary NaN/Inf
        # checks, so it is opt-in through DETECT_ANOMALY.
        self._anomaly_detection_previous = torch.is_anomaly_enabled()
        if getattr(self.config, 'DETECT_ANOMALY', False):
            torch.autograd.set_detect_anomaly(True)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
            betas=(0.9, 0.999)
        )

        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5,
            patience=int(getattr(self.config, 'LR_SCHEDULER_PATIENCE', 2))
        )

        objective = _build_moe_objective(
            self.config, self.train_data.y, self.device_manager.target_device
        )

        # 早停机制
        best_auc = 0
        best_selection_score = float('-inf')
        best_epoch = 0
        patience_counter = 0

        logger.info("开始归纳式学习训练...")

        pbar = tqdm(range(self.config.EPOCHS), desc="Inductive Training")

        for epoch in pbar:
            self.model.train()
            optimizer.zero_grad()

            try:
                simulated_new_gene_mask = _sample_inductive_node_mask(
                    self.train_data, self.config, generator=self.sampling_generator
                )
                # 前向传播（传入训练集基因ID）
                _, edge_pred, _, expert_details = self.model(
                    self.train_data, self.kg_data, self.gene_mapping,
                    mode='train',
                    gene_ids=self.train_gene_ids,
                    train_gene_ids=self.train_gene_ids or None,
                    return_details=True,
                    cancer_index=_balanced_training_cancer_index(
                        self.train_data, generator=self.sampling_generator
                    ),
                    simulated_new_gene_mask=simulated_new_gene_mask,
                )

                # 检查edge_pred
                if self.config.CHECK_NAN_INF and (torch.isnan(edge_pred).any() or torch.isinf(edge_pred).any()):
                    logger.warning(f"Epoch {epoch}: edge_pred contains NaN or Inf, skipping")
                    self.device_manager.clear_cache()
                    continue

                # 计算训练损失
                train_loss, loss_components = objective(
                    edge_pred, self.train_data.y, expert_details
                )

                if torch.isnan(train_loss) or torch.isinf(train_loss):
                    logger.warning(f"Epoch {epoch}: 无效的损失值，跳过")
                    self.device_manager.clear_cache()
                    continue

                # 反向传播（添加异常处理）
                try:
                    train_loss.backward()
                except RuntimeError as e:
                    logger.error(f"Epoch {epoch}: 反向传播失败 - {e}")
                    optimizer.zero_grad(set_to_none=True)
                    self.device_manager.clear_cache()
                    raise RuntimeError(f"Epoch {epoch} backward pass failed") from e

                # 梯度裁剪
                if self.config.USE_GRADIENT_CLIPPING:
                    if self.config.GRADIENT_CLIP_VALUE > 0:
                        torch.nn.utils.clip_grad_value_(self.model.parameters(), self.config.GRADIENT_CLIP_VALUE)
                    if self.config.GRADIENT_CLIP_NORM > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP_NORM)

                optimizer.step()

                self.train_losses.append(train_loss.item())
                current_lr = optimizer.param_groups[0]['lr']
                self.learning_rates.append(current_lr)

                monitor_interval = max(
                    1, int(getattr(self.config, 'GATE_MONITOR_INTERVAL', 10))
                )
                monitor_epoch = (
                    epoch % monitor_interval == 0
                    or epoch == self.config.EPOCHS - 1
                )

                # 验证（每10个epoch）
                if monitor_epoch:
                    try:
                        train_gate_stats = _summarize_gate_details(
                            expert_details, self.train_data.y, 'train', self.config
                        )
                        # 修改：evaluate现在返回更多指标
                        val_loss, auc, auprc, accuracy, precision, recall, f1, p_at_10 = self.evaluate(plot_curves=False)

                        # 确保只记录与epoch对应的验证指标
                        self.val_losses.append(val_loss)
                        self.val_aucs.append(auc)
                        self.val_auprcs.append(auprc)

                        # 记录详细历史 - 现在所有指标都是基于同一个验证点
                        self.train_history['epoch'].append(epoch)
                        self.train_history['train_loss'].append(train_loss.item())
                        self.train_history['val_loss'].append(val_loss)
                        self.train_history['val_auc'].append(auc)
                        self.train_history['val_auprc'].append(auprc)
                        self.train_history['val_f1'].append(f1)
                        self.train_history['val_precision_at_10'].append(p_at_10)
                        selection_score = _validation_selection_score(
                            self.config, auc, auprc, f1
                        )
                        self.train_history['val_selection_score'].append(
                            selection_score
                        )
                        self.train_history['learning_rate'].append(current_lr)

                        # 新增：记录详细指标
                        self.detailed_metrics['epoch'].append(epoch)
                        self.detailed_metrics['train_loss'].append(train_loss.item())
                        self.detailed_metrics['val_loss'].append(val_loss)
                        self.detailed_metrics['val_auc'].append(auc)
                        self.detailed_metrics['val_auprc'].append(auprc)
                        self.detailed_metrics['val_accuracy'].append(accuracy)
                        self.detailed_metrics['val_precision'].append(precision)
                        self.detailed_metrics['val_recall'].append(recall)
                        self.detailed_metrics['val_f1'].append(f1)
                        self.detailed_metrics['val_precision_at_10'].append(p_at_10)
                        self.detailed_metrics['val_selection_score'].append(
                            selection_score
                        )
                        self.detailed_metrics['learning_rate'].append(current_lr)
                        self._record_gate_epoch_metrics(epoch, train_gate_stats, {
                            'val_loss': val_loss,
                            'val_auc': auc,
                            'val_aupr': auprc,
                            'val_accuracy': accuracy,
                            'val_precision': precision,
                            'val_recall': recall,
                            'val_f1': f1,
                            'val_precision_at_10': p_at_10,
                            'val_selection_score': selection_score,
                        })
                        # 更新学习率
                        scheduler.step(selection_score)

                        # 更新进度条
                        pbar.set_description(
                            f"Epoch {epoch}: Train Loss={train_loss.item():.4f}, Val Loss={val_loss:.4f}, "
                            f"AUC={auc:.4f}, AUPR={auprc:.4f}, F1={f1:.4f}, P@10={p_at_10:.4f}"
                        )

                        # 早停机制
                        if selection_score > best_selection_score:
                            best_selection_score = selection_score
                            best_auc = auc
                            best_epoch = epoch
                            patience_counter = 0

                            # 保存最佳模型
                            merged_model_config, model_config_sha256 = (
                                _checkpoint_model_config(self.config)
                            )
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'train_loss': train_loss.item(),
                                'val_loss': val_loss,
                                'auc': auc,
                                'selection_score': selection_score,
                                'auprc': auprc,
                                'aupr': auprc,
                                'accuracy': accuracy,
                                'precision': precision,
                                'recall': recall,
                                'f1': f1,
                                'precision_at_10': p_at_10,
                                'train_history': self.train_history,
                                'detailed_metrics': self.detailed_metrics,
                                'model_config': merged_model_config,
                                'model_config_sha256': model_config_sha256,
                                'train_gene_ids': self.train_gene_ids,
                                'all_gene_ids': self.all_gene_ids,
                                'val_gene_ids': self.val_gene_ids,
                                'strict_inductive': True,
                                'architecture_version': '4.0-MoE',
                                'expert_architecture': 'sl_omics_plus_hgt',
                                'pair_decoder': 'symmetric_sum_absdiff_product',
                                'cancer_vocabulary': self.cancer_vocabulary_state,
                                'train_positive_pairs': list(
                                    getattr(self.model, 'train_positive_pairs', [])
                                ),
                                'loss_components': {
                                    key: float(value.cpu())
                                    for key, value in loss_components.items()
                                },
                                'kg_schema': self.kg_schema,
                            }, os.path.join(self.config.OUTPUT_DIR, 'best_inductive_model.pth'))
                        else:
                            patience_counter += 1

                        if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                            logger.info(
                                f"早停在epoch {epoch}, 最佳选择分数: "
                                f"{best_selection_score:.4f}, 对应AUC: {best_auc:.4f}"
                            )
                            break

                    except Exception as e:
                        logger.exception(f"Epoch {epoch}: 验证失败")
                        raise RuntimeError(f"Epoch {epoch} validation failed") from e
                else:
                    pbar.set_description(f"Epoch {epoch}: Train Loss={train_loss.item():.4f}")

            except Exception as e:
                logger.exception(f"Epoch {epoch}: 训练失败")
                self.device_manager.clear_cache()
                raise RuntimeError(f"Epoch {epoch} training failed") from e

            # 定期清理内存
            if epoch % self.config.MEMORY_CLEANUP_FREQUENCY == 0:
                self.device_manager.clear_cache()

        # 加载最佳模型
        best_model_path = os.path.join(self.config.OUTPUT_DIR, 'best_inductive_model.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device_manager.target_device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if hasattr(self.model, 'restore_train_sl_context'):
                self.model.restore_train_sl_context(
                    checkpoint.get('train_gene_ids', self.train_gene_ids),
                    checkpoint.get('train_positive_pairs', []),
                )

            # 恢复训练集基因ID和所有基因ID
            if 'train_gene_ids' in checkpoint:
                self.train_gene_ids = list(checkpoint['train_gene_ids'])
                self.train_gene_ids_set = set(self.train_gene_ids)
                logger.info(f"恢复训练集基因ID集合，大小: {len(self.train_gene_ids_set)}")

            if 'val_gene_ids' in checkpoint or 'all_gene_ids' in checkpoint:
                self.val_gene_ids = list(
                    checkpoint.get('val_gene_ids', checkpoint.get('all_gene_ids', []))
                )
                self.all_gene_ids = self.val_gene_ids
                logger.info(f"恢复所有基因ID列表，大小: {len(self.all_gene_ids)}")

            if 'detailed_metrics' in checkpoint:
                self.detailed_metrics = checkpoint['detailed_metrics']

            logger.info(
                f"加载最佳归纳模型 (Epoch {checkpoint['epoch']}), "
                f"选择分数: {checkpoint.get('selection_score', checkpoint['auc']):.4f}, "
                f"AUC: {checkpoint['auc']:.4f}"
            )

            # 获取训练集基因嵌入和连接性分数
            self.model.eval()
            with torch.no_grad():
                fused_embeddings, sl_connectivity = self.model(
                    self.train_data, self.kg_data, self.gene_mapping,
                    mode='embedding_only',
                    gene_ids=self.train_gene_ids,
                    train_gene_ids=self.train_gene_ids or None,
                    refresh_train_memory=True
                )
                torch.save(fused_embeddings.cpu(), os.path.join(self.config.OUTPUT_DIR, 'train_gene_embeddings.pt'))
                torch.save(sl_connectivity.cpu(), os.path.join(self.config.OUTPUT_DIR, 'sl_connectivity.pt'))
                if self.model.train_gene_embeddings is not None:
                    checkpoint['train_gene_reference_embeddings'] = \
                        self.model.train_gene_embeddings.detach().cpu()
                    torch.save(checkpoint, best_model_path)
                logger.info("训练集基因嵌入和连接性分数已保存")

            # 保存注意力权重
            self.model.eval()
            with torch.no_grad():
                result = self.model(
                    self.train_data, self.kg_data, self.gene_mapping,
                    mode='eval', return_attention=True,
                    gene_ids=self.train_gene_ids,
                    train_gene_ids=self.train_gene_ids or None
                )
                # 当 return_attention=True 且 mode='eval' 时，应返回4个值
                if isinstance(result, tuple) and len(result) == 2:
                    (fused_embeddings, edge_pred, sl_connectivity), attention_dict = result
                else:
                    logger.warning("模型未返回预期的4个返回值，请检查 return_attention 实现。")
                    attention_dict = None

                if attention_dict is not None:
                    cpu_attention = {}
                    for key, value in attention_dict.items():
                        if isinstance(value, torch.Tensor):
                            cpu_attention[key] = value.cpu()
                        elif isinstance(value, tuple) and len(value) == 2:
                            cpu_attention[key] = (value[0].cpu(), value[1].cpu())
                        else:
                            cpu_attention[key] = value
                    save_path = os.path.join(self.config.OUTPUT_DIR, 'attention_weights.pt')
                    torch.save(cpu_attention, save_path)
                    logger.info(f"注意力权重已保存至 {save_path}")
                else:
                    logger.warning("未获取到注意力字典，跳过保存。")

        # Restore the caller's anomaly-detection state instead of globally
        # disabling it after training.
        if getattr(self.config, 'DETECT_ANOMALY', False):
            torch.autograd.set_detect_anomaly(
                self._anomaly_detection_previous
            )

        # 绘制训练过程曲线
        self.plot_inductive_training_curves()

        # 保存详细指标
        self.save_detailed_metrics()

        # 只在非交叉验证时保存预处理模型
        output_dir_str = str(self.config.OUTPUT_DIR)
        if 'fold' not in output_dir_str and 'cv' not in output_dir_str.lower():
            # 单次训练模式，保存预处理模型
            if self.processor is not None:
                try:
                    # 保存标准化器和PCA
                    self.processor.save_scaler_and_pca(self.config.OUTPUT_DIR)

                    # 保存基因映射
                    self.processor.save_gene_mapping(self.config.OUTPUT_DIR)

                    logger.info("预处理模型和基因映射已保存")
                except Exception as e:
                    logger.warning(f"保存预处理模型失败: {e}")
        else:
            logger.info("交叉验证模式，跳过保存预处理模型")


        # 最终评估
        final_val_loss, final_val_auc, final_val_auprc, final_accuracy, final_precision, final_recall, final_f1, final_p_at_10 = self.evaluate()
        logger.info(
            f"最终验证集性能: AUC={final_val_auc:.4f}, AUPR={final_val_auprc:.4f}, "
            f"F1={final_f1:.4f}, Precision@10={final_p_at_10:.4f}"
        )
        logger.info(
            f"准确率: {final_accuracy:.4f}, 精确率: {final_precision:.4f}, 召回率: {final_recall:.4f}, F1分数: {final_f1:.4f}")

        return self.train_losses, self.val_losses, self.val_aucs, self.val_auprcs

    def evaluate(self, plot_curves=True):
        """评估模型性能（支持新基因），返回详细指标"""
        self.model.eval()
        self.last_validation_gate_stats = {}

        with torch.no_grad():
            # 传入所有基因ID
            _, edge_pred, _, expert_details = self.model(
                self.val_data, self.val_kg_data, self.val_gene_mapping,
                mode='eval',
                gene_ids=self.val_gene_ids,
                return_details=True,
            )
            self.last_validation_gate_stats = _summarize_gate_details(
                expert_details, self.val_data.y, 'val', self.config
            )

            y_true = self.val_data.y.cpu().numpy()
            y_pred = torch.sigmoid(edge_pred).cpu().numpy()

            try:
                # 计算验证损失
                criterion = nn.BCEWithLogitsLoss()
                val_loss = criterion(edge_pred, self.val_data.y).item()

                # 计算评估指标
                auc = roc_auc_score(y_true, y_pred)
                auprc = average_precision_score(y_true, y_pred)

                # 新增：寻找最佳阈值
                best_threshold, best_f1 = self.find_best_threshold(y_true, y_pred)
                self.best_threshold_metrics['threshold'] = best_threshold

                # 使用最佳阈值计算分类指标
                y_pred_binary = (y_pred > best_threshold).astype(int)
                accuracy = accuracy_score(y_true, y_pred_binary)
                precision = precision_score(y_true, y_pred_binary, zero_division=0)
                recall = recall_score(y_true, y_pred_binary, zero_division=0)
                f1 = f1_score(y_true, y_pred_binary, zero_division=0)
                p_at_10 = precision_at_k(y_true, y_pred, k=10)

                self.best_threshold_metrics['accuracy'] = accuracy
                self.best_threshold_metrics['precision'] = precision
                self.best_threshold_metrics['recall'] = recall
                self.best_threshold_metrics['f1'] = f1
                self.best_threshold_metrics['precision_at_10'] = p_at_10

                # 分析新基因和已知基因的性能差异
                if self.train_gene_mask is not None and self.all_gene_ids:
                    self._analyze_performance_by_gene_type(y_true, y_pred, self.all_gene_ids)

                # 绘制评估曲线
                if plot_curves and len(y_true) > 0 and len(np.unique(y_true)) > 1:
                    self.plot_inductive_curves(y_true, y_pred, self.all_gene_ids)

                logger.info(
                    "验证集性能 - AUC: %.4f, AUPR: %.4f, F1: %.4f, Precision@10: %.4f",
                    auc, auprc, f1, p_at_10,
                )
                return val_loss, auc, auprc, accuracy, precision, recall, f1, p_at_10
            except Exception as e:
                logger.error(f"评估指标计算错误: {e}")
                return 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0

    def find_best_threshold(self, y_true, y_pred):
        """寻找最佳阈值（基于F1分数）"""
        thresholds = np.linspace(0.0, 1.0, 101)
        best_threshold = 0.5
        best_f1 = 0

        for threshold in thresholds:
            y_pred_binary = (y_pred > threshold).astype(int)
            try:
                f1 = f1_score(y_true, y_pred_binary, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            except:
                continue

        return best_threshold, best_f1

    def plot_inductive_training_curves(self):
        """绘制归纳式学习的训练过程曲线"""
        try:
            fig, axes = plt.subplots(3, 2, figsize=(16, 12))

            # 1. 训练损失曲线
            if self.train_losses:
                epochs_train = list(range(len(self.train_losses)))
                axes[0, 0].plot(epochs_train, self.train_losses, 'b-', linewidth=2, alpha=0.7, label='Train Loss')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].set_title('Training Loss Curve')
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].legend()

            # 2. 验证损失曲线 - 修复：只使用有对应验证指标的epoch
            if self.val_losses and self.detailed_metrics['epoch']:
                # 确保长度匹配
                min_len = min(len(self.val_losses), len(self.detailed_metrics['epoch']))
                val_epochs = self.detailed_metrics['epoch'][:min_len]
                val_losses = self.val_losses[:min_len]

                axes[0, 1].plot(val_epochs, val_losses, 'r-', linewidth=2, label='Val Loss')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Loss')
                axes[0, 1].set_title('Validation Loss Curve')
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].legend()

            # 3. 验证AUC曲线 - 修复：确保长度匹配
            if self.val_aucs and self.detailed_metrics['epoch']:
                min_len = min(len(self.val_aucs), len(self.detailed_metrics['epoch']))
                val_epochs = self.detailed_metrics['epoch'][:min_len]
                val_aucs = self.val_aucs[:min_len]

                axes[1, 0].plot(val_epochs, val_aucs, 'g-', linewidth=2, label='AUC')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('AUC Score')
                axes[1, 0].set_title('Validation AUC Curve')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].set_ylim(0.0, 1.0)
                axes[1, 0].legend()

            # 4. 验证AUPRC曲线 - 修复：确保长度匹配
            if self.val_auprcs and self.detailed_metrics['epoch']:
                min_len = min(len(self.val_auprcs), len(self.detailed_metrics['epoch']))
                val_epochs = self.detailed_metrics['epoch'][:min_len]
                val_auprcs = self.val_auprcs[:min_len]

                axes[1, 1].plot(val_epochs, val_auprcs, 'purple', linewidth=2, label='AUPRC')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('AUPRC Score')
                axes[1, 1].set_title('Validation AUPRC Curve')
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].set_ylim(0.0, 1.0)
                axes[1, 1].legend()

            # 5. F1分数曲线 - 修复：确保长度匹配
            if self.detailed_metrics['val_f1'] and self.detailed_metrics['epoch']:
                min_len = min(len(self.detailed_metrics['val_f1']), len(self.detailed_metrics['epoch']))
                val_epochs = self.detailed_metrics['epoch'][:min_len]
                val_f1_scores = self.detailed_metrics['val_f1'][:min_len]

                axes[2, 0].plot(val_epochs, val_f1_scores, 'orange', linewidth=2, label='F1 Score')
                axes[2, 0].set_xlabel('Epoch')
                axes[2, 0].set_ylabel('F1 Score')
                axes[2, 0].set_title('Validation F1 Score Curve')
                axes[2, 0].grid(True, alpha=0.3)
                axes[2, 0].set_ylim(0.0, 1.0)
                axes[2, 0].legend()

            # 6. 多指标对比 - 修复：确保长度匹配
            if (self.detailed_metrics['val_auc'] and self.detailed_metrics['val_auprc'] and
                    self.detailed_metrics['val_f1'] and self.detailed_metrics['epoch']):
                # 找到所有数组的最小长度
                min_len = min(
                    len(self.detailed_metrics['val_auc']),
                    len(self.detailed_metrics['val_auprc']),
                    len(self.detailed_metrics['val_f1']),
                    len(self.detailed_metrics['epoch'])
                )

                val_epochs = self.detailed_metrics['epoch'][:min_len]
                val_aucs = self.detailed_metrics['val_auc'][:min_len]
                val_auprcs = self.detailed_metrics['val_auprc'][:min_len]
                val_f1_scores = self.detailed_metrics['val_f1'][:min_len]

                axes[2, 1].plot(val_epochs, val_aucs, 'g-', linewidth=2, label='AUC')
                axes[2, 1].plot(val_epochs, val_auprcs, 'purple', linewidth=2, label='AUPRC')
                axes[2, 1].plot(val_epochs, val_f1_scores, 'orange', linewidth=2, label='F1')
                axes[2, 1].set_xlabel('Epoch')
                axes[2, 1].set_ylabel('Score')
                axes[2, 1].set_title('Validation Metrics Comparison')
                axes[2, 1].grid(True, alpha=0.3)
                axes[2, 1].set_ylim(0.0, 1.0)
                axes[2, 1].legend()

            plt.tight_layout()

            save_path = os.path.join(self.config.OUTPUT_DIR, 'inductive_training_curves.pdf')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"归纳学习训练曲线已保存至: {save_path}")

        except Exception as e:
            logger.warning(f"绘制归纳学习训练曲线失败: {e}")
            logger.error(f"详细错误信息: {traceback.format_exc()}")

    def save_detailed_metrics(self):
        """保存详细指标到CSV文件"""
        if self.detailed_metrics['epoch']:
            metrics_df = pd.DataFrame(self.detailed_metrics)
            metrics_path = os.path.join(self.config.OUTPUT_DIR, 'inductive_detailed_metrics.csv')
            metrics_df.to_csv(metrics_path, index=False)
            logger.info(f"详细训练指标已保存至: {metrics_path}")

            # 打印最佳指标
            if len(metrics_df) > 0:
                best_auc_idx = metrics_df['val_auc'].idxmax()
                best_auc_row = metrics_df.iloc[best_auc_idx]

                best_f1_idx = metrics_df['val_f1'].idxmax()
                best_f1_row = metrics_df.iloc[best_f1_idx]

                logger.info("\n" + "=" * 60)
                logger.info("最佳训练指标汇总:")
                logger.info("=" * 60)
                logger.info(f"最佳AUC - Epoch {best_auc_row['epoch']}:")
                logger.info(f"  AUC: {best_auc_row['val_auc']:.4f}")
                logger.info(f"  AUPRC: {best_auc_row['val_auprc']:.4f}")
                logger.info(f"  F1: {best_auc_row['val_f1']:.4f}")
                logger.info(f"  准确率: {best_auc_row['val_accuracy']:.4f}")
                logger.info(f"  精确率: {best_auc_row['val_precision']:.4f}")
                logger.info(f"  召回率: {best_auc_row['val_recall']:.4f}")

                logger.info(f"\n最佳F1 - Epoch {best_f1_row['epoch']}:")
                logger.info(f"  F1: {best_f1_row['val_f1']:.4f}")
                logger.info(f"  AUC: {best_f1_row['val_auc']:.4f}")
                logger.info(f"  AUPRC: {best_f1_row['val_auprc']:.4f}")
                logger.info(f"  准确率: {best_f1_row['val_accuracy']:.4f}")
                logger.info(f"  精确率: {best_f1_row['val_precision']:.4f}")
                logger.info(f"  召回率: {best_f1_row['val_recall']:.4f}")
                logger.info("=" * 60)

                # 保存最佳阈值指标
                threshold_df = pd.DataFrame([self.best_threshold_metrics])
                threshold_path = os.path.join(self.config.OUTPUT_DIR, 'best_threshold_metrics.csv')
                threshold_df.to_csv(threshold_path, index=False)
                logger.info(f"最佳阈值指标已保存至: {threshold_path}")

                logger.info(f"最佳分类阈值: {self.best_threshold_metrics['threshold']:.4f}")
                logger.info(f"基于最佳阈值的性能:")
                logger.info(f"  准确率: {self.best_threshold_metrics['accuracy']:.4f}")
                logger.info(f"  精确率: {self.best_threshold_metrics['precision']:.4f}")
                logger.info(f"  召回率: {self.best_threshold_metrics['recall']:.4f}")
                logger.info(f"  F1分数: {self.best_threshold_metrics['f1']:.4f}")

    def plot_inductive_curves(self, y_true, y_pred, all_gene_ids=None):
        """绘制归纳学习的评估曲线（包含更多指标）"""
        try:
            from sklearn.metrics import roc_curve, precision_recall_curve

            fig, axes = plt.subplots(2, 3, figsize=(18, 10))

            # 1. 整体ROC曲线
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            auc_score = roc_auc_score(y_true, y_pred)
            axes[0, 0].plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc_score:.4f}')
            axes[0, 0].plot([0, 1], [0, 1], 'r--', linewidth=1)
            axes[0, 0].set_xlabel('False Positive Rate')
            axes[0, 0].set_ylabel('True Positive Rate')
            axes[0, 0].set_title('ROC Curve')
            axes[0, 0].legend(loc='lower right')
            axes[0, 0].grid(True, alpha=0.3)

            # 2. 整体PR曲线
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            auprc_score = average_precision_score(y_true, y_pred)
            axes[0, 1].plot(recall, precision, 'g-', linewidth=2, label=f'AUPRC = {auprc_score:.4f}')
            axes[0, 1].set_xlabel('Recall')
            axes[0, 1].set_ylabel('Precision')
            axes[0, 1].set_title('Precision-Recall Curve')
            axes[0, 1].legend(loc='upper right')
            axes[0, 1].grid(True, alpha=0.3)

            # 3. 预测分数分布
            axes[0, 2].hist(y_pred, bins=50, alpha=0.7, color='blue', edgecolor='black')
            axes[0, 2].axvline(x=self.best_threshold_metrics['threshold'], color='red',
                               linestyle='--', label=f'best_threshold={self.best_threshold_metrics["threshold"]:.3f}')
            axes[0, 2].set_xlabel('Prediction Score')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].set_title('Prediction Score Distribution')
            axes[0, 2].grid(True, alpha=0.3)
            axes[0, 2].legend()

            # 4. 不同阈值下的指标变化
            thresholds = np.linspace(0.0, 1.0, 101)
            accuracies = []
            precisions = []
            recalls = []
            f1_scores = []

            for threshold in thresholds:
                y_pred_binary = (y_pred > threshold).astype(int)
                accuracies.append(accuracy_score(y_true, y_pred_binary))
                precisions.append(precision_score(y_true, y_pred_binary, zero_division=0))
                recalls.append(recall_score(y_true, y_pred_binary, zero_division=0))
                f1_scores.append(f1_score(y_true, y_pred_binary, zero_division=0))

            axes[1, 0].plot(thresholds, accuracies, 'b-', label='Accuracy')
            axes[1, 0].plot(thresholds, precisions, 'g-', label='Precision')
            axes[1, 0].plot(thresholds, recalls, 'r-', label='Recall')
            axes[1, 0].plot(thresholds, f1_scores, 'orange', label='F1 Score')
            axes[1, 0].axvline(x=self.best_threshold_metrics['threshold'], color='black',
                               linestyle='--', label='Best Threshold')
            axes[1, 0].set_xlabel('Threshold')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].set_title('Metrics vs Threshold')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
            axes[1, 0].set_ylim(0.0, 1.0)

            # 5. 混淆矩阵
            y_pred_binary = (y_pred > self.best_threshold_metrics['threshold']).astype(int)
            cm = confusion_matrix(y_true, y_pred_binary)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Predicted Negative', 'Predicted Positive'],
                        yticklabels=['Actual Negative', 'Actual Positive'], ax=axes[1, 1])
            axes[1, 1].set_title(f'Confusion Matrix (Threshold={self.best_threshold_metrics["threshold"]:.3f})')
            axes[1, 1].set_ylabel('Actual')
            axes[1, 1].set_xlabel('Predicted')

            # 6. 按基因类型分析（如果可用）
            if self.train_gene_mask is not None:
                new_gene_count = (~self.train_gene_mask).sum().item()
                known_gene_count = len(self.train_gene_mask) - new_gene_count

                labels = ['Known Genes', 'New Genes']
                sizes = [known_gene_count, new_gene_count]
                colors = ['lightblue', 'lightcoral']

                axes[1, 2].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                axes[1, 2].axis('equal')
                axes[1, 2].set_title('Gene Type Distribution')

            plt.tight_layout()

            save_path = os.path.join(self.config.OUTPUT_DIR, 'inductive_evaluation_curves.pdf')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"归纳学习评估曲线已保存至: {save_path}")

        except Exception as e:
            logger.warning(f"绘制归纳学习评估曲线失败: {e}")

    def _analyze_performance_by_gene_type(self, y_true, y_pred, all_gene_ids):
        """按基因类型（新基因/已知基因）分析性能"""
        if self.train_gene_mask is None or len(all_gene_ids) != len(self.train_gene_mask):
            return

        is_new_gene_np = ~self.train_gene_mask.cpu().numpy()

        # 计算每条边是否包含新基因
        edge_contains_new_gene = []
        edge_indices = self.val_data.edge_index.cpu().numpy().T

        for src_idx, dst_idx in edge_indices:
            if src_idx < len(is_new_gene_np) and dst_idx < len(is_new_gene_np):
                is_new = (is_new_gene_np[src_idx] or is_new_gene_np[dst_idx])
                edge_contains_new_gene.append(is_new)
            else:
                edge_contains_new_gene.append(False)

        edge_contains_new_gene = np.array(edge_contains_new_gene)

        if edge_contains_new_gene.any():
            # 新基因边的性能
            new_edges_mask = edge_contains_new_gene
            if new_edges_mask.sum() > 0:
                new_edges_y_true = y_true[new_edges_mask]
                new_edges_y_pred = y_pred[new_edges_mask]

                if len(new_edges_y_true) > 0 and len(np.unique(new_edges_y_true)) > 1:
                    try:
                        new_edges_auc = roc_auc_score(new_edges_y_true, new_edges_y_pred)
                        new_edges_auprc = average_precision_score(new_edges_y_true, new_edges_y_pred)

                        logger.info(
                            f"新基因边性能: AUC={new_edges_auc:.4f}, AUPRC={new_edges_auprc:.4f} ({new_edges_mask.sum()}条边)")
                    except Exception as e:
                        logger.warning(f"无法计算新基因边的AUC: {e}")

            # 已知基因边的性能
            known_edges_mask = ~edge_contains_new_gene
            if known_edges_mask.any():
                known_edges_y_true = y_true[known_edges_mask]
                known_edges_y_pred = y_pred[known_edges_mask]

                if len(known_edges_y_true) > 0 and len(np.unique(known_edges_y_true)) > 1:
                    try:
                        known_edges_auc = roc_auc_score(known_edges_y_true, known_edges_y_pred)
                        known_edges_auprc = average_precision_score(known_edges_y_true, known_edges_y_pred)

                        logger.info(
                            f"已知基因边性能: AUC={known_edges_auc:.4f}, AUPRC={known_edges_auprc:.4f} ({known_edges_mask.sum()}条边)")
                    except Exception as e:
                        logger.warning(f"无法计算已知基因边的AUC: {e}")
