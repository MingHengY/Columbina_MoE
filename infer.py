"""
合成致死关系预测推理脚本
"""

import gc
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import sys
import logging
import argparse
from pathlib import Path
import warnings
from itertools import combinations


from torch_geometric.data import Data
from tqdm import tqdm
try:
    import matplotlib.pyplot as plt
except ImportError:  # plotting is optional for CLI and prediction execution
    plt = None
from itertools import combinations
try:
    import seaborn as sns
except ImportError:
    sns = None
from collections import defaultdict

# Allow both ``python infer.py`` and package-style imports without depending
# on the author's old remote checkout path.
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from config import Config
from device_manager import DeviceManager
from models import Inductive_Columbina_Model
from data_processor import InductiveSLDataProcessor
from knowledge_graph_builder import KnowledgeGraphBuilder
from gene_mapping import GeneIDMapper
from artifacts import apply_model_config, validate_deployment_manifest
from calibration import apply_temperature, load_calibration

warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('inference.log')
    ]
)
logger = logging.getLogger(__name__)


class DeviceManagerForInfer(DeviceManager):
    """修复版设备管理器，专门用于推理场景"""

    def ensure_all_on_device_inference(self, model, sl_data, kg_data):
        """
        推理专用：确保所有组件都在目标设备上

        注意：这是原训练代码中ensure_all_on_device的修复版，
        专门针对推理场景进行了优化
        """
        target_device_type = self.target_device.split(':')[0]

        # 1. 确保模型在正确设备上
        if hasattr(model, 'parameters'):
            model_device = next(model.parameters()).device
            if model_device.type != target_device_type:
                logger.warning(f"[推理] 模型在 {model_device}，移动到 {self.target_device}")
                model = self.move_model(model)

        # 2. 确保SL数据在正确设备上
        if sl_data is not None:
            # 检查多种可能的特征属性
            for attr in ['x', 'features', 'node_features', 'feat']:
                if hasattr(sl_data, attr):
                    tensor = getattr(sl_data, attr)
                    if isinstance(tensor, torch.Tensor):
                        if tensor.device.type != target_device_type:
                            logger.warning(f"[推理] SL数据{attr}在 {tensor.device}，移动到 {self.target_device}")
                            sl_data = self.move_data(sl_data)
                        break

        # 3. 确保知识图谱数据在正确设备上
        if kg_data is not None:
            # 使用修复版的移动方法
            kg_data = self.move_hetero_data_robust(kg_data)

        return model, sl_data, kg_data

    def move_hetero_data_robust(self, hetero_data):
        """
        更健壮的 HeteroData 移动方法，确保所有张量都移动到目标设备
        """
        if hetero_data is None:
            return hetero_data

        try:
            # 首先尝试内置方法
            if hasattr(hetero_data, 'to'):
                try:
                    hetero_data = hetero_data.to(self.target_device)
                    logger.info(f"[推理] 使用内置方法移动 HeteroData 到 {self.target_device}")
                    return hetero_data
                except Exception as e:
                    logger.warning(f"[推理] 内置方法失败: {e}，尝试手动移动")

            # 手动移动所有节点特征
            if hasattr(hetero_data, 'node_types'):
                for node_type in hetero_data.node_types:
                    node_store = hetero_data[node_type]
                    if hasattr(node_store, '_mapping'):
                        # 遍历所有属性
                        for key in node_store.keys():
                            tensor = node_store[key]
                            if torch.is_tensor(tensor) and tensor is not None:
                                if tensor.device.type != self.target_device.split(':')[0]:
                                    node_store[key] = tensor.to(self.target_device)
                                    logger.debug(f"[推理] 移动 {node_type}.{key} 到 {self.target_device}")

            # 手动移动所有边索引和特征
            if hasattr(hetero_data, 'edge_types'):
                for edge_type in hetero_data.edge_types:
                    edge_store = hetero_data[edge_type]
                    if hasattr(edge_store, '_mapping'):
                        for key in edge_store.keys():
                            tensor = edge_store[key]
                            if torch.is_tensor(tensor) and tensor is not None:
                                if tensor.device.type != self.target_device.split(':')[0]:
                                    edge_store[key] = tensor.to(self.target_device)
                                    logger.debug(f"[推理] 移动 {edge_type}.{key} 到 {self.target_device}")

            logger.info(f"[推理] 手动移动 HeteroData 到 {self.target_device}")

        except Exception as e:
            logger.error(f"[推理] 移动 HeteroData 失败: {e}")
            # 尝试最后的恢复：创建一个新的 HeteroData 并复制数据
            logger.warning("[推理] 尝试恢复性移动...")
            try:
                from torch_geometric.data import HeteroData
                new_hetero_data = HeteroData()

                # 复制节点数据
                if hasattr(hetero_data, 'node_types'):
                    for node_type in hetero_data.node_types:
                        node_store = hetero_data[node_type]
                        for key in node_store.keys():
                            tensor = node_store[key]
                            if torch.is_tensor(tensor):
                                new_hetero_data[node_type][key] = tensor.to(self.target_device)

                # 复制边数据
                if hasattr(hetero_data, 'edge_types'):
                    for edge_type in hetero_data.edge_types:
                        edge_store = hetero_data[edge_type]
                        for key in edge_store.keys():
                            tensor = edge_store[key]
                            if torch.is_tensor(tensor):
                                new_hetero_data[edge_type][key] = tensor.to(self.target_device)

                hetero_data = new_hetero_data
                logger.info(f"[推理] 恢复性移动完成")
            except Exception as e2:
                logger.error(f"[推理] 恢复性移动也失败: {e2}")

        return hetero_data

    def check_device_consistency(self, *args):
        """
        检查多个张量或对象是否在同一设备上

        Returns:
            bool: 所有对象都在目标设备上返回True，否则返回False
        """
        target_device_type = self.target_device.split(':')[0]

        for i, obj in enumerate(args):
            if obj is None:
                continue

            if isinstance(obj, torch.Tensor):
                if obj.device.type != target_device_type:
                    logger.warning(f"[检查] 张量 {i} 在 {obj.device}，目标设备是 {target_device_type}")
                    return False
            elif hasattr(obj, 'parameters'):
                # 模型对象
                try:
                    param_device = next(obj.parameters()).device.type
                    if param_device != target_device_type:
                        logger.warning(f"[检查] 模型参数在 {param_device}，目标设备是 {target_device_type}")
                        return False
                except StopIteration:
                    pass
            elif hasattr(obj, 'x'):
                # Data对象
                if hasattr(obj.x, 'device'):
                    if obj.x.device.type != target_device_type:
                        logger.warning(f"[检查] Data对象特征在 {obj.x.device}，目标设备是 {target_device_type}")
                        return False

        logger.info(f"[检查] 所有对象都在目标设备 {target_device_type} 上")
        return True

    def log_device_info(self, **kwargs):
        """记录设备信息"""
        logger.info("=" * 50)
        logger.info("设备信息汇总:")
        logger.info(f"目标设备: {self.target_device}")

        for name, obj in kwargs.items():
            if obj is None:
                logger.info(f"{name}: None")
            elif isinstance(obj, torch.Tensor):
                logger.info(f"{name} (Tensor): {obj.device}")
            elif hasattr(obj, 'parameters'):
                try:
                    device = next(obj.parameters()).device
                    logger.info(f"{name} (Model): {device}")
                except StopIteration:
                    logger.info(f"{name} (Model): 无参数")
            elif hasattr(obj, 'x') and hasattr(obj.x, 'device'):
                logger.info(f"{name} (Data): {obj.x.device}")
            elif hasattr(obj, 'node_types'):
                # HeteroData
                device_info = []
                for node_type in obj.node_types[:2]:  # 只检查前两种节点类型
                    if hasattr(obj[node_type], 'x') and obj[node_type].x is not None:
                        device_info.append(f"{node_type}: {obj[node_type].x.device}")
                logger.info(f"{name} (HeteroData): {', '.join(device_info) if device_info else '未知'}")

        logger.info("=" * 50)


class SLDataProcessorForInference(InductiveSLDataProcessor):
    """推理专用数据处理器"""

    def __init__(self, config=None, device_manager=None):
        super().__init__(config, device_manager)
        # 初始化基因ID映射器
        self.gene_id_mapper = GeneIDMapper(config)
        logger.info("基因ID映射器初始化完成")

    def create_gene_mapping_from_list(self, gene_list):

        logger.info(f"从基因列表创建映射: {len(gene_list)} 个基因")

        # 重置映射
        self.gene_id_to_idx = {}
        self.idx_to_gene_id = {}

        # 标准化基因ID
        standardized_genes = []
        gene_type_stats = {'symbol': 0, 'entrez_id': 0, 'unknown': 0}

        for gene in gene_list:
            gene_str = str(gene).strip()
            if not gene_str:
                continue

            # 尝试转换为Entrez ID
            entrez_id = None

            # 首先检查是否是数字（可能是Entrez ID）
            if gene_str.isdigit():
                entrez_id = gene_str
                gene_type_stats['entrez_id'] += 1
            else:
                # 尝试通过符号获取Entrez ID
                entrez_id = self.gene_id_mapper.get_id_by_symbol(gene_str)
                if entrez_id:
                    gene_type_stats['symbol'] += 1
                else:
                    # 尝试通过任何名称获取
                    entrez_id = self.gene_id_mapper.get_id_by_any_name(gene_str)
                    if entrez_id:
                        gene_type_stats['symbol'] += 1
                    else:
                        # 无法识别，保持原样
                        entrez_id = gene_str
                        gene_type_stats['unknown'] += 1
                        logger.warning(f"无法识别基因: {gene_str}")

            if entrez_id and entrez_id not in standardized_genes:
                standardized_genes.append(entrez_id)

        # 创建映射
        for idx, gene_id in enumerate(standardized_genes):
            self.gene_id_to_idx[gene_id] = idx
            self.idx_to_gene_id[idx] = gene_id

        # 统计信息
        logger.info(f"基因类型统计: 符号={gene_type_stats['symbol']}, "
                    f"Entrez ID={gene_type_stats['entrez_id']}, "
                    f"未知={gene_type_stats['unknown']}")
        logger.info(f"创建映射: {len(standardized_genes)} 个唯一基因")

        # 显示一些映射示例
        if len(standardized_genes) > 0:
            logger.info("映射示例（前5个）:")
            for i in range(min(5, len(standardized_genes))):
                gene_id = standardized_genes[i]
                symbol = self.gene_id_mapper.get_symbol_by_id(gene_id)
                gene_name = self.gene_id_mapper.get_name_by_id(gene_id)
                logger.info(f"  {i}: {gene_id} -> {symbol} ({gene_name})")

        return standardized_genes

    def analyze_gene_mapping(self, gene_list):
        """分析基因映射覆盖情况"""
        results = []

        for gene in gene_list:
            gene_str = str(gene).strip()
            if not gene_str:
                continue

            # 检查映射状态
            is_entrez_id = gene_str.isdigit()
            symbol = None
            entrez_id = None

            if is_entrez_id:
                entrez_id = gene_str
                symbol = self.gene_id_mapper.get_symbol_by_id(entrez_id)
            else:
                entrez_id = self.gene_id_mapper.get_id_by_symbol(gene_str)
                if not entrez_id:
                    entrez_id = self.gene_id_mapper.get_id_by_any_name(gene_str)
                symbol = gene_str if entrez_id else None

            results.append({
                'input': gene_str,
                'is_entrez_id': is_entrez_id,
                'entrez_id': entrez_id,
                'symbol': symbol,
                'gene_name': self.gene_id_mapper.get_name_by_id(entrez_id) if entrez_id else None,
                'mapped': entrez_id is not None
            })

        return pd.DataFrame(results)

    def load_preprocessing_models(self, preprocessing_dir):
        """加载训练时保存的预处理模型"""
        logger.info(f"加载预处理模型从: {preprocessing_dir}")

        # 加载基因映射
        if not self.load_gene_mapping(preprocessing_dir):
            logger.warning("无法加载基因映射，将使用新创建的映射")

        # 加载标准化器和PCA
        if not self.load_scaler_and_pca(preprocessing_dir):
            logger.error("无法加载标准化器和PCA模型，推理将失败！")
            raise ValueError("必须先训练模型以拟合标准化器")

        scaler = getattr(self, 'scaler', None)
        if not self.is_scaler_fitted or not hasattr(scaler, 'n_features_in_'):
            raise ValueError(
                "加载的标准化器未拟合，拒绝继续推理；请提供训练阶段保存的预处理目录"
            )
        logger.info("标准化器已验证为拟合状态")

        logger.info("预处理模型加载完成")
        return True

    def load_multi_omics_features_per_cancer(self, gene_ids, cancer_type, cell_line_to_cancer):
        """
        为指定癌症类型加载多组学特征：保留所有细胞系的列，但只填充属于该癌症的细胞系的值，其余为0
        """
        logger.info(f"加载癌症类型 {cancer_type} 的多组学特征（保留所有列，填充该癌症的值）...")
        all_features = []

        for omics_type, file_path in self.config.FEATURE_FILES.items():
            if not os.path.exists(file_path):
                logger.warning(f"文件不存在: {file_path}，跳过 {omics_type}")
                continue

            df = pd.read_csv(file_path, index_col=0)
            if df.shape[0] < df.shape[1]:
                df = df.T

            if self.config.USE_ID_ANCHORING:
                df.index = [self.gene_id_mapper.get_id_by_any_name(g) or g for g in df.index]
                df = df.groupby(df.index).mean()

            # 获取该癌症类型的细胞系列表
            cancer_cell_lines = [cl for cl, cancer in cell_line_to_cancer.items() if cancer == cancer_type]
            # 创建掩码：哪些列属于该癌症
            mask = np.isin(df.columns, cancer_cell_lines)

            # 构建特征矩阵：形状 (len(gene_ids), df.shape[1])
            feature_matrix = np.zeros((len(gene_ids), df.shape[1]), dtype=np.float32)
            for i, gid in enumerate(gene_ids):
                if gid in df.index:
                    # 提取该基因在所有细胞系的值（可能为NaN）
                    row_vals = df.loc[gid].fillna(0).values.astype(np.float32)
                    # 只保留属于该癌症的细胞系的值，其他列已经初始化为0，无需额外操作
                    # 可以直接赋值整行，因为其他列会被覆盖为原始值，但我们希望非癌症的列保持0，所以不能直接赋值整行
                    # 正确做法：只将癌症对应的列位置赋值为原始值
                    feature_matrix[i, mask] = row_vals[mask]
                # 基因缺失则保持0
            all_features.append(feature_matrix)
            logger.info(f"{omics_type}: 特征形状 {feature_matrix.shape}")

        if len(all_features) == 0:
            raise FileNotFoundError(
                "没有找到任何组学特征文件，无法为推理构造节点特征"
            )

        combined_features = np.concatenate(all_features, axis=1)
        logger.info(f"拼接后特征形状: {combined_features.shape}")

        if not hasattr(self, 'scaler') or not self.is_scaler_fitted:
            raise ValueError("标准化器未加载")

        normalized = self.scaler.transform(combined_features)
        if self.pca is not None:
            normalized = self.pca.transform(normalized)

        return torch.tensor(normalized, dtype=torch.float).to(self.device_manager.target_device)

    def create_zero_features(self, num_nodes, feature_dim=None):
        """创建零节点特征（备用，强制GPU）"""
        if feature_dim is None:
            feature_dim = int(getattr(self.config, 'OMICS_FEATURE_DIM', 262))
        logger.info(f"使用零节点特征，节点数: {num_nodes}, 维度: {feature_dim}")
        # 创建张量时直接放到目标设备
        return torch.zeros(num_nodes, feature_dim).to(self.device_manager.target_device)

    def load_multi_omics_features_for_inference(self, gene_ids):
        """
        加载全局多组学特征（与训练一致：拼接所有细胞系的所有特征）
        Args:
            gene_ids: 基因ID列表（已标准化）
        Returns:
            torch.Tensor: 节点特征矩阵，形状 (len(gene_ids), total_features)
        """
        logger.info("加载全局多组学特征（拼接所有细胞系数据）...")
        all_features = []  # 每个元素为 (num_genes, num_cell_lines_for_omics)

        for omics_type, file_path in self.config.FEATURE_FILES.items():
            if not os.path.exists(file_path):
                logger.warning(f"文件不存在: {file_path}，跳过 {omics_type}")
                continue

            df = pd.read_csv(file_path, index_col=0)
            # 确保基因为行，细胞系为列
            if df.shape[0] < df.shape[1]:
                df = df.T  # 如果基因在列中，转置

            # 转换基因索引为 Entrez ID（如果使用 ID 锚定）
            if self.config.USE_ID_ANCHORING:
                df.index = [self.gene_id_mapper.get_id_by_any_name(g) or g for g in df.index]
                # 合并重复基因（取均值）
                df = df.groupby(df.index).mean()

            # 构建特征矩阵：对于每个基因，提取所有细胞系的值
            # 初始化全零矩阵，然后填充
            feature_matrix = np.zeros((len(gene_ids), df.shape[1]), dtype=np.float32)
            for i, gid in enumerate(gene_ids):
                if gid in df.index:
                    # 取出该基因所有细胞系的值（可能为NaN），填充0
                    vals = df.loc[gid].fillna(0).values.astype(np.float32)
                    feature_matrix[i, :] = vals
                # 基因缺失则保持0（已初始化为0）
            all_features.append(feature_matrix)
            logger.info(f"{omics_type}: 特征形状 {feature_matrix.shape}")

        if len(all_features) == 0:
            raise FileNotFoundError(
                "没有找到任何组学特征文件，无法为推理构造节点特征"
            )

        # 按列拼接所有组学特征
        combined_features = np.concatenate(all_features, axis=1)
        logger.info(f"拼接后特征形状: {combined_features.shape}")

        # 检查标准化器是否已加载
        if not hasattr(self, 'scaler') or not self.is_scaler_fitted:
            raise ValueError("标准化器未加载，请先调用 load_preprocessing_models")

        # 应用标准化和PCA
        normalized = self.scaler.transform(combined_features)
        if self.pca is not None:
            normalized = self.pca.transform(normalized)

        return torch.tensor(normalized, dtype=torch.float).to(self.device_manager.target_device)


class SLInference:
    """合成致死关系推理器（修复版）"""

    def __init__(self, model_path=None, config_path=None, device='auto', preprocessing_dir=None):
        """
        初始化推理器

        Args:
            model_path: 模型权重文件路径
            config_path: 配置文件路径
            device: 运行设备 ('cuda', 'cpu', 或 'auto')
            preprocessing_dir: 预处理模型目录（包含scaler.pkl, pca.pkl等）
        """
        # 创建设备管理器
        self.device_manager = DeviceManagerForInfer(device)
        logger.info(f"使用设备: {self.device_manager.target_device}")

        # 清理缓存
        self.device_manager.clear_cache()

        # 加载配置
        self.config = Config()
        if config_path:
            # JSON 只覆盖非路径的模型参数；数据/输出路径保持本机配置。
            import json
            with open(config_path, 'r', encoding='utf-8') as handle:
                config_payload = json.load(handle)
            apply_model_config(
                self.config,
                config_payload.get('config', config_payload),
            )

        # 设置预处理目录
        self.preprocessing_dir = preprocessing_dir
        if preprocessing_dir is None:
            # 默认使用C3场景的输出目录
            self.preprocessing_dir = os.path.join(
                str(self.config.OUTPUT_DIR),
                "C3"
            )
            logger.info(f"使用默认预处理目录: {self.preprocessing_dir}")

        # 加载模型
        self.model = self._load_model(model_path)

        # 初始化处理器
        self.processor = SLDataProcessorForInference(
            config=self.config,
            device_manager=self.device_manager
        )
        if getattr(self, 'cancer_vocabulary_state', None):
            self.processor.cancer_vocabulary.load_state_dict(
                self.cancer_vocabulary_state
            )

        # 知识图谱相关
        self.kg_data = None
        self.gene_mapping = None
        self.node_features = None

        self.precomputed_fused_embeddings = None
        self.precomputed_sl_connectivity = None
        self.gene_id_to_idx = None  # 从 gene_mapping 映射而来

    def _load_model(self, model_path):
        """加载训练好的模型"""
        if model_path is None:
            # 默认使用C3场景的最佳模型
            default_model_path = os.path.join(
                str(self.config.OUTPUT_DIR),
                "C3",
                "C3_cv",
                "best_overall_model.pth"
            )

            # 如果交叉验证模型不存在，尝试单次训练模型
            if not os.path.exists(default_model_path):
                default_model_path = os.path.join(
                    str(self.config.OUTPUT_DIR),
                    "C3",
                    "best_inductive_model.pth"
                )

            if os.path.exists(default_model_path):
                model_path = default_model_path
                logger.info(f"使用默认模型: {model_path}")
            else:
                raise FileNotFoundError("未找到预训练模型，请指定模型路径")

        # 创建模型实例
        model = Inductive_Columbina_Model(
            config=self.config,
            device_manager=self.device_manager
        )

        # 加载模型权重
        logger.info(f"加载模型权重: {model_path}")
        manifest_path = os.path.join(
            os.path.dirname(model_path), 'deployment_manifest.json'
        )
        if os.path.isfile(manifest_path):
            manifest = validate_deployment_manifest(manifest_path)
            if os.path.basename(model_path) != manifest.get('checkpoint'):
                raise ValueError(
                    "模型路径与 deployment manifest 中的 checkpoint 不一致"
                )
            logger.info(f"部署清单校验通过: {manifest_path}")
        checkpoint = torch.load(
            model_path,
            map_location=self.device_manager.target_device
        )
        checkpoint_config = checkpoint.get('model_config', {}) \
            if isinstance(checkpoint, dict) else {}
        checkpoint_feature_dim = checkpoint_config.get('omics_feature_dim')
        configured_feature_dim = int(
            getattr(self.config, 'OMICS_FEATURE_DIM', 262)
        )
        if (
            checkpoint_feature_dim is not None
            and int(checkpoint_feature_dim) != configured_feature_dim
        ):
            raise ValueError(
                "模型与当前配置的组学特征维度不一致: "
                f"checkpoint={checkpoint_feature_dim}, "
                f"config={configured_feature_dim}"
            )
        checkpoint_kg_dim = checkpoint_config.get('kg_feature_dim')
        configured_kg_dim = int(getattr(self.config, 'KG_FEATURE_DIM', 768))
        if (
            checkpoint_kg_dim is not None
            and int(checkpoint_kg_dim) != configured_kg_dim
        ):
            raise ValueError(
                "模型与当前配置的 KG 特征维度不一致: "
                f"checkpoint={checkpoint_kg_dim}, config={configured_kg_dim}"
            )
        # 维度强校验通过后，用 checkpoint 内的全量配置快照恢复其余训练参数。
        if isinstance(checkpoint, dict):
            apply_model_config(self.config, checkpoint.get('model_config'))
            architecture = checkpoint.get('architecture_version')
            if architecture and architecture != self.config.ARCHITECTURE_VERSION:
                raise ValueError(
                    f"checkpoint架构为 {architecture}，运行时配置为 "
                    f"{self.config.ARCHITECTURE_VERSION}"
                )
        self.cancer_vocabulary_state = checkpoint.get('cancer_vocabulary') \
            if isinstance(checkpoint, dict) else None

        # 先按训练时保存的schema物化KG层，再加载包含这些层的权重。
        if isinstance(checkpoint, dict) and checkpoint.get('kg_schema'):
            model.prepare_kg_schema(checkpoint['kg_schema'])

        # 处理不同的checkpoint格式
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        # 设置模型为评估模式
        model.eval()

        # 如果有训练集基因信息，恢复它
        if 'train_gene_ids' in checkpoint:
            train_gene_ids = checkpoint['train_gene_ids']
            all_gene_ids = checkpoint.get('all_gene_ids', [])

            self.train_gene_ids = train_gene_ids
            self.all_gene_ids = all_gene_ids
            model.restore_train_sl_context(
                train_gene_ids,
                checkpoint.get('train_positive_pairs', []),
            )
            logger.info(f"恢复训练集基因: {len(train_gene_ids)}个")

        reference_embeddings = checkpoint.get('train_gene_reference_embeddings')
        if reference_embeddings is not None:
            model.train_gene_embeddings = reference_embeddings.to(
                self.device_manager.target_device
            )
            logger.info(
                f"恢复训练基因参考嵌入: {model.train_gene_embeddings.shape[0]}个"
            )

        # 校准参数：优先读取与权重同目录的 calibration.json，并与 checkpoint
        # 内嵌温度交叉校验。REQUIRE_CALIBRATION 开启时缺少校准即拒绝推理。
        self.calibration = (
            checkpoint.get('posthoc_calibration')
            if isinstance(checkpoint, dict) else None
        )
        calibration_path = os.path.join(
            os.path.dirname(model_path), 'calibration.json'
        )
        if os.path.isfile(calibration_path):
            file_calibration = load_calibration(calibration_path)
            if self.calibration and not np.isclose(
                float(self.calibration['temperature']),
                float(file_calibration['temperature']),
            ):
                raise ValueError("checkpoint与 calibration.json 的温度不一致")
            self.calibration = file_calibration
            logger.info(
                f"温度缩放校准已加载: T={self.calibration['temperature']:.4f}"
            )
        if not self.calibration:
            if getattr(self.config, 'REQUIRE_CALIBRATION', False):
                raise ValueError(
                    "缺少验证集温度缩放参数，REQUIRE_CALIBRATION=True 时拒绝输出未校准概率"
                )
            logger.warning(
                "未找到校准参数，输出未校准概率（温度=1.0）"
            )

        logger.info("模型加载完成")
        return model

    def _calibrated_scores(self, logits):
        """把原始 logits 转成概率；有校准参数时输出温度缩放后的概率。"""
        logits_np = logits.detach().cpu().numpy() if torch.is_tensor(
            logits
        ) else np.asarray(logits)
        if self.calibration:
            return apply_temperature(logits_np, self.calibration)
        return 1.0 / (1.0 + np.exp(-np.clip(logits_np, -60.0, 60.0)))

    def prepare_inference(self, gene_list, rebuild_kg=False):
        """
        准备推理环境
        """
        logger.info("准备推理环境...")

        # 清理缓存
        self.device_manager.clear_cache()

        # 1. 加载预处理模型（标准化器、PCA等）
        if self.preprocessing_dir:
            try:
                self.processor.load_preprocessing_models(self.preprocessing_dir)
                logger.info("预处理模型加载成功")
            except Exception as e:
                raise RuntimeError(
                    "加载训练阶段预处理模型失败，无法进行可信推理"
                ) from e
        else:
            raise ValueError(
                "推理必须提供包含 scaler.pkl 的 preprocessing_dir"
            )

        # 2. 创建基因映射
        standardized_genes = self.processor.create_gene_mapping_from_list(gene_list)
        self.gene_mapping = self.processor.gene_id_to_idx

        logger.info(f"基因映射创建完成: {len(standardized_genes)}个基因")

        # 3. 生成映射报告
        mapping_report = self.processor.analyze_gene_mapping(gene_list)
        mapped_count = mapping_report['mapped'].sum()
        logger.info(f"基因映射成功率: {mapped_count}/{len(gene_list)} ({mapped_count / len(gene_list) * 100:.1f}%)")

        # 4. 加载节点特征
        self._load_node_features(standardized_genes)

        # 5. 构建知识图谱
        if rebuild_kg or self.kg_data is None:
            logger.info("构建知识图谱...")
            kg_builder = KnowledgeGraphBuilder(
                edges_file=self.config.KG_EDGES_FILE,
                nodes_file=self.config.KG_NODES_FILE,
                string_file=self.config.STRING_FILE,
                local_model_path=self.config.LOCAL_MODEL_PATH,
                use_local_model=self.config.USE_PRETRAINED,
                config=self.config,
                device_manager=self.device_manager
            )

            # 为给定的基因构建知识图谱
            self.kg_data, kg_gene_mapping = kg_builder.create_knowledge_graph_data(standardized_genes)

            # 更新基因映射以包含知识图谱中的基因
            for gene_id, idx in kg_gene_mapping.items():
                if gene_id not in self.gene_mapping:
                    new_idx = len(self.gene_mapping)
                    self.gene_mapping[gene_id] = new_idx
                    self.processor.idx_to_gene_id[new_idx] = gene_id

            logger.info(f"知识图谱构建完成，包含 {len(self.kg_data['Gene'])} 个基因节点")

        # 6. 确保所有数据在正确设备上（使用修复版方法）
        logger.info("检查并确保所有数据在正确设备上...")
        self.model, self.node_features, self.kg_data = self.device_manager.ensure_all_on_device_inference(
            self.model, self.node_features, self.kg_data
        )

        # 记录设备信息
        self.device_manager.log_device_info(
            model=self.model,
            node_features=self.node_features,
            kg_data=self.kg_data
        )

        # 7. 如果模型有记忆训练集基因的功能，需要设置
        if hasattr(self.model, 'train_gene_ids_set') and hasattr(self, 'train_gene_ids'):
            # 设置训练集基因
            self.model.train_gene_ids_set = set(self.train_gene_ids)
            logger.info(f"设置模型记忆的训练集基因: {len(self.train_gene_ids)}个")

        logger.info("推理环境准备完成")
        return standardized_genes

    def _load_node_features(self, gene_ids):
        """加载节点特征"""
        logger.info("加载多组学特征...")

        try:
            scaler = getattr(self.processor, 'scaler', None)
            if (
                not getattr(self.processor, 'is_scaler_fitted', False)
                or not hasattr(scaler, 'n_features_in_')
            ):
                raise ValueError(
                    "标准化器未拟合；请先在训练输出目录加载预处理模型"
                )

            # 使用推理专用的特征加载方法
            self.node_features = self.processor.load_multi_omics_features_for_inference(gene_ids)

            logger.info(f"节点特征加载完成，维度: {self.node_features.shape}")
            expected_dim = int(getattr(
                self.config, 'OMICS_FEATURE_DIM', 262
            ))
            if self.node_features.ndim != 2 or self.node_features.shape[1] != expected_dim:
                raise ValueError(
                    "推理特征维度与模型不一致: "
                    f"actual={tuple(self.node_features.shape)}, "
                    f"expected=(*, {expected_dim})"
                )

            # 确保特征在正确设备上
            self.node_features = self.device_manager.move_data(self.node_features)

        except Exception as e:
            logger.error(f"加载节点特征失败: {e}")
            raise RuntimeError(
                "节点特征加载失败，已停止推理以避免使用伪造特征"
            ) from e

    def predict_gene_pairs(self, gene_pairs, batch_size=256):
        """
        预测基因对的合成致死关系
        """

        # 如果已经预计算了嵌入，直接使用快速预测
        if self.precomputed_fused_embeddings is not None and self.precomputed_sl_connectivity is not None:
            return self._predict_with_precomputed(gene_pairs, batch_size)

        # 否则走原有逻辑（完整前向传播）
        logger.info(f"预测 {len(gene_pairs)} 对基因...")

        # 确保推理环境已准备
        if self.processor is None or self.kg_data is None:
            raise ValueError("请先调用prepare_inference准备推理环境")

        # 确保已加载节点特征
        if self.node_features is None:
            raise ValueError("节点特征未加载，请先调用prepare_inference")

        # 清理缓存
        self.device_manager.clear_cache()

        # 检查设备一致性
        logger.info("检查设备一致性...")
        if not self.device_manager.check_device_consistency(
                self.model, self.node_features, self.kg_data
        ):
            logger.warning("设备不一致，尝试修复...")
            self.model, self.node_features, self.kg_data = self.device_manager.ensure_all_on_device_inference(
                self.model, self.node_features, self.kg_data
            )

        all_results = []

        # 分批处理以避免内存溢出
        for i in tqdm(range(0, len(gene_pairs), batch_size), desc="批量预测"):
            batch_pairs = gene_pairs[i:i + batch_size]

            try:
                # 为当前批次创建图数据
                batch_df = pd.DataFrame(batch_pairs, columns=['Gene.A', 'Gene.B'])
                batch_df['label'] = 0  # 占位标签

                # 创建图数据
                batch_graph_data = self.processor._create_single_graph_data_with_features(
                    batch_df,
                    self.node_features,
                    mode='inference'
                )

                # 确保数据在正确设备上
                batch_graph_data = self.device_manager.move_data(batch_graph_data)

                # 确保知识图谱数据在正确设备上（使用修复版方法）
                self.kg_data = self.device_manager.move_hetero_data_robust(self.kg_data)

                # 最终设备检查
                self.device_manager.log_device_info(
                    model=self.model,
                    batch_data=batch_graph_data,
                    kg_data=self.kg_data
                )

                # 获取基因ID列表（从映射中）
                gene_ids = list(self.gene_mapping.keys())

                # 推理
                with torch.no_grad():
                    _, predictions, _ = self.model(
                        batch_graph_data,
                        self.kg_data,
                        self.gene_mapping,
                        mode='eval',
                        gene_ids=gene_ids
                    )

                # 处理预测结果
                pred_scores = self._calibrated_scores(predictions)

                for j, (gene_a, gene_b) in enumerate(batch_pairs):
                    if j < len(pred_scores):
                        # 获取基因符号（用于可读性）
                        symbol_a = self.processor.gene_id_mapper.get_symbol_by_id(gene_a) or gene_a
                        symbol_b = self.processor.gene_id_mapper.get_symbol_by_id(gene_b) or gene_b

                        all_results.append({
                            'Gene.A': gene_a,
                            'Gene.B': gene_b,
                            'Gene.A_Symbol': symbol_a,
                            'Gene.B_Symbol': symbol_b,
                            'Gene.A_Name': self.processor.gene_id_mapper.get_name_by_id(gene_a),
                            'Gene.B_Name': self.processor.gene_id_mapper.get_name_by_id(gene_b),
                            'prediction_score': float(pred_scores[j]),
                            'confidence': self._score_to_confidence(pred_scores[j])
                        })

            except Exception as e:
                logger.error(f"批次 {i // batch_size + 1} 预测失败: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # 清理缓存并继续
                self.device_manager.clear_cache()
                continue

        # 清理缓存
        self.device_manager.clear_cache()

        # 创建结果DataFrame
        if len(all_results) == 0:
            logger.warning("没有生成任何预测结果")
            # 返回包含正确列的空DataFrame，避免KeyError
            results_df = pd.DataFrame(columns=[
                'Gene.A', 'Gene.B', 'Gene.A_Symbol', 'Gene.B_Symbol',
                'Gene.A_Name', 'Gene.B_Name', 'prediction_score', 'confidence'
            ])
        else:
            results_df = pd.DataFrame(all_results)
            if len(results_df) > 0:
                results_df = self._add_priority_score(results_df, 'prediction_score')
            # 按预测分数排序
            results_df = results_df.sort_values('prediction_score', ascending=False)

        return results_df

    def predict_all_pairs(self, gene_list, min_score=0.5):
        """
        预测给定基因列表中所有可能的基因对

        Args:
            gene_list: 基因ID列表
            min_score: 最小预测分数阈值

        Returns:
            预测结果DataFrame
        """
        # 生成所有可能的基因对（排除自环）
        gene_pairs = list(combinations(gene_list, 2))
        logger.info(f"生成 {len(gene_pairs)} 个可能的基因对")

        # 预测所有基因对
        results_df = self.predict_gene_pairs(gene_pairs)

        # 过滤低分数结果
        filtered_df = results_df[results_df['prediction_score'] >= min_score]
        logger.info(f"过滤后保留 {len(filtered_df)} 个高置信度结果 (分数 ≥ {min_score})")

        return filtered_df

    def _check_device_consistency(self):
        """检查设备一致性"""
        target_device = self.device_manager.target_device
        target_device_type = target_device.split(':')[0]

        logger.info("检查设备一致性...")

        # 检查模型设备
        model_device = next(self.model.parameters()).device
        if model_device.type != target_device_type:
            logger.warning(f"模型在 {model_device}，但目标设备是 {target_device}")
            self.model = self.device_manager.move_model(self.model)

        # 检查节点特征设备
        if self.node_features is not None:
            if self.node_features.device.type != target_device_type:
                logger.warning(f"节点特征在 {self.node_features.device}，移动到 {target_device}")
                self.node_features = self.node_features.to(target_device)

        # 检查知识图谱数据设备
        if self.kg_data is not None:
            # 检查kg_data的所有张量
            for node_type in self.kg_data.node_types:
                node_data = self.kg_data[node_type]
                for key, value in node_data.items():
                    if torch.is_tensor(value):
                        if value.device.type != target_device_type:
                            logger.warning(f"kg_data[{node_type}][{key}] 在 {value.device}")

    def _log_device_info(self, batch_graph_data, kg_data):
        """记录设备信息"""
        logger.info("设备信息:")
        logger.info(f"  模型参数设备: {next(self.model.parameters()).device}")
        logger.info(f"  节点特征设备: {self.node_features.device if self.node_features is not None else 'None'}")

        if hasattr(batch_graph_data, 'x'):
            logger.info(f"  批处理图数据x设备: {batch_graph_data.x.device}")
        if hasattr(batch_graph_data, 'edge_index'):
            logger.info(f"  批处理图数据edge_index设备: {batch_graph_data.edge_index.device}")

        # 检查kg_data的关键张量
        if hasattr(kg_data, 'node_types'):
            for node_type in kg_data.node_types[:3]:  # 只检查前3种节点类型
                if hasattr(kg_data[node_type], 'x') and kg_data[node_type].x is not None:
                    logger.info(f"  kg_data[{node_type}].x设备: {kg_data[node_type].x.device}")
                break

    def _score_to_confidence(self, score):
        """将预测分数转换为置信度级别"""
        if score >= 0.9:
            return "非常高"
        elif score >= 0.7:
            return "高"
        elif score >= 0.5:
            return "中等"
        elif score >= 0.3:
            return "低"
        else:
            return "非常低"

    def analyze_predictions(self, results_df, top_k=50, save_dir=None):
        """
        分析预测结果（支持多列预测分数）
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # 检测预测列
        pred_cols = [col for col in results_df.columns if col.startswith('prediction_')]
        if not pred_cols:
            pred_cols = ['prediction_score']  # 兼容旧版单列

        # 基本统计
        print("\n" + "=" * 60)
        print("预测结果统计")
        print("=" * 60)
        for col in pred_cols:
            cancer = col.replace('prediction_', '') if col != 'prediction_score' else 'global'
            scores = results_df[col].dropna()
            if len(scores) == 0:
                continue
            print(f"{cancer}: 总数 {len(scores)}, 均值 {scores.mean():.4f}, 标准差 {scores.std():.4f}")
            print(f"  高置信度 (≥0.7): {(scores >= 0.7).sum()} ({((scores >= 0.7).sum()/len(scores))*100:.1f}%)")

        # 显示 Top-K（按第一个预测列排序）
        if pred_cols:
            sort_col = pred_cols[0]
            top_results = results_df.nlargest(top_k, sort_col)
            print(f"\nTop-{top_k} 潜在合成致死基因对 (按 {sort_col} 排序):")
            print("-" * 80)
            for i, (_, row) in enumerate(top_results.iterrows(), 1):
                gene_a = row['Gene.A_Symbol'] if row['Gene.A_Symbol'] else row['Gene.A']
                gene_b = row['Gene.B_Symbol'] if row['Gene.B_Symbol'] else row['Gene.B']
                scores_str = " | ".join([f"{col}: {row[col]:.4f}" for col in pred_cols[:3]])  # 只显示前3个
                print(f"{i:3d}. {gene_a:10s} - {gene_b:10s}: {scores_str}")

        # 可视化：如果有多列，绘制箱线图
        if len(pred_cols) > 1 and save_dir:
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 6))
                data_to_plot = [results_df[col].dropna().values for col in pred_cols]
                plt.boxplot(data_to_plot, labels=[col.replace('prediction_', '') for col in pred_cols])
                plt.ylabel('Prediction Score')
                plt.title('Prediction Score Distribution by Cancer Type')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, 'cancer_comparison_boxplot.png'), dpi=300)
                plt.close()
                logger.info(f"癌症类型比较箱线图已保存")
            except Exception as e:
                logger.warning(f"绘制箱线图失败: {e}")

        # 保存结果
        if save_dir:
            # 保存所有结果
            all_results_path = os.path.join(save_dir, "all_predictions.csv")
            results_df.to_csv(all_results_path, index=False)
            logger.info(f"所有预测结果已保存至: {all_results_path}")

            # 保存Top-K结果
            top_results_path = os.path.join(save_dir, f"top_{top_k}_predictions.csv")
            top_results.to_csv(top_results_path, index=False)
            logger.info(f"Top-{top_k} 结果已保存至: {top_results_path}")

            # 生成原有可视化（可选）
            try:
                self._visualize_results(results_df, save_dir)
            except Exception as e:
                logger.warning(f"原有可视化失败: {e}")

            # 生成详细报告
            self._generate_report(results_df, save_dir, top_k)

    def _visualize_results(self, results_df, save_dir):
        """可视化预测结果"""
        try:
            # 修复：使用正确的样式名称
            # 尝试多种可能的 seaborn 样式名称
            possible_styles = ['seaborn-v0_8', 'seaborn-whitegrid', 'seaborn', 'ggplot', 'default']
            style_used = None

            for style in possible_styles:
                try:
                    plt.style.use(style)
                    style_used = style
                    logger.info(f"使用样式: {style}")
                    break
                except:
                    continue

            if style_used is None:
                plt.style.use('default')
                logger.warning("无法设置任何样式，使用默认样式")

            # 1. 预测分数分布
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # 预测分数直方图
            axes[0, 0].hist(results_df['prediction_score'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='阈值=0.5')
            axes[0, 0].axvline(x=0.7, color='orange', linestyle='--', alpha=0.7, label='阈值=0.7')
            axes[0, 0].set_xlabel('预测分数')
            axes[0, 0].set_ylabel('频率')
            axes[0, 0].set_title('预测分数分布')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # 置信度级别分布
            confidence_counts = results_df['confidence'].value_counts()
            colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
            axes[0, 1].pie(confidence_counts.values, labels=confidence_counts.index,
                           autopct='%1.1f%%', colors=colors[:len(confidence_counts)])
            axes[0, 1].set_title('置信度级别分布')

            # Top-20基因对分数
            top_20 = results_df.head(20)
            axes[1, 0].barh(range(len(top_20)), top_20['prediction_score'].values, color='lightcoral')
            axes[1, 0].set_yticks(range(len(top_20)))
            axes[1, 0].set_yticklabels(
                [f"{row['Gene.A_Symbol']}-{row['Gene.B_Symbol']}" for _, row in top_20.iterrows()],
                fontsize=8
            )
            axes[1, 0].set_xlabel('预测分数')
            axes[1, 0].set_title('Top-20 基因对预测分数')
            axes[1, 0].invert_yaxis()  # 最高的在顶部

            # 累计分布函数
            sorted_scores = np.sort(results_df['prediction_score'])
            cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
            axes[1, 1].plot(sorted_scores, cdf, 'b-', linewidth=2)
            axes[1, 1].fill_between(sorted_scores, cdf, alpha=0.3)
            axes[1, 1].set_xlabel('预测分数')
            axes[1, 1].set_ylabel('累计概率')
            axes[1, 1].set_title('预测分数累计分布')
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'prediction_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"可视化图表已保存至: {os.path.join(save_dir, 'prediction_analysis.png')}")

        except Exception as e:
            logger.warning(f"可视化失败: {e}")
            # 记录详细错误信息
            import traceback
            logger.warning(traceback.format_exc())

    def _generate_report(self, results_df, save_dir, top_k):
        """生成详细报告"""
        report_path = os.path.join(save_dir, 'prediction_report.txt')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("合成致死关系预测报告\n")
            f.write("=" * 80 + "\n\n")

            f.write("1. 预测概要\n")
            f.write(f"   总预测基因对数量: {len(results_df)}\n")
            f.write(f"   预测时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"   设备: {self.device_manager.target_device}\n")
            f.write(f"   预处理目录: {self.preprocessing_dir}\n\n")

            f.write("2. 置信度统计\n")
            if len(results_df) > 0:
                for conf_level in ['非常高', '高', '中等', '低', '非常低']:
                    count = len(results_df[results_df['confidence'] == conf_level])
                    if count > 0:
                        percentage = count / len(results_df) * 100
                        f.write(f"   {conf_level}: {count} 对 ({percentage:.1f}%)\n")
            else:
                f.write("   无预测结果\n\n")

            f.write("\n3. Top-K 潜在合成致死基因对\n")
            f.write("-" * 80 + "\n")
            if len(results_df) > 0:
                top_results = results_df.head(top_k)
                for i, (_, row) in enumerate(top_results.iterrows(), 1):
                    gene_a = row['Gene.A_Symbol'] if row['Gene.A_Symbol'] else row['Gene.A']
                    gene_b = row['Gene.B_Symbol'] if row['Gene.B_Symbol'] else row['Gene.B']
                    f.write(f"{i:3d}. {gene_a:10s} - {gene_b:10s}: "
                            f"{row['prediction_score']:.4f} ({row['confidence']})\n")
            else:
                f.write("无预测结果\n")

            f.write("\n4. 建议\n")
            if len(results_df) > 0:
                f.write("   - 高置信度结果 (≥0.7): 建议优先进行实验验证\n")
                f.write("   - 中等置信度结果 (0.5-0.7): 可作为候选进行进一步研究\n")
                f.write("   - 低置信度结果 (<0.5): 建议谨慎对待，可能需要更多证据\n")
            else:
                f.write("   - 未发现高置信度的合成致死关系\n")
                f.write("   - 建议降低预测阈值或检查输入基因\n")

            f.write("\n5. 输出文件\n")
            if len(results_df) > 0:
                f.write(f"   所有预测结果: {os.path.join(save_dir, 'all_predictions.csv')}\n")
                f.write(f"   Top-K 预测结果: {os.path.join(save_dir, f'top_{top_k}_predictions.csv')}\n")
                f.write(f"   可视化图表: {os.path.join(save_dir, 'prediction_analysis.png')}\n")
            else:
                f.write(f"   无预测结果文件\n")
            f.write(f"   日志文件: inference.log\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("报告生成完成\n")
            f.write("=" * 80 + "\n")

        logger.info(f"详细报告已保存至: {report_path}")

    def predict_gene_pairs_per_cancer(self, gene_pairs, cancer_types, cell_line_to_cancer, batch_size=256):
        """
        对每种癌症类型分别预测基因对的合成致死关系。

        Args:
            gene_pairs: list of (gene_a, gene_b)
            cancer_types: list of cancer type strings
            cell_line_to_cancer: dict, cell line ID -> cancer type
            batch_size: 批处理大小

        Returns:
            DataFrame with columns: Gene.A, Gene.B, Gene.A_Symbol, Gene.B_Symbol, prediction_<cancer1>, ...
        """
        logger.info(f"按癌症类型预测，共 {len(cancer_types)} 种癌症")

        # 获取所有基因的ID列表（从映射中）
        all_gene_ids = list(self.gene_mapping.keys())

        # 为每种癌症类型存储预测结果（列表，每个元素是 numpy 数组，形状 (len(gene_pairs),)）
        cancer_predictions = {cancer: [] for cancer in cancer_types}

        # 分批处理基因对
        num_batches = (len(gene_pairs) + batch_size - 1) // batch_size

        for cancer in cancer_types:
            logger.info(f"处理癌症类型: {cancer}")
            # 为该癌症类型加载节点特征
            node_features = self.processor.load_multi_omics_features_per_cancer(
                all_gene_ids, cancer, cell_line_to_cancer
            )
            node_features = self.device_manager.move_tensor(node_features)

            # 分批预测
            all_preds = []
            for i in range(num_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, len(gene_pairs))
                batch_pairs = gene_pairs[start:end]

                batch_df = pd.DataFrame(batch_pairs, columns=['Gene.A', 'Gene.B'])
                batch_df['label'] = 0
                batch_df['cancer'] = cancer
                batch_graph = self.processor._create_single_graph_data_with_features(
                    batch_df, node_features, mode='inference'
                )
                batch_graph = self.device_manager.move_data(batch_graph)

                # 确保知识图谱数据在设备上
                self.kg_data = self.device_manager.move_hetero_data_robust(self.kg_data)

                with torch.no_grad():
                    _, preds, _ = self.model(
                        batch_graph, self.kg_data, self.gene_mapping,
                        mode='eval', gene_ids=all_gene_ids
                    )
                all_preds.append(self._calibrated_scores(preds))

            cancer_predictions[cancer] = np.concatenate(all_preds, axis=0)
            logger.info(f"{cancer} 预测完成，共 {len(cancer_predictions[cancer])} 条")

        # 构建最终 DataFrame
        results = []
        for i, (gene_a, gene_b) in enumerate(gene_pairs):
            row = {
                'Gene.A': gene_a,
                'Gene.B': gene_b,
                'Gene.A_Symbol': self.processor.gene_id_mapper.get_symbol_by_id(gene_a),
                'Gene.B_Symbol': self.processor.gene_id_mapper.get_symbol_by_id(gene_b),
            }
            for cancer in cancer_types:
                row[f'prediction_{cancer}'] = float(cancer_predictions[cancer][i])
            results.append(row)

        df = pd.DataFrame(results)

        # 为每个癌症列生成 priority_score
        for cancer in cancer_types:
            pred_col = f'prediction_{cancer}'
            if pred_col in df.columns:
                eps = 1e-8
                p = df[pred_col].clip(eps, 1 - eps)
                df[f'priority_score_{cancer}'] = -np.log(1 - p)

        return df

    def precompute_embeddings(self, batch_size=1000):
        """
        预计算所有基因节点的融合嵌入和连接性分数，分批处理避免显存溢出。
        batch_size: 每批处理的节点数（不是边数），默认1000。
        """
        if self.node_features is None or self.kg_data is None:
            raise RuntimeError("请先调用 prepare_inference 准备数据")
        logger.info(f"预计算所有基因节点的融合嵌入（分批处理，每批 {batch_size} 节点）...")
        self.model.eval()
        # 确保所有数据在正确设备上
        self.model, self.node_features, self.kg_data = self.device_manager.ensure_all_on_device_inference(
            self.model, self.node_features, self.kg_data
        )
        gene_ids = list(self.gene_mapping.keys())
        self.gene_id_to_idx = {gid: idx for idx, gid in enumerate(gene_ids)}  # 建立索引映射

        # 1. 先获取所有节点的 sl_embeddings 和 knowledge_embeddings
        # 创建一个空的SL图数据（无边），仅用于获取基础嵌入
        empty_edge_index = torch.tensor([[], []], dtype=torch.long, device=self.device_manager.target_device)
        sl_data = Data(
            x=self.node_features,
            edge_index=empty_edge_index,
            y=torch.tensor([], dtype=torch.float, device=self.device_manager.target_device),
            num_nodes=len(gene_ids)
        )
        with torch.no_grad():
            sl_embeddings, knowledge_embeddings = self.model(
                sl_data, self.kg_data, self.gene_mapping,
                mode='embedding',  # 假设模型支持 mode='embedding' 返回 (sl_emb, kg_emb)
                gene_ids=gene_ids
            )
        # sl_embeddings, knowledge_embeddings 形状均为 [num_nodes, embedding_dim]

        # 2. 分批计算融合嵌入
        num_nodes = sl_embeddings.shape[0]
        all_fused = []
        all_connectivity = []
        for i in range(0, num_nodes, batch_size):
            end = min(i + batch_size, num_nodes)
            batch_sl = sl_embeddings[i:end]
            batch_kg = knowledge_embeddings[i:end]
            batch_omics = self.node_features[i:end]  # 原始组学特征

            # 调用融合模块
            fused_batch = self.model.multimodal_fusion(batch_sl, batch_kg, batch_omics)
            # 计算连接性
            conn_batch = self.model.sl_connectivity(fused_batch).squeeze()
            all_fused.append(fused_batch)
            all_connectivity.append(conn_batch)
            logger.debug(f"处理节点批次 [{i}:{end}] 完成")

        self.precomputed_fused_embeddings = torch.cat(all_fused, dim=0)
        self.precomputed_sl_connectivity = torch.cat(all_connectivity, dim=0)
        logger.info(
            f"预计算完成: 融合嵌入 {self.precomputed_fused_embeddings.shape}, 连接性 {self.precomputed_sl_connectivity.shape}")

    def _predict_with_precomputed(self, gene_pairs, batch_size=256):
        """
        使用预计算的融合嵌入和连接性分数快速预测边。
        """
        if self.precomputed_fused_embeddings is None or self.precomputed_sl_connectivity is None:
            raise RuntimeError("请先调用 precompute_embeddings()")
        logger.info(f"使用预计算嵌入预测 {len(gene_pairs)} 对基因...")
        device = self.device_manager.target_device
        fused = self.precomputed_fused_embeddings.to(device)
        connectivity = self.precomputed_sl_connectivity.to(device)
        idx_map = self.gene_id_to_idx  # 已在 precompute_embeddings 中建立

        results = []
        # 分批处理基因对
        for i in tqdm(range(0, len(gene_pairs), batch_size), desc="批量边预测"):
            batch_pairs = gene_pairs[i:i + batch_size]
            src_indices = []
            dst_indices = []
            valid_pairs = []
            for g1, g2 in batch_pairs:
                if g1 in idx_map and g2 in idx_map:
                    src_indices.append(idx_map[g1])
                    dst_indices.append(idx_map[g2])
                    valid_pairs.append((g1, g2))
            if not src_indices:
                continue

            edge_index = torch.tensor(
                [src_indices, dst_indices], dtype=torch.long, device=device
            )
            batch_graph = Data(
                x=self.node_features.to(device),
                edge_index=edge_index,
                y=torch.zeros(len(src_indices), dtype=torch.float, device=device),
                num_nodes=self.node_features.shape[0],
            )
            batch_graph.cancer_index = torch.ones(
                len(src_indices), dtype=torch.long, device=device
            )
            with torch.no_grad():
                _, pred_logits, _ = self.model(
                    batch_graph, self.kg_data, self.gene_mapping,
                    mode='eval', gene_ids=list(self.gene_mapping.keys())
                )
                pred_scores = self._calibrated_scores(
                    pred_logits.reshape(-1)
                )

            # 组装结果
            for j, (g1, g2) in enumerate(valid_pairs):
                results.append({
                    'Gene.A': g1,
                    'Gene.B': g2,
                    'Gene.A_Symbol': self.processor.gene_id_mapper.get_symbol_by_id(g1) or g1,
                    'Gene.B_Symbol': self.processor.gene_id_mapper.get_symbol_by_id(g2) or g2,
                    'Gene.A_Name': self.processor.gene_id_mapper.get_name_by_id(g1),
                    'Gene.B_Name': self.processor.gene_id_mapper.get_name_by_id(g2),
                    'prediction_score': float(pred_scores[j]),
                    'confidence': self._score_to_confidence(pred_scores[j])
                })

        if not results:
            return pd.DataFrame(columns=['Gene.A', 'Gene.B', 'Gene.A_Symbol', 'Gene.B_Symbol',
                                         'Gene.A_Name', 'Gene.B_Name', 'prediction_score', 'confidence'])
        df = pd.DataFrame(results)
        if len(df) > 0:
            df = self._add_priority_score(df, 'prediction_score')
        df = df.sort_values('prediction_score', ascending=False)
        logger.info(f"预测完成，生成 {len(df)} 条结果")
        return df

    def _add_priority_score(self, df, pred_col='prediction_score'):
        """为 DataFrame 添加 priority_score = -log(1 - p)"""
        eps = 1e-8
        p = df[pred_col].clip(eps, 1 - eps)
        df['priority_score'] = -np.log(1 - p)
        return df



def main():
    """主函数"""
    from itertools import combinations


    parser = argparse.ArgumentParser(description='合成致死关系预测推理')
    parser.add_argument('--gene-list', type=str, default=None,
                        help='基因列表文件路径（支持.txt, .csv, .xlsx格式）。若不提供，则必须提供 --gene-pair-file')
    parser.add_argument('--model-path', type=str, default=None,
                        help='模型权重文件路径（默认使用C3场景最佳模型）')
    parser.add_argument('--output-dir', type=str, default='predictions',
                        help='输出目录（默认: predictions）')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='运行设备（默认: auto）')
    parser.add_argument('--top-k', type=int, default=50,
                        help='显示前K个结果（默认: 50）')
    parser.add_argument('--min-score', type=float, default=0.3,
                        help='最小预测分数阈值（默认: 0.3）')
    parser.add_argument('--rebuild-kg', action='store_true',
                        help='重新构建知识图谱')
    parser.add_argument('--gene-pair-file', type=str, default=None,
                        help='基因对文件（CSV格式，包含Gene.A和Gene.B列）')
    parser.add_argument('--save-mapping', action='store_true',
                        help='保存基因映射报告')
    parser.add_argument('--preprocessing-dir', type=str, default=None,
                        help='预处理模型目录（包含scaler.pkl, pca.pkl等）')
    parser.add_argument('--cancer-types', type=str, default=None,
                        help='癌症类型列表，逗号分隔（如 "LUAD,BRCA"）或 "all" 表示所有类型')
    args = parser.parse_args()

    def load_gene_list(file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"基因列表文件不存在: {file_path}")
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                with open(file_path, 'r') as f:
                    gene_ids = [line.strip() for line in f if line.strip()]
                return gene_ids
            gene_id_columns = ['GeneID', 'gene_id', 'Gene_ID', 'geneid', 'ID', 'id',
                               'Symbol', 'symbol', 'Gene', 'gene']
            for col in gene_id_columns:
                if col in df.columns:
                    gene_ids = df[col].dropna().astype(str).tolist()
                    logger.info(f"从列 '{col}' 读取 {len(gene_ids)} 个基因")
                    return gene_ids
            gene_ids = df.iloc[:, 0].dropna().astype(str).tolist()
            logger.info(f"从第一列读取 {len(gene_ids)} 个基因")
            return gene_ids
        except Exception as e:
            logger.error(f"加载基因列表失败: {e}")
            raise

    print("\n" + "=" * 60)
    print("合成致死关系预测推理")
    print("=" * 60)

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()

        # ---------- 确定基因列表 ----------
        if args.gene_pair_file:
            logger.info(f"从基因对文件提取唯一基因: {args.gene_pair_file}")
            try:
                gene_pairs_df = pd.read_csv(args.gene_pair_file)
                if 'Gene.A' not in gene_pairs_df.columns or 'Gene.B' not in gene_pairs_df.columns:
                    raise ValueError("基因对文件必须包含 'Gene.A' 和 'Gene.B' 列")
                all_genes = pd.concat([gene_pairs_df['Gene.A'], gene_pairs_df['Gene.B']]).unique()
                gene_list = [str(g).strip() for g in all_genes if str(g).strip()]
                logger.info(f"从基因对文件中提取到 {len(gene_list)} 个唯一基因")
                if args.gene_list:
                    logger.warning("同时提供了 --gene-list 和 --gene-pair-file，将忽略 --gene-list")
            except Exception as e:
                logger.error(f"读取基因对文件失败: {e}")
                sys.exit(1)
        elif args.gene_list:
            logger.info(f"加载基因列表: {args.gene_list}")
            gene_list = load_gene_list(args.gene_list)
        else:
            logger.error("必须提供 --gene-list 或 --gene-pair-file 之一")
            sys.exit(1)

        if len(gene_list) < 2:
            logger.error("基因数量不足2个，无法进行有效预测")
            sys.exit(1)

        print(f"加载基因: {len(gene_list)} 个")

        if args.gene_list and not args.gene_pair_file and len(gene_list) > 500:
            logger.warning(f"基因列表较大 ({len(gene_list)}个)，预测时间可能较长")
            gene_pair_count = len(list(combinations(gene_list, 2)))
            print(f"提示: 输入基因数量为 {len(gene_list)} 个，生成基因对数量: {gene_pair_count}")
            response = input("是否继续预测？(y/n): ").strip().lower()
            if response != 'y':
                print("预测取消")
                return

        # ---------- 细胞系映射 ----------
        cell_line_map_path = 'depmap_model.csv'
        cell_line_to_cancer = None
        if os.path.exists(cell_line_map_path):
            try:
                map_df = pd.read_csv(cell_line_map_path)
                if 'ModelID' in map_df.columns and 'TCGA_Code' in map_df.columns:
                    cell_line_to_cancer = dict(zip(map_df['ModelID'], map_df['TCGA_Code']))
                    logger.info(f"加载了 {len(cell_line_to_cancer)} 个细胞系映射")
                else:
                    logger.error("映射文件必须包含 ModelID 和 TCGA_Code 列")
                    cell_line_to_cancer = None
            except Exception as e:
                logger.error(f"加载细胞系映射失败: {e}")
                cell_line_to_cancer = None
        else:
            logger.warning(f"细胞系映射文件 {cell_line_map_path} 不存在，将无法使用癌症类型预测功能")

        cancer_types = None
        if args.cancer_types:
            if args.cancer_types.lower() == 'all':
                if cell_line_to_cancer is not None:
                    unique_cancers = set(cell_line_to_cancer.values())
                    cancer_types = sorted(list(unique_cancers))
                    logger.info(f"从映射文件中获取到 {len(cancer_types)} 种癌症类型")
                else:
                    logger.error("无法获取所有癌症类型：细胞系映射未正确加载")
                    sys.exit(1)
            else:
                cancer_types = [c.strip() for c in args.cancer_types.split(',')]
                logger.info(f"将预测以下癌症类型: {cancer_types}")

        # ---------- 初始化推理器 ----------
        logger.info("初始化推理器...")
        inference = SLInference(
            model_path=args.model_path,
            device=args.device,
            preprocessing_dir=args.preprocessing_dir
        )
        standardized_genes = inference.prepare_inference(gene_list, rebuild_kg=args.rebuild_kg)

        if args.save_mapping:
            mapping_df = inference.processor.analyze_gene_mapping(gene_list)
            mapping_path = os.path.join(args.output_dir, 'gene_mapping_report.csv')
            os.makedirs(args.output_dir, exist_ok=True)
            mapping_df.to_csv(mapping_path, index=False)
            logger.info(f"基因映射报告已保存至: {mapping_path}")

        # ---------- 预测 ----------
        if cancer_types is not None and cell_line_to_cancer is not None:
            # ===== 分癌症预测 =====
            if args.gene_pair_file:
                logger.info(f"从文件读取基因对: {args.gene_pair_file}")
                gene_pairs_df = pd.read_csv(args.gene_pair_file)
                if 'Gene.A' not in gene_pairs_df.columns and 'gene_a' in gene_pairs_df.columns:
                    gene_pairs_df.rename(columns={'gene_a': 'Gene.A', 'gene_b': 'Gene.B'}, inplace=True)
                if 'Gene.A' not in gene_pairs_df.columns:
                    raise ValueError("基因对文件必须包含 'Gene.A' 和 'Gene.B' 列")
                mapper = inference.processor.gene_id_mapper
                valid_pairs = []
                unmapped = set()
                for _, row in gene_pairs_df.iterrows():
                    a_raw = str(row['Gene.A']).strip()
                    b_raw = str(row['Gene.B']).strip()
                    a_std = mapper.get_id_by_any_name(a_raw)
                    if a_std is None:
                        a_std = a_raw
                        unmapped.add(a_raw)
                    b_std = mapper.get_id_by_any_name(b_raw)
                    if b_std is None:
                        b_std = b_raw
                        unmapped.add(b_raw)
                    if a_std in inference.gene_mapping and b_std in inference.gene_mapping:
                        valid_pairs.append((a_std, b_std))
                if unmapped:
                    logger.warning(f"以下基因无法映射: {unmapped}")
                if len(valid_pairs) == 0:
                    logger.error("没有有效的基因对可预测")
                    sys.exit(1)
                gene_pairs = valid_pairs
            else:
                if len(standardized_genes) < 2:
                    logger.error("基因数量不足2个，无法生成基因对")
                    sys.exit(1)
                gene_pairs = list(combinations(standardized_genes, 2))
                logger.info(f"生成 {len(gene_pairs)} 个基因对")
            results_df = inference.predict_gene_pairs_per_cancer(gene_pairs, cancer_types, cell_line_to_cancer, batch_size=256)
        else:
            # ===== 全局预测 =====
            if args.gene_pair_file:
                logger.info(f"从文件读取基因对: {args.gene_pair_file}")
                gene_pairs_df = pd.read_csv(args.gene_pair_file)
                if 'Gene.A' not in gene_pairs_df.columns and 'gene_a' in gene_pairs_df.columns:
                    gene_pairs_df.rename(columns={'gene_a': 'Gene.A', 'gene_b': 'Gene.B'}, inplace=True)
                if 'Gene.A' not in gene_pairs_df.columns:
                    raise ValueError("基因对文件必须包含 'Gene.A' 和 'Gene.B' 列")
                mapper = inference.processor.gene_id_mapper
                valid_pairs = []
                unmapped = set()
                for _, row in gene_pairs_df.iterrows():
                    a_raw = str(row['Gene.A']).strip()
                    b_raw = str(row['Gene.B']).strip()
                    a_std = mapper.get_id_by_any_name(a_raw)
                    if a_std is None:
                        a_std = a_raw
                        unmapped.add(a_raw)
                    b_std = mapper.get_id_by_any_name(b_raw)
                    if b_std is None:
                        b_std = b_raw
                        unmapped.add(b_raw)
                    if a_std in inference.gene_mapping and b_std in inference.gene_mapping:
                        valid_pairs.append((a_std, b_std))
                if unmapped:
                    logger.warning(f"以下基因无法映射: {unmapped}")
                if len(valid_pairs) == 0:
                    logger.error("没有有效的基因对可预测")
                    sys.exit(1)
                if len(valid_pairs) < len(gene_pairs_df):
                    logger.info(f"有效基因对数量: {len(valid_pairs)} / {len(gene_pairs_df)}")
                results_df = inference.predict_gene_pairs(valid_pairs)
            else:
                logger.info("未提供基因对文件，预测所有可能组合...")
                results_df = inference.predict_all_pairs(standardized_genes, min_score=args.min_score)

        # ---------- 结果分析 ----------
        logger.info("分析预测结果...")
        inference.analyze_predictions(results_df, top_k=args.top_k, save_dir=args.output_dir)

        print("\n" + "=" * 60)
        print("推理完成!")
        print("=" * 60)

        if len(results_df) > 0:
            pred_cols = [col for col in results_df.columns if col.startswith('prediction_')]
            if pred_cols:
                best_col = pred_cols[0]
                best_row = results_df.loc[results_df[best_col].idxmax()]
                gene_a = best_row['Gene.A_Symbol'] if best_row['Gene.A_Symbol'] else best_row['Gene.A']
                gene_b = best_row['Gene.B_Symbol'] if best_row['Gene.B_Symbol'] else best_row['Gene.B']
                print(f"\n最高置信度结果 (按 {best_col}):")
                print(f"{gene_a} - {gene_b}: {best_row[best_col]:.4f}")
            else:
                best_row = results_df.iloc[0]
                gene_a = best_row['Gene.A_Symbol'] if best_row['Gene.A_Symbol'] else best_row['Gene.A']
                gene_b = best_row['Gene.B_Symbol'] if best_row['Gene.B_Symbol'] else best_row['Gene.B']
                print(f"\n最高置信度结果:")
                print(f"{gene_a} - {gene_b}: {best_row['prediction_score']:.4f}")

        print(f"\n建议:")
        if len(results_df) > 0:
            print(f"1. 查看 {args.output_dir}/prediction_report.txt 获取详细报告")
            print(f"2. 使用 {args.output_dir}/all_predictions.csv 进行进一步分析")
            print(f"3. 高置信度结果建议进行实验验证")
        else:
            print(f"1. 未发现高置信度的合成致死关系")
            print(f"2. 建议降低 --min-score 参数值重新尝试")
            print(f"3. 检查预处理目录是否正确: {args.preprocessing_dir or inference.preprocessing_dir}")

        if args.save_mapping:
            print(f"4. 基因映射报告: {args.output_dir}/gene_mapping_report.csv")

    except Exception as e:
        logger.error(f"推理过程出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
