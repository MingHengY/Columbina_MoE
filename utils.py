import os
import traceback
import random
import numpy as np
import pandas as pd
import torch
import logging
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data

from data_processor import SLDataProcessor, InductiveSLDataProcessor, UniversalDataSplitter
from models import Columbina_Model, Inductive_Columbina_Model
from train import Trainer, InductiveTrainer
from config import Config

from config import Config
from device_manager import DeviceManager


# ==================== 全局logger定义 ====================
logger = logging.getLogger(__name__)


def CM4SL_main(config=None, device_choice=None):
    """主执行流程"""
    # 设置随机种子
    torch.manual_seed(config.RANDOM_SEED if config else 42)
    np.random.seed(config.RANDOM_SEED if config else 42)
    random.seed(config.RANDOM_SEED if config else 42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_SEED if config else 42)

    if config is None:
        config = Config()

    # 设备选择逻辑
    if device_choice is None:
        # 交互式设备选择
        if torch.cuda.is_available():
            print("\n" + "=" * 60)
            print("设备选择")
            print("=" * 60)
            print(f"检测到GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print("\n请选择运行设备:")
            print("1. GPU (CUDA)")
            print("2. CPU")
            print("3. 自动选择 (如有GPU则使用GPU)")

            choice = input("请输入选择 (1/2/3, 默认3): ").strip()
            if choice == '1':
                device = 'cuda'
            elif choice == '2':
                device = 'cpu'
            else:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            print("未检测到GPU，将使用CPU运行")
            device = 'cpu'
    else:
        device = device_choice

    # 创建设备管理器
    device_manager = DeviceManager(device)
    logger.info(f"使用设备管理器，目标设备: {device_manager.target_device}")

    # 1. 数据预处理（带划分）
    logger.info("正在加载和预处理数据...")
    processor = SLDataProcessor(config=config, device_manager=device_manager)

    train_graph_data, val_graph_data, kg_data, gene_mapping, train_data, val_data = processor.create_integrated_graph_data_with_split()

    # 2. 创建模型
    logger.info("\n创建ColumbinaSL模型...")
    model = Columbina_Model(config=config, device_manager=device_manager)

    # 打印模型信息
    def safe_count_parameters(module):
        total = 0
        for param in module.parameters():
            try:
                if hasattr(param, 'is_uninitialized') and param.is_uninitialized():
                    continue
                total += param.numel()
            except Exception as e:
                logger.warning(f"统计参数时出错: {e}")
                continue
        return total

    logger.info(f"模型结构:")
    logger.info(f"- SL/omics专家参数: {safe_count_parameters(model.sl_omics_expert)}")
    logger.info(f"- HGT知识图谱专家参数: {safe_count_parameters(model.kg_expert)}")
    logger.info(f"- SL pair专家参数: {safe_count_parameters(model.sl_pair_expert)}")
    logger.info(f"- KG pair专家参数: {safe_count_parameters(model.kg_pair_expert)}")
    logger.info(f"- cancer embedding参数: {safe_count_parameters(model.cancer_embedding)}")
    logger.info(f"- SL连接性模块参数: {safe_count_parameters(model.sl_connectivity)}")
    logger.info(f"- MoE门控参数: {safe_count_parameters(model.expert_gate)}")
    total_params = safe_count_parameters(model)
    logger.info(f"模型总参数: {total_params}")

    # 3. 创建训练器
    logger.info("开始训练...")
    trainer = Trainer(model, train_graph_data, val_graph_data, kg_data, gene_mapping,
                      config=config, device_manager=device_manager)

    # 传递ID映射器到训练器
    if hasattr(processor, 'id_mapper') and hasattr(processor, 'idx_to_gene_id'):
        trainer.set_id_mapper(processor.id_mapper, processor.idx_to_gene_id)

    train_losses, val_losses, val_aucs, val_auprcs = trainer.train()

    # 4. 最终评估
    logger.info("\n最终评估:")

    # 验证集评估
    val_loss, final_val_auc, final_val_auprc = trainer.evaluate(plot_curves=True)
    logger.info(f"验证集性能:")
    logger.info(f"  Loss: {val_loss:.4f}")
    logger.info(f"  AUC: {final_val_auc:.4f}")
    logger.info(f"  AUPRC: {final_val_auprc:.4f}")

    # 训练集评估（检查过拟合）
    train_loss, train_auc, train_auprc = trainer.evaluate_on_train()
    logger.info(f"训练集性能:")
    logger.info(f"  Loss: {train_loss:.4f}")
    logger.info(f"  AUC: {train_auc:.4f}")
    logger.info(f"  AUPRC: {train_auprc:.4f}")

    # 5. 保存最终结果总结
    logger.info("\n保存结果...")
    model.eval()
    with torch.no_grad():
        # 获取所有数据的嵌入
        all_graph_data = Data(
            x=train_graph_data.x,
            edge_index=torch.cat([train_graph_data.edge_index, val_graph_data.edge_index], dim=1),
            y=torch.cat([train_graph_data.y, val_graph_data.y]),
            num_nodes=train_graph_data.num_nodes
        )

        final_embeddings, sl_connectivity = model(all_graph_data, kg_data, gene_mapping, mode='embedding_only')

    # 保存嵌入（包含ID和符号）
    if hasattr(processor, 'id_mapper') and hasattr(processor, 'idx_to_gene_id'):
        embedding_data = []
        for i in range(final_embeddings.shape[0]):
            gene_id = processor.idx_to_gene_id[i]
            symbol = processor.id_mapper.get_symbol_by_id(gene_id)

            row_data = {
                'node_index': i,
                'gene_id': gene_id,
                'gene_symbol': symbol,
                'sl_connectivity': float(sl_connectivity[i]) if sl_connectivity.dim() > 0 else float(sl_connectivity)
            }

            # 添加嵌入向量
            for j in range(final_embeddings.shape[1]):
                row_data[f'embedding_{j}'] = float(final_embeddings[i, j])

            embedding_data.append(row_data)

        embedding_df = pd.DataFrame(embedding_data)
        embedding_path = os.path.join(config.OUTPUT_DIR, 'enhanced_gene_embeddings_with_ids_split.csv')
        embedding_df.to_csv(embedding_path, index=False)
        logger.info(f"集成基因嵌入已保存至: {embedding_path}，维度: {final_embeddings.shape}")

    # 保存预测结果
    trainer.save_predictions(train_data, val_data)

    # 保存最终结果摘要
    save_final_summary(config, trainer, final_val_auc, final_val_auprc, train_auc, train_auprc)

    return processor, model, trainer, final_embeddings, sl_connectivity


def inductive_main(config=None, device_choice=None, scenario='C3'):
    """Compatibility wrapper for the canonical strict inductive entry point.

    The old implementation below predates the unified evaluation flow and did
    not produce the same test report as ``main.universal_main``.  Keep this
    public name for existing callers, but route it through the validated path.
    """
    if scenario not in {'C2', 'C3'}:
        raise ValueError("inductive_main only supports strict C2 or C3 scenarios")
    logger.warning(
        "utils.inductive_main is deprecated; forwarding to main.universal_main"
    )
    from main import universal_main
    return universal_main(
        scenario=scenario,
        config=config,
        device_choice=device_choice,
        enable_cv=False,
    )

    # Retained below only as historical source context; unreachable legacy
    # code is intentionally left in the backup rather than used at runtime.
    """归纳式学习入口；默认使用严格C3基因级划分。"""
    # 设置随机种子
    torch.manual_seed(config.RANDOM_SEED if config else 42)
    np.random.seed(config.RANDOM_SEED if config else 42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_SEED if config else 42)

    if config is None:
        config = Config()

    # 创建设备管理器
    if device_choice is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_choice

    device_manager = DeviceManager(device)
    logger.info(f"使用设备管理器，目标设备: {device_manager.target_device}")

    # 1. 数据预处理（归纳式版本）
    logger.info("正在加载和预处理归纳学习数据...")
    processor = InductiveSLDataProcessor(config=config, device_manager=device_manager)

    if scenario not in {'C2', 'C3'}:
        raise ValueError("inductive_main only supports strict C2 or C3 scenarios")

    sl_data = processor.load_sl_data()
    non_sl_data = processor.load_non_sl_data(sample_size=len(sl_data))
    balanced_data = processor.combine_and_balance_data(sl_data, non_sl_data)
    splitter = UniversalDataSplitter(config, enable_cv=False)
    train_data, val_data, test_data = splitter.split_data_by_scenario(
        balanced_data, scenario=scenario, seed=config.RANDOM_SEED
    )

    (
        train_graph_data,
        val_graph_data,
        kg_data,
        gene_mapping,
        val_gene_mapping,
        train_gene_ids,
        val_gene_ids,
        train_data,
        val_data
    ) = processor.create_strict_inductive_training_data(train_data, val_data)
    val_kg_data = processor.get_strict_stage_kg('val')
    test_kg_schema_context = processor.prepare_strict_stage_kg(
        test_data[['Gene.A', 'Gene.B']], mode='test'
    )
    all_gene_ids = val_gene_ids

    logger.info(f"训练集基因数量: {len(train_gene_ids)}")
    logger.info(f"所有基因数量: {len(all_gene_ids) if all_gene_ids else 'N/A'}")

    # 2. 创建归纳式模型
    logger.info("\n创建归纳式ColumbinaSL模型...")
    model = Inductive_Columbina_Model(config=config, device_manager=device_manager)

    # 3. 创建归纳式训练器
    logger.info("开始归纳式学习训练...")
    trainer = InductiveTrainer(
        model, train_graph_data, val_graph_data, kg_data, gene_mapping,
        train_gene_ids=train_gene_ids,
        val_gene_mapping=val_gene_mapping,
        val_gene_ids=val_gene_ids,
        val_kg_data=val_kg_data,
        schema_kg_data=[test_kg_schema_context] if test_kg_schema_context is not None else None,
        processor=processor,
        config=config, device_manager=device_manager
    )

    # 保存训练数据和验证数据
    processor.release_strict_stage_kg('test')
    del test_kg_schema_context
    device_manager.clear_cache()

    train_data_path = os.path.join(config.OUTPUT_DIR, 'inductive_train_data.csv')
    val_data_path = os.path.join(config.OUTPUT_DIR, 'inductive_val_data.csv')
    train_data.to_csv(train_data_path, index=False)
    val_data.to_csv(val_data_path, index=False)
    logger.info(f"训练数据保存至: {train_data_path}")
    logger.info(f"验证数据保存至: {val_data_path}")

    train_losses, val_losses, val_aucs, val_auprcs = trainer.train()

    # 确保绘制训练曲线
    try:
        trainer.plot_inductive_training_curves()
    except Exception as e:
        logger.error(f"绘制训练曲线失败: {e}")
        logger.error(f"详细错误: {traceback.format_exc()}")

    # 4. 详细性能分析
    logger.info("\n详细性能分析:")

    # 分析新基因预测性能
    if hasattr(processor, 'new_genes_detected') and processor.new_genes_detected:
        logger.info("分析新基因预测性能...")

        # 获取预测结果
        trainer.model.eval()
        with torch.no_grad():
            _, val_pred, _ = trainer.model(
                trainer.val_data, trainer.val_kg_data, trainer.val_gene_mapping,
                mode='eval', gene_ids=trainer.val_gene_ids
            )

            y_true = trainer.val_data.y.cpu().numpy()
            y_pred = torch.sigmoid(val_pred).cpu().numpy()

            logger.info(f"验证集总体性能: AUC={roc_auc_score(y_true, y_pred):.4f}")

    # 5. 保存结果
    logger.info("\n保存归纳式学习结果...")

    # 保存模型和结果
    # 在函数末尾，加载最佳模型进行最终评估
    best_model_path = os.path.join(config.OUTPUT_DIR, 'best_inductive_model.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device_manager.target_device)

        # 生成最终评估报告
        final_report_path = os.path.join(config.OUTPUT_DIR, 'final_evaluation_report.txt')
        with open(final_report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("归纳式学习合成致死预测模型 - 最终评估报告\n")
            f.write("=" * 80 + "\n\n")

            f.write("1. 模型配置信息:\n")
            f.write(f"   训练轮次: {config.EPOCHS}\n")
            f.write(f"   隐藏维度: {config.HIDDEN_DIM}\n")
            f.write(f"   嵌入维度: {config.EMBEDDING_DIM}\n")
            f.write(f"   学习率: {config.LEARNING_RATE}\n")
            f.write(f"   数据划分比例: {config.TRAIN_VAL_SPLIT_RATIO}\n")
            f.write(f"   最佳epoch: {checkpoint.get('epoch', 'N/A')}\n\n")

            f.write("2. 数据集统计:\n")
            f.write(f"   训练集边数: {train_graph_data.num_edges}\n")
            f.write(f"   验证集边数: {val_graph_data.num_edges}\n")
            f.write(f"   训练集基因数: {len(train_gene_ids)}\n")
            f.write(f"   总基因数: {len(all_gene_ids) if all_gene_ids else 'N/A'}\n")
            f.write(
                f"   新基因数: {len(all_gene_ids) - len(train_gene_ids) if all_gene_ids and train_gene_ids else 'N/A'}\n")
            f.write(
                f"   新基因比例: {(len(all_gene_ids) - len(train_gene_ids)) / len(all_gene_ids) * 100:.1f}% (如果适用)\n\n")

            f.write("3. 最佳模型性能指标:\n")
            f.write(f"   验证集损失: {checkpoint.get('val_loss', 'N/A'):.4f}\n")
            f.write(f"   验证集AUC: {checkpoint.get('auc', 'N/A'):.4f}\n")
            f.write(f"   验证集AUPRC: {checkpoint.get('auprc', 'N/A'):.4f}\n")
            f.write(f"   验证集准确率: {checkpoint.get('accuracy', 'N/A'):.4f}\n")
            f.write(f"   验证集精确率: {checkpoint.get('precision', 'N/A'):.4f}\n")
            f.write(f"   验证集召回率: {checkpoint.get('recall', 'N/A'):.4f}\n")
            f.write(f"   验证集F1分数: {checkpoint.get('f1', 'N/A'):.4f}\n")
            f.write(f"   最佳分类阈值: {trainer.best_threshold_metrics.get('threshold', 0.5):.4f}\n\n")

            f.write("4. 训练过程摘要:\n")
            if 'detailed_metrics' in checkpoint:
                metrics = checkpoint['detailed_metrics']
                if metrics['val_auc']:
                    best_auc = max(metrics['val_auc'])
                    best_auc_epoch = metrics['epoch'][metrics['val_auc'].index(best_auc)]
                    f.write(f"   最佳AUC: {best_auc:.4f} (epoch {best_auc_epoch})\n")

                if metrics['val_f1']:
                    best_f1 = max(metrics['val_f1'])
                    best_f1_epoch = metrics['epoch'][metrics['val_f1'].index(best_f1)]
                    f.write(f"   最佳F1: {best_f1:.4f} (epoch {best_f1_epoch})\n")

            f.write("\n5. 文件输出:\n")
            f.write("   - inductive_training_curves.png: 训练过程曲线图\n")
            f.write("   - inductive_evaluation_curves.png: 评估曲线图\n")
            f.write("   - inductive_detailed_metrics.csv: 详细训练指标\n")
            f.write("   - best_threshold_metrics.csv: 最佳阈值指标\n")
            f.write("   - best_inductive_model.pth: 最佳模型权重\n")
            f.write("   - final_inductive_model.pth: 最终模型权重\n")
            f.write("   - inductive_train_data.csv: 训练数据\n")
            f.write("   - inductive_val_data.csv: 验证数据\n")
            f.write("   - final_evaluation_report.txt: 本报告\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("报告生成完成\n")
            f.write("=" * 80 + "\n")

        logger.info(f"最终评估报告已保存至: {final_report_path}")

        # 打印最终指标摘要
        logger.info("\n" + "=" * 60)
        logger.info("最终性能指标摘要")
        logger.info("=" * 60)
        logger.info(f"验证集AUC: {checkpoint.get('auc', 'N/A'):.4f}")
        logger.info(f"验证集AUPRC: {checkpoint.get('auprc', 'N/A'):.4f}")
        logger.info(f"验证集准确率: {checkpoint.get('accuracy', 'N/A'):.4f}")
        logger.info(f"验证集精确率: {checkpoint.get('precision', 'N/A'):.4f}")
        logger.info(f"验证集召回率: {checkpoint.get('recall', 'N/A'):.4f}")
        logger.info(f"验证集F1分数: {checkpoint.get('f1', 'N/A'):.4f}")
        logger.info(f"最佳分类阈值: {trainer.best_threshold_metrics.get('threshold', 0.5):.4f}")
        logger.info("=" * 60)

    return processor, model, trainer


def generate_inductive_report(config, trainer, processor):
    """生成归纳学习报告"""
    report_path = os.path.join(config.OUTPUT_DIR, 'inductive_learning_report.txt')

    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("归纳式学习合成致死预测模型 - 结果报告\n")
        f.write("=" * 60 + "\n\n")

        f.write("模型配置:\n")
        f.write(f"  训练轮次: {config.EPOCHS}\n")
        f.write(f"  隐藏维度: {config.HIDDEN_DIM}\n")
        f.write(f"  嵌入维度: {config.EMBEDDING_DIM}\n")
        f.write(f"  数据划分比例: {config.TRAIN_VAL_SPLIT_RATIO}\n\n")

        f.write("归纳学习统计:\n")
        if hasattr(processor, 'new_genes_detected'):
            f.write(f"  检测到新基因: {processor.new_genes_detected}\n")

        if trainer.train_gene_ids_set:
            f.write(f"  训练集基因数量: {len(trainer.train_gene_ids_set)}\n")

        if trainer.all_gene_ids:
            f.write(f"  所有基因数量: {len(trainer.all_gene_ids)}\n")
            new_gene_count = len(trainer.all_gene_ids) - len(
                trainer.train_gene_ids_set) if trainer.train_gene_ids_set else 0
            f.write(f"  新基因数量: {new_gene_count}\n")
            f.write(f"  新基因比例: {new_gene_count / len(trainer.all_gene_ids) * 100:.1f}%\n\n")

        f.write("性能指标:\n")
        if len(trainer.val_aucs) > 0:
            f.write(f"  最佳验证AUC: {max(trainer.val_aucs):.4f}\n")
            f.write(f"  最佳验证AUPRC: {max(trainer.val_auprcs):.4f}\n")

        f.write("\n新基因处理策略:\n")
        f.write("  1. 基于知识图谱邻居的特征初始化\n")
        f.write("  2. 基于训练集相似基因的特征增强\n")
        f.write("  3. 归纳式GNN架构\n")

        f.write("\n生成的文件:\n")
        f.write("  1. final_inductive_model.pth - 最终模型\n")
        f.write("  2. best_inductive_model.pth - 最佳模型\n")
        f.write("  3. inductive_evaluation_curves.png - 评估曲线\n")
        f.write("  4. inductive_learning_report.txt - 本报告\n")
        f.write("  5. inductive_train_data.csv - 训练数据\n")
        f.write("  6. inductive_val_data.csv - 验证数据\n")

    logger.info(f"归纳学习报告已保存至: {report_path}")


def save_final_summary(config, trainer, val_auc, val_auprc, train_auc, train_auprc):
    """保存最终结果摘要"""
    summary_path = os.path.join(config.OUTPUT_DIR, 'final_results_summary.txt')

    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("合成致死预测模型 - 最终结果摘要\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"数据划分比例: {config.TRAIN_VAL_SPLIT_RATIO}\n")
        f.write(f"训练轮次: {config.EPOCHS}\n")
        f.write(f"随机种子: {config.RANDOM_SEED}\n\n")

        f.write("数据集大小:\n")
        f.write(f"  训练集: {trainer.train_data.num_edges}\n")
        f.write(f"  验证集: {trainer.val_data.num_edges}\n\n")

        f.write("模型性能:\n")
        f.write(f"  训练集 AUC: {train_auc:.4f}\n")
        f.write(f"  训练集 AUPRC: {train_auprc:.4f}\n")
        f.write(f"  验证集 AUC: {val_auc:.4f}\n")
        f.write(f"  验证集 AUPRC: {val_auprc:.4f}\n\n")

        f.write("知识图谱信息:\n")
        f.write(f"  基因节点数: {trainer.kg_data['Gene'].x.shape[0] if 'Gene' in trainer.kg_data.node_types else 0}\n")
        f.write(f"  边类型数: {len(trainer.kg_data.edge_types)}\n\n")

        if hasattr(trainer, 'train_history') and len(trainer.train_history['val_auc']) > 0:
            f.write("训练历史:\n")
            best_epoch = np.argmax(trainer.train_history['val_auc'])
            f.write(f"  最佳epoch: {trainer.train_history['epoch'][best_epoch]}\n")
            f.write(f"  最佳验证AUC: {trainer.train_history['val_auc'][best_epoch]:.4f}\n")
            f.write(f"  最佳验证AUPRC: {trainer.train_history['val_auprc'][best_epoch]:.4f}\n\n")

        f.write("生成的文件:\n")
        f.write("  1. best_model.pth - 最佳模型权重\n")
        f.write("  2. training_history.csv - 训练历史记录\n")
        f.write("  3. final_training_progress.png - 训练进度图\n")
        f.write("  4. val_roc_pr_curves.png - 验证集ROC/PR曲线\n")
        f.write("  5. train_roc_pr_curves.png - 训练集ROC/PR曲线\n")
        f.write("  6. enhanced_gene_embeddings_with_ids_split.csv - 基因嵌入向量\n")
        f.write("  7. predictions_with_split.csv - 预测结果\n")
        f.write("  8. train_data.csv - 训练数据集\n")
        f.write("  9. val_data.csv - 验证数据集\n")

    logger.info(f"最终结果摘要已保存至: {summary_path}")

