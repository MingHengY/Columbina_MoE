任务

  合成致死性 (Synthetic Lethality, SL) 预测 —— 判断两个基因是否具有"单独敲除可存活、同时敲除致死"的关系。这是癌症靶向药
  发现的核心问题：找到可被药物同时抑制的基因对，选择性杀死癌细胞。

  模型支持三种评估场景：
  - C1（直推式）：测试基因在训练中见过，泛化到新的 SL 关系
  - C2（归纳式-新基因）：测试集包含训练中完全未见过的基因
  - C3（归纳式-新基因对）：更严格，基因对的两个成员都可能未见

  ---
  架构

  Columbina 4.0-MoE（Mixture of Experts），核心思路是融合两个互补的专家：

  组学特征(5种) + SL图 ──► SLomicsExpert (GraphSAGE + MLP)
                                                      │
  知识图谱(HGT) ─────────► HGTKnowledgeEncoder         ├──► ExpertGate ──► 最终预测
                                │                      │    (softmax门控)
                                ▼                      │
                           SymmetricPairExpert ────────┘
                           (顺序无关解码器 + CancerFiLM)

  关键模块（models.py）：

  ┌─────────────────────┬───────────────────────────────────────────────────────────────────┐
  │        模块         │                               作用                                │
  ├─────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ SLomicsExpert       │ 从基因表达/CNV/突变等 262 维特征 + SL 正边 GraphSAGE 编码基因表征 │
  ├─────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ HGTKnowledgeEncoder │ 在基因-通路-疾病-药物异质图上运行 HGTConv，捕获全局知识           │
  ├─────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ SymmetricPairExpert │ 顺序无关解码器，输入 [a+b, |a-b|, a*b, ...]，保证 SL 对称性       │
  ├─────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ CancerFiLM          │ 特征线性调制，用癌症类型嵌入条件化解码器                          │
  ├─────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ ExpertGate          │ 2 层 MLP，根据覆盖率/可靠性动态融合 SL 和 KG 两个专家的输出       │
  └─────────────────────┴───────────────────────────────────────────────────────────────────┘

  训练技巧：
  - 伪归纳掩蔽：C2/C3 训练时以 50% 概率随机掩蔽部分训练节点的 SL 边，模拟新基因场景
  - 多目标损失：L = L_final + 0.25*L_sl + 0.25*L_kg + 0.10*L_rank + 0.10*L_route + 0.05*L_sym
  - 10 折交叉验证 + 严格的数据泄露隔离（split_protocol.py 保证 C2/C3 划分正确性）
  - 模型选择基于 0.4*AUC + 0.4*AUPR + 0.2*F1

  数据流：原始 CSV → InductiveSLDataProcessor（基因映射 + 癌症类型词汇 + 组学加载）→ KnowledgeGraphBuilder（构建 HGT
  异质图）→ SplitProtocol（无泄漏划分）→ InductiveTrainer → 评估/推理

  运行约束（4.0-MoE）：
  - 训练和交叉验证统一从 `main.py` 的 `universal_main` 进入；旧的 `utils.inductive_main` 仅作为兼容包装器。
  - 推理必须提供训练输出目录中的 `scaler.pkl`（以及可选的 `pca.pkl`）；预处理缺失或特征维度不一致会直接报错，不会使用随机特征。
  - `RANDOM_SEED` 同时控制数据划分、门控/癌症采样和排序损失采样；`TOTAL_SAMPLES_PER_CLASS`、`OMICS_FEATURE_DIM` 与 KG 相似度阈值均来自配置。
  - `DETECT_ANOMALY` 默认关闭以避免全量训练的显著性能损失；需要定位梯度问题时显式打开，`CHECK_NAN_INF` 仍独立生效。
  - 知识图谱嵌入模型不可用时默认使用确定性的零特征；随机 KG 特征仅可通过 `ALLOW_RANDOM_KG_FEATURES` 显式用于消融实验。
  - 路径优先读取 `CM4SL_BASE_DIR`、`CM4SL_DATA_DIR`、`CM4SL_OUTPUT_DIR`；未设置时自动识别当前工作区的数据目录，否则回退到原远程目录布局。
  - KG 构建默认启用进程内 LRU 缓存（`KG_CACHE_MAX_ENTRIES=4`），缓存按基因顺序、输入文件签名和关键配置隔离，并以 CPU 图副本保存；可通过 `ENABLE_KG_CACHE=False` 关闭。

  合并的工程模块（详见 MERGE_NOTES.md）
  - artifacts.py / calibration.py：验证集温度缩放校准、带 SHA-256 的部署清单（deployment_manifest.json）与 checkpoint 全量配置快照。
  - 测试期 KG schema 严格校验：测试专用边类型丢弃+审计，未知节点类型 fail-closed。
  - batch_encode_texts 真批量文本编码（CLS 池化、单批失败回退零矩阵）。
  - 组学矩阵同处理器缓存 + 单文件缺失立即报错；TOTAL_SAMPLES_PER_CLASS 正数校验。
  - 推理端：config_path 覆盖、部署清单哈希校验、校准概率输出（REQUIRE_CALIBRATION 控制是否强制校准文件）。

  性能与显存优化（详见 OPTIMIZATION_NOTES.md，均可经 config 关闭）
  - HGT 梯度检查点：反向时重算前向，梯度与关闭时逐位一致；实测大幅降低百万边 KG 的激活显存峰值。
  - 数据侧进程级缓存：KG/STRING 源文件解析一次、文本向量缓存、PubMedBERT 实例共享、基因映射与组学帧跨实例共享——交叉验证各折不再重复解析相同数据。
  - PubMedBERT 构建后暂存 CPU、验证/测试 KG 非常驻 GPU、折间显存清理。
  - 上述优化均有等价性回归测试（tests/test_performance_optimizations.py）。
