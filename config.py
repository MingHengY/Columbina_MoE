"""配置文件"""

import os

# Resolve paths from the current checkout when the bundled local data is
# present, while keeping the original remote layout as a fallback.  Explicit
# environment variables always win, which keeps cluster jobs configurable.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REMOTE_ROOT = "/root/autodl-tmp/CM4SL"
_LOCAL_DATA_ROOT = os.path.join(_PROJECT_ROOT, "cm4sl_input_data")
_LOCAL_OUTPUT_ROOT = os.path.join(_PROJECT_ROOT, "cm4sl_outputs")
_DEFAULT_BASE_DIR = (
    _PROJECT_ROOT if os.path.isdir(_LOCAL_DATA_ROOT) else _REMOTE_ROOT
)

# ==================== 配置文件 ====================
class Config:
    """配置文件类"""

    # 基础路径
    BASE_DIR = os.environ.get("CM4SL_BASE_DIR", _DEFAULT_BASE_DIR)

    # 模型配置
    USE_PRETRAINED = True
    MODEL_TYPE = "local"  # "local", "huggingface", "simple"
    LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "pretrained_models/pubmedbert")

    # 数据文件路径
    DATA_DIR = os.environ.get(
        "CM4SL_DATA_DIR",
        _LOCAL_DATA_ROOT if BASE_DIR == _PROJECT_ROOT
        else os.path.join(BASE_DIR, "input_data"),
    )
    OUTPUT_DIR = os.environ.get(
        "CM4SL_OUTPUT_DIR",
        os.path.join(_LOCAL_OUTPUT_ROOT, "output_v4_merged")
        if BASE_DIR == _PROJECT_ROOT and os.path.isdir(_LOCAL_OUTPUT_ROOT)
        else os.path.join(BASE_DIR, "output_v4_merged"),
    )

    # 文件路径 - 更新为新的CSV格式
    SL_PAIRS_FILE = os.path.join(DATA_DIR, "integrated_sl_pairs_3.csv")
    NON_SL_PAIRS_FILE = os.path.join(DATA_DIR, "gene_nonsl_gene.tsv")
    KG_EDGES_FILE = os.path.join(DATA_DIR, "reformed_KG_edges.csv")
    KG_NODES_FILE = os.path.join(DATA_DIR, "reformed_KG_nodes.csv")
    STRING_FILE = os.path.join(DATA_DIR, "string_interactions.tsv")
    KG_EDGES_SEPARATOR = ','
    ENABLE_KG_CACHE = True
    KG_CACHE_MAX_ENTRIES = 4

    # 组学特征文件
    FEATURE_FILES = {
        'expression': os.path.join(DATA_DIR, 'expression.csv'),
        'cnv': os.path.join(DATA_DIR, 'cnv.csv'),
        'mutation': os.path.join(DATA_DIR, 'mutation.csv'),
        'methylation': os.path.join(DATA_DIR, 'methylation.csv'),
        'dependency': os.path.join(DATA_DIR, 'dependency.csv')
    }

    # FEATURE_FILES = {
    #     'expression': os.path.join(DATA_DIR, 'expression.csv'),
    #     'cnv': os.path.join(DATA_DIR, 'cnv.csv'),
    #     'mutation': os.path.join(DATA_DIR, 'mutation.csv'),
    #     'methylation': os.path.join(DATA_DIR, 'methylation.csv'),
    #     'dependency': os.path.join(DATA_DIR, 'dependency.csv')
    # }

    # 基因ID映射文件
    GENE_ID_MAPPING_FILE = os.path.join(DATA_DIR, "human_gene_mapping.csv")

    # ID锚定模式
    USE_ID_ANCHORING = True  # 启用ID锚定模式
    GENE_ID_COLUMN = 'ENTREZID'  # 使用的ID列名
    GENE_SYMBOL_COLUMN = 'SYMBOL'  # 基因符号列名

    # 训练参数
    EPOCHS = 100
    LEARNING_RATE = 0.0005
    BATCH_SIZE = 160
    HIDDEN_DIM = 128
    EMBEDDING_DIM = 64
    DROPOUT = 0.3
    WEIGHT_DECAY = 1e-5
    EARLY_STOPPING_PATIENCE = 4
    LR_SCHEDULER_PATIENCE = 2

    # Columbina 4.0-MoE
    ARCHITECTURE_VERSION = "4.0-MoE"
    CANCER_COLUMN_CANDIDATES = ('cancer', 'cancer_type', 'TCGA_Code', 'tissue')
    CANCER_CELL_LINE_MAPPING_FILE = os.path.join(DATA_DIR, 'depmap_model.csv')
    CANCER_EMBEDDING_DIM = 32
    MAX_CANCER_TYPES = 128
    HGT_HIDDEN_DIM = 128
    HGT_NUM_LAYERS = 2
    HGT_NUM_HEADS = 4
    MOE_EXPERT_DIM = 128
    # Keep the model input width and feature fallbacks in one place.  The
    # training pipeline currently produces 262-dimensional omics vectors.
    OMICS_FEATURE_DIM = 262
    KG_FEATURE_DIM = 768

    # Multi-objective loss weights. Weighted BCE remains the primary objective.
    LOSS_FINAL_WEIGHT = 1.0
    LOSS_SL_EXPERT_WEIGHT = 0.25
    LOSS_KG_EXPERT_WEIGHT = 0.25
    LOSS_RANK_WEIGHT = 0.10
    LOSS_SYMMETRY_WEIGHT = 0.05
    LOSS_GATE_BALANCE_WEIGHT = 0.0
    LOSS_GATE_ROUTING_WEIGHT = 0.10
    RANKING_MARGIN = 0.20
    RANKING_MAX_PAIRS = 1024
    GATE_MONITOR_INTERVAL = 10

    # Train-only pseudo-inductive augmentation. A 50% node mask yields a mix of
    # C1-like, C2-like, and C3-like training pairs without using held-out data.
    INDUCTIVE_AUGMENTATION_PROBABILITY = 0.50
    INDUCTIVE_AUGMENTATION_NODE_FRACTION = 0.50
    KG_NOVEL_EXPERT_BOOST = 1.0
    GATE_KG_PRIOR_STRENGTH = 0.40
    GATE_TARGET_MIN = 0.10
    GATE_TARGET_MAX = 0.90

    # Gate routing target. 'distill' supervises the gate on every sample with a
    # soft target derived from each expert's detached per-sample loss; 'coverage'
    # is the legacy prior that is a constant 0.5 on non-augmented train data.
    GATE_ROUTING_MODE = 'distill'
    GATE_DISTILL_TEMPERATURE = 0.25

    # Gate reliability input. 10 = 4 legacy coverage features + per-pair KG
    # connectivity (direct edge, shared neighbors) + symmetric log-degree
    # features; 4 restores the legacy coverage-only input.
    RELIABILITY_DIM = 10
    KG_PAIR_FEATURE_SCALE = 5.0

    # Keep a stable omics path even when positive training-SL context exists.
    SL_CONTEXT_MAX_WEIGHT = 0.65

    # Validation-only model selection. Precision@10 remains a reported metric
    # but is excluded because ten samples make it too coarse for early stopping.
    MODEL_SELECTION_AUC_WEIGHT = 0.40
    MODEL_SELECTION_AUPR_WEIGHT = 0.40
    MODEL_SELECTION_F1_WEIGHT = 0.20

    # 知识图谱构建参数
    MAX_NODES_PER_TYPE = 10000  # 每种节点类型的最大节点数
    MAX_EDGES_PER_RELATION = 100000  # 每种关系的最大边数
    DEBUG_MODE = True  # 启用调试模式

    # 性能优化参数
    CHUNKSIZE = 100000  # 分块处理大小
    MAX_STRING_EDGES = 10000  # 最大STRING边数
    MAX_KG_EDGES_PER_RELATION = 5000  # 每种关系的最大KG边数
    PROCESS_SAMPLE_SIZE = 50000  # 处理时的采样大小
    MAX_NODE_TYPES = 20  # 最大处理的节点类型数
    MAX_RELATION_TYPES = 30  # 最大处理的关系类型数

    # 知识图谱特定配置
    KG_GENE_TYPES = ['Gene', 'gene', 'Gene/Protein', 'Protein']  # 基因类型识别
    KG_FILTER_ONLY_GENE_EDGES = True  # 只保留至少一端是基因的边
    KG_USE_ID_FOR_MAPPING = True  # 优先使用x_id/y_id进行映射

    # 基因名称标准化参数（与ID锚定兼容）
    USE_GENE_NAME_NORMALIZATION = True  # 启用基因名称标准化
    GENE_SYNONYM_FILE = os.path.join(DATA_DIR, "gene_synonyms.csv")  # 基因同义词文件（可选）

    # 设备配置
    FORCE_GPU_PROCESSING = True  # 强制在GPU上处理所有数据
    GPU_MEMORY_LIMIT = 1  # GPU内存使用限制（0-1）

    # 训练稳定性参数
    GRADIENT_CLIP_VALUE = 1.0  # 梯度裁剪阈值
    GRADIENT_CLIP_NORM = 1.0  # 梯度裁剪范数
    USE_GRADIENT_CLIPPING = True  # 启用梯度裁剪
    USE_MIXED_PRECISION = False  # 使用混合精度训练
    CHECK_NAN_INF = True  # 检查NaN和Inf
    MEMORY_CLEANUP_FREQUENCY = 10  # 内存清理频率
    # Autograd anomaly detection is useful while debugging but is several
    # times slower on a full training run.  NaN/Inf checks remain enabled
    # independently through CHECK_NAN_INF.
    DETECT_ANOMALY = False


    # 数据划分参数
    TRAIN_VAL_SPLIT_RATIO = 0.8  # 训练集比例
    RANDOM_SEED = 42  # 随机种子
    BALANCE_CLASSES = True  # 是否平衡正负样本
    STRATIFIED_SPLIT = True  # 是否分层抽样
    TRAIN_VAL_TEST_RATIOS = (0.8, 0.1, 0.1)  # 训练集:验证集:测试集 = 8:1:1
    TOTAL_SAMPLES_PER_CLASS = 29000  # 每类样本总数
    USE_SAMPLING = True  # 是否启用抽样（默认为True）
    # Zero-filled omics are only appropriate for controlled ablation tests;
    # never hide a production data-loading failure by default.
    ALLOW_ZERO_FEATURE_FALLBACK = False
    ALLOW_RANDOM_KG_FEATURES = False
    # When True, inference refuses to emit probabilities unless a
    # calibration.json (validation-fitted temperature scaling) is available.
    # Default False keeps pre-merge checkpoints without calibration loadable.
    REQUIRE_CALIBRATION = False

    # New-gene KG augmentation.  Keeping this threshold configurable makes
    # experiments reproducible and avoids a hidden value in the processor.
    KG_SIMILARITY_THRESHOLD = 0.1

    # 交叉验证配置
    ENABLE_CROSS_VALIDATION = False  # 是否启用交叉验证
    N_FOLDS = 10  # 交叉验证折数
    SAVE_FOLD_DATA = True  # 保存每折数据
    SAVE_FOLD_MODELS = True  # 保存每折模型
    SAVE_DETAILED_REPORTS = True  # 保存详细报告
    SAVE_COMPARISON_CHARTS = True  # 保存比较图表

    USE_EXTERNAL_SPLIT = False  # 默认关闭
    EXTERNAL_TRAIN_FILE = None
    EXTERNAL_VAL_FILE = None
    EXTERNAL_TEST_FILE = None

    # ==================== 性能与显存优化 ====================
    # 以下开关均不改变数值语义（与关闭时数学等价）：
    # HGT反向传播时重算前向（梯度检查点）：等价数学，大幅降低百万边KG的激活显存。
    HGT_GRADIENT_CHECKPOINTING = True
    # KG构建完成后把冻结的PubMedBERT暂存到CPU（约0.5GB显存），下次编码自动搬回。
    PARK_EMBEDDING_MODEL_AFTER_BUILD = True
    # 冻结编码器的文本向量按文本做进程级缓存（交叉验证折间存在大量重复文本）。
    ENABLE_TEXT_EMBEDDING_CACHE = True
    TEXT_EMBEDDING_CACHE_MAX_ENTRIES = 100000
    # KG边/STRING源文件按签名只解析一次，进程内共享原始数据块，每次构建仅按基因集过滤。
    ENABLE_KG_FILE_CACHE = True
    KG_SOURCE_CACHE_MAX_ENTRIES = 4
    # 基因映射文件（19万行）解析结果跨实例共享，避免每折重复解析。
    ENABLE_GENE_MAPPING_CACHE = True
    # 组学矩阵帧跨处理器实例共享（以约2GB CPU内存换每折数分钟的重复读取）。
    ENABLE_OMICS_FRAME_CACHE = True
    OMICS_FRAME_CACHE_MAX_ENTRIES = 12
    # 验证/测试阶段的KG常驻CPU，仅在前向时按需传输到计算设备。
    KEEP_STAGE_KG_ON_CPU = True







