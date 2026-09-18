"""数据处理器"""

import torch
import pandas as pd
import numpy as np
import os
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torch_geometric.data import Data
import logging

from config import Config
from device_manager import DeviceManager
from gene_mapping import GeneIDMapper, GeneNameNormalizer
from knowledge_graph_builder import KnowledgeGraphBuilder, InductiveKnowledgeGraphBuilder
from split_protocol import (
    build_cross_validation_splits,
    build_single_split,
    validate_protocol_split
)

logger = logging.getLogger(__name__)


class CancerTypeVocabulary:
    """Training-only cancer vocabulary with stable unknown/global indices."""

    UNKNOWN_TOKEN = '<UNK>'
    GLOBAL_TOKEN = '<GLOBAL>'

    def __init__(self, candidates=None, max_types=128, cell_line_mapping_file=None):
        self.candidates = tuple(candidates or ('cancer', 'cancer_type', 'TCGA_Code', 'tissue'))
        self.max_types = int(max_types)
        self.cell_line_mapping_file = cell_line_mapping_file
        self.cell_line_to_cancer = None
        self.token_to_idx = {self.UNKNOWN_TOKEN: 0, self.GLOBAL_TOKEN: 1}
        self.column = None
        self.fitted = False

    @staticmethod
    def _normalize(value):
        if pd.isna(value):
            return CancerTypeVocabulary.GLOBAL_TOKEN
        token = str(value).strip().upper()
        return token if token else CancerTypeVocabulary.GLOBAL_TOKEN

    def _find_column(self, data_df):
        return next((name for name in self.candidates if name in data_df.columns), None)

    def _load_cell_line_mapping(self):
        if self.cell_line_to_cancer is not None:
            return self.cell_line_to_cancer
        self.cell_line_to_cancer = {}
        mapping_file = self.cell_line_mapping_file
        if not mapping_file or not os.path.exists(mapping_file):
            mapping_file = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'infer', 'depmap_model.csv'
            )
        if not os.path.exists(mapping_file):
            return self.cell_line_to_cancer
        try:
            mapping_df = pd.read_csv(mapping_file)
            name_columns = [
                column for column in
                ('ModelID', 'CellLineName', 'StrippedCellLineName', 'CCLEName')
                if column in mapping_df.columns
            ]
            if 'TCGA_Code' not in mapping_df.columns:
                return self.cell_line_to_cancer
            for _, row in mapping_df.iterrows():
                cancer = self._normalize(row['TCGA_Code'])
                if cancer == self.GLOBAL_TOKEN:
                    continue
                for column in name_columns:
                    value = row[column]
                    if not pd.isna(value):
                        self.cell_line_to_cancer[str(value).strip().upper()] = cancer
        except Exception as error:
            logger.warning("Failed to load cancer cell-line mapping: %s", error)
        return self.cell_line_to_cancer

    def _tokens(self, data_df):
        column = self._find_column(data_df)
        if column is None:
            tokens = pd.Series(
                [self.GLOBAL_TOKEN] * len(data_df), index=data_df.index, dtype=object
            )
        else:
            tokens = data_df[column].map(self._normalize)
        if 'cell_line' in data_df.columns:
            mapping = self._load_cell_line_mapping()
            inferred = data_df['cell_line'].map(
                lambda value: mapping.get(str(value).strip().upper())
                if not pd.isna(value) else None
            )
            missing = tokens.eq(self.GLOBAL_TOKEN) & inferred.notna()
            tokens.loc[missing] = inferred.loc[missing]
        return tokens.tolist()

    def fit(self, data_df):
        self.column = self._find_column(data_df)
        values = self._tokens(data_df)
        observed = sorted({value for value in values if value != self.GLOBAL_TOKEN})
        capacity = max(self.max_types - len(self.token_to_idx), 0)
        for token in observed[:capacity]:
            self.token_to_idx[token] = len(self.token_to_idx)
        self.fitted = True
        return self

    def transform(self, data_df):
        tokens = self._tokens(data_df)
        return np.asarray(
            [self.token_to_idx.get(token, 0) for token in tokens], dtype=np.int64
        )

    def state_dict(self):
        return {
            'token_to_idx': dict(self.token_to_idx),
            'column': self.column,
            'fitted': self.fitted,
            'max_types': self.max_types,
            'candidates': list(self.candidates),
            'cell_line_mapping_file': self.cell_line_mapping_file,
        }

    def load_state_dict(self, state):
        if state:
            self.token_to_idx = dict(state.get('token_to_idx', self.token_to_idx))
            self.column = state.get('column')
            self.fitted = bool(state.get('fitted', True))
            self.cell_line_mapping_file = state.get(
                'cell_line_mapping_file', self.cell_line_mapping_file
            )
        return self

# ==================== 通用数据划分器 ====================
class UniversalDataSplitter:
    """通用数据划分器，支持C1、C2、C3三种场景，支持交叉验证"""

    def __init__(self, config=None, enable_cv=False, n_folds=10):
        self.config = config if config else Config()
        self.id_mapper = GeneIDMapper(config)
        self.enable_cv = enable_cv  # 是否启用交叉验证
        self.n_folds = n_folds if enable_cv else 1  # 折数
        self.total_samples_per_class = int(
            getattr(self.config, 'TOTAL_SAMPLES_PER_CLASS', 29000)
        )
        if self.total_samples_per_class <= 0:
            raise ValueError("TOTAL_SAMPLES_PER_CLASS must be positive")

        self.train_ratio = 0.8  # 训练集比例
        self.val_ratio = 0.1  # 验证集比例
        self.test_ratio = 0.1  # 测试集比例

    def split_data_by_scenario(self, data_df, scenario='C1', seed=42):
        """
        根据场景划分数据（支持交叉验证）

        Args:
            data_df: 包含['Gene.A', 'Gene.B', 'label']的DataFrame
            scenario: 'C1', 'C2', 或 'C3'
            seed: 随机种子

        Returns:
            如果enable_cv=True: 返回fold_splits列表，每个元素是(train_df, val_df, test_df)
            如果enable_cv=False: 返回单个(train_df, val_df, test_df)
        """
        logger.info(f"开始{scenario}场景的数据划分...")
        logger.info(f"交叉验证: {'启用' if self.enable_cv else '禁用'}, 折数: {self.n_folds}")
        logger.info(
            f"目标划分比例: 训练集{self.train_ratio * 100:.0f}%, 验证集{self.val_ratio * 100:.0f}%, 测试集{self.test_ratio * 100:.0f}%")

        np.random.seed(seed)

        # 分离正负样本
        positive_data = data_df[data_df['label'] == 1]
        negative_data = data_df[data_df['label'] == 0]

        logger.info(f"原始数据 - 正样本: {len(positive_data)}条, 负样本: {len(negative_data)}条")

        # 1. 抽样：从正负样本中各抽取20000条
        positive_sampled = self._sample_data(positive_data, self.total_samples_per_class, seed, "正样本")
        negative_sampled = self._sample_data(negative_data, self.total_samples_per_class, seed, "负样本")

        # 合并并打乱
        balanced_data = pd.concat([positive_sampled, negative_sampled], ignore_index=True)
        balanced_data = balanced_data.sample(frac=1, random_state=seed).reset_index(drop=True)

        logger.info(f"抽样后数据: {len(balanced_data)}条 (正负样本各{self.total_samples_per_class}条)")

        # 2. 根据是否启用交叉验证选择划分方式
        if self.enable_cv:
            original_folds = self.n_folds
            if self.n_folds != 10:
                logger.warning(
                    f"C1/C2/C3的8:1:1轮转协议固定使用10折，"
                    f"将折数从{self.n_folds}调整为10"
                )
                self.n_folds = 10
            result = self._cross_validation_split(balanced_data, scenario, seed)
            self.n_folds = original_folds
            return result
        else:
            return self._single_split(balanced_data, scenario, seed)

    def _sample_data(self, data_df, sample_size, seed, data_type):
        """抽样数据，如果数据量不足则使用全部"""
        if len(data_df) <= sample_size:
            logger.warning(f"{data_type}数据量({len(data_df)})不足{sample_size}，使用全部数据")
            return data_df.copy()
        else:
            return data_df.sample(n=sample_size, random_state=seed, replace=False)

    def _single_split(self, data_df, scenario, seed):
        """单次划分（训练:验证:测试=8:1:1）"""
        logger.info(
            f"执行单次划分 ({self.train_ratio * 100:.0f}:{self.val_ratio * 100:.0f}:{self.test_ratio * 100:.0f})...")

        if scenario == 'C1':
            return self._split_c1(data_df, seed)
        elif scenario == 'C2':
            return self._split_c2(data_df, seed)
        elif scenario == 'C3':
            return self._split_c3(data_df, seed)
        else:
            raise ValueError(f"未知的场景: {scenario}")

    def _cross_validation_split(self, data_df, scenario, seed):
        """执行交叉验证划分"""
        logger.info(f"执行{self.n_folds}折交叉验证...")

        fold_splits = []


        # 根据场景选择不同的交叉验证策略
        if scenario == 'C1':
            fold_splits = self._cv_split_c1(data_df, seed)
        elif scenario == 'C2':
            fold_splits = self._cv_split_c2(data_df, seed)
        elif scenario == 'C3':
            fold_splits = self._cv_split_c3(data_df, seed)
        else:
            raise ValueError(f"未知的场景: {scenario}")

        # 验证每个fold的划分
        for fold_idx, (train_df, val_df, test_df) in enumerate(fold_splits):
            logger.info(f"Fold {fold_idx + 1}: 训练集{len(train_df)}条, 验证集{len(val_df)}条, 测试集{len(test_df)}条")
            self._validate_fold_split(train_df, val_df, test_df, scenario, fold_idx + 1)

        return fold_splits

    def _cv_split_c1(self, data_df, seed):
        """C1: rotate ten mutually exclusive undirected gene-pair buckets."""
        logger.info("C1交叉验证：无向基因对10折轮转，8折训练/1折验证/1折测试")
        return build_cross_validation_splits(
            data_df, scenario='C1', seed=seed, n_folds=10
        )

    def _cv_split_c2(self, data_df, seed):
        """C2: rotate gene buckets and retain one-seen/one-new edges."""
        logger.info(
            "C2交叉验证：基因10折轮转，验证/测试每条边严格一个训练已见基因"
        )
        return build_cross_validation_splits(
            data_df, scenario='C2', seed=seed, n_folds=10
        )

    def _cv_split_c3(self, data_df, seed):
        """C3: rotate gene buckets and retain fully unseen evaluation pairs."""
        logger.info(
            "C3交叉验证：基因10折轮转，验证/测试每条边的两个基因均为未见"
        )
        return build_cross_validation_splits(
            data_df, scenario='C3', seed=seed, n_folds=10
        )

    def _assign_data_by_pairs(self, data_df, train_pairs, val_pairs, test_pairs):
        """根据基因对分配数据"""
        train_data, val_data, test_data = [], [], []

        for _, row in data_df.iterrows():
            pair = (row['Gene.A'], row['Gene.B'])
            if pair in train_pairs:
                train_data.append(row)
            elif pair in val_pairs:
                val_data.append(row)
            elif pair in test_pairs:
                test_data.append(row)

        return (
            pd.DataFrame(train_data) if train_data else pd.DataFrame(columns=data_df.columns),
            pd.DataFrame(val_data) if val_data else pd.DataFrame(columns=data_df.columns),
            pd.DataFrame(test_data) if test_data else pd.DataFrame(columns=data_df.columns)
        )

    def _assign_data_c2_c3(self, data_df, train_genes, val_genes, test_genes, scenario='C2'):
        """为C2/C3场景分配数据"""
        train_data, val_data, test_data = [], [], []

        for _, row in data_df.iterrows():
            gene_a = row['Gene.A']
            gene_b = row['Gene.B']

            if scenario == 'C2':
                # C2条件：两个基因都在训练集中 -> 训练集
                if gene_a in train_genes and gene_b in train_genes:
                    train_data.append(row)
                # 一个在训练集，一个在验证集 -> 验证集
                elif (gene_a in train_genes and gene_b in val_genes) or \
                        (gene_b in train_genes and gene_a in val_genes):
                    val_data.append(row)
                # 一个在训练集，一个在测试集 -> 测试集
                elif (gene_a in train_genes and gene_b in test_genes) or \
                        (gene_b in train_genes and gene_a in test_genes):
                    test_data.append(row)
                # 其他情况丢弃
            elif scenario == 'C3':
                # C3条件：两个基因都在训练集中 -> 训练集
                if gene_a in train_genes and gene_b in train_genes:
                    train_data.append(row)
                # 两个基因都在验证集 -> 验证集
                elif gene_a in val_genes and gene_b in val_genes:
                    val_data.append(row)
                # 两个基因都在测试集 -> 测试集
                elif gene_a in test_genes and gene_b in test_genes:
                    test_data.append(row)
                # 其他情况丢弃

        return (
            pd.DataFrame(train_data) if train_data else pd.DataFrame(columns=data_df.columns),
            pd.DataFrame(val_data) if val_data else pd.DataFrame(columns=data_df.columns),
            pd.DataFrame(test_data) if test_data else pd.DataFrame(columns=data_df.columns)
        )

    def _validate_fold_split(self, train_df, val_df, test_df, scenario, fold_idx):
        """Validate a fold against the complete protocol before training."""
        report = validate_protocol_split(train_df, val_df, test_df, scenario)
        logger.info(
            f"Fold {fold_idx} {scenario}划分校验通过: "
            f"train={report.train_size} ({report.train_ratio:.2%}), "
            f"val={report.val_size} ({report.val_ratio:.2%}), "
            f"test={report.test_size} ({report.test_ratio:.2%}), "
            f"train_genes={report.train_gene_count}, "
            f"val_new={report.val_new_gene_count}, "
            f"test_new={report.test_new_gene_count}"
        )

    # 单次划分方法

    def _split_c1(self, data_df, seed):
        """C1 single split using fold zero of the rotating pair protocol."""
        return build_single_split(data_df, scenario='C1', seed=seed)

    def _split_c2(self, data_df, seed):
        """C2 single split using fold zero of the rotating gene protocol."""
        return build_single_split(data_df, scenario='C2', seed=seed)

    def _split_c3(self, data_df, seed):
        """C3 single split using fold zero of the rotating gene protocol."""
        return build_single_split(data_df, scenario='C3', seed=seed)

    # 分析划分
    def analyze_split(self, train_df, val_df, test_df, scenario):
        """分析划分结果"""
        logger.info(f"\n{scenario}场景划分分析:")

        # 统计基因
        train_genes = set(train_df['Gene.A']).union(set(train_df['Gene.B']))
        val_genes = set(val_df['Gene.A']).union(set(val_df['Gene.B']))
        test_genes = set(test_df['Gene.A']).union(set(test_df['Gene.B']))

        logger.info(f"训练集基因数: {len(train_genes)}")
        logger.info(f"验证集基因数: {len(val_genes)}")
        logger.info(f"测试集基因数: {len(test_genes)}")

        # 检查重叠
        train_val_overlap = train_genes.intersection(val_genes)
        train_test_overlap = train_genes.intersection(test_genes)
        val_test_overlap = val_genes.intersection(test_genes)

        logger.info(f"训练集-验证集基因重叠: {len(train_val_overlap)}")
        logger.info(f"训练集-测试集基因重叠: {len(train_test_overlap)}")
        logger.info(f"验证集-测试集基因重叠: {len(val_test_overlap)}")

        if scenario in {'C2', 'C3'}:
            val_new_genes = val_genes - train_genes
            test_new_genes = test_genes - train_genes
            logger.info(
                f"验证集-测试集held-out基因重叠: "
                f"{len(val_new_genes.intersection(test_new_genes))}"
            )

        # 检查场景条件
        if scenario == 'C2':
            # 验证集和测试集中的基因对应该恰好有一个基因在训练集中
            val_condition = all(
                (row['Gene.A'] in train_genes) ^ (row['Gene.B'] in train_genes)
                for _, row in val_df.iterrows()
            ) if len(val_df) > 0 else True
            test_condition = all(
                (row['Gene.A'] in train_genes) ^ (row['Gene.B'] in train_genes)
                for _, row in test_df.iterrows()
            ) if len(test_df) > 0 else True
            logger.info(f"验证集C2条件满足: {val_condition}")
            logger.info(f"测试集C2条件满足: {test_condition}")

        elif scenario == 'C3':
            # 验证集和测试集中的基因对应该在训练集中都未出现
            val_condition = all(
                (row['Gene.A'] not in train_genes) and (row['Gene.B'] not in train_genes)
                for _, row in val_df.iterrows()
            ) if len(val_df) > 0 else True
            test_condition = all(
                (row['Gene.A'] not in train_genes) and (row['Gene.B'] not in train_genes)
                for _, row in test_df.iterrows()
            ) if len(test_df) > 0 else True
            logger.info(f"验证集C3条件满足: {val_condition}")
            logger.info(f"测试集C3条件满足: {test_condition}")

    def validate_split_ratio(self, train_df, val_df, test_df, scenario):
        """验证划分比例是否符合8:1:1"""
        total = len(train_df) + len(val_df) + len(test_df)
        if total == 0:
            return

        train_ratio = len(train_df) / total
        val_ratio = len(val_df) / total
        test_ratio = len(test_df) / total

        logger.info(f"{scenario}场景划分比例验证:")
        logger.info(f"  训练集: {len(train_df)} ({train_ratio:.2%})")
        logger.info(f"  验证集: {len(val_df)} ({val_ratio:.2%})")
        logger.info(f"  测试集: {len(test_df)} ({test_ratio:.2%})")

        # 检查是否接近8:1:1
        target_train = self.train_ratio
        target_val = self.val_ratio
        target_test = self.test_ratio

        tolerance = 0.05  # 5%的容忍度

        if (abs(train_ratio - target_train) < tolerance and
                abs(val_ratio - target_val) < tolerance and
                abs(test_ratio - target_test) < tolerance):
            logger.info("✓ 划分比例符合8:1:1要求")
        else:
            logger.warning("⚠ 划分比例偏离8:1:1要求")



class SLDataProcessor:
    # Parsed omics frames shared across processor instances.  Cross-validation
    # creates a new InductiveSLDataProcessor per fold, which previously
    # re-parsed the same ~2GB of CSVs every fold; frames are read-only after
    # keying, so sharing them is safe (config: ENABLE_OMICS_FRAME_CACHE).
    _omics_frame_shared_cache = OrderedDict()

    @classmethod
    def clear_omics_frame_cache(cls):
        """Drop shared parsed omics frames (tests / explicit invalidation)."""
        cls._omics_frame_shared_cache.clear()

    def __init__(self, config=None, device_manager=None):
        """使用配置初始化数据处理器"""
        if config is None:
            self.config = config = Config()
        else:
            self.config = config

        # 设备管理器
        self.device_manager = device_manager or DeviceManager()

        # 基因ID映射器
        self.id_mapper = GeneIDMapper(config)

        # 基因名称标准化器
        self.gene_normalizer = GeneNameNormalizer(config)

        # 知识图谱构建器（使用本地模型）
        self.kg_builder = KnowledgeGraphBuilder(
            edges_file=self.config.KG_EDGES_FILE,
            nodes_file=self.config.KG_NODES_FILE,
            string_file=self.config.STRING_FILE,
            local_model_path=self.config.LOCAL_MODEL_PATH,
            use_local_model=self.config.USE_PRETRAINED,
            config=self.config,
            gene_normalizer=self.gene_normalizer,
            device_manager=self.device_manager
        )

        # 使用ID作为键的映射
        self.gene_id_to_idx = {}  # ENTREZID -> 索引
        self.idx_to_gene_id = {}  # 索引 -> ENTREZID

        # 标准化和PCA处理器（初始为空，将在训练集上拟合）
        self.scaler = StandardScaler()
        self.pca = None
        self.is_scaler_fitted = False  # 标记是否已拟合

        self.original_to_normalized = {}  # 原始基因名到标准化名称的映射
        # 同一处理器实例内的组学矩阵缓存，按（路径，锚定，归一化）隔离。
        self._omics_frame_cache = {}
        self.cancer_vocabulary = CancerTypeVocabulary(
            candidates=getattr(self.config, 'CANCER_COLUMN_CANDIDATES', None),
            max_types=getattr(self.config, 'MAX_CANCER_TYPES', 128),
            cell_line_mapping_file=getattr(
                self.config, 'CANCER_CELL_LINE_MAPPING_FILE', None
            ),
        )

    def fit_cancer_vocabulary(self, train_data):
        self.cancer_vocabulary.fit(train_data)
        logger.info(
            "Cancer vocabulary fitted from training data: %d entries",
            len(self.cancer_vocabulary.token_to_idx),
        )
        return dict(self.cancer_vocabulary.token_to_idx)

    def _attach_cancer_context(self, graph_data, data_df, mode='train'):
        if not hasattr(self, 'cancer_vocabulary'):
            self.cancer_vocabulary = CancerTypeVocabulary(
                candidates=getattr(self.config, 'CANCER_COLUMN_CANDIDATES', None),
                max_types=getattr(self.config, 'MAX_CANCER_TYPES', 128),
                cell_line_mapping_file=getattr(
                    self.config, 'CANCER_CELL_LINE_MAPPING_FILE', None
                ),
            )
        if mode == 'train' and not self.cancer_vocabulary.fitted:
            self.fit_cancer_vocabulary(data_df)
        graph_data.cancer_index = torch.tensor(
            self.cancer_vocabulary.transform(data_df),
            dtype=torch.long,
            device=graph_data.x.device,
        )
        return graph_data

    def load_sl_data(self):
        """加载合成致死对数据（转换为ID）"""
        try:
            sl_data = pd.read_csv(self.config.SL_PAIRS_FILE)
        except Exception as e:
            logger.error(f"加载SL数据失败: {e}")
            # 尝试不同的分隔符
            try:
                sl_data = pd.read_csv(self.config.SL_PAIRS_FILE, sep='\t')
            except:
                raise ValueError(f"无法加载SL数据文件: {self.config.SL_PAIRS_FILE}")

        logger.info(f"原始数据量: {len(sl_data)}")

        # 检查必要的列
        required_columns = ['Gene.A', 'Gene.B']
        for col in required_columns:
            if col not in sl_data.columns:
                # 尝试从其他可能的列名中查找
                if 'x:START_ID' in sl_data.columns and 'y:END_ID' in sl_data.columns:
                    sl_data = sl_data.rename(columns={
                        'x_name': 'Gene.A',
                        'y_name': 'Gene.B'
                    })
                    break
                else:
                    raise ValueError(f"数据文件中缺少必要的列: {col}")

        # 转换基因名称为ID
        if self.config.USE_ID_ANCHORING:
            logger.info("将SL数据中的基因名称转换为ENTREZID...")

            # 转换Gene.A
            original_genes_a = sl_data['Gene.A'].tolist()
            gene_ids_a = []
            missing_a = 0

            for gene in original_genes_a:
                gene_id = self.id_mapper.get_id_by_any_name(gene)
                if gene_id:
                    gene_ids_a.append(gene_id)
                else:
                    gene_ids_a.append(gene)  # 保持原名称
                    missing_a += 1

            sl_data['Gene.A'] = gene_ids_a

            # 转换Gene.B
            original_genes_b = sl_data['Gene.B'].tolist()
            gene_ids_b = []
            missing_b = 0

            for gene in original_genes_b:
                gene_id = self.id_mapper.get_id_by_any_name(gene)
                if gene_id:
                    gene_ids_b.append(gene_id)
                else:
                    gene_ids_b.append(gene)
                    missing_b += 1

            sl_data['Gene.B'] = gene_ids_b

            logger.info(f"基因转换完成: Gene.A缺失{missing_a}条, Gene.B缺失{missing_b}条")

            # 统计唯一基因数
            unique_ids = len(set(gene_ids_a + gene_ids_b))
            logger.info(f"唯一ENTREZID数: {unique_ids}")
        else:
            # 使用原有的标准化逻辑
            if self.config.USE_GENE_NAME_NORMALIZATION:
                logger.info("标准化SL数据中的基因名称...")

                # 标准化Gene.A
                original_genes_a = sl_data['Gene.A'].tolist()
                normalized_genes_a = self.gene_normalizer.batch_normalize(original_genes_a)
                sl_data['Gene.A'] = normalized_genes_a

                # 标准化Gene.B
                original_genes_b = sl_data['Gene.B'].tolist()
                normalized_genes_b = self.gene_normalizer.batch_normalize(original_genes_b)
                sl_data['Gene.B'] = normalized_genes_b

                logger.info(f"标准化了 {len(sl_data)} 条SL数据中的基因名称")

                # 统计唯一基因数
                unique_original = len(set(original_genes_a + original_genes_b))
                unique_normalized = len(set(normalized_genes_a + normalized_genes_b))
                logger.info(f"SL基因名称标准化: {unique_original} 个原始名称 -> {unique_normalized} 个标准化名称")

        # 根据GEMINI.sensitive <= -1 定义标签（如果有这个列）
        if 'GEMINI.sensitive' in sl_data.columns:
            sl_data['label'] = (sl_data['GEMINI.sensitive'] <= -1).astype(int)
            logger.info(f"正样本数: {sl_data['label'].sum()}, 负样本数: {len(sl_data) - sl_data['label'].sum()}")
        else:
            # 如果没有GEMINI.sensitive列，假设所有都是正样本
            sl_data['label'] = 1
            logger.info("未找到GEMINI.sensitive列，假设所有样本为正样本")

        return sl_data

    def load_non_sl_data(self, sample_size=None):
        """加载非合成致死对数据（转换为ID）"""
        if not os.path.exists(self.config.NON_SL_PAIRS_FILE):
            logger.warning(f"非SL数据文件不存在: {self.config.NON_SL_PAIRS_FILE}")
            return None

        try:
            non_sl_data = pd.read_csv(self.config.NON_SL_PAIRS_FILE, sep='\t')
        except Exception as e:
            logger.error(f"加载非SL数据失败: {e}")
            return None

        logger.info(f"原始非SL数据量: {len(non_sl_data)}")

        # 检查必要的列
        if 'x:START_ID' in non_sl_data.columns and 'y:END_ID' in non_sl_data.columns:
            non_sl_data = non_sl_data.rename(columns={
                'x_name': 'Gene.A',
                'y_name': 'Gene.B'
            })

        # 转换基因名称为ID
        if self.config.USE_ID_ANCHORING:
            logger.info("将非SL数据中的基因名称转换为ENTREZID...")

            # 转换Gene.A
            if 'Gene.A' in non_sl_data.columns:
                original_genes_a = non_sl_data['Gene.A'].tolist()
                gene_ids_a = []
                missing_a = 0

                for gene in original_genes_a:
                    gene_id = self.id_mapper.get_id_by_any_name(gene)
                    if gene_id:
                        gene_ids_a.append(gene_id)
                    else:
                        gene_ids_a.append(gene)
                        missing_a += 1

                non_sl_data['Gene.A'] = gene_ids_a
                logger.info(f"Gene.A转换完成: 缺失{missing_a}条")

            # 转换Gene.B
            if 'Gene.B' in non_sl_data.columns:
                original_genes_b = non_sl_data['Gene.B'].tolist()
                gene_ids_b = []
                missing_b = 0

                for gene in original_genes_b:
                    gene_id = self.id_mapper.get_id_by_any_name(gene)
                    if gene_id:
                        gene_ids_b.append(gene_id)
                    else:
                        gene_ids_b.append(gene)
                        missing_b += 1

                non_sl_data['Gene.B'] = gene_ids_b
                logger.info(f"Gene.B转换完成: 缺失{missing_b}条")
        elif self.config.USE_GENE_NAME_NORMALIZATION:
            logger.info("标准化非SL数据中的基因名称...")

            # 标准化Gene.A
            if 'Gene.A' in non_sl_data.columns:
                original_genes_a = non_sl_data['Gene.A'].tolist()
                normalized_genes_a = self.gene_normalizer.batch_normalize(original_genes_a)
                non_sl_data['Gene.A'] = normalized_genes_a

            # 标准化Gene.B
            if 'Gene.B' in non_sl_data.columns:
                original_genes_b = non_sl_data['Gene.B'].tolist()
                normalized_genes_b = self.gene_normalizer.batch_normalize(original_genes_b)
                non_sl_data['Gene.B'] = normalized_genes_b

        # 添加标签（负样本）
        non_sl_data['label'] = 0

        # 随机抽样
        if sample_size and len(non_sl_data) > sample_size:
            non_sl_data = non_sl_data.sample(n=sample_size, random_state=self.config.RANDOM_SEED)
            logger.info(f"抽样后非SL数据量: {len(non_sl_data)}")

        return non_sl_data

    def combine_and_balance_data(self, sl_data, non_sl_data):
        """合并正负样本数据并平衡"""

        if self.config.USE_SAMPLING:
            # 抽样逻辑
            sample_size = min(
                self.config.TOTAL_SAMPLES_PER_CLASS,
                len(sl_data),
                len(non_sl_data) if non_sl_data is not None else float('inf')
            )

            # 从正负样本中各抽取指定数量的样本
            positive_samples = sl_data.sample(n=sample_size, random_state=self.config.RANDOM_SEED)
            negative_samples = non_sl_data.sample(n=sample_size, random_state=self.config.RANDOM_SEED)

            balanced_data = pd.concat([positive_samples, negative_samples], ignore_index=True)
            balanced_data = balanced_data.sample(frac=1, random_state=self.config.RANDOM_SEED).reset_index(drop=True)

            logger.info(f"抽样平衡后数据: {len(balanced_data)}条 (正负样本各{sample_size}条)")

            return balanced_data

        else:
            if non_sl_data is not None:
                # 合并数据
                combined_data = pd.concat([sl_data, non_sl_data], ignore_index=True)

                # 平衡正负样本（确保1:1）
                positive_samples = combined_data[combined_data['label'] == 1]
                negative_samples = combined_data[combined_data['label'] == 0]

                logger.info(f"合并后数据 - 正样本: {len(positive_samples)}, 负样本: {len(negative_samples)}")

                # 取较小数量作为平衡后的样本数
                min_count = min(len(positive_samples), len(negative_samples))

                if min_count > 0:
                    # 从正负样本中各取min_count个
                    positive_balanced = positive_samples.sample(n=min_count, random_state=self.config.RANDOM_SEED)
                    negative_balanced = negative_samples.sample(n=min_count, random_state=self.config.RANDOM_SEED)

                    # 合并平衡后的数据
                    balanced_data = pd.concat([positive_balanced, negative_balanced], ignore_index=True)
                    balanced_data = balanced_data.sample(frac=1, random_state=self.config.RANDOM_SEED).reset_index(
                        drop=True)

                    logger.info(f"平衡后总数据量: {len(balanced_data)}")
                    logger.info(f"平衡后正样本: {len(balanced_data[balanced_data['label'] == 1])}")
                    logger.info(f"平衡后负样本: {len(balanced_data[balanced_data['label'] == 0])}")
                    logger.info(f"正负样本比例: 1:1")

                    return balanced_data
                else:
                    logger.warning("正样本或负样本数量为0，无法平衡")
                    return combined_data
            else:
                logger.info("只使用正样本数据")
                return sl_data

    def split_data(self, data_df):
        """划分数据集为训练集和验证集"""
        try:
            # 分层划分数据
            train_data, val_data = train_test_split(
                data_df,
                test_size=1 - self.config.TRAIN_VAL_SPLIT_RATIO,
                random_state=self.config.RANDOM_SEED,
                stratify=data_df['label'] if self.config.STRATIFIED_SPLIT else None
            )

            logger.info(f"训练集大小: {len(train_data)} ({len(train_data) / len(data_df) * 100:.1f}%)")
            logger.info(f"验证集大小: {len(val_data)} ({len(val_data) / len(data_df) * 100:.1f}%)")
            logger.info(
                f"训练集中正负样本比例: {len(train_data[train_data['label'] == 1])}:{len(train_data[train_data['label'] == 0])}")
            logger.info(
                f"验证集中正负样本比例: {len(val_data[val_data['label'] == 1])}:{len(val_data[val_data['label'] == 0])}")

            return train_data, val_data

        except Exception as e:
            logger.error(f"数据划分失败: {e}")
            # 如果失败，使用简单随机划分
            train_data = data_df.sample(frac=self.config.TRAIN_VAL_SPLIT_RATIO, random_state=self.config.RANDOM_SEED)
            val_data = data_df.drop(train_data.index)
            return train_data, val_data

    def _fit_scaler_pca_on_train(self, train_features):
        """仅在训练集上拟合标准化器和PCA"""
        logger.info("在训练集上拟合标准化器和PCA...")

        # 1. 标准化特征（只在训练集）
        self.scaler.fit(train_features)
        normalized_features = self.scaler.transform(train_features)

        # 记录标准化器的详细信息
        logger.info(f"标准化器拟合完成 - 均值: {self.scaler.mean_.shape}, 标准差: {self.scaler.scale_.shape}")
        logger.info(f"标准化前特征范围: [{train_features.min():.4f}, {train_features.max():.4f}]")
        logger.info(f"标准化后特征范围: [{normalized_features.min():.4f}, {normalized_features.max():.4f}]")

        # 确保is_scaler_fitted设置为True
        self.is_scaler_fitted = True
        logger.info(f"标准化器拟合标志已设置为: {self.is_scaler_fitted}")

        # 2. PCA降维（严格在训练集）
        if normalized_features.shape[1] > 300:
            logger.info(f"在训练集({train_features.shape[0]}个样本)上进行PCA降维...")
            # 确保n_components不超过训练样本数-1
            n_samples = train_features.shape[0]
            n_components = min(
                int(getattr(self.config, 'OMICS_FEATURE_DIM', 262)),
                n_samples - 1,
            )
            self.pca = PCA(n_components=n_components)
            normalized_features = self.pca.fit_transform(normalized_features)  # 只在训练集拟合

            # 记录PCA的详细信息
            explained_variance = np.sum(self.pca.explained_variance_ratio_)
            logger.info(f"PCA降维: {normalized_features.shape[1]} -> {n_components} 维")
            logger.info(f"PCA保留方差: {explained_variance:.3f} ({explained_variance * 100:.1f}%)")
            logger.info(f"主成分形状: {self.pca.components_.shape}")

            # 保存每个主成分解释的方差
            if hasattr(self.pca, 'explained_variance_ratio_'):
                variance_df = pd.DataFrame({
                    'component': range(1, len(self.pca.explained_variance_ratio_) + 1),
                    'explained_variance_ratio': self.pca.explained_variance_ratio_,
                    'cumulative_variance': np.cumsum(self.pca.explained_variance_ratio_)
                })
                variance_path = os.path.join(self.config.OUTPUT_DIR, 'pca_variance.csv')
                variance_df.to_csv(variance_path, index=False)
                logger.info(f"PCA方差解释率已保存至: {variance_path}")

        return normalized_features

    def _transform_with_fitted_scaler_pca(self, features):
        """使用训练集拟合的标准化器和PCA转换特征"""
        if not self.is_scaler_fitted:
            logger.error("标准化器尚未拟合！")
            # 尝试从scaler属性检查
            if hasattr(self.scaler, 'n_features_in_'):
                logger.info("scaler有n_features_in_属性，设置为已拟合")
                self.is_scaler_fitted = True
            else:
                raise ValueError("必须先调用_fit_scaler_pca_on_train方法拟合标准化器")

        # 标准化特征
        try:
            normalized_features = self.scaler.transform(features)
        except Exception as e:
            logger.error(f"标准化转换失败: {e}")
            # 返回原始特征作为备选
            logger.warning("返回原始特征作为备选")
            return features

        # 如果训练时进行了PCA，则应用PCA转换
        if self.pca is not None and hasattr(self.pca, 'components_'):
            try:
                normalized_features = self.pca.transform(normalized_features)
            except Exception as e:
                logger.error(f"PCA转换失败: {e}")
                # 返回标准化后的特征
                logger.warning("返回标准化特征（跳过PCA）")

        return normalized_features

    @staticmethod
    def _omics_file_signature(path):
        if not path:
            return None
        try:
            stat = os.stat(path)
        except OSError:
            return (os.fspath(path), None, None)
        return (os.fspath(path), stat.st_mtime_ns, stat.st_size)

    def _shared_omics_cache_key(self, file_path):
        # 帧内容取决于：源文件 + ID锚定开关（依赖基因映射文件）+ 名称归一化（依赖同义词文件）。
        return (
            self._omics_file_signature(file_path),
            bool(self.config.USE_ID_ANCHORING),
            bool(self.config.USE_GENE_NAME_NORMALIZATION),
            self._omics_file_signature(getattr(self.config, 'GENE_ID_MAPPING_FILE', None)),
            self._omics_file_signature(getattr(self.config, 'GENE_SYNONYM_FILE', None)),
        )

    def _load_omics_frame(self, omics_type, file_path):
        """读取、转置并键定一个组学矩阵；解析结果跨处理器实例共享。"""
        cache_key = (
            str(file_path),
            bool(self.config.USE_ID_ANCHORING),
            bool(self.config.USE_GENE_NAME_NORMALIZATION),
        )
        cached = self._omics_frame_cache.get(cache_key)
        if cached is not None:
            logger.info(f"复用已处理 {omics_type} 数据: {cached.shape}")
            return cached

        shared_enabled = bool(getattr(self.config, 'ENABLE_OMICS_FRAME_CACHE', True))
        shared_key = self._shared_omics_cache_key(file_path)
        if shared_enabled:
            shared = SLDataProcessor._omics_frame_shared_cache.get(shared_key)
            if shared is not None:
                SLDataProcessor._omics_frame_shared_cache.move_to_end(shared_key)
                logger.info(f"复用共享 {omics_type} 数据帧: {shared.shape}")
                self._omics_frame_cache[cache_key] = shared
                return shared

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"组学文件不存在: {file_path}")

        logger.info(f"加载 {omics_type} 数据从 {file_path}")
        frame = None
        for encoding in ('utf-8', 'latin-1', 'cp1252'):
            try:
                frame = pd.read_csv(file_path, index_col=0, encoding=encoding)
                logger.info(f"使用编码 {encoding} 成功加载数据")
                break
            except UnicodeDecodeError:
                continue
        if frame is None:
            raise ValueError(f"无法读取组学文件: {file_path}")

        logger.info(f"原始数据形状: {frame.shape}")
        if frame.shape[0] <= frame.shape[1]:
            logger.info("检测到基因在列中，进行转置")
            frame = frame.T
        else:
            logger.info("检测到基因在行中，保持原格式")
        logger.info(f"处理后数据形状: {frame.shape}")

        if self.config.USE_ID_ANCHORING:
            logger.info(f"将 {omics_type} 数据中的基因名称转换为ENTREZID...")
            mapped_index = []
            missing_count = 0
            for gene_name in frame.index:
                gene_id = self.id_mapper.get_id_by_any_name(gene_name)
                mapped_index.append(gene_id or gene_name)
                missing_count += int(not gene_id)
            frame.index = mapped_index
            if len(set(mapped_index)) != len(mapped_index):
                logger.warning(f"{omics_type} 数据有重复的基因ID，将取平均值")
                frame = frame.groupby(frame.index).mean()
            logger.info(f"ID转换后基因数: {len(frame)}，缺失映射: {missing_count}条")
        elif self.config.USE_GENE_NAME_NORMALIZATION:
            logger.info(f"标准化 {omics_type} 数据中的基因名称...")
            normalized_index = self.gene_normalizer.batch_normalize(
                frame.index.tolist()
            )
            frame.index = normalized_index
            if len(set(normalized_index)) != len(normalized_index):
                logger.warning(f"{omics_type} 数据有重复的标准化基因名称，将取平均值")
                frame = frame.groupby(frame.index).mean()
            logger.info(f"标准化后基因数: {len(frame)}")

        self._omics_frame_cache[cache_key] = frame
        if shared_enabled:
            cls = SLDataProcessor
            max_entries = int(getattr(
                self.config, 'OMICS_FRAME_CACHE_MAX_ENTRIES', 12
            ))
            cls._omics_frame_shared_cache[shared_key] = frame
            cls._omics_frame_shared_cache.move_to_end(shared_key)
            while len(cls._omics_frame_shared_cache) > max_entries:
                cls._omics_frame_shared_cache.popitem(last=False)
        return frame

    def load_multi_omics_features(self, gene_ids, mode='train'):
        """加载多组学数据并整合为节点特征（支持ID锚定，强制GPU）

        Args:
            gene_ids: 基因ID列表
            mode: 'train' - 训练模式（拟合标准化器）
                  'val' - 验证模式（仅转换）
                  'test' - 测试模式（仅转换）
        """
        logger.info(f"加载多组学特征 (模式: {mode})...")

        all_features = []

        for omics_type, file_path in self.config.FEATURE_FILES.items():
            # 单个模态加载失败立即报错：静默跳过会造成特征宽度漂移，
            # 最终在维度校验处以更含糊的方式失败。
            try:
                df = self._load_omics_frame(omics_type, file_path)
            except Exception as e:
                raise RuntimeError(f"加载 {omics_type} 数据失败") from e

            # 为每个基因收集特征
            gene_features_list = []
            missing_count = 0

            for gene_id in gene_ids:
                if gene_id in df.index:
                    feature_vector = df.loc[gene_id].fillna(0).astype(np.float32).values
                    gene_features_list.append(feature_vector)
                else:
                    # 单个基因缺失用零表示该模态未观测，不代替整个特征管线。
                    feature_dim = df.shape[1]
                    gene_features_list.append(np.zeros(feature_dim, dtype=np.float32))
                    missing_count += 1

            if gene_features_list:
                feature_matrix = np.array(gene_features_list)
                all_features.append(feature_matrix)
                logger.info(
                    f"{omics_type}: 匹配基因数 {len(gene_features_list) - missing_count}/{len(gene_ids)}")

        if len(all_features) == 0:
            message = "所有组学数据加载失败，无法构造节点特征"
            if not getattr(self.config, 'ALLOW_ZERO_FEATURE_FALLBACK', False):
                raise RuntimeError(message)
            logger.warning("%s；按配置使用零特征", message)
            return self.create_zero_features(len(gene_ids))

        # 拼接所有特征
        logger.info("拼接多组学特征...")
        try:
            combined_features = np.concatenate(all_features, axis=1)
            logger.info(f"合并后特征维度: {combined_features.shape}")

            # 根据模式处理特征
            if mode == 'train':
                # 训练模式：拟合并转换
                normalized_features = self._fit_scaler_pca_on_train(combined_features)
            else:
                # 验证/测试模式：仅转换（使用训练集拟合的参数）
                if not self.is_scaler_fitted:
                    raise ValueError("必须先训练模型以拟合标准化器")
                normalized_features = self._transform_with_fitted_scaler_pca(combined_features)

            expected_dim = int(getattr(
                self.config, 'OMICS_FEATURE_DIM', 262
            ))
            if normalized_features.ndim != 2 or normalized_features.shape[1] != expected_dim:
                raise ValueError(
                    "预处理后的组学特征维度与模型配置不一致: "
                    f"actual={tuple(normalized_features.shape)}, "
                    f"expected=(*, {expected_dim})"
                )

            # 创建张量时直接放到目标设备
            return torch.tensor(normalized_features, dtype=torch.float).to(self.device_manager.target_device)

        except Exception as e:
            logger.error(f"特征处理失败: {e}")
            if not getattr(self.config, 'ALLOW_ZERO_FEATURE_FALLBACK', False):
                raise RuntimeError("组学特征处理失败") from e
            return self.create_zero_features(len(gene_ids))

    def create_zero_features(self, num_nodes, feature_dim=None):
        """创建零节点特征（备用，强制GPU）"""
        if feature_dim is None:
            feature_dim = int(getattr(self.config, 'OMICS_FEATURE_DIM', 262))
        logger.info(f"使用零节点特征，节点数: {num_nodes}, 维度: {feature_dim}")
        # 创建张量时直接放到目标设备
        return torch.zeros(num_nodes, feature_dim).to(self.device_manager.target_device)

    def _create_single_graph_data(self, data_df, mode='train'):
        """为单个数据集创建图数据

        Args:
            data_df: 数据DataFrame
            mode: 'train' - 训练模式（拟合标准化器）
                  'val' - 验证模式（仅转换）
        """
        # 创建边索引，直接放到目标设备
        edge_index = torch.tensor([
            [self.gene_id_to_idx[g1] for g1 in data_df['Gene.A']],
            [self.gene_id_to_idx[g2] for g2 in data_df['Gene.B']]
        ], dtype=torch.long).to(self.device_manager.target_device)

        # 边标签，直接放到目标设备
        edge_label = torch.tensor(data_df['label'].values, dtype=torch.float).to(
            self.device_manager.target_device)

        # 获取当前数据集的基因ID列表
        gene_ids_in_data = list(self.idx_to_gene_id.values())

        # 节点特征，根据模式调用不同的加载方法
        node_features = self.load_multi_omics_features(gene_ids_in_data, mode=mode)

        # 创建PyG Data对象
        graph_data = Data(
            x=node_features,
            edge_index=edge_index,
            y=edge_label,
            num_nodes=len(self.gene_id_to_idx)
        )
        self._attach_cancer_context(graph_data, data_df, mode=mode)

        logger.info(f"图数据创建完成 (模式: {mode}):")
        logger.info(f"- 节点数: {graph_data.num_nodes}")
        logger.info(f"- 边数: {graph_data.num_edges}")
        logger.info(f"- 节点特征维度: {graph_data.x.shape}")
        logger.info(f"- 数据设备: {graph_data.x.device}")

        return graph_data

    def create_gene_mapping(self, data_df):
        """创建基因ID到索引的映射（仅基于输入数据）"""
        # 获取所有唯一基因
        all_genes = set(data_df['Gene.A']).union(set(data_df['Gene.B']))

        # 如果是ID模式，确保我们处理的是ID
        if self.config.USE_ID_ANCHORING:
            # 验证ID格式
            valid_ids = []
            for gene_id in all_genes:
                if isinstance(gene_id, (int, float, str)) and str(gene_id).isdigit():
                    valid_ids.append(str(int(gene_id) if isinstance(gene_id, float) else gene_id))
                else:
                    logger.warning(f"非标准ID格式: {gene_id}")
                    valid_ids.append(str(gene_id))

            all_gene_ids = sorted(set(valid_ids), key=str)
        else:
            # 进一步标准化（确保一致性）
            if self.config.USE_GENE_NAME_NORMALIZATION:
                all_genes_list = sorted(all_genes, key=str)
                normalized_genes = self.gene_normalizer.batch_normalize(all_genes_list)
                all_gene_ids = sorted(set(normalized_genes), key=str)
            else:
                all_gene_ids = sorted(all_genes, key=str)

        # 创建映射
        self.gene_id_to_idx = {gene_id: idx for idx, gene_id in enumerate(all_gene_ids)}
        self.idx_to_gene_id = {idx: gene_id for gene_id, idx in self.gene_id_to_idx.items()}

        logger.info(f"创建了 {len(self.gene_id_to_idx)} 个基因的映射")

        # 调试：显示ID和符号的对应关系
        if self.config.DEBUG_MODE and self.config.USE_ID_ANCHORING:
            logger.debug("基因ID映射示例（前10个）:")
            for i, (gene_id, idx) in enumerate(list(self.gene_id_to_idx.items())[:10]):
                symbol = self.id_mapper.get_symbol_by_id(gene_id)
                logger.debug(f"  ID: {gene_id}, 符号: {symbol}, 索引: {idx}")

    def filter_data_by_gene_mapping(self, data_df):
        """过滤数据，只保留在基因映射中的基因对"""
        filtered_rows = []
        removed_count = 0

        for idx, row in data_df.iterrows():
            gene_a = row['Gene.A']
            gene_b = row['Gene.B']

            if gene_a in self.gene_id_to_idx and gene_b in self.gene_id_to_idx:
                filtered_rows.append(row)
            else:
                removed_count += 1

        filtered_df = pd.DataFrame(filtered_rows) if filtered_rows else pd.DataFrame(columns=data_df.columns)

        logger.info(f"数据过滤: 原始 {len(data_df)} 条, 过滤后 {len(filtered_df)} 条, 移除 {removed_count} 条")

        return filtered_df

    def create_graph_data_with_split(self):
        """创建带有训练/验证划分的图数据"""
        # 加载SL数据
        sl_data = self.load_sl_data()

        # 加载非SL数据
        non_sl_data = self.load_non_sl_data(sample_size=len(sl_data))

        # 合并并平衡数据
        balanced_data = self.combine_and_balance_data(sl_data, non_sl_data)

        # 划分数据
        train_data, val_data = self.split_data(balanced_data)

        # 保存划分后的数据
        train_data_path = os.path.join(self.config.OUTPUT_DIR, 'train_data.csv')
        val_data_path = os.path.join(self.config.OUTPUT_DIR, 'val_data.csv')
        train_data.to_csv(train_data_path, index=False)
        val_data.to_csv(val_data_path, index=False)
        logger.info(f"训练数据已保存至: {train_data_path}")
        logger.info(f"验证数据已保存至: {val_data_path}")

        # 1. 只使用训练集创建基因映射
        logger.info("基于训练集创建基因映射...")
        self.create_gene_mapping(train_data)  # 只传训练集

        # 2. 过滤验证集，只保留训练集中出现的基因
        logger.info("过滤验证集（只保留训练集中出现的基因）...")
        val_data_filtered = self.filter_data_by_gene_mapping(val_data)

        # 检查验证集是否为空
        if len(val_data_filtered) == 0:
            raise ValueError("验证集为空，请检查数据划分策略及基因映射")
        else:
            logger.info(f"验证集过滤后: {len(val_data_filtered)} 条边")

        # 3. 为训练集基因加载特征（拟合标准化器和PCA）
        logger.info("为训练集基因加载多组学特征（拟合标准化器和PCA）...")
        train_gene_ids = list(self.idx_to_gene_id.values())
        train_node_features = self.load_multi_omics_features(train_gene_ids, mode='train')

        # 保存特征名称
        self.save_feature_names(self.config.OUTPUT_DIR)

        # 4. 创建训练图数据
        train_graph_data = self._create_single_graph_data_with_features(
            train_data, train_node_features, 'train'
        )

        # 5. 为验证集创建图数据（使用训练集拟合的标准化器转换）
        val_graph_data = self._create_single_graph_data_with_features(
            val_data_filtered, train_node_features, 'val'
        )

        # 6. 构建知识图谱（只使用训练集基因）
        train_gene_id_set = set(train_gene_ids)
        logger.info(f"使用训练集基因构建知识图谱，基因数: {len(train_gene_id_set)}")
        # Preserve the stable mapping order so equivalent C1 calls can reuse
        # the process-wide KG cache without changing node indices.
        kg_data, _ = self.kg_builder.create_knowledge_graph_data(train_gene_ids)

        # Public model contract: gene_mapping always maps gene ID -> SL node index.
        gene_mapping = dict(self.gene_id_to_idx)
        return train_graph_data, val_graph_data, kg_data, gene_mapping, train_data, val_data_filtered

    def create_integrated_graph_data_with_split(self):
        """创建集成图数据"""
        # 直接调用修复后的 create_graph_data_with_split 方法
        # 它会返回所有我们需要的数据
        train_graph_data, val_graph_data, kg_data, gene_mapping, train_data, val_data = self.create_graph_data_with_split()

        # 调试信息：打印知识图谱的边类型
        logger.info(f"知识图谱边类型数量: {len(kg_data.edge_types)}")
        for edge_type in kg_data.edge_types:
            if hasattr(kg_data[edge_type], 'edge_index'):
                num_edges = kg_data[edge_type].edge_index.shape[1] if hasattr(kg_data[edge_type].edge_index,
                                                                              'shape') else 0
                edge_str = str(edge_type).replace("'", "").replace("(", "").replace(")", "")
                logger.info(f"  边类型 {edge_str}: {num_edges}条边")

        return train_graph_data, val_graph_data, kg_data, gene_mapping, train_data, val_data

    def _create_single_graph_data_with_features(self, data_df, node_features, mode='train'):
        """使用给定的节点特征创建图数据"""
        # 确保数据中的基因都在特征矩阵中
        edge_src_indices = []
        edge_dst_indices = []
        edge_labels = []
        valid_count = 0
        skipped_count = 0
        valid_row_indices = []

        for row_index, row in data_df.iterrows():
            gene_a = row['Gene.A']
            gene_b = row['Gene.B']

            if gene_a in self.gene_id_to_idx and gene_b in self.gene_id_to_idx:
                edge_src_indices.append(self.gene_id_to_idx[gene_a])
                edge_dst_indices.append(self.gene_id_to_idx[gene_b])
                edge_labels.append(row['label'])
                valid_row_indices.append(row_index)
                valid_count += 1
            else:
                skipped_count += 1

        if skipped_count > 0:
            logger.warning(f"跳过 {skipped_count} 条边（基因不在映射中）")

        if valid_count == 0:
            logger.error(f"没有有效的边可以创建 {mode} 图数据")
            # 创建一个空的图数据
            empty_graph = Data(
                x=node_features,
                edge_index=torch.tensor([[], []], dtype=torch.long).to(node_features.device),
                y=torch.tensor([], dtype=torch.float).to(node_features.device),
                num_nodes=node_features.shape[0]
            )
            return empty_graph

        # 创建边索引
        edge_index = torch.tensor([edge_src_indices, edge_dst_indices], dtype=torch.long)

        # 边标签
        edge_label = torch.tensor(edge_labels, dtype=torch.float)

        # 创建图数据，确保所有张量在同一个设备上
        device = node_features.device
        graph_data = Data(
            x=node_features,
            edge_index=edge_index.to(device),
            y=edge_label.to(device),
            num_nodes=node_features.shape[0]
        )
        valid_frame = data_df.loc[valid_row_indices].reset_index(drop=True)
        self._attach_cancer_context(graph_data, valid_frame, mode=mode)

        logger.info(f"创建 {mode} 图数据: {valid_count} 条边")
        return graph_data

    def save_scaler_and_pca(self, save_dir=None):
        """保存标准化器和PCA模型到指定目录"""
        if save_dir is None:
            save_dir = self.config.OUTPUT_DIR

        os.makedirs(save_dir, exist_ok=True)

        try:
            # 保存StandardScaler
            scaler_path = os.path.join(save_dir, 'scaler.pkl')
            import joblib

            # 修改：不再仅仅依赖is_scaler_fitted标志，而是检查scaler的实际属性
            if hasattr(self.scaler, 'n_features_in_') or hasattr(self.scaler, 'mean_'):
                # scaler有拟合属性，即使is_scaler_fitted为False也可能已经拟合了
                logger.info(f"标准化器有拟合属性，将保存。n_features_in_: {getattr(self.scaler, 'n_features_in_', '无')}")
                logger.info(
                    f"标准化器mean形状: {getattr(self.scaler, 'mean_', '无').shape if hasattr(self.scaler, 'mean_') else '无'}")
                logger.info(f"is_scaler_fitted标志: {self.is_scaler_fitted}")

                joblib.dump(self.scaler, scaler_path)
                logger.info(f"标准化器已保存至: {scaler_path}")
            else:
                logger.error("标准化器没有拟合属性，无法保存！")
                return False

            # 保存PCA（如果存在）
            if self.pca is not None and hasattr(self.pca, 'components_'):
                pca_path = os.path.join(save_dir, 'pca.pkl')
                joblib.dump(self.pca, pca_path)
                logger.info(f"PCA模型已保存至: {pca_path}")
                logger.info(f"PCA主成分数: {self.pca.n_components_}")
            else:
                logger.info("未使用PCA，跳过保存PCA模型")

            # 保存其他相关信息
            info_path = os.path.join(save_dir, 'preprocessing_info.pkl')
            preprocessing_info = {
                'is_scaler_fitted': hasattr(self.scaler, 'n_features_in_') or hasattr(self.scaler, 'mean_'),
                'feature_dim': self.scaler.n_features_in_ if hasattr(self.scaler, 'n_features_in_') else None,
                'pca_components': self.pca.n_components_ if self.pca is not None and hasattr(self.pca,
                                                                                             'n_components_') else None,
                'pca_explained_variance': self.pca.explained_variance_ratio_.sum() if self.pca is not None and hasattr(
                    self.pca, 'explained_variance_ratio_') else None,
                'scaler_mean_shape': self.scaler.mean_.shape if hasattr(self.scaler, 'mean_') else None,
                'scaler_scale_shape': self.scaler.scale_.shape if hasattr(self.scaler, 'scale_') else None,
                'scaler_actual_fitted': hasattr(self.scaler, 'n_features_in_'),
                'cancer_vocabulary': self.cancer_vocabulary.state_dict(),
            }
            joblib.dump(preprocessing_info, info_path)
            logger.info(f"预处理信息已保存至: {info_path}")

            # 更新is_scaler_fitted标志
            self.is_scaler_fitted = preprocessing_info['is_scaler_fitted']

            # 同时保存为JSON便于查看
            json_path = os.path.join(save_dir, 'preprocessing_info.json')
            import json
            with open(json_path, 'w') as f:
                # 转换numpy数组为列表以便JSON序列化
                json_info = {}
                for k, v in preprocessing_info.items():
                    if isinstance(v, np.ndarray):
                        json_info[k] = v.tolist()
                    elif hasattr(v, 'shape'):
                        json_info[k] = str(v)
                    else:
                        json_info[k] = v
                json.dump(json_info, f, indent=2)
            logger.info(f"预处理信息JSON已保存至: {json_path}")

            return True
        except Exception as e:
            logger.error(f"保存标准化器和PCA失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def load_scaler_and_pca(self, load_dir=None):
        """从指定目录加载标准化器和PCA模型"""
        if load_dir is None:
            load_dir = self.config.OUTPUT_DIR

        try:
            scaler_path = os.path.join(load_dir, 'scaler.pkl')
            pca_path = os.path.join(load_dir, 'pca.pkl')
            info_path = os.path.join(load_dir, 'preprocessing_info.pkl')

            if not os.path.exists(scaler_path):
                logger.warning(f"标准化器文件不存在: {scaler_path}")
                return False

            import joblib

            # 加载标准化器
            self.scaler = joblib.load(scaler_path)
            logger.info(f"标准化器已加载: {scaler_path}")

            # 验证scaler是否已拟合
            if not hasattr(self.scaler, 'n_features_in_'):
                logger.error("加载的标准化器未拟合！")
                self.is_scaler_fitted = False
                # 重新初始化为新scaler
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
                logger.warning("已重新初始化为新标准化器")
                return False
            else:
                self.is_scaler_fitted = True
                logger.info(f"标准化器特征维度: {self.scaler.n_features_in_}")

            # 加载PCA（如果存在）
            if os.path.exists(pca_path):
                self.pca = joblib.load(pca_path)
                if hasattr(self.pca, 'components_'):
                    logger.info(f"PCA模型已加载: {pca_path}")
                    logger.info(f"PCA主成分数: {self.pca.n_components_}")
                else:
                    logger.warning("PCA模型未正确拟合")
                    self.pca = None
            else:
                self.pca = None
                logger.info("未找到PCA模型，使用原始特征")

            # 加载预处理信息
            if os.path.exists(info_path):
                preprocessing_info = joblib.load(info_path)
                self.is_scaler_fitted = preprocessing_info.get('is_scaler_fitted', False)
                feature_dim = preprocessing_info.get('feature_dim', None)
                logger.info(f"预处理信息已加载，特征维度: {feature_dim}")

                # 验证一致性
                if feature_dim is not None and hasattr(self.scaler, 'n_features_in_'):
                    if feature_dim != self.scaler.n_features_in_:
                        logger.warning(f"特征维度不一致: 预处理信息={feature_dim}, scaler={self.scaler.n_features_in_}")

            return True
        except Exception as e:
            logger.error(f"加载标准化器和PCA失败: {e}")
            # 初始化新scaler
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            self.pca = None
            self.is_scaler_fitted = False
            return False

    def save_gene_mapping(self, save_dir=None):
        """保存基因映射到指定目录"""
        if save_dir is None:
            save_dir = self.config.OUTPUT_DIR

        os.makedirs(save_dir, exist_ok=True)

        try:
            # 保存基因ID到索引的映射
            mapping_path = os.path.join(save_dir, 'gene_mapping.pkl')
            import joblib
            mapping_data = {
                'gene_id_to_idx': self.gene_id_to_idx,
                'idx_to_gene_id': self.idx_to_gene_id
            }
            joblib.dump(mapping_data, mapping_path)
            logger.info(f"基因映射已保存至: {mapping_path}")

            # 同时保存为CSV便于查看
            csv_path = os.path.join(save_dir, 'gene_mapping.csv')
            mapping_df = pd.DataFrame({
                'gene_id': list(self.gene_id_to_idx.keys()),
                'idx': list(self.gene_id_to_idx.values())
            })
            mapping_df.to_csv(csv_path, index=False)
            logger.info(f"基因映射CSV已保存至: {csv_path}")

            # 额外保存一个CSV文件，包含基因ID、符号和索引的对应关系
            gene_ids = list(self.gene_id_to_idx.keys())
            symbols = [self.id_mapper.get_symbol_by_id(gid) for gid in gene_ids]
            df = pd.DataFrame({
                'gene_id': gene_ids,
                'symbol': symbols,
                'idx': [self.gene_id_to_idx[gid] for gid in gene_ids]
            })
            csv_path = os.path.join(save_dir, 'gene_id_to_symbol.csv')
            df.to_csv(csv_path, index=False)
            logger.info(f"基因ID-符号映射CSV已保存至: {csv_path}")

            return True
        except Exception as e:
            logger.error(f"保存基因映射失败: {e}")
            return False

    def load_gene_mapping(self, load_dir=None):
        """从指定目录加载基因映射"""
        if load_dir is None:
            load_dir = self.config.OUTPUT_DIR

        try:
            mapping_path = os.path.join(load_dir, 'gene_mapping.pkl')

            if not os.path.exists(mapping_path):
                logger.warning(f"基因映射文件不存在: {mapping_path}")
                return False

            import joblib
            mapping_data = joblib.load(mapping_path)

            self.gene_id_to_idx = mapping_data['gene_id_to_idx']
            self.idx_to_gene_id = mapping_data['idx_to_gene_id']

            logger.info(f"基因映射已加载，共 {len(self.gene_id_to_idx)} 个基因")
            return True
        except Exception as e:
            logger.error(f"加载基因映射失败: {e}")
            return False

    def save_processing_models(self, save_dir=None):
        """保存标准化器和PCA模型（仅在非交叉验证的单次训练中调用）"""
        if save_dir is None:
            save_dir = self.config.OUTPUT_DIR

        # 确保输出目录存在
        os.makedirs(save_dir, exist_ok=True)

        # 保存基因映射
        self.save_gene_mapping(save_dir)

        # 保存标准化器和PCA
        scaler_saved = self.save_scaler_and_pca(save_dir)

        if scaler_saved:
            logger.info(f"预处理模型已成功保存到: {save_dir}")
        else:
            logger.warning(f"标准化器保存失败，但继续保存其他预处理信息")

        # 保存额外的处理信息
        self._save_processing_info(save_dir)

        return scaler_saved

    def _save_processing_info(self, save_dir):
        """保存额外的处理信息"""
        import joblib

        processing_info = {
            'total_genes': len(self.gene_id_to_idx),
            'feature_dim': self.scaler.n_features_in_ if hasattr(self.scaler, 'n_features_in_') else None,
            'is_scaler_fitted': self.is_scaler_fitted,
            'pca_components': self.pca.n_components_ if self.pca is not None else None,
            'use_id_anchoring': self.config.USE_ID_ANCHORING,
            'use_gene_name_normalization': self.config.USE_GENE_NAME_NORMALIZATION,
            'config_info': {
                'TOTAL_SAMPLES_PER_CLASS': self.config.TOTAL_SAMPLES_PER_CLASS,
                'TRAIN_VAL_SPLIT_RATIO': self.config.TRAIN_VAL_SPLIT_RATIO,
                'RANDOM_SEED': self.config.RANDOM_SEED
            }
        }

        info_path = os.path.join(save_dir, 'processing_info.pkl')
        joblib.dump(processing_info, info_path)
        logger.info(f"处理信息已保存: {info_path}")

    def get_preprocessing_info(self):
        """获取预处理信息（用于保存）"""
        info = {
            'gene_id_to_idx': self.gene_id_to_idx,
            'idx_to_gene_id': self.idx_to_gene_id,
            'is_scaler_fitted': self.is_scaler_fitted,
            'scaler_mean': self.scaler.mean_ if hasattr(self.scaler, 'mean_') else None,
            'scaler_scale': self.scaler.scale_ if hasattr(self.scaler, 'scale_') else None,
            'pca_components': self.pca.components_ if self.pca is not None else None,
            'pca_explained_variance': self.pca.explained_variance_ratio_ if self.pca is not None else None,
            'config': {
                'USE_ID_ANCHORING': self.config.USE_ID_ANCHORING,
                'USE_GENE_NAME_NORMALIZATION': self.config.USE_GENE_NAME_NORMALIZATION,
                'FEATURE_FILES': dict(self.config.FEATURE_FILES)
            }
        }
        return info

    def save_feature_names(self, save_dir):
        """保存所有组学特征的特征名称到 CSV 文件"""
        names = []
        for omics_type, file_path in self.config.FEATURE_FILES.items():
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, nrows=0)  # 只读列名
                if df.shape[1] > 0:
                    cols = df.columns.tolist()
                    names.extend([f"{omics_type}_{col}" for col in cols])
                else:
                    names.append(omics_type)
        feature_df = pd.DataFrame({'feature_name': names, 'index': range(len(names))})
        feature_df.to_csv(os.path.join(save_dir, 'feature_names.csv'), index=False)
        logger.info(f"特征名称已保存至 {os.path.join(save_dir, 'feature_names.csv')}")


class InductiveSLDataProcessor(SLDataProcessor):
    """支持归纳式学习的数据处理器"""

    def __init__(self, config=None, device_manager=None):
        super().__init__(config, device_manager)

        # 替换为归纳式知识图谱构建器
        self.kg_builder = InductiveKnowledgeGraphBuilder(
            base_builder=self.kg_builder,
            config=self.config,
            device_manager=self.device_manager
        )

        # 新增：用于新基因特征初始化的邻居缓存
        self.gene_neighbor_cache = {}

        # 新增：新基因检测和处理的标记
        self.new_genes_detected = False
        self.new_gene_embeddings = {}  # 缓存新基因的嵌入

        # Strict-inductive state. Only training genes/features are retained here;
        # validation and test contexts are built independently from this state.
        self.strict_train_gene_ids = None
        self.strict_train_gene_mapping = None
        self.strict_train_node_features = None
        self.strict_stage_contexts = {}
        self.strict_stage_kgs = {}
        self.strict_stage_kg_gene_ids = {}

    def _canonicalize_gene_id(self, gene_id):
        """Return the same stable gene representation used by gene mappings."""
        if not self.config.USE_ID_ANCHORING:
            return gene_id

        if pd.isna(gene_id):
            raise ValueError("Gene ID cannot be missing in strict inductive mode")

        if isinstance(gene_id, (int, np.integer)):
            return str(int(gene_id))
        if isinstance(gene_id, (float, np.floating)) and float(gene_id).is_integer():
            return str(int(gene_id))
        return str(gene_id).strip()

    def _canonicalize_pair_frame(self, data_df):
        """Normalize pair identifiers without mutating the authoritative split."""
        required_columns = {'Gene.A', 'Gene.B', 'label'}
        missing_columns = required_columns.difference(data_df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
        if data_df.empty:
            raise ValueError("Strict inductive stages cannot be built from an empty split")

        normalized = data_df.copy().reset_index(drop=True)
        normalized['Gene.A'] = normalized['Gene.A'].map(self._canonicalize_gene_id)
        normalized['Gene.B'] = normalized['Gene.B'].map(self._canonicalize_gene_id)
        return normalized

    def _create_graph_for_mapping(self, data_df, node_features, gene_mapping, mode):
        """Create one stage-local graph and reject cross-stage index mismatches."""
        missing_genes = (
            set(data_df['Gene.A']).union(set(data_df['Gene.B']))
            - set(gene_mapping)
        )
        if missing_genes:
            examples = sorted(missing_genes, key=str)[:5]
            raise ValueError(f"{mode} graph contains unmapped genes: {examples}")
        if node_features.shape[0] != len(gene_mapping):
            raise ValueError(
                f"{mode} node feature rows ({node_features.shape[0]}) do not match "
                f"mapping size ({len(gene_mapping)})"
            )

        edge_index = torch.tensor([
            [gene_mapping[gene_id] for gene_id in data_df['Gene.A']],
            [gene_mapping[gene_id] for gene_id in data_df['Gene.B']]
        ], dtype=torch.long, device=self.device_manager.target_device)
        edge_label = torch.tensor(
            data_df['label'].values,
            dtype=torch.float,
            device=self.device_manager.target_device
        )

        graph_data = Data(
            x=node_features,
            edge_index=edge_index,
            y=edge_label,
            num_nodes=len(gene_mapping)
        )
        self._attach_cancer_context(graph_data, data_df, mode=mode)
        logger.info(
            f"严格归纳图创建完成 ({mode}): nodes={graph_data.num_nodes}, "
            f"edges={graph_data.num_edges}, features={tuple(graph_data.x.shape)}"
        )
        return graph_data

    def _restrict_kg_to_train_genes(self, kg_data, train_gene_ids):
        """Remove every non-training Gene node and its incident KG edges."""
        if 'Gene' not in kg_data.node_types:
            raise ValueError("Knowledge graph has no Gene node type")

        gene_store = kg_data['Gene']
        gene_mapping = dict(getattr(gene_store, 'node_name_to_idx', {}))
        allowed_gene_ids = {str(gene_id) for gene_id in train_gene_ids}
        kept_items = sorted(
            (
                (str(gene_id), int(old_idx))
                for gene_id, old_idx in gene_mapping.items()
                if str(gene_id) in allowed_gene_ids
            ),
            key=lambda item: item[1]
        )
        if not kept_items:
            raise ValueError("No training genes remain after restricting the knowledge graph")

        kept_old_indices = torch.tensor(
            [old_idx for _, old_idx in kept_items],
            dtype=torch.long,
            device=gene_store.x.device
        )
        old_to_new = torch.full(
            (gene_store.x.shape[0],), -1,
            dtype=torch.long,
            device=gene_store.x.device
        )
        old_to_new[kept_old_indices] = torch.arange(
            len(kept_items), device=gene_store.x.device
        )

        for edge_type in list(kg_data.edge_types):
            head_type, _, tail_type = edge_type
            edge_index = kg_data[edge_type].edge_index
            keep_mask = torch.ones(
                edge_index.shape[1], dtype=torch.bool, device=edge_index.device
            )

            if head_type == 'Gene':
                head_remap = old_to_new.to(edge_index.device)
                keep_mask &= head_remap[edge_index[0]] >= 0
            if tail_type == 'Gene':
                tail_remap = old_to_new.to(edge_index.device)
                keep_mask &= tail_remap[edge_index[1]] >= 0

            restricted_edges = edge_index[:, keep_mask].clone()
            if head_type == 'Gene' and restricted_edges.numel() > 0:
                restricted_edges[0] = old_to_new.to(edge_index.device)[restricted_edges[0]]
            if tail_type == 'Gene' and restricted_edges.numel() > 0:
                restricted_edges[1] = old_to_new.to(edge_index.device)[restricted_edges[1]]
            kg_data[edge_type].edge_index = restricted_edges

        gene_store.x = gene_store.x.index_select(0, kept_old_indices)
        gene_store.node_name_to_idx = {
            gene_id: new_idx for new_idx, (gene_id, _) in enumerate(kept_items)
        }
        gene_store.num_nodes = len(kept_items)

        leaked_genes = set(gene_store.node_name_to_idx).difference(allowed_gene_ids)
        if leaked_genes:
            raise RuntimeError("Non-training genes remain in the strict training KG")

        logger.info(
            f"训练KG限制完成: Gene节点 {len(gene_mapping)} -> {len(kept_items)}"
        )
        return kg_data

    def create_strict_inductive_training_data(self, train_data, val_data):
        """Build leak-free train and validation contexts for C2/C3."""
        train_data = self._canonicalize_pair_frame(train_data)
        val_data = self._canonicalize_pair_frame(val_data)

        self.create_gene_mapping(train_data)
        train_gene_mapping = dict(self.gene_id_to_idx)
        train_gene_ids = [
            self.idx_to_gene_id[idx] for idx in range(len(self.idx_to_gene_id))
        ]
        train_node_features = self.load_multi_omics_features(train_gene_ids, mode='train')
        train_graph_data = self._create_graph_for_mapping(
            train_data, train_node_features, train_gene_mapping, mode='train'
        )

        self.strict_train_gene_ids = list(train_gene_ids)
        self.strict_train_gene_mapping = dict(train_gene_mapping)
        self.strict_train_node_features = train_node_features
        self.strict_stage_contexts = {
            'train': {
                'gene_ids': list(train_gene_ids),
                'gene_mapping': dict(train_gene_mapping)
            }
        }
        self.strict_stage_kgs = {}
        self.strict_stage_kg_gene_ids = {}

        kg_data, _ = self.kg_builder.build_train_kg(train_gene_ids)
        kg_data = self._restrict_kg_to_train_genes(kg_data, train_gene_ids)
        self.strict_stage_kgs['train'] = kg_data
        self.strict_stage_kg_gene_ids['train'] = list(train_gene_ids)

        val_graph_data, val_gene_mapping, val_gene_ids, val_data = \
            self.create_strict_inductive_evaluation_data(val_data, mode='val')

        logger.info(
            "严格归纳训练上下文已建立: "
            f"train_nodes={len(train_gene_ids)}, val_nodes={len(val_gene_ids)}, "
            "test_nodes=0"
        )
        return (
            train_graph_data,
            val_graph_data,
            kg_data,
            train_gene_mapping,
            val_gene_mapping,
            train_gene_ids,
            val_gene_ids,
            train_data,
            val_data
        )

    def prepare_strict_stage_kg(self, eval_data, mode='test'):
        """Build a label-free train+evaluation KG context from gene identities only."""
        if mode not in {'val', 'test'}:
            raise ValueError("Strict KG mode must be 'val' or 'test'")
        if self.strict_train_gene_ids is None:
            raise RuntimeError("Training context must be created before KG contexts")

        required_columns = {'Gene.A', 'Gene.B'}
        missing_columns = required_columns.difference(eval_data.columns)
        if missing_columns:
            raise ValueError(f"Missing required gene columns: {sorted(missing_columns)}")
        normalized = eval_data[['Gene.A', 'Gene.B']].copy().reset_index(drop=True)
        normalized['Gene.A'] = normalized['Gene.A'].map(self._canonicalize_gene_id)
        normalized['Gene.B'] = normalized['Gene.B'].map(self._canonicalize_gene_id)
        train_gene_set = set(self.strict_train_gene_ids)
        eval_gene_set = set(normalized['Gene.A']).union(set(normalized['Gene.B']))
        new_gene_ids = sorted(eval_gene_set.difference(train_gene_set), key=str)
        stage_gene_ids = list(self.strict_train_gene_ids) + new_gene_ids

        cached_gene_ids = getattr(self, 'strict_stage_kg_gene_ids', {}).get(mode)
        if cached_gene_ids == stage_gene_ids and mode in getattr(self, 'strict_stage_kgs', {}):
            return self.strict_stage_kgs[mode]

        if not hasattr(self, 'kg_builder') or self.kg_builder is None:
            return None

        if not hasattr(self, 'strict_stage_kgs'):
            self.strict_stage_kgs = {}
        if not hasattr(self, 'strict_stage_kg_gene_ids'):
            self.strict_stage_kg_gene_ids = {}
        stage_kg = self.kg_builder.build_all_genes_kg(stage_gene_ids)
        if bool(getattr(self.config, 'KEEP_STAGE_KG_ON_CPU', True)) \
                and hasattr(stage_kg, 'to'):
            # Stage KGs are only consumed inside no_grad forwards (validation
            # every GATE_MONITOR_INTERVAL epochs, calibration, test eval);
            # parking them on CPU avoids holding several million-edge graphs
            # on the accelerator.  The model encoder moves tensors to the
            # compute device per forward, so only residency changes.
            stage_kg = stage_kg.to('cpu')
        self.strict_stage_kgs[mode] = stage_kg
        self.strict_stage_kg_gene_ids[mode] = list(stage_gene_ids)
        return self.strict_stage_kgs[mode]

    def create_strict_inductive_evaluation_data(self, eval_data, mode='test'):
        """Build train+eval nodes without importing nodes from another eval stage."""
        if mode not in {'val', 'test'}:
            raise ValueError("Strict evaluation mode must be 'val' or 'test'")
        if self.strict_train_gene_ids is None or self.strict_train_node_features is None:
            raise RuntimeError("Training context must be created before evaluation contexts")

        eval_data = self._canonicalize_pair_frame(eval_data)
        train_gene_set = set(self.strict_train_gene_ids)
        eval_gene_set = set(eval_data['Gene.A']).union(set(eval_data['Gene.B']))
        new_gene_ids = sorted(eval_gene_set.difference(train_gene_set), key=str)
        if mode == 'val':
            self.new_genes_detected = bool(new_gene_ids)
        stage_gene_ids = list(self.strict_train_gene_ids) + new_gene_ids
        stage_gene_mapping = {
            gene_id: idx for idx, gene_id in enumerate(stage_gene_ids)
        }

        if new_gene_ids:
            new_gene_features = self.load_multi_omics_features(new_gene_ids, mode=mode)
            stage_node_features = torch.cat(
                [self.strict_train_node_features, new_gene_features], dim=0
            )
        else:
            stage_node_features = self.strict_train_node_features

        graph_data = self._create_graph_for_mapping(
            eval_data, stage_node_features, stage_gene_mapping, mode=mode
        )
        self.strict_stage_contexts[mode] = {
            'gene_ids': list(stage_gene_ids),
            'gene_mapping': dict(stage_gene_mapping),
            'new_gene_ids': list(new_gene_ids)
        }
        # Static KG structure has no SL labels. It may be prepared before
        # training so every data-dependent module is registered in the optimizer.
        self.prepare_strict_stage_kg(eval_data, mode=mode)
        logger.info(
            f"严格归纳{mode}上下文: train={len(self.strict_train_gene_ids)}, "
            f"new={len(new_gene_ids)}, total={len(stage_gene_ids)}"
        )
        return graph_data, stage_gene_mapping, stage_gene_ids, eval_data

    def get_strict_stage_kg(self, mode):
        """Return the isolated static KG context for train, validation, or test."""
        if mode not in getattr(self, 'strict_stage_kgs', {}):
            raise RuntimeError(f"Strict KG context has not been built for stage: {mode}")
        return self.strict_stage_kgs[mode]

    def release_strict_stage_kg(self, mode):
        """Release a cached stage KG while preserving the registered model schema."""
        if hasattr(self, 'strict_stage_kgs'):
            self.strict_stage_kgs.pop(mode, None)
        if hasattr(self, 'strict_stage_kg_gene_ids'):
            self.strict_stage_kg_gene_ids.pop(mode, None)

    def _extend_knowledge_graph_with_new_genes(self, kg_data, new_genes):
        """Register new-gene nodes in the KG before adding edges to them."""
        if not new_genes:
            return kg_data

        gene_store = kg_data['Gene']
        kg_gene_mapping = dict(getattr(gene_store, 'node_name_to_idx', {}))
        missing_genes = [str(gene_id) for gene_id in new_genes if str(gene_id) not in kg_gene_mapping]
        if not missing_genes:
            return kg_data

        base_builder = self.kg_builder.base_builder
        feature_dim = gene_store.x.shape[1]
        device = gene_store.x.device

        if base_builder.embedding_model:
            gene_texts = []
            for gene_id in missing_genes:
                symbol = self.id_mapper.get_symbol_by_id(gene_id)
                gene_texts.append(f"Gene {symbol}" if symbol and symbol != gene_id else f"Gene ID {gene_id}")
            new_features = base_builder.batch_encode_texts(gene_texts)
            new_features = torch.tensor(new_features, dtype=gene_store.x.dtype, device=device)
        else:
            # A deterministic fallback is preferable to introducing random inference state.
            new_features = torch.zeros(
                len(missing_genes), feature_dim,
                dtype=gene_store.x.dtype, device=device
            )

        start_idx = gene_store.x.shape[0]
        gene_store.x = torch.cat([gene_store.x, new_features], dim=0)
        for offset, gene_id in enumerate(missing_genes):
            kg_gene_mapping[gene_id] = start_idx + offset
        setattr(gene_store, 'node_name_to_idx', kg_gene_mapping)

        logger.info(f"向知识图谱注册了 {len(missing_genes)} 个新基因节点")
        return kg_data

    def add_similarity_edges_for_new_genes(self, kg_data, train_gene_ids, new_genes,
                                           train_node_features, all_node_features, top_k=3):
        """为新基因添加基于特征相似度的边（避免孤立节点，无数据泄露）

        Args:
            kg_data: 知识图谱数据（只包含训练集基因）
            train_gene_ids: 训练集基因ID列表
            new_genes: 新基因ID列表
            train_node_features: 训练集基因的特征张量
            all_node_features: 所有基因的特征张量
            top_k: 每个新基因连接的最相似训练集基因数量

        Returns:
            更新后的知识图谱数据
        """
        logger.info(f"为新基因添加基于特征相似度的边，top_k={top_k}...")

        if len(new_genes) == 0 or len(train_gene_ids) == 0:
            logger.info("没有新基因或训练集基因为空，跳过相似度边添加")
            return kg_data

        try:
            kg_gene_mapping = getattr(kg_data['Gene'], 'node_name_to_idx', {})

            # Keep the SL feature index space separate from the KG node index space.
            available_train_genes = [
                gene_id for gene_id in train_gene_ids
                if gene_id in self.gene_id_to_idx and str(gene_id) in kg_gene_mapping
            ]
            available_new_genes = [
                gene_id for gene_id in new_genes
                if gene_id in self.gene_id_to_idx and str(gene_id) in kg_gene_mapping
            ]

            if len(available_train_genes) == 0 or len(available_new_genes) == 0:
                logger.warning("无法计算相似度：训练特征或新基因特征索引为空")
                return kg_data

            # 提取训练集特征和新基因特征
            train_feature_indices = [self.gene_id_to_idx[gene_id] for gene_id in available_train_genes]
            new_feature_indices = [self.gene_id_to_idx[gene_id] for gene_id in available_new_genes]
            train_features = all_node_features[train_feature_indices]
            new_features = all_node_features[new_feature_indices]

            # 计算余弦相似度
            logger.info(f"计算相似度矩阵: {new_features.shape[0]}个新基因 × {train_features.shape[0]}个训练基因")

            # 使用PyTorch计算余弦相似度
            train_features_norm = torch.nn.functional.normalize(train_features, dim=1)
            new_features_norm = torch.nn.functional.normalize(new_features, dim=1)
            similarity_matrix = torch.mm(new_features_norm, train_features_norm.t())

            # 为每个新基因添加与最相似的训练集基因的边
            new_edges_src = []
            new_edges_dst = []
            edge_count = 0

            for i, new_gene_id in enumerate(available_new_genes):
                new_gene_kg_idx = kg_gene_mapping[str(new_gene_id)]
                # 获取最相似的k个训练集基因
                similarities = similarity_matrix[i]
                top_k_actual = min(top_k, len(similarities))
                _, top_indices = torch.topk(similarities, top_k_actual)

                # 只添加相似度大于阈值的边
                similarity_threshold = float(getattr(
                    self.config, 'KG_SIMILARITY_THRESHOLD', 0.1
                ))
                for j in range(top_k_actual):
                    train_idx_in_features = top_indices[j].item()
                    similarity_score = similarities[train_idx_in_features].item()

                    if similarity_score > similarity_threshold:
                        # 获取训练集基因在图中的索引
                        train_gene_id = available_train_genes[train_idx_in_features]
                        train_gene_kg_idx = kg_gene_mapping.get(str(train_gene_id))

                        if train_gene_kg_idx is not None:
                            # 添加双向边
                            new_edges_src.append(new_gene_kg_idx)
                            new_edges_dst.append(train_gene_kg_idx)
                            new_edges_src.append(train_gene_kg_idx)
                            new_edges_dst.append(new_gene_kg_idx)
                            edge_count += 2

            if new_edges_src:
                # 创建边索引张量
                edge_index = torch.tensor([new_edges_src, new_edges_dst], dtype=torch.long)

                # 确保边索引与特征张量在同一设备上
                device = kg_data['Gene'].x.device if hasattr(kg_data['Gene'],
                                                             'x') else self.device_manager.target_device
                edge_index = edge_index.to(device)

                # 添加到知识图谱中
                edge_type = ('Gene', 'similar_to', 'Gene')

                if edge_type in kg_data.edge_types:
                    # 合并现有边
                    existing_edges = kg_data[edge_type].edge_index
                    kg_data[edge_type].edge_index = torch.cat([existing_edges, edge_index], dim=1)
                else:
                    # 创建新边类型
                    kg_data[edge_type].edge_index = edge_index

                logger.info(f"添加了 {len(new_edges_src)} 条相似度边，连接 {len(available_new_genes)} 个新基因到训练集基因")

                # 缓存新基因的连接信息
                kg_idx_to_gene = {idx: gene_id for gene_id, idx in kg_gene_mapping.items()}
                for gene_id in available_new_genes:
                    gene_kg_idx = kg_gene_mapping[str(gene_id)]
                    connected_train_genes = [
                        kg_idx_to_gene[dst]
                        for src, dst in zip(new_edges_src, new_edges_dst)
                        if src == gene_kg_idx and dst in kg_idx_to_gene
                    ]
                    self.gene_neighbor_cache[gene_id] = connected_train_genes

            else:
                logger.info("没有找到足够相似的基因对，未添加相似度边")

            return kg_data

        except Exception as e:
            logger.error(f"添加相似度边时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return kg_data

    def detect_new_genes(self, train_data, val_data):
        """检测验证集中训练集未出现的新基因"""
        train_genes = set(train_data['Gene.A']).union(set(train_data['Gene.B']))
        val_genes = set(val_data['Gene.A']).union(set(val_data['Gene.B']))

        return sorted(val_genes - train_genes, key=str)

    def initialize_new_gene_features(self, new_gene_ids, reference_genes, reference_features):
        """为新基因初始化特征（基于邻居的均值）"""
        logger.info(f"为新基因初始化特征，数量: {len(new_gene_ids)}")

        # 注意：这个方法现在可能不再需要，因为我们在add_similarity_edges_for_new_genes中处理了连接
        # 保持方法用于兼容性，但主要使用相似度边方法

        # 获取所有基因的知识图谱（用于查找邻居）
        reference_gene_ids = sorted(reference_genes, key=str)
        new_gene_list = sorted(new_gene_ids, key=str)
        all_gene_ids = reference_gene_ids + new_gene_list
        self.kg_builder.build_all_genes_kg(all_gene_ids)

        new_gene_features = []

        for gene_id in new_gene_list:
            # 查找基因在知识图谱中的邻居
            neighbors = self.kg_builder.get_gene_neighbors(gene_id, max_neighbors=5)
            self.gene_neighbor_cache[gene_id] = neighbors

            if neighbors:
                # 查找邻居在参考基因中的索引和特征
                neighbor_features = []
                for neighbor_id in neighbors:
                    if neighbor_id in reference_genes:
                        neighbor_idx = reference_genes.index(neighbor_id)
                        neighbor_features.append(reference_features[neighbor_idx])

                if neighbor_features:
                    # 使用邻居特征的平均值作为新基因的特征
                    avg_feature = torch.stack(neighbor_features).mean(dim=0)
                else:
                    # 没有找到已知邻居，使用零向量
                    avg_feature = torch.zeros_like(reference_features[0])
            else:
                # 没有邻居，使用零向量
                avg_feature = torch.zeros_like(reference_features[0])

            new_gene_features.append(avg_feature)

            # 缓存嵌入
            self.new_gene_embeddings[gene_id] = avg_feature

        if new_gene_features:
            return torch.stack(new_gene_features)
        else:
            return torch.zeros(len(new_gene_ids), reference_features.shape[1])

    def create_inductive_graph_data(self, train_data=None, val_data=None, test_data=None):
        """Create inductive graph data from one authoritative split.

        When pre-split frames are supplied by the scenario orchestrator, this method
        must not reload or repartition the dataset. The optional test frame only
        contributes unlabeled node identities/features to the shared node space.
        """
        # 1. Load legacy data only when no authoritative split was supplied.
        if train_data is None and val_data is None:
            sl_data = self.load_sl_data()
            non_sl_data = self.load_non_sl_data(sample_size=len(sl_data))
            balanced_data = self.combine_and_balance_data(sl_data, non_sl_data)
            train_data, val_data = self.split_data(balanced_data)
        elif train_data is None or val_data is None:
            raise ValueError("train_data and val_data must be supplied together")
        else:
            train_data = train_data.copy().reset_index(drop=True)
            val_data = val_data.copy().reset_index(drop=True)
            test_data = test_data.copy().reset_index(drop=True) if test_data is not None else None
            logger.info(
                f"使用上层场景划分: train={len(train_data)}, val={len(val_data)}, "
                f"test={len(test_data) if test_data is not None else 0}"
            )

        # 2. 检测新基因
        train_genes = set(train_data['Gene.A']).union(set(train_data['Gene.B']))
        val_genes = set(val_data['Gene.A']).union(set(val_data['Gene.B']))
        test_genes = set()
        if test_data is not None:
            test_genes = set(test_data['Gene.A']).union(set(test_data['Gene.B']))
        new_genes = sorted((val_genes | test_genes) - train_genes, key=str)
        self.new_genes_detected = bool(new_genes)

        logger.info(f"训练集基因数: {len(train_genes)}")
        logger.info(f"验证集基因数: {len(val_genes)}")
        logger.info(f"新基因数: {len(new_genes)}")

        # 3. 基因映射（严格只使用训练集）
        logger.info("基于训练集创建基因映射...")
        self.create_gene_mapping(train_data)

        # 4. 为新基因添加映射
        if new_genes:
            start_idx = len(self.gene_id_to_idx)
            for i, gene_id in enumerate(new_genes):
                new_idx = start_idx + i
                self.gene_id_to_idx[gene_id] = new_idx
                self.idx_to_gene_id[new_idx] = gene_id
            logger.info(f"扩展基因映射，新增 {len(new_genes)} 个新基因")

        # 5. 特征处理
        all_gene_ids = list(self.idx_to_gene_id.values())

        # 5.1 为训练集基因加载特征（拟合标准化器和PCA）
        logger.info("为训练集基因加载多组学特征...")
        train_gene_id_list = [gene_id for gene_id in all_gene_ids if gene_id in train_genes]
        train_node_features = self.load_multi_omics_features(train_gene_id_list, mode='train')

        # 5.2 为新基因加载特征（使用训练集拟合的参数）
        if new_genes:
            logger.info("为新基因加载多组学特征...")
            new_gene_features = self.load_multi_omics_features(new_genes, mode='val')

        # 5.3 构建完整特征矩阵
        feature_dim = train_node_features.shape[1]
        all_node_features = torch.zeros(len(all_gene_ids), feature_dim).to(self.device_manager.target_device)

        # 放置训练集特征
        for i, gene_id in enumerate(train_gene_id_list):
            if gene_id in self.gene_id_to_idx:
                idx = self.gene_id_to_idx[gene_id]
                all_node_features[idx] = train_node_features[i]

        # 放置新基因特征
        if new_genes:
            for i, gene_id in enumerate(new_genes):
                if gene_id in self.gene_id_to_idx:
                    idx = self.gene_id_to_idx[gene_id]
                    all_node_features[idx] = new_gene_features[i]

        # 6. 构建知识图谱（只使用训练集基因）
        logger.info("构建训练集知识图谱...")
        kg_data, _ = self.kg_builder.build_train_kg(train_gene_id_list)

        # New-gene edges are valid only after the corresponding KG nodes exist.
        kg_data = self._extend_knowledge_graph_with_new_genes(kg_data, new_genes)

        # 7. 为新基因添加相似度边（避免孤立节点）
        if new_genes:
            kg_data = self.add_similarity_edges_for_new_genes(
                kg_data, train_gene_id_list, new_genes,
                train_node_features, all_node_features, top_k=3
            )

        # 8. 创建图数据
        train_graph_data = self._create_single_graph_data_with_features(
            train_data, all_node_features, 'train'
        )

        val_graph_data = self._create_single_graph_data_with_features(
            val_data, all_node_features, 'val'
        )

        # 9. 验证数据完整性
        logger.info(f"\n数据完整性检查:")
        logger.info(f"总基因数: {len(all_gene_ids)}")
        logger.info(f"训练集基因数: {len(train_genes)}")
        logger.info(f"新基因数: {len(new_genes)}")
        logger.info(f"知识图谱基因节点数: {kg_data['Gene'].x.shape[0] if hasattr(kg_data['Gene'], 'x') else 'N/A'}")

        # 检查知识图谱中的连接
        total_edges = 0
        for edge_type in kg_data.edge_types:
            if hasattr(kg_data[edge_type], 'edge_index'):
                num_edges = kg_data[edge_type].edge_index.shape[1]
                total_edges += num_edges
                edge_str = str(edge_type).replace("'", "").replace("(", "").replace(")", "")
                logger.info(f"  边类型 {edge_str}: {num_edges}条边")

        logger.info(f"知识图谱总边数: {total_edges}")

        gene_mapping = dict(self.gene_id_to_idx)
        return train_graph_data, val_graph_data, kg_data, gene_mapping, train_data, val_data

    def _create_single_graph_data_with_features(self, data_df, node_features, mode='train'):
        """使用给定的节点特征创建图数据"""
        # 创建边索引
        edge_index = torch.tensor([
            [self.gene_id_to_idx[g1] for g1 in data_df['Gene.A']],
            [self.gene_id_to_idx[g2] for g2 in data_df['Gene.B']]
        ], dtype=torch.long).to(self.device_manager.target_device)

        # 边标签
        edge_label = torch.tensor(data_df['label'].values, dtype=torch.float).to(
            self.device_manager.target_device
        )

        # 创建PyG Data对象
        graph_data = Data(
            x=node_features,
            edge_index=edge_index,
            y=edge_label,
            num_nodes=len(self.gene_id_to_idx)
        )
        self._attach_cancer_context(graph_data, data_df, mode=mode)

        logger.info(f"图数据创建完成 (模式: {mode}):")
        logger.info(f"- 节点数: {graph_data.num_nodes}")
        logger.info(f"- 边数: {graph_data.num_edges}")
        logger.info(f"- 节点特征维度: {graph_data.x.shape}")

        return graph_data

    def _analyze_new_genes(self, new_genes, kg_data):
        """分析新基因在知识图谱中的连接情况"""
        logger.info("分析新基因在知识图谱中的连接...")

        for gene_id in new_genes[:5]:  # 只分析前5个作为示例
            symbol = self.id_mapper.get_symbol_by_id(gene_id)
            neighbors = self.gene_neighbor_cache.get(gene_id, [])

            logger.info(f"基因 {symbol} ({gene_id}):")
            if neighbors:
                logger.info(f"  在知识图谱中找到 {len(neighbors)} 个邻居:")
                for neighbor_id in neighbors[:3]:  # 显示前3个邻居
                    neighbor_symbol = self.id_mapper.get_symbol_by_id(neighbor_id)
                    logger.info(f"    - {neighbor_symbol} ({neighbor_id})")
            else:
                logger.info("  在知识图谱中没有找到连接邻居")

        # 统计新基因的连接情况
        connected_count = sum(1 for gene_id in new_genes if self.gene_neighbor_cache.get(gene_id))
        logger.info(f"新基因连接统计: {connected_count}/{len(new_genes)} 个基因在知识图谱中有连接")

