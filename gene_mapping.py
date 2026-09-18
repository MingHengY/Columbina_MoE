"""基因ID映射管理器"""

import pandas as pd
import numpy as np
import os
import re
import logging
import traceback

from config import Config

logger = logging.getLogger(__name__)

# ==================== 基因ID映射管理器 ====================
class GeneIDMapper:
    """基因ID映射管理器"""

    # Parsed mapping dictionaries shared across instances.  A CV fold
    # constructs several processors/builders that each re-parsed this
    # 193k-row file through iterrows (~7s per parse); the parse result is a
    # pure function of the file, so share it.  Dictionaries are handed out
    # as copies - nothing mutates them after load, but copies keep every
    # consumer independent.
    _shared_mappings = {}

    @classmethod
    def clear_shared_mappings(cls):
        """Drop cached parses (tests / explicit cache invalidation)."""
        cls._shared_mappings.clear()

    @staticmethod
    def _mapping_signature(config):
        def file_sig(path):
            if not path:
                return None
            try:
                stat = os.stat(path)
            except OSError:
                return (os.fspath(path), None, None)
            return (os.fspath(path), stat.st_mtime_ns, stat.st_size)
        return (
            file_sig(getattr(config, 'GENE_ID_MAPPING_FILE', None)),
            str(getattr(config, 'GENE_ID_COLUMN', 'ENTREZID')).lower().strip(),
            str(getattr(config, 'GENE_SYMBOL_COLUMN', 'SYMBOL')).lower().strip(),
        )

    def __init__(self, config=None):
        self.config = config if config else Config()

        # 映射字典
        self.symbol_to_id = {}  # SYMBOL -> ENTREZID
        self.id_to_symbol = {}  # ENTREZID -> SYMBOL
        self.id_to_name = {}  # ENTREZID -> GENENAME
        self.all_names_to_id = {}  # 所有名称变体 -> ENTREZID

        signature = self._mapping_signature(self.config)
        shared = None
        if bool(getattr(self.config, 'ENABLE_GENE_MAPPING_CACHE', True)):
            shared = self.__class__._shared_mappings.get(signature)
        if shared is not None:
            logger.info("复用已解析的基因映射缓存: %d 个ID", len(shared['id_to_symbol']))
            self.symbol_to_id = dict(shared['symbol_to_id'])
            self.id_to_symbol = dict(shared['id_to_symbol'])
            self.id_to_name = dict(shared['id_to_name'])
            self.all_names_to_id = dict(shared['all_names_to_id'])
            return

        # 加载映射文件
        self.load_gene_mapping()

        if bool(getattr(self.config, 'ENABLE_GENE_MAPPING_CACHE', True)) \
                and self.id_to_symbol:
            self.__class__._shared_mappings[signature] = {
                'symbol_to_id': dict(self.symbol_to_id),
                'id_to_symbol': dict(self.id_to_symbol),
                'id_to_name': dict(self.id_to_name),
                'all_names_to_id': dict(self.all_names_to_id),
            }

    def load_gene_mapping(self):
        """加载基因ID映射文件"""
        try:
            if os.path.exists(self.config.GENE_ID_MAPPING_FILE):
                mapping_df = pd.read_csv(self.config.GENE_ID_MAPPING_FILE)
                logger.info(f"加载基因映射文件: {len(mapping_df)} 条记录")
                logger.info(f"映射文件列名: {mapping_df.columns.tolist()}")

                # 检查必要的列是否存在
                required_cols = [self.config.GENE_ID_COLUMN, self.config.GENE_SYMBOL_COLUMN]

                # 检查列名，处理可能的空格或大小写问题
                available_cols = [col for col in mapping_df.columns]
                logger.info(f"可用列名: {available_cols}")

                # 建立列名映射，不区分大小写
                col_mapping = {}
                for col in available_cols:
                    col_mapping[col.lower().strip()] = col

                # 获取正确的列名
                id_col = None
                symbol_col = None
                name_col = None

                # 查找ID列
                id_col_key = self.config.GENE_ID_COLUMN.lower().strip()
                if id_col_key in col_mapping:
                    id_col = col_mapping[id_col_key]
                else:
                    # 尝试其他可能的名称
                    for key in ['entrezid', 'entrez', 'entrez_id', 'geneid', 'gene_id']:
                        if key in col_mapping:
                            id_col = col_mapping[key]
                            break

                # 查找符号列
                symbol_col_key = self.config.GENE_SYMBOL_COLUMN.lower().strip()
                if symbol_col_key in col_mapping:
                    symbol_col = col_mapping[symbol_col_key]
                else:
                    # 尝试其他可能的名称
                    for key in ['symbol', 'gene_symbol', 'genesymbol', 'gene']:
                        if key in col_mapping:
                            symbol_col = col_mapping[key]
                            break

                # 查找全名列
                name_keys = ['genename', 'gene_name', 'name', 'description']
                for key in name_keys:
                    if key in col_mapping:
                        name_col = col_mapping[key]
                        break

                if not id_col:
                    raise ValueError(f"找不到ID列，期望的列名: {self.config.GENE_ID_COLUMN}")
                if not symbol_col:
                    raise ValueError(f"找不到符号列，期望的列名: {self.config.GENE_SYMBOL_COLUMN}")

                logger.info(f"使用列名 - ID: {id_col}, 符号: {symbol_col}, 全名: {name_col}")

                # 建立基本映射
                for idx, row in mapping_df.iterrows():
                    try:
                        # 获取ENTREZID
                        entrezid = str(row[id_col]) if pd.notna(row[id_col]) else ""
                        if not entrezid:
                            continue

                        # 获取基因符号
                        symbol = str(row[symbol_col]) if pd.notna(row[symbol_col]) else ""

                        # 获取基因全名
                        genename = ""
                        if name_col and pd.notna(row[name_col]):
                            genename = str(row[name_col])

                        # 符号到ID的映射
                        if symbol:
                            self.symbol_to_id[symbol.upper()] = entrezid
                            self.all_names_to_id[symbol.upper()] = entrezid

                        # ID到符号的映射
                        self.id_to_symbol[entrezid] = symbol
                        self.id_to_name[entrezid] = genename

                        # 其他可能的映射（如果文件中有同义词列）
                        for col in available_cols:
                            # 跳过已处理的列
                            if col in [id_col, symbol_col, name_col]:
                                continue

                            if pd.notna(row[col]):
                                # 处理可能的分隔符
                                cell_value = str(row[col])
                                # 可能的分隔符：逗号、分号、竖线等
                                separators = [';', ',', '|', '/']

                                synonyms = []
                                for sep in separators:
                                    if sep in cell_value:
                                        synonyms = [s.strip() for s in cell_value.split(sep)]
                                        break

                                # 如果没有分隔符，整个字符串作为一个同义词
                                if not synonyms:
                                    synonyms = [cell_value.strip()]

                                for synonym in synonyms:
                                    if synonym:
                                        synonym_upper = synonym.upper()
                                        self.all_names_to_id[synonym_upper] = entrezid

                    except Exception as e:
                        logger.warning(f"处理第{idx}行时出错: {e}")
                        continue

                logger.info(f"建立映射: {len(self.symbol_to_id)} 个符号 -> {len(self.id_to_symbol)} 个ID")

                # 验证：确保ID唯一性
                if len(set(self.id_to_symbol.keys())) != len(mapping_df):
                    logger.warning("存在重复的ENTREZID！")

                # 显示一些示例映射
                if self.config.DEBUG_MODE and len(self.id_to_symbol) > 0:
                    logger.debug("映射示例（前5个）:")
                    for i, (entrezid, symbol) in enumerate(list(self.id_to_symbol.items())[:5]):
                        logger.debug(f"  {entrezid}: {symbol}")
            else:
                logger.warning(f"基因映射文件不存在: {self.config.GENE_ID_MAPPING_FILE}")

        except Exception as e:
            logger.error(f"加载基因映射文件失败: {e}")
            logger.error(traceback.format_exc())
            raise

    def get_id_by_symbol(self, symbol):
        """通过基因符号获取ENTREZID"""
        if not symbol or pd.isna(symbol):
            return None

        symbol_upper = str(symbol).strip().upper()
        return self.symbol_to_id.get(symbol_upper)

    def get_id_by_any_name(self, name):
        """通过任何名称获取ENTREZID"""
        if not name or pd.isna(name):
            return None

        name_str = str(name).strip().upper()

        # 尝试直接匹配
        if name_str in self.all_names_to_id:
            return self.all_names_to_id[name_str]

        # 尝试移除版本号等修饰符
        name_clean = re.sub(r'\.\d+$', '', name_str)
        if name_clean in self.all_names_to_id:
            return self.all_names_to_id[name_clean]

        # 尝试其他常见变体
        # 移除"-AS1", "-AS2"等反义标记
        if name_str.endswith(('-AS1', '-AS2', '-AS3', '-AS4')):
            name_base = name_str[:-4]
            if name_base in self.all_names_to_id:
                return self.all_names_to_id[name_base]

        # 处理LOC前缀
        if name_str.startswith('LOC'):
            # 如果是LOC数字，尝试直接匹配
            if name_str in self.all_names_to_id:
                return self.all_names_to_id[name_str]

        return None

    def get_symbol_by_id(self, entrezid):
        """通过ENTREZID获取基因符号"""
        if not entrezid:
            return ""
        entrezid_str = str(entrezid)
        return self.id_to_symbol.get(entrezid_str, entrezid_str)  # 返回符号或ID本身

    def get_name_by_id(self, entrezid):
        """通过ENTREZID获取基因全名"""
        if not entrezid:
            return ""
        entrezid_str = str(entrezid)
        return self.id_to_name.get(entrezid_str, "")

    def batch_symbol_to_id(self, symbols):
        """批量转换符号到ID"""
        return [self.get_id_by_symbol(s) for s in symbols]

    def batch_id_to_symbol(self, ids):
        """批量转换ID到符号"""
        return [self.get_symbol_by_id(id_) for id_ in ids]

    def create_id_mapping_report(self, ids):
        """创建ID映射报告"""
        found = []
        not_found = []

        for entrezid in ids:
            if entrezid in self.id_to_symbol:
                found.append(entrezid)
            else:
                not_found.append(entrezid)

        return {
            'total': len(ids),
            'found': len(found),
            'not_found': len(not_found),
            'coverage': len(found) / len(ids) if len(ids) > 0 else 0
        }


# ==================== 基因名称标准化模块（ID锚定版） ====================
class GeneNameNormalizer:
    """基因名称标准化器（支持ID模式）"""

    def __init__(self, config=None):
        self.config = config if config else Config()

        # 基因同义词映射
        self.gene_synonyms = {}
        self.standardized_names = {}

        # ID映射器
        self.id_mapper = GeneIDMapper(config) if self.config.USE_ID_ANCHORING else None

        # 加载基因同义词表（如果存在）
        if os.path.exists(self.config.GENE_SYNONYM_FILE):
            self.load_gene_synonyms()

    def load_gene_synonyms(self):
        """加载基因同义词表"""
        try:
            synonyms_df = pd.read_csv(self.config.GENE_SYNONYM_FILE)
            logger.info(f"加载基因同义词表: {len(synonyms_df)} 条记录")

            for _, row in synonyms_df.iterrows():
                standard_name = row.get('standard_name', '').strip()
                if pd.isna(standard_name):
                    continue

                # 收集所有同义词
                synonyms = []
                for col in synonyms_df.columns:
                    if col != 'standard_name' and col in row and not pd.isna(row[col]):
                        synonym = str(row[col]).strip()
                        if synonym:
                            synonyms.append(synonym)

                # 建立同义词到标准名称的映射
                for synonym in synonyms:
                    self.gene_synonyms[synonym] = standard_name

                self.standardized_names[standard_name] = standard_name

            logger.info(f"加载了 {len(self.gene_synonyms)} 个基因同义词映射")
        except Exception as e:
            logger.warning(f"加载基因同义词表失败: {e}")

    def normalize_gene_name(self, gene_name):
        """标准化基因名称（如果启用ID锚定，返回ENTREZID）"""
        if not gene_name or pd.isna(gene_name):
            return ""

        gene_name = str(gene_name).strip()

        # 如果启用ID锚定，优先返回ENTREZID
        if self.config.USE_ID_ANCHORING and self.id_mapper:
            entrezid = self.id_mapper.get_id_by_any_name(gene_name)
            if entrezid:
                return entrezid

        # 1. 首先检查同义词映射
        if gene_name in self.gene_synonyms:
            return self.gene_synonyms[gene_name]

        # 2. 常见的标准化操作
        # 移除版本号 (如.1, .2)
        normalized = re.sub(r'\.\d+$', '', gene_name)

        # 移除括号内容
        normalized = re.sub(r'\([^)]*\)', '', normalized).strip()

        # 转换为大写（但保留一些特殊情况）
        if not normalized.startswith(('hsa-', 'miR-', 'ENS', 'LOC')):
            normalized = normalized.upper()

        return normalized

    def get_symbol_for_id(self, entrezid):
        """获取ID对应的符号（用于调试显示）"""
        if self.config.USE_ID_ANCHORING and self.id_mapper:
            return self.id_mapper.get_symbol_by_id(entrezid)
        return entrezid

    def get_id_for_symbol(self, symbol):
        """获取符号对应的ID"""
        if self.config.USE_ID_ANCHORING and self.id_mapper:
            return self.id_mapper.get_id_by_symbol(symbol)
        return symbol

    def batch_normalize(self, gene_names):
        """批量标准化基因名称"""
        return [self.normalize_gene_name(name) for name in gene_names]
