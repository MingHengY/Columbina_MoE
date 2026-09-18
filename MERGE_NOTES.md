# 合并说明：cm4sl_code_v4 + 交接包工程模块

生成日期：2026-08-23

本目录 `cm4sl_code_v4_merged` 以生物学改进版 `cm4sl_code_v4` 为基础（未改动其
任何模型/生物学逻辑），把交接包 `columbina_xlab_biological_handoff_20260820/code_v4`
中的工程模块合并进来。`cm4sl_code_v4` 原目录保持原样。

## 合并内容一览

| 模块 | 来源 | 接入点 |
|---|---|---|
| `artifacts.py` | 交接包（原样移植） | train/main/infer |
| `calibration.py` | 交接包（原样移植） | main/infer |
| 验证集温度缩放 | 交接包 `main.py` | `_fit_and_persist_calibration` |
| 测试期 KG schema 严格校验 | 交接包 `main.py` | `_drop_untrained_test_edge_types` / `_validate_test_kg_schema` |
| 真批量文本编码 | 交接包 `knowledge_graph_builder.py` | `batch_encode_texts` |
| 组学矩阵缓存 + 严格加载 | 交接包 `data_processor.py` | `_load_omics_frame` |
| `TOTAL_SAMPLES_PER_CLASS` 正数校验 | 交接包 `data_processor.py` | `UniversalDataSplitter.__init__` |
| `__init__.py` 包标识 | 交接包 | 替代拼写错误的 `_init_.py` |
| 测试 4 组 | 交接包（适配） | `tests/` |

## 各接入点说明

### 1. checkpoint 配置元数据（train.py）

两处最佳模型保存点（`Trainer` → `best_model.pth`、`InductiveTrainer` →
`best_inductive_model.pth`）的 `model_config` 改为**合并字典**：

- 小写维度键（`hidden_dim` / `embedding_dim` / `omics_feature_dim` /
  `kg_feature_dim` / `node_types` / `edge_types`）：`infer.py` 的维度强校验依赖；
- 全量大写配置快照（`artifacts.serialize_config`）：加载时经
  `apply_model_config` 恢复全部训练参数；
- 新增 `model_config_sha256`（快照的规范化 SHA-256）。

`apply_model_config` 只应用 Config 上已存在的大写属性并跳过全部路径键，因此
小写维度键与其互不干扰。

### 2. 校准（main.py / calibration.py）

- `_fit_and_persist_calibration(trainer, scenario)`：仅在**验证集 logits** 上
  网格搜索温度（600 个 log 间隔候选，NLL 最小化），写 `calibration.json` 并
  原子回写最佳 checkpoint 的 `posthoc_calibration` 字段。
- C1 与 C2/C3 的测试阈值统一改为 `calibration['decision_threshold']`
  （缩放后验证 F1 最优）；测试标签从不参与阈值选择。
  `strict_inductive_manifest.json` 记录 `decision_threshold_source:
  validation_f1_after_temperature_scaling` 与校准摘要。
- `evaluate_test_set` / `evaluate_inductive_test_set` 新增 `calibration`
  参数：概率经 `apply_temperature` 产生，结果新增
  `calibration_nll` / `calibration_brier_score` / `calibration_ece` /
  `temperature` 键（无校准时也输出，温度=1.0，便于报告对齐）。
- 交叉验证：每折各自拟合校准；`select_best_fold_model` 会把获胜折的
  `calibration.json` 与 `best_overall_model.pth` 成对复制到 CV 根目录。

### 3. 部署清单（main.py / artifacts.py）

单次（非 CV）运行结束时：

- `write_model_config` 写出带 SHA-256 的全量配置快照（替代原先手工挑子集的
  `model_config.json`）；
- `write_deployment_manifest` 校验 8 项必需产物（C2/C3 另加
  `strict_inductive_manifest.json`）齐全后生成 `deployment_manifest.json`，
  含每个产物的字节数与 SHA-256；产物缺失直接抛 `FileNotFoundError`
  （fail-closed）。CV 折目录不保存 scaler，因此不写清单。

与交接包应用的衔接：`scripts/validate_columbina_deployment.py` 校验的
`deployment_manifest.json` 即此清单；`validate_deployment_manifest` 提供
加载侧哈希复核。

### 4. 测试期 KG schema 严格校验（main.py）

`_run_c2c3_scenario` 在取得测试 KG 后：

1. `_drop_untrained_test_edge_types`：丢弃训练+验证阶段从未见过（无学习参数）
   的测试专用边类型，记录审计（写入 strict manifest 的
   `dropped_untrained_test_edge_types`）；
2. `_validate_test_kg_schema`：未知**节点**类型或节点维度变化直接抛
   `ValueError`（节点无法像边那样丢弃，删除会孤立边）。

### 5. 校准推理（infer.py）

- `config_path` 参数真正生效：JSON 只覆盖非路径的模型参数
  （`apply_model_config`，接受 `model_config.json` 或裸配置字典）。
- `_load_model`：若存在 `deployment_manifest.json` 则做哈希校验与 checkpoint
  名一致性校验；维度强校验通过后，用 checkpoint 内配置快照恢复运行时参数，
  并交叉校验 `architecture_version`。
- 校准加载：优先同目录 `calibration.json`，与 checkpoint 内嵌温度交叉校验；
  三个预测出口（`predict_gene_pairs` / `predict_gene_pairs_per_cancer` /
  `_predict_with_precomputed`）经 `_calibrated_scores` 输出温度缩放概率。
- 新配置 `REQUIRE_CALIBRATION`（默认 **False**）：开启后缺少校准文件即拒绝
  推理。默认关闭是为了合并前的正式 C3 资产（无 calibration.json）仍可加载；
  新训练产物建议开启。

### 6. 真批量文本编码（knowledge_graph_builder.py）

`batch_encode_texts` 由"逐条调用 `get_text_embedding`"改为真实模型批次：
整批 tokenize（truncation/max_length=128/padding）、CLS 池化、单批失败仅该批
回退零矩阵（保持行数与顺序）、`batch_size<=0` 抛错。空输入返回 `(0, hidden)`。
KG 构建走 LRU 缓存的未缓存路径，批编码只在缓存未命中时执行。

### 7. 组学缓存与严格加载（data_processor.py）

- 新增 `SLDataProcessor._load_omics_frame`：读取（utf-8/latin-1/cp1252 依次
  尝试）→ 自动转置 → ID 锚定/名称归一化，结果按
  `(路径, USE_ID_ANCHORING, USE_GENE_NAME_NORMALIZATION)` 在同一处理器实例内
  缓存。交叉验证折间重复读取同一矩阵的开销显著降低。
- `load_multi_omics_features` 改用该函数，**单文件缺失/不可读立即抛错**
  （原 warn-and-skip 会在特征宽度漂移后以更含糊的维度错误失败）。逐基因零
  向量语义不变；`ALLOW_ZERO_FEATURE_FALLBACK` 仅在未配置任何特征文件时兜底。
- `UniversalDataSplitter`：`TOTAL_SAMPLES_PER_CLASS` 必须为正，否则抛
  `ValueError`（getattr 兜底默认从 28000 对齐为配置值 29000）。

## 有意不合并 / 差异项

- 交接包的 `COLUMBINA_*` 环境变量与路径体系**不**合并：保留 cm4sl 的
  `CM4SL_BASE_DIR/DATA_DIR/OUTPUT_DIR` 与本地/远程双布局。
- 交接包 main.py 的 argparse CLI（`build_arg_parser`/`configure_from_args`）
  不合并：cm4sl 版入口是 `universal_main` + `interactive_main`；相应 2 个 CLI
  测试不移植。
- 交接包 `models.py` 的动态维度构建（从运行时数据推断编码器输入宽度）不合并：
  cm4sl 版按 `config.OMICS_FEATURE_DIM` 构建并在 PCA 后强校验维度；动态维度
  测试已适配为"从 checkpoint 配置快照恢复宽度"（经 `apply_model_config`）。
- 推理侧 `load_multi_omics_features_per_cancer` 仍是 cm4sl 原实现（其本身已
  fail-closed）；组学缓存只覆盖训练侧 `load_multi_omics_features`。

## 测试

全部测试位于 `tests/`（unittest，无需 pytest）：

```bash
cd cm4sl_code_v4_merged
python -m unittest discover -s tests -p "test_*.py"
```

共 39 项，包含新增：

- `test_training_artifacts.py`：checkpoint 元数据完整性、部署清单哈希/损坏
  检测/缺失拒绝、温度缩放确定性与 NLL 不劣化、splitter 正数校验、推理
  fail-closed（无随机张量回退）。
- `test_kg_text_batching.py`：真实分批（tokenize/前向各调用 2 次而非 3 次）、
  顺序保持、空输入。
- `test_strict_inductive.py::StrictTestKgSchemaTests`：测试专用边类型
  drop+审计；未知节点类型 fail-closed。
- `test_moe_architecture.py::DynamicOmicsDimensionTests`：7 维组学前向/回传/
  从 checkpoint 快照重载。

## 对既有产物的影响

- 已有 `cm4sl_outputs/output_v4/C3_cv` 等资产不受影响；旧 checkpoint 无
  `calibration.json`，推理按温度=1.0 输出并记录 warning（除非
  `REQUIRE_CALIBRATION=True`）。
- 用本目录重新训练后，单次运行产物目录将新增 `calibration.json`、
  `deployment_manifest.json`，`model_config.json` 变为全量快照格式。
