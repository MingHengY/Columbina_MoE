# 性能与显存优化说明（2026-08-24）

针对两个问题在不改变模型数值语义的前提下优化（备份：`cm4sl_code_v4_merged_backup_20260824`）：

- **显存**：C1 ~90GB / C2 ~60GB；
- **速度**：C2 10折交叉验证 ~6 小时。

## 根因（基于真实运行日志与代码核查）

**显存**：HGT 专家在每次前向都对整张 KG 做带梯度的全图注意力。KG 有 12 种节点、
29 种边类型、约 96–219 万条边；`HGTKnowledgeEncoder` 为每种边自动加反向边后，
HGTConv 需要为每条边物化 `[E, heads, head_dim]` 激活并保留到反向传播。C1 的 KG
来自几乎全量的训练基因（最大 ~219 万边 → 反向后 ~438 万），因此 C1 > C2。
次要因素：PubMedBERT 在数据准备后常驻 GPU（~440MB）、验证 KG 常驻 GPU。

**速度**：训练本身每折仅 45–90 秒。6 小时几乎全部是每折重复的数据侧工作：
每折 3–4 次全量 KG 构建（每次重扫 904MB 边表 + 239MB STRING + ~2 万条文本过
PubMedBERT，服务器上每次 ~5 分钟）、每折重读 2.05GB 组学 CSV、每折 4 次解析
19 万行基因映射、每折重新从磁盘加载 PubMedBERT。KG 图缓存的键含精确基因序，
C2 每折轮换基因桶，跨折必然 miss。

## 优化内容

### 显存（数学等价）

| 优化 | 位置 | 说明 |
|---|---|---|
| HGT 梯度检查点 | `models.py::HGTKnowledgeEncoder`（`HGT_GRADIENT_CHECKPOINTING=True`） | 反向时重算各 HGT 层前向（dropout 掩码由保留的 RNG 状态重放），梯度与关闭时**逐位一致**（有测试保证）。本地 180 万边合成 KG 实测峰值 8.16GB→5.40GB（-34%）。 |
| PubMedBERT 构建后暂存 CPU | `knowledge_graph_builder.py::._park_embedding_model`（`PARK_EMBEDDING_MODEL_AFTER_BUILD=True`） | 每次构建完成后把冻结编码器移到 CPU（省 ~440MB），下次编码自动搬回。 |
| 验证/测试 KG 常驻 CPU | `train.py::InductiveTrainer`、`data_processor.py::prepare_strict_stage_kg`（`KEEP_STAGE_KG_ON_CPU=True`） | 这些 KG 只在 no_grad 前向中被消费；编码器按张量上卡，只改变常驻位置。 |
| 折间清理 | `main.py` CV 折循环 | 每折结束 `del trainer` + `empty_cache` + `gc.collect()`，降低跨折峰值累积。 |

### 速度（语义等价，均有等价性测试）

| 优化 | 位置 | 服务器估计收益 |
|---|---|---|
| KG/STRING 源文件解析一次 | `knowledge_graph_builder.py` `_source_cache`（`ENABLE_KG_FILE_CACHE=True`） | 每次构建只按基因集过滤缓存块，不再重扫 1.1GB 源文件、不再重建名称映射（~25s）。本地实测：同基因集图缓存命中 0.0s；不同基因集 46.7s→25.2s（服务器冷解析更慢，收益更大）。 |
| 文本向量进程级缓存 | `_text_embedding_cache`（`ENABLE_TEXT_EMBEDDING_CACHE=True`） | 冻结编码器输出确定，折间节点文本大量重复；失败批次不写缓存。省掉大部分 ~3–5 分钟/次的 PubMedBERT 编码。 |
| PubMedBERT 实例共享 | `_shared_embedding_models` | 每折不再从磁盘重新加载权重。 |
| 基因映射解析共享 | `gene_mapping.py`（`ENABLE_GENE_MAPPING_CACHE=True`） | 每折 4 次 iterrows 解析 19 万行（每次 ~7s）→ 全 CV 仅首次解析。 |
| 组学帧跨实例共享 | `data_processor.py` `_omics_frame_shared_cache`（`ENABLE_OMICS_FRAME_CACHE=True`） | 每折 2.05GB CSV 重读 → 全 CV 仅一次（代价 ~2GB CPU 内存）。 |
| 验证时免渲染曲线 | `train.py` | 训练循环内每个验证点不再渲染 dpi=300 曲线（C1 由训练后评估渲染一次；归纳式由循环后最终评估渲染），每折省 ~10 次渲染。 |
| 折图数据保存可关 | `main.py`（`SAVE_FOLD_DATA`，默认仍 True） | 每折 213MB `kg_data.pt` 可选关闭。 |

**预期总效果**：折 1 付出冷解析成本后，折 2–10 的数据准备时间从 ~15–20 分钟/折
降至 ~2–4 分钟/折，C2 10 折总时长预计从 ~6h 降至 ~1.5–2h；C1/C2 显存峰值预计
下降 1/3 左右（HGT 检查点）再叠加 BERT/验证 KG 常驻释放的 ~1GB。

## 等价性保证（tests/test_performance_optimizations.py，9 项）

1. **HGT 检查点**：开启/关闭时全部参数梯度 `torch.equal` 逐位一致；eval 前向输出一致。
2. **源文件缓存**：精心构造覆盖关键语义分支（ID 命中、名称映射改写、未知名称、
   y_id 命中、两端非目标、多块边界）的夹具，断言"缓存命中路径 == 逐次解析路径"
   逐帧相等（`assert_frame_equal`），并断言原始筛选语义本身不变。
3. **文本向量缓存**：命中返回相同向量且不再前向；部分命中只编码缺失文本；
   失败批次不写缓存、下次重试。
4. **组学帧/基因映射共享**：跨实例内容一致；副本隔离（改一个不影响另一个）。

已知微小区别（记录在案）：warm 缓存下文本的批分组可能与冷运行不同，GPU 上
cuBLAS 不同批形状的浮点归约顺序可能产生 ULP 级差异——冷启动运行与旧代码
逐批一致；这不影响模型性能，仅在最严格的逐位复现语境下有区别。

## 开关回退

全部优化可通过 config 关闭以复现旧行为：`HGT_GRADIENT_CHECKPOINTING`、
`PARK_EMBEDDING_MODEL_AFTER_BUILD`、`ENABLE_TEXT_EMBEDDING_CACHE`、
`ENABLE_KG_FILE_CACHE`、`ENABLE_GENE_MAPPING_CACHE`、`ENABLE_OMICS_FRAME_CACHE`、
`KEEP_STAGE_KG_ON_CPU`（默认全部开启）。

## 本机验证记录

- 48 项单元测试全部通过（含 9 项新增优化回归测试）。
- 真实数据（F:/projects/comp/cm4sl_input_data，904MB 边表 + 239MB STRING）：
  冷构建 46.7s → 同基因集 0.0s（图缓存）→ 不同基因集 25.2s（源缓存命中），
  图结构逐位一致。
- 本地 RTX 4070 合成 KG（180 万边）HGT 峰值显存 8.16GB→5.40GB。
- 本机未安装 `transformers`，PubMedBERT 路径按零特征回退验证（服务器不受影响）。
