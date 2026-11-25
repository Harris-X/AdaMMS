# RAPT (V0.3 交付版)

`merge/rapt/` 是 `merge/rapt_V0_3/` 的**方法 + 合并 + 评测**迁移版，与 `rapt_V0_3` 同级。

- `rapt_V0_3` 保留为历史归档（1.6T merged_models、51G work、eval_crossarch 等结果全部原地不动）。
- 本目录只迁入代码/脚本/文档/配置，不含权重或历史评测结果。
- 迁移采用复制而非移动：`rapt_V0_3` 的旧脚本仍可运行。

## 配置分层：全局默认 + 每组一个 YAML

```
config.yaml                    全局默认: paths / merge / eval
configs/groups/<group>.yaml    单个模型组: anchor / host / donor / family /
                               interp_alpha / mrc_interp_alpha / backend / status
                               (可选覆盖全局 merge/eval/paths)
```

`--group` 的解析与合并由 [tools/cfg.py](./tools/cfg.py) 完成：
**组文件优先于全局默认**，合并结果才是脚本实际使用的配置；脚本内不写任何实验超参默认值。
`tools/cfg_bootstrap.sh` 是 bash 侧的统一引导（预扫描 `--group`、生成 `--shell` 变量、计算配置指纹）。

```bash
python merge/rapt/tools/cfg.py --list-groups                 # 列出所有组
python merge/rapt/tools/cfg.py --group g1 --get groups.g1.host
python merge/rapt/tools/cfg.py --group g1 --shell            # 合并后的全部键值
python merge/rapt/tools/cfg.py --group g1 --sha256           # 有效配置指纹
```

- 环境变量（`RANK=256`、`TASKS=...`）只用于单次临时覆盖。
- 实际生效值写进产物 `merge_meta*.json` / 评测 `run_manifest.json` / `rapt_eval_manifest.json`，
  并记录 `config_sha256`，可追溯到组文件的具体版本。
- 改口径：改组 YAML 或 `config.yaml`，不要改脚本。

## `--group` 参数的作用

`--group <name|path>` 是**模型组选择器**，本身不改变 TA_hyb/Stage-1~3 的算法，只决定：

| 维度 | 来源字段 | 影响 |
|---|---|---|
| 三份权重 | `anchor` / `host` / `donor` | anchor 决定 task vector 零点 W_0；host 决定输出骨架与保护基；donor 决定被吸收的知识 |
| 评测适配 | `family` | lmms-eval `qwen2_vl/llava/mplug_owl2`、VLMEvalKit `Qwen2VLChat/LLaVA/mPLUG_Owl2/CogVlm` |
| 组特有超参 | `interp_alpha` / `mrc_interp_alpha` | TA_hyb 插值系数；方案 B 的 R3 自选 ia |
| 后端 | `backend` | `ta_hyb`（单文件无缓存）或 `pipeline`（Stage-1→2→3） |
| 覆盖项 | `merge:` 段 | 该组的 rank/alpha/target/gate/device 等，深合并到全局默认 |
| 产物/缓存 | 组名 | `merged_models/<group>/`、`work/<group>/`、评测 tag `<group>_base/_merged` |

三种使用形态：

```bash
--group g1                                   # configs/groups/g1.yaml
--group merge/rapt/configs/groups/g1.yaml    # 显式路径
--group /tmp/my_new_group.yaml               # 任意实验配置, 不必改主 config
```

以前组定义硬编码在 `config.groups`、`--group` 只接受三个固定名，新增异构组合必须改主配置；
现在每个组合一个文件，可独立版本化/复制/覆盖，`run_merge_all.sh --groups "a b c"` 一次跑多组。

## 需求：异构模型融合

融合的前提不是"模型同名"，而是**语言侧模块名与形状可对齐，且存在共同 anchor**：

| 组合 | anchor | 语言侧对齐 | 状态 |
|---|---|---|---|
| Qwen2-VL-7B + LLaVA-OneVision-7B | Qwen2-7B | 同属 Qwen2 系，模块名/形状一致 | `g1` ready |
| LLaVA-v1.5 ↔ mPLUG-Owl2 | Llama-2-7b-hf | 同属 Llama-2/Vicuna 系；mPLUG 有 `.multiway` 碰撞（已处理） | `g3_*` ready |
| CogVLM (grounding) + LLaVA-v1.5 | Llama-2-7b-hf | 语言侧同为 Llama-2/Vicuna 系，但 CogVLM 是 expert 命名 + qkv 拼接形状 | `needs-adapter` |

CogVLM 组（`cogvlm_llava_fwd` / `llava_cogvlm_rev`）已写好配置，但默认**不运行**：

- 模块命名：`self_attn.language_expert_query_key_value` / `language_expert_dense` /
  `mlp.language_mlp.*`，与 LLaVA 的 `q/k/v/o_proj`、`mlp.*` 不同；
- 形状：expert qkv 是 `3h×h`，LLaVA 的 q/k/v 是 `h×h`，且是 1:3 关系；
- 因此 Stage-1 anchor 匹配与 TA_hyb 单文件配对会跳过这些模块，**结果不能作为完整 CogVLM 合并且报**。
- 需要后续加"语言侧 expert 适配器"（anchor q/k/v 拼接、donor 侧按 q/k/v 拆分/拼接）。
- 强制试验：`ALLOW_UNREADY=1` 或 `--allow-unready`（`run_merge_all` 会打印警告）。

视觉塔本身不做融合：输出以 host 的视觉塔为骨架，只吸收 donor 的语言侧（与 RAPT 现有语义一致）。

## 固定评测口径（base 与 merged 完全相同）

| 框架 | 参数 | 值 | 来源 |
|---|---|---|---|
| lmms-eval | min_pixels | 3136 | `eval.lmms_eval.min_pixels` |
| lmms-eval | max_pixels | 12845056 | `eval.lmms_eval.max_pixels` |
| lmms-eval | max_num_frames | 16 | `eval.lmms_eval.max_num_frames` |
| lmms-eval | max_new_tokens | 256 | `eval.lmms_eval.max_new_tokens` |
| lmms-eval | temperature | 0 | `eval.lmms_eval.temperature` |
| lmms-eval | tasks | 8 任务，逐任务、样本数升序 | `eval.lmms_eval.tasks` |
| VLMEvalKit | min_pixels | 1003520 | `eval.vlmeval.min_pixels` |
| VLMEvalKit | max_pixels | 12845056 | `eval.vlmeval.max_pixels` |
| VLMEvalKit | max_new_tokens | 不传，保持框架默认 2048 | `eval.vlmeval.max_new_tokens: null` |
| VLMEvalKit | datasets | TextVQA_VAL / OCRVQA_TESTCORE / VizWiz / GQA_TestDev_Balanced / ChartQA_TEST | `eval.vlmeval.datasets` |

lmms-eval 的像素/帧参数只对 Qwen2-VL 系生效（LLaVA/mPLUG adapter 不接受）；`max_new_tokens=256`
对三族统一。VLMEvalKit 的像素参数只对 `Qwen2VLChat` 生效。

## 目录结构

```
merge/rapt/
├── config.yaml                  # 全局默认配置
├── configs/groups/              # 每组一个文件
│   ├── g1.yaml                  # Qwen2-VL <- LLaVA-OneVision
│   ├── g3_fwd.yaml              # LLaVA-1.5 <- mPLUG
│   ├── g3_rev.yaml              # mPLUG <- LLaVA-1.5
│   ├── cogvlm_llava_fwd.yaml    # CogVLM <- LLaVA (needs-adapter)
│   └── llava_cogvlm_rev.yaml    # LLaVA <- CogVLM (needs-adapter)
├── VERSION
├── rapt_v3_common.py            # 方法: 公共算子/SVD/缓存
├── rapt_v3_stage1_subspace.py   # 方法: Stage-1
├── rapt_v3_stage2_mapping.py    # 方法: Stage-2
├── rapt_v3_stage3_injection.py  # 方法: Stage-3
├── ta_hyb_standalone.py         # 方法: TA_hyb/MRC 单文件
├── utils.py
├── test_rapt_v3_synthetic.py
├── docs/                        # 方法说明
├── tools/
│   ├── cfg.py                   # YAML 合并/查询
│   └── cfg_bootstrap.sh         # bash 统一引导
├── merge/
│   ├── run_stage1.sh            # Stage-1 (支持 --group)
│   ├── run_stage2.sh            # Stage-2
│   ├── run_stage3.sh            # Stage-3
│   ├── run_ta_hyb.sh            # TA_hyb/MRC 单文件
│   └── run_merge_all.sh         # 按组配置批量合并
└── eval/
    ├── eval_lmms.sh             # lmms-eval 8 任务
    ├── eval_vlmeval.sh          # VLMEvalKit 5 数据集
    ├── run_base_and_merged.sh   # base+merged 同参配对
    ├── summarize_eval8.py
    └── paired_bootstrap.py
```

## 使用

### 合并

```bash
# 默认组列表 (config.yaml merge.groups = g1 g3_fwd g3_rev)
bash merge/rapt/merge/run_merge_all.sh

# 单组 / 多组
bash merge/rapt/merge/run_merge_all.sh --group g1
bash merge/rapt/merge/run_merge_all.sh --groups "g1 g3_rev"

# 任意实验组 (不改主 config)
bash merge/rapt/merge/run_merge_all.sh --group merge/rapt/configs/groups/g3_fwd.yaml

# 临时覆盖组配置
RANK=256 ALPHA=0.4 bash merge/rapt/merge/run_merge_all.sh --group g1
ORTHO_SCALE=1 bash merge/rapt/merge/run_merge_all.sh --group g3_fwd   # 方案 B
```

`run_merge_all.sh` 按每组 YAML 的 `backend` 分流：`ta_hyb` 走单文件，`pipeline` 依次调用
`run_stage1/2/3.sh`。产物：`merge/rapt/merged_models/<group>/<host>/...`。

新增一个组：

```bash
cp merge/rapt/configs/groups/g1.yaml merge/rapt/configs/groups/my_exp.yaml
# 改 name/anchor/host/donor/family/interp_alpha
bash merge/rapt/merge/run_merge_all.sh --group my_exp
```

若该组需要不同的 rank/alpha/target/gate，在 YAML 中加：

```yaml
merge:
  rank: 256
  alpha: 0.4
  target: task_delta
  gate: apt
```

### 评测（base 与 merged 同参）

```bash
bash merge/rapt/eval/run_base_and_merged.sh \
    --group g1 \
    --merged merge/rapt/merged_models/g1/Qwen2-VL-7B-Instruct/ta_hyb_standalone_r128_a0p5_ia0p1
```

脚本按 `eval.pairs.roles = [base, merged]` 顺序调用同一份 `eval_lmms.sh` / `eval_vlmeval.sh`；
base 取自组配置的 `host`，因此不存在“base 与 merged 参数不同”的可能。

单独评测：

```bash
bash merge/rapt/eval/eval_lmms.sh <model_path> <run_name>
bash merge/rapt/eval/eval_vlmeval.sh <model_path> <run_name>
```

### 汇总

```bash
python merge/rapt/eval/summarize_eval8.py \
  merge/rapt/eval_results/lmms/g1_base \
  merge/rapt/eval_results/lmms/g1_merged

RAPT_TASKS=8 python merge/rapt/eval/paired_bootstrap.py \
  merge/rapt/eval_results/lmms/g1_merged
```

## 与 rapt_V0_3 的对应关系

| 本目录 | 来源 |
|---|---|
| 顶层 `rapt_v3_*.py`, `ta_hyb_standalone.py`, `utils.py`, `test_rapt_v3_synthetic.py` | 同名文件原样复制 |
| `docs/RAPT_V0_3_方法说明_*.md` | 原样复制 |
| `merge/run_stage1.sh` | 由 `run_ta_gen.sh` 的 Stage-1 段参数化 |
| `merge/run_stage2.sh` | 由 `run_ta_gen.sh` 的 Stage-2 段参数化 |
| `merge/run_stage3.sh` | 由 `run_ta_gen.sh` 的 TA-HYB 段参数化 |
| `merge/run_ta_hyb.sh` | 由 `run_crossarch_batch.sh` (P1/P2) + `run_mrc_gen.sh` 参数化 |
| `eval/eval_lmms.sh` | 由 `eval8.sh` + `run/eval_single_model_{qwenvl,llava_v15,mplug}.sh` 合并，超参改 config 固定值 |
| `eval/eval_vlmeval.sh` | 由 `merge/baseline/run_vlmeval_*` 参数化，固定 min/max pixels |
| `eval/run_base_and_merged.sh` | 新增（base/merged 同参配对） |

未迁移：历史实验队列（`run_*.sh` 队列/诊断）、`merged_models*`、`work`、`logs`、
`eval_crossarch`、`paper_data`。需要历史结果时直接查 `merge/rapt_V0_3/`。

## 变更记录

- 2026-11-24: 新建 `merge/rapt/`；超参集中到 `config.yaml`；统一评测口径；base/merged 同参配对。
- 2026-11-25: 模型组配置从主 `config.groups` 拆为 `configs/groups/<group>.yaml`（每组一个文件）；
  `cfg.py` 支持 `--group NAME|PATH`、深合并、`--list-groups`；`run_merge_all` 支持按组 backend/status；
  新增 CogVLM ↔ LLaVA 异构组配置（`status: needs-adapter`，默认不运行）。
