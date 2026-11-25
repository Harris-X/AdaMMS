# RAPT (V0.3 交付版)

`merge/rapt/` 是 `merge/rapt_V0_3/` 的**方法 + 合并 + 评测**迁移版，与 `rapt_V0_3` 同级。

- `rapt_V0_3` 保留为历史归档（1.6T merged_models、51G work、eval_crossarch 等结果全部原地不动）。
- 本目录只迁入**代码/脚本/文档/统一配置**，不含任何权重或历史评测结果。
- 迁移采用复制而非移动：`rapt_V0_3` 的旧脚本仍可运行，不会被破坏。

## 唯一事实来源：`config.yaml`

所有超参数集中在 [config.yaml](./config.yaml)。`merge/`、`eval/` 下的每个脚本都通过
[tools/cfg.py](./tools/cfg.py) 读取它，**脚本内不再写任何实验超参默认值**。

```bash
# 查看当前生效值
python merge/rapt/tools/cfg.py --get eval.lmms_eval.max_pixels
python merge/rapt/tools/cfg.py --shell          # 全部扁平化键值
python merge/rapt/tools/cfg.py --sha256         # 配置指纹
```

- 环境变量（如 `RANK=256`、`TASKS="mmu_val mme"`）只用于单次临时覆盖。
- 实际生效值写进产物 `merge_meta*.json` / 评测 `run_manifest.json` / `rapt_eval_manifest.json`，
  并记录 `config_sha256`，保证结果可追溯到具体配置。
- 要改口径，改 `config.yaml` 并在下方"变更记录"登记，不要改脚本。

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
| VLMEvalKit | max_new_tokens | **不传，保持框架默认 2048** | `eval.vlmeval.max_new_tokens: null` |
| VLMEvalKit | datasets | TextVQA_VAL / OCRVQA_TESTCORE / VizWiz / GQA_TestDev_Balanced / ChartQA_TEST | `eval.vlmeval.datasets` |

`lmms-eval` 的四个像素/帧参数只对 Qwen2-VL 系生效（LLaVA/mPLUG 的 model adapter 不接受）；
`max_new_tokens=256` 对三族统一。VLMEvalKit 的像素参数只对 `Qwen2VLChat` 生效。

## 目录结构

```
merge/rapt/
├── config.yaml                  # 唯一超参来源
├── VERSION                      # 版本记录 (复制自 rapt_V0_3)
├── rapt_v3_common.py            # 方法: 公共算子/SVD/缓存工具
├── rapt_v3_stage1_subspace.py   # 方法: Stage-1 SVD 子空间
├── rapt_v3_stage2_mapping.py    # 方法: Stage-2 算子对应/双 target
├── rapt_v3_stage3_injection.py  # 方法: Stage-3 target×gate 注入
├── ta_hyb_standalone.py         # 方法: TA_hyb/MRC 单文件最优实现
├── utils.py                     # 权重加载
├── test_rapt_v3_synthetic.py    # 生产路径单测
├── docs/                        # 方法说明 (完整历史推导与结论)
├── tools/cfg.py                 # YAML 读取
├── merge/                       # 合并入口
│   ├── run_stage1.sh            # Stage-1 (host/donor τ SVD)
│   ├── run_stage2.sh            # Stage-2 (Sinkhorn 对应)
│   ├── run_stage3.sh            # Stage-3 (完整三阶段 + 消融开关)
│   ├── run_ta_hyb.sh            # TA_hyb / MRC (推荐, 单文件无缓存)
│   └── run_merge_all.sh         # 三组一键合并
└── eval/                        # 评测入口
    ├── eval_lmms.sh             # lmms-eval 8 任务 (逐任务落盘/幂等)
    ├── eval_vlmeval.sh          # VLMEvalKit 5 数据集
    ├── run_base_and_merged.sh   # base+merged 同参配对评测
    ├── summarize_eval8.py       # 8 任务汇总
    └── paired_bootstrap.py      # 配对 bootstrap 显著性
```

## 使用

### 1. 合并（三组）

```bash
# 默认: TA_hyb 单文件实现, 组/超参全部来自 config.yaml
bash merge/rapt/merge/run_merge_all.sh

# 只跑一组
bash merge/rapt/merge/run_merge_all.sh --group g1

# 方案 B (MRC, 零超参): 使用 config 的 merge.mrc_interp_alpha
bash merge/rapt/merge/run_merge_all.sh --backend ta_hyb   # + config variants.ortho_scale=true
# 或临时:
ORTHO_SCALE=1 bash merge/rapt/merge/run_merge_all.sh
```

组（`config.groups`，路径相对 `paths.models`）：

| group | anchor | host (base) | donor |
|---|---|---|---|
| g1 | Qwen2-7B | Qwen2-VL-7B-Instruct | llava-onevision-qwen2-7b-si-hf |
| g3_fwd | Llama-2-7b-hf | llava-v1.5-7b | mplug-owl2-llama2-7b |
| g3_rev | Llama-2-7b-hf | mplug-owl2-llama2-7b | llava-v1.5-7b |

产物：`merge/rapt/merged_models/<group>/<host>/ta_hyb_standalone_r{R}_a{A}_ia{IA}[_suffix]`。

### 2. 评测（base 与 merged 同参）

```bash
# merged 目录填 run_merge_all.sh 输出的那个
bash merge/rapt/eval/run_base_and_merged.sh \
    --group g1 \
    --merged merge/rapt/merged_models/g1/Qwen2-VL-7B-Instruct/ta_hyb_standalone_r128_a0p5_ia0p1
```

脚本按 `eval.pairs.roles = [base, merged]` 顺序调用同一份 `eval_lmms.sh` /
`eval_vlmeval.sh`，base 取自 `groups.<g>.host`，因此不存在"base 与 merged 参数不同"的可能。

输出：

```
merge/rapt/eval_results/
├── lmms/<tag>_base/<task>/<model>/<ts>_results.json
├── lmms/<tag>_merged/<task>/...
├── vlmeval/<tag>_base/<ts>/...
└── vlmeval/<tag>_merged/<ts>/...
```

单独评测：

```bash
bash merge/rapt/eval/eval_lmms.sh <model_path> <run_name>
bash merge/rapt/eval/eval_vlmeval.sh <model_path> <run_name>
```

### 3. 汇总

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
| `eval/eval_lmms.sh` | 由 `eval8.sh` + `run/eval_single_model_{qwenvl,llava_v15,mplug}.sh` 合并，超参改为 config 固定值 |
| `eval/eval_vlmeval.sh` | 由 `merge/baseline/run_vlmeval_*` 参数化，固定 min/max pixels |
| `eval/run_base_and_merged.sh` | 新增（base/merged 同参配对） |

**未迁移**：历史实验队列（`run_*.sh` 队列/诊断）、`merged_models*`、`work`、`logs`、
`eval_crossarch`、`paper_data`。需要历史结果时直接查 `merge/rapt_V0_3/`。

## 变更记录

- 2026-11-24: 新建 `merge/rapt/`；超参集中到 `config.yaml`；评测统一为
  lmms-eval `min_pixels=3136 / max_pixels=12845056 / max_num_frames=16 / max_new_tokens=256`，
  VLMEvalKit `min_pixels=1003520 / max_pixels=12845056 / max_new_tokens=框架默认 2048`；
  base 与 merged 同参配对。
