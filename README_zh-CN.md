<div align="center">

# VascFusion：MICCAI 2026 GAVE2 Challenge 第 8 名方案

[English](README.md) | [简体中文](README_zh-CN.md)

</div>

本仓库包含 **VascFusion** 的训练与推理代码。VascFusion 是我们参加 MICCAI
2026 GAVE2 Challenge 的第 8 名方案，覆盖三个任务：

1. 仅使用 CFP 的视网膜动静脉分割；
2. CFP + FFA 跨模态动静脉分割；
3. CRAE、CRVE、AVR、动静脉密度和动静脉分形维数测量。

## 比赛结果

| 方法 | Task 1 | Task 2 | Task 3 | 总分 |
|---|---:|---:|---:|---:|
| VascFusion | 8.29660 | 8.31706 | 7.39370 | **7.94362** |

## 方法概览

VascFusion 将模态自适应血管分割与解剖结构引导的生物标志物测量结合起来，
在保留原分辨率血管细节的同时，仅在规则允许的任务中利用造影时相信息。

### VascFusion-Seg：模态自适应动静脉分割

- ConvNeXt-Tiny 多尺度 CFP 编码器；
- U-Net 风格解码器与原分辨率细节分支；
- 语义、通用血管及分类中心线多头预测；
- 共享参数的概率空间五步递归修正；
- 面向动静脉拓扑的训练损失；
- Task 1 使用 CFP optical-density 细节特征；
- Task 2 在多个尺度门控融合配准后的早期/晚期 FFA。

Task 1 **严格仅使用 CFP**，训练和推理均不会读取 FFA。

### VascFusion-Quant：解剖结构引导的指标量化

Task 3 并非直接预测七个标量的黑盒回归器。系统先从分割概率图中按 SIVA 风格
几何规则提取指标，并分别针对管径、密度和网络复杂度采用专门的测量分支：

| 生物标志物 | 测量策略 |
|---|---|
| CRAE、CRVE、AVR | 管径导向的动静脉掩膜与修订 Knudtson–Hubbard 公式 |
| 动脉密度 | 血管感知动脉掩膜与仅由训练集 OOF 结果确定的校准 |
| 静脉密度 | 轻量 C 区静脉修正分支 |
| 动脉分形维数 | 分支一致性动脉骨架与计盒法 |
| 静脉分形维数 | 三个固定阈值下的时相感知静脉几何均值 |

## 仓库结构

```text
.
├── README.md / README_zh-CN.md
├── code/
│   ├── run_inference.sh             # 端到端推理与打包
│   ├── assemble_biomarkers.py       # 七项生物标志物组装
│   ├── biomarker_tools/             # 视盘与血管几何测量
│   ├── external/MINIMA/             # Apache-2.0 MINIMA 源码快照
│   └── gave2_solution/              # 模型、训练、预测与后处理
├── configs/                         # 训练配置、路由和权重清单
├── weights/                         # 不提交权重，只保留路径说明
├── environment.yml
└── requirements.txt
```

## 环境安装

正式训练与推理环境为 Python 3.11、PyTorch 2.4.0、CUDA 12.1 和 RTX 4090。

```bash
git clone git@github.com:KawhiQaQ/GAVE2-Challenge-8th-Solution.git
cd GAVE2-Challenge-8th-Solution

conda env create -f environment.yml
conda activate gave2
export PYTHONPATH="$PWD/code/gave2_solution"
```

也可以先按自己的 CUDA 版本安装 PyTorch，再执行：

```bash
pip install -r requirements.txt
```

## 数据准备

请从比赛官网下载 GAVE2 数据，并按以下结构组织。本仓库不会下载或分发比赛数据。

```text
GAVE2_preliminary/
├── training/
│   ├── images/g_xxx.png
│   ├── masks/g_xxx.png
│   ├── av/g_xxx.png
│   ├── FFA_A/g_xxx.png
│   ├── FFA_AV/g_xxx.png
│   └── biomarker/g_xxx.txt
└── validation/
    ├── images/g_xxx.png
    ├── masks/g_xxx.png
    ├── FFA_A/g_xxx.png
    └── FFA_AV/g_xxx.png
```

其中 `masks/` 是视野 ROI，`av/` 是 RGB 动脉/全血管/静脉标注。图像应保持
官方 1536×1024 分辨率。

## 权重

模型权重可从
[百度网盘](https://pan.baidu.com/s/1tqCe97lvXGM0zn1VvUlPMA?pwd=29v6)
下载，提取码为 `29v6`。请下载完整的 `weights/` 目录，并直接放在仓库根目录下。
网盘保留了训练时的原始归档目录名，因此还需执行以下命令，为公开推理代码创建
语义化权重路径：

```bash
ln -sf v1_task1_fold0/best.pt weights/initialization/task1_fold0.pt
ln -sf v1_task2_fold0/best.pt weights/initialization/task2_fold0.pt
ln -sf tj009_v54_full/final.pt weights/task1/segmentation_cfp.pt
ln -sf tj009_v54_full/final.pt weights/task2/segmentation_multimodal.pt
ln -sf v2_task2_full/final.pt weights/task3/caliber_expert.pt
ln -sf v3_task2_full/final.pt weights/task3/artery_density_expert.pt
ln -sf v8_task2_full/final.pt weights/task3/vein_density_parent.pt
ln -sf v21_v12_fd_full/final.pt weights/task3/phase_geometry_expert.pt
ln -sf d0044_vein_density_full/final.pt weights/task3/vein_density_refiner.pt
```

Task 1/2/3 的七个正式权重是推理必需项；两个 initialization 权重仅在完整
复现训练时使用。预期路径与 SHA256 记录在
[configs/weights_manifest.json](configs/weights_manifest.json)，也可查看
[weights/README.md](weights/README.md)。

网盘目录不包含两个第三方权重。请从
[MINIMA 官方 Release](https://github.com/LSXI7/storage/releases/download/MINIMA/minima_loftr.ckpt)
下载 MINIMA-LoFTR，并放到 `weights/external/minima_loftr.ckpt`；另将
MNet/DeepCDR 视盘权重放到
`weights/external/Model_DiscSeg_ORIGA.h5`。

## 复现 VascFusion 推理

准备好数据和权重后运行：

```bash
bash code/run_inference.sh \
  --data-root /absolute/path/GAVE2_private \
  --work-root /absolute/path/vascfusion_work \
  --output-zip /absolute/path/kawhi00.zip \
  --python "$(which python)" \
  --device cuda
```

输入根目录必须包含 `validation/{images,masks,FFA_A,FFA_AV}`，病例 ID 必须是
连续的 `g_###` 区间。脚本将依次完成：

1. 用 MINIMA-LoFTR 将早/晚期 FFA 分别配准至 CFP；
2. 检测视盘并生成 SIVA C 区；
3. 使用 VascFusion-Seg 完成 Task 1/Task 2 推理；
4. 运行管径、密度和时相几何测量分支；
5. 使用 VascFusion-Quant 确定性提取并组装指标；
6. 不改变 0.5 阈值判断的概率图压缩；
7. 校验并生成提交 ZIP。

输出结构为：

```text
Task1/g_xxx.png  # RGB = 动脉、全部血管、静脉概率
Task2/g_xxx.png  # RGB = 动脉、全部血管、静脉概率
Task3/g_xxx.txt  # 七项生物标志物
```

脚本不会覆盖已有工作目录或 ZIP。如果已有经过核验的配准、视盘或 C 区缓存，
可通过 `bash code/run_inference.sh --help` 查看可选缓存参数。

## 从头训练

以下命令同时支持交叉验证与全数据训练，均对应仓库内真实入口。若希望精确复现，
可以修改路径，但不要修改超参数。

### 1. 训练第一阶段初始化权重

VascFusion-Seg 正式模型分别从 Task 1/Task 2 的 Fold-0 第一阶段权重初始化。
训练 Fold 0：
（如需完整 CV，将 fold 依次设为 0–4。）

```bash
export DATA=/absolute/path/GAVE2_preliminary
export RUNS=$PWD/runs

python code/gave2_solution/train.py \
  --task 1 --fold 0 --n-folds 5 --data-root "$DATA" \
  --output-dir "$RUNS/init/task1/fold0" \
  --size 768x1152 --epochs 80 --patience 15 --seed 77 --device cuda

python code/gave2_solution/train.py \
  --task 2 --fold 0 --n-folds 5 --data-root "$DATA" \
  --output-dir "$RUNS/init/task2/fold0" \
  --size 768x1152 --epochs 80 --patience 15 --seed 77 --device cuda
```

### 2. 为 Task 2 配准 FFA

```bash
export REGISTERED=$PWD/cache/registered_ffa

python code/gave2_solution/register_ffa_minima.py \
  --data-root "$DATA" \
  --output-root "$REGISTERED" \
  --minima-root code/external/MINIMA \
  --checkpoint weights/external/minima_loftr.ckpt \
  --splits training validation
```

训练前应检查生成的 QA 图像与 `registration_manifest.json`。配准过程不读取血管
或生物标志物标签。

### 3. 训练最终 Task 1 与 Task 2

```bash
python code/gave2_solution/train_segmentation.py \
  --task 1 --data-root "$DATA" \
  --output-dir "$RUNS/vascfusion/task1" \
  --init-checkpoint "$RUNS/init/task1/fold0/best.pt" \
  --epochs 48 --size 1024x1536 --batch-size 1 \
  --accumulation-steps 2 --learning-rate 1.5e-4 \
  --encoder-lr-ratio 0.10 --weight-decay 1e-4 \
  --warmup-epochs 2 --freeze-encoder-epochs 1 \
  --ema-decay 0.995 --refinement-steps 5 --hard-gap-weight 0.25 \
  --seed 77 --device cuda --num-workers 0

python code/gave2_solution/train_segmentation.py \
  --task 2 --data-root "$DATA" --ffa-root "$REGISTERED" \
  --output-dir "$RUNS/vascfusion/task2" \
  --init-checkpoint "$RUNS/init/task2/fold0/best.pt" \
  --epochs 37 --size 1024x1536 --batch-size 1 \
  --accumulation-steps 2 --learning-rate 1.5e-4 \
  --encoder-lr-ratio 0.10 --weight-decay 1e-4 \
  --warmup-epochs 2 --freeze-encoder-epochs 1 \
  --ema-decay 0.995 --refinement-steps 5 --hard-gap-weight 0.25 \
  --seed 77 --device cuda --num-workers 0
```

对应 Task 1 的 1,200 次、Task 2 的 925 次 optimizer update。

### 4. 准备视盘与 SIVA C 区

```bash
python code/biomarker_tools/local_optic_disc.py \
  --data-root "$DATA" \
  --weights weights/external/Model_DiscSeg_ORIGA.h5 \
  --output-dir "$PWD/cache/disc_masks" --split all --device cuda

python code/gave2_solution/prepare_zone_c_masks.py \
  --disc-dir "$PWD/cache/disc_masks/training" \
  --output-dir "$PWD/cache/zone_c/training" \
  --report-output "$PWD/cache/zone_c/training_report.json"
```

### 5. 训练 VascFusion-Quant

指标量化系统使用多个针对不同测量目标的分割派生分支，并统一从 Task 2 Fold-0
第一阶段权重初始化：

```bash
export INIT_T2="$RUNS/init/task2/fold0/best.pt"
```

**管径分支——使用原始 FFA 几何：**

```bash
python code/gave2_solution/train_quantification.py \
  --task 2 --data-root "$DATA" --output-dir "$RUNS/quant/caliber" \
  --epochs 47 --size 1024x1536 --batch-size 1 --accumulation-steps 2 \
  --learning-rate 1.5e-4 --encoder-lr-ratio 0.10 --weight-decay 1e-4 \
  --warmup-epochs 2 --freeze-encoder-epochs 1 --ema-decay 0.995 \
  --num-refinements 3 --hard-gap-weight 0 --seed 77 --device cuda \
  --num-workers 0 --no-pretrained --init-checkpoint "$INIT_T2"
```

**动脉密度分支——使用原始 FFA 几何：**

```bash
python code/gave2_solution/train_quantification.py \
  --task 2 --data-root "$DATA" --output-dir "$RUNS/quant/artery_density" \
  --epochs 47 --size 1024x1536 --batch-size 1 --accumulation-steps 2 \
  --learning-rate 1.5e-4 --encoder-lr-ratio 0.10 --weight-decay 1e-4 \
  --warmup-epochs 2 --freeze-encoder-epochs 1 --ema-decay 0.995 \
  --num-refinements 3 --hard-gap-weight 0.25 --seed 77 --device cuda \
  --num-workers 0 --no-pretrained --init-checkpoint "$INIT_T2"
```

`configs/artery_density_calibration.json` 仅由训练集 OOF 预测确定，推理阶段保持固定。

**静脉密度的配准 FFA 父网络：**

```bash
python code/gave2_solution/train_quantification.py \
  --task 2 --data-root "$DATA" --ffa-root "$REGISTERED" \
  --output-dir "$RUNS/quant/vein_parent" \
  --epochs 48 --size 1024x1536 --batch-size 1 --accumulation-steps 2 \
  --learning-rate 1.5e-4 --encoder-lr-ratio 0.10 --weight-decay 1e-4 \
  --warmup-epochs 2 --freeze-encoder-epochs 1 --ema-decay 0.995 \
  --num-refinements 3 --hard-gap-weight 0.25 --seed 77 --device cuda \
  --num-workers 0 --no-pretrained --init-checkpoint "$INIT_T2"
```

**分形维数的时相几何分支：**

```bash
python code/gave2_solution/train_quantification.py \
  --task 2 --phase-geometry --data-root "$DATA" \
  --ffa-root "$REGISTERED" --output-dir "$RUNS/quant/phase_geometry" \
  --epochs 48 --size 1024x1536 --batch-size 1 --accumulation-steps 2 \
  --learning-rate 1.5e-4 --encoder-lr-ratio 0.10 --weight-decay 1e-4 \
  --warmup-epochs 2 --freeze-encoder-epochs 1 --ema-decay 0.995 \
  --num-refinements 3 --hard-gap-weight 0.25 --seed 77 --device cuda \
  --num-workers 0 --init-checkpoint "$INIT_T2"
```

**轻量 C 区静脉修正器：**

```bash
python code/gave2_solution/train_vein_refiner.py \
  --data-root "$DATA" --ffa-root "$REGISTERED" \
  --zone-c-dir "$PWD/cache/zone_c/training" \
  --parent-checkpoint "$RUNS/quant/vein_parent/final.pt" \
  --output-dir "$RUNS/quant/vein_refiner" \
  --size 1024x1536 --epochs 24 --accumulation-steps 2 \
  --learning-rate 3e-4 --weight-decay 1e-4 --warmup-epochs 2 \
  --ema-decay 0.995 --seed 77 --device cuda --num-workers 0 \
  --loss-policy vessel_argmax
```

最后阶段仅更新轻量 C 区修正器，其父网络保持冻结。

## 评测与使用约束

对于本地有标签的预测，可复现公开 A/V 指标：

```bash
python code/gave2_solution/evaluate_av.py \
  --predictions-dir /path/to/probability_pngs \
  --labels-dir "$DATA/training/av" \
  --masks-dir "$DATA/training/masks" \
  --output /path/to/metrics.json
```

Task 3 的 `local_biomarker_eval.py` 会在有标签训练折上报告原始 MAE 与 SMAPE；
本地评测不会猜测主办方未公开的 P/Q 归一化函数。

- Task 1 绝不能读取 FFA；
- Task 1/2 PNG 通道顺序为 `[动脉, 全部血管, 静脉]`；
- Task 3 对所有病例使用完全相同的字段路由；
- 不使用测试标签或逐病例榜单选择；
- 不在验证集/私榜测试集重新拟合校准因子；

## 第三方代码

为保证 FFA 配准可复现，本仓库保留了
[MINIMA](https://github.com/LSXI7/MINIMA) 的未修改源码快照及其 Apache-2.0
许可证。视盘权重来源于 MNet/DeepCDR，本仓库不重新分发该权重。详见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 引用

我们的 Challenge 报告正式发表后会补充引用信息。如果本代码对你有帮助，也请
引用 GAVE2 Challenge 及 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
中列出的上游工作。

## 许可证

本仓库原创代码采用 [MIT License](LICENSE)。第三方组件继续遵循各自的许可证，
详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
