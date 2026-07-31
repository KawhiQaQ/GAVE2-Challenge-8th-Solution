# 训练细节

本文档补充根目录 README 中的训练命令。首先设置：

```bash
export PYTHONPATH="$PWD/code/gave2_solution"
export DATA=/absolute/path/GAVE2_preliminary
export REGISTERED=$PWD/cache/registered_ffa
export RUNS=$PWD/runs
```

## 交叉验证

`train.py` 使用 seed 77，根据血管密度与网络复杂度构造平衡折。完整五折 V1：

```bash
for fold in 0 1 2 3 4; do
  python code/gave2_solution/train.py \
    --task 1 --fold "$fold" --n-folds 5 --data-root "$DATA" \
    --output-dir "$RUNS/cv/task1/fold${fold}" \
    --size 768x1152 --epochs 80 --patience 15 --seed 77 --device cuda

  python code/gave2_solution/train.py \
    --task 2 --fold "$fold" --n-folds 5 --data-root "$DATA" \
    --output-dir "$RUNS/cv/task2/fold${fold}" \
    --size 768x1152 --epochs 80 --patience 15 --seed 77 --device cuda
done
```

TJ009 只使用各 Task 的 Fold-0 checkpoint 作为全量训练初始化，并非五模型推理融合。

## Task 1/Task 2 全量训练

请使用根目录中文 README 中的两条 V54 命令。精确配置同时保存在
`configs/training/task1_v54.json` 与 `configs/training/task2_v54.json`。

## 视盘和 C 区准备

将 MNet/DeepCDR HDF5 权重放到 `weights/external/` 后运行：

```bash
python code/biomarker_tools/local_optic_disc.py \
  --data-root "$DATA" \
  --weights weights/external/Model_DiscSeg_ORIGA.h5 \
  --output-dir "$PWD/cache/disc_masks" --split all --device cuda

python code/gave2_solution/prepare_zone_c_masks.py \
  --disc-dir "$PWD/cache/disc_masks/training" \
  --output-dir "$PWD/cache/zone_c/training" \
  --report-output "$PWD/cache/zone_c/training_report.json"

python code/gave2_solution/prepare_zone_c_masks.py \
  --disc-dir "$PWD/cache/disc_masks/validation" \
  --output-dir "$PWD/cache/zone_c/validation" \
  --report-output "$PWD/cache/zone_c/validation_report.json"
```

## Task 3 来源训练

所有来源均从 Task 2 V1 Fold-0 初始化：

```bash
export INIT_T2="$RUNS/init/task2/fold0/best.pt"
```

### V2：原始 FFA，负责 CRAE/CRVE/AVR

```bash
python code/gave2_solution/train_full_v2.py \
  --task 2 --data-root "$DATA" --output-dir "$RUNS/task3/v2" \
  --epochs 47 --size 1024x1536 --batch-size 1 --accumulation-steps 2 \
  --learning-rate 1.5e-4 --encoder-lr-ratio 0.10 --weight-decay 1e-4 \
  --warmup-epochs 2 --freeze-encoder-epochs 1 --ema-decay 0.995 \
  --num-refinements 3 --hard-gap-weight 0 --seed 77 --device cuda \
  --num-workers 0 --no-pretrained --init-checkpoint "$INIT_T2"
```

### V3：原始 FFA，负责动脉密度

```bash
python code/gave2_solution/train_full_v2.py \
  --task 2 --data-root "$DATA" --output-dir "$RUNS/task3/v3" \
  --epochs 47 --size 1024x1536 --batch-size 1 --accumulation-steps 2 \
  --learning-rate 1.5e-4 --encoder-lr-ratio 0.10 --weight-decay 1e-4 \
  --warmup-epochs 2 --freeze-encoder-epochs 1 --ema-decay 0.995 \
  --num-refinements 3 --hard-gap-weight 0.25 --seed 77 --device cuda \
  --num-workers 0 --no-pretrained --init-checkpoint "$INIT_T2"
```

`configs/v3_density_calibration.json` 的密度校准因子来自训练集 OOF 预测，不能在
测试数据上重新拟合。

### V8：配准 FFA，作为静脉密度父模型

```bash
python code/gave2_solution/train_full_v2.py \
  --task 2 --data-root "$DATA" --ffa-root "$REGISTERED" \
  --output-dir "$RUNS/task3/v8" \
  --epochs 48 --size 1024x1536 --batch-size 1 --accumulation-steps 2 \
  --learning-rate 1.5e-4 --encoder-lr-ratio 0.10 --weight-decay 1e-4 \
  --warmup-epochs 2 --freeze-encoder-epochs 1 --ema-decay 0.995 \
  --num-refinements 3 --hard-gap-weight 0.25 --seed 77 --device cuda \
  --num-workers 0 --no-pretrained --init-checkpoint "$INIT_T2"
```

### V12：配准 FFA，负责静脉分形维数

```bash
python code/gave2_solution/train_full_v2.py \
  --task 2 --v12-phase-geometry --data-root "$DATA" \
  --ffa-root "$REGISTERED" --output-dir "$RUNS/task3/v12" \
  --epochs 48 --size 1024x1536 --batch-size 1 --accumulation-steps 2 \
  --learning-rate 1.5e-4 --encoder-lr-ratio 0.10 --weight-decay 1e-4 \
  --warmup-epochs 2 --freeze-encoder-epochs 1 --ema-decay 0.995 \
  --num-refinements 3 --hard-gap-weight 0.25 --seed 77 --device cuda \
  --num-workers 0 --init-checkpoint "$INIT_T2"
```

V12 的新 FFA phase encoder 保留 ImageNet 初始化，因此不要添加
`--no-pretrained`。

### D0044：低容量 C 区静脉密度修正器

```bash
python code/gave2_solution/train_full_v17_task3_vein.py \
  --data-root "$DATA" --ffa-root "$REGISTERED" \
  --zone-c-dir "$PWD/cache/zone_c/training" \
  --parent-checkpoint "$RUNS/task3/v8/final.pt" \
  --output-dir "$RUNS/task3/d0044" \
  --size 1024x1536 --epochs 24 --accumulation-steps 2 \
  --learning-rate 3e-4 --weight-decay 1e-4 --warmup-epochs 2 \
  --ema-decay 0.995 --seed 77 --device cuda --num-workers 0 \
  --loss-policy vessel_argmax
```

训练时 V8 完全冻结，只更新小型 refiner。

## 使用自训练权重推理

将最终 checkpoint 复制或软链接到 `weights/README.md` 规定的位置，核对配置后
运行 `code/run_tj009_inference.sh`。统一脚本会完成预测、几何测量、字段路由、
压缩和打包。

如果开发的是新版本而非复现 TJ009，应同时更新路由和权重清单，不能悄悄用新
文件覆盖旧版本名。

