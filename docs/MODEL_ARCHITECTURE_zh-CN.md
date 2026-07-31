# TJ009 模型架构说明

## 总览

TJ009 由两个正式分割模型和一个固定字段式 biomarker 系统组成：

1. Task1：V54 CFP-only 动静脉分割；
2. Task2：V54 CFP + 配准 FFA 动静脉分割；
3. Task3：V2/V3/V12/V17 的几何结果按字段固定路由。

所有输出保持挑战赛 RGB 语义：R=动脉，G=全部血管，B=静脉。

## Task1/Task2：GAVEV54

代码：`code/gave2_solution/gave2v1/model_v54.py`。

共同主干：

- ConvNeXt-Tiny 多尺度 CFP 编码器；
- U-Net 式多尺度解码器与原生分辨率 detail stem；
- semantic、generic-vessel、class-centerline 多头结构；
- 固定 generic-vessel 概率作为条件，使用共享的深层概率 U-Net 对 A/V
  分类递归修正 5 步；
- 输出层以零残差初始化，使新增递归模块从父模型行为平滑开始，而不是破坏
  已学到的血管几何。

Task1 专有部分：

- 严格只输入 CFP；
- detail stem 除归一化 RGB 外，加入红/绿通道的局部 optical-density 特征；
- 使用 native-resolution A/V final refiner，重点改善细血管和分支拓扑。

Task2 专有部分：

- CFP 使用 ConvNeXt-Tiny 编码；
- 早期 FFA、晚期 FFA 与正差分组成 3 通道 FFA；
- 轻量 FFA pyramid 在每个编码尺度通过 gated cross-modal fusion 注入 CFP；
- 输入 FFA 先由 MINIMA-LoFTR 独立估计早/晚期到 CFP 的单应变换。

训练参数来自 checkpoint 配置：

| 项目 | Task1 | Task2 |
|---|---:|---:|
| 分辨率 | 1024×1536 | 1024×1536 |
| epochs | 48 | 37 |
| optimizer updates | 1200 | 925 |
| batch / accumulation | 1 / 2 | 1 / 2 |
| learning rate | 1.5e-4 | 1.5e-4 |
| encoder LR ratio | 0.10 | 0.10 |
| weight decay | 1e-4 | 1e-4 |
| warmup | 2 epochs | 2 epochs |
| encoder freeze | 1 epoch | 1 epoch |
| EMA | 0.995 | 0.995 |
| seed | 77 | 77 |

两者从各自 V1 Fold0 第一阶段 checkpoint 做结构兼容初始化，但之后使用全部
50 个带标签样本正式训练；没有加载公开 R2-V2 比赛权重。

## Task3：固定字段几何系统

Task3 不依赖一个统一回归头。先从概率图按官方/基线几何公式提取指标，再按
全局固定规则选择每个字段的已验证来源：

| 字段 | 来源 | 模态/处理 |
|---|---|---|
| CRAE、CRVE、AVR | V2 Task2 full | 原始未配准 FFA；`vessel_argmax`/管径公式 |
| artery_density | V3 Task2 full | 原始未配准 FFA；全局 OOF 比例校准 |
| vein_density | D0044 | 注册 FFA；冻结 V8 + C 区低容量静脉残差头 |
| artery_fractal_dimension | D0040 | 对 V2 概率图做确定性 branch consistency 后计盒 |
| vein_fractal_dimension | V21/V12 | 注册 FFA；0.4/0.5/0.6 三阈值标量 FD 算术平均 |

V12 在 V2 的 CFP/轻量 FFA 路径之外增加一个完整 ConvNeXt-Tiny phase
encoder，并以小残差在四个尺度融合注册 FFA。D0044 冻结 V8 Task2，只训练
一个低容量 U-Net 式静脉 logit 残差头，而且只在 SIVA C 区改变静脉概率。

最终路由不读取标签、不按病例切换、不扫描阈值、不根据私榜结果做逐样本选择。
机器可读定义见 `routes/tj009_route.json`。

## 后处理与打包

- Task1/Task2：`postprocess_av.py --radius 3`；
- 概率图：step-4 量化，保证每个像素在 `128/255` 两侧的判定完全不变；
- Task3：7 行固定键序 TXT；
- ZIP：`Task1/`、`Task2/`、`Task3/`，总计 150 个文件。
