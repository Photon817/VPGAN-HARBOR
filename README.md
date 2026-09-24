# VPGAN / HARBOR — 非配对组织虚拟染色

把一条常规 **H&E** 染色切片，翻译成 PAS / MAS / PASM 等特殊染色图像，无需重新切片机、无需额外试剂。核心是两件事：

- **VPGAN**：CycleGAN 骨架的非配对图像翻译，负责 H&E ⇄ 特殊染色域之间的双向映射，并注入 VLM 文本提示做方向约束。
- **HARBOR**：在 VPGAN 生成结果之上，用一个预训练 DDIM 扩散模型做推理端增强，抑制伪影、稳定纹理。



---

## 染色域

四个域两两组合成非配对翻译任务，域内图像按 `train/<域>/`、`val/<域>/` 存放：

| 代号 | 染色 | 说明 |
|------|------|------|
| `H&E` | 苏木精-伊红 | 常规病理主染，本仓库的默认源域 |
| `PAS` | 过碘酸雪夫 | 基膜 / 系膜基质阳性 |
| `MAS` | 马松三色 | 胶原偏蓝绿 |
| `PASM` | 六胺银 | 嗜银基底膜 |

源域到目标域即 `netG_A`（如 `he2pas`），反向为 `netG_B`。

---

## 安装

已在 Linux/Ubuntu 22.04 与 macOS（Apple Silicon）下验证。不再强制要求多卡：无 CUDA 时自动回退 MPS，再回退 CPU。

```bash
conda env create -f VPGAN/environment.yml    # 或 HARBOR/environment.yaml
conda activate vpgan
pip install -r VPGAN/requirements.txt
```

依赖要点：`torch>=2.0`、`numpy`、`opencv`、`Pillow`、`openai-clip`、`blobfile`（HARBOR 侧 DDIM 用）。CONCH 相关代码已 vendored 在 `VPGAN/vlm/`，无需额外拉取。

---

## 数据准备

真实训练数据取自 [ANHIR](https://anhir.grand-challenge.org/) 肾活检 WSI，按 [UMDST](https://ojs.aaai.org/index.php/AAAI/article/view/20054) 的方法切 256×256 patch 并划分 train/val：

```
DATA_ROOT_DIR/
├── train/
│   ├── H&E/    ├── PAS/    ├── MAS/    ├── PASM/
└── val/
    ├── H&E/    ├── PAS/    ├── MAS/    ├── PASM/
```

---

## 文本提示（.pt）

VPGAN 训练前需先离线生成 VLM 文本嵌入，放到 `VPGAN/text/`：

```
VPGAN/text/
├── concept/         # {he,mas,pas,pasm,same}_concepts.pt  概念锚点
├── HE/contra.pt     # 各域对比提示（contrastive）
├── PAS/contra.pt
├── MAS/contra.pt
└── PASM/contra.pt
```

- 概念锚点：用视觉-语言模型对每个域的形态学描述做嵌入，参考 [CLIP-LIT](https://github.com/ZhexinLiang/CLIP-LIT) 的二分类初始化写法，把 CLIP 换成 [CONCH](https://github.com/mahmoodlab/CONCH) 得到病理专用提示。
- 读取路径与格式可在 `train.py` 里按需调整。
- 描述文本本身交给当下任一多模态/大语言模型生成即可，重点是措辞稳定、可复现，不绑定特定模型版本。

---

## 训练

```bash
python VPGAN/train.py \
  --dataroot ./DATA_ROOT_DIR/ \
  --name <task_name> \
  --gpu_ids 0            # 无 CUDA 留空或传 -1，自动走 MPS/CPU \
  --checkpoint_dir <CONCH权重目录>
```

权重输出在 `VPGAN/checkpoints/<task_name>/`。不同数据集/域对的超参见论文正文与补充材料。

## 推理增强（HARBOR）

先用 [guided-diffusion](https://github.com/openai/guided-diffusion) 预训练一个类别条件 DDIM，再对 VPGAN 结果做增强：

```bash
CUDA_VISIBLE_DEVICES=0 python HARBOR/main.py \
  --model_path ./pretrained_model/<DDIM权重> \
  --target_domain MAS \
  --target_name <task_name> \
  --data_dir ./DATA_ROOT_DIR/val/ \
  --class_cond True
```

---

## 本地适配（相对上游）

为让仓库在单机、Apple Silicon、无多卡环境下开箱即跑，做了如下改动（源码内以 `PathAI 本地适配` 注释标注，检索该关键字可定位全部改动）：

| 文件 | 改动 |
|------|------|
| `VPGAN/util/device.py` | **新增**：`pick_device()` 统一 CUDA→MPS→CPU 选择 |
| `VPGAN/models/base_model.py` | 设备解析改用 `pick_device` |
| `VPGAN/models/networks.py` | 无 CUDA 时将网络移至 MPS/CPU；`torch.load` 兼容 `map_location` |
| `VPGAN/options/base_options.py` | `--gpu_ids` 允许 `mps`/`cpu`/空值，不再硬性 `set_device` |
| `VPGAN/clip_score.py` | 提示嵌入在无 GPU 时的设备对齐 |
| `HARBOR/guided_diffusion/dist_util.py` | 单进程/无 NCCL 时的通信后端回退 |
| `HARBOR/main.py` | 参数与设备路径适配单机推理 |

这些改动**不改变算法**，只放宽运行环境假设。

---

## 复现边界（如实说明）

- 本仓库**不含训练权重**，需自行训练或按论文配置复现。
- 论文级结果依赖：ANHIR WSI、CONCH 提示嵌入、以及域特定的 DDIM 预训练——三者缺一即退化为演示效果。
- 若只需一个能点开就看的 Demo，请配合上层 PathAI 应用（自带一份小样本 demo 权重与合成/真实切片），本仓库负责"从哪来"。

## 致谢

构建于 [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)、[CLIP-LIT](https://github.com/ZhexinLiang/CLIP-LIT)、[DPI](https://github.com/DianaNerualNetwork/StainPromptInversion)、[guided-diffusion](https://github.com/openai/guided-diffusion) 之上。VPGAN 目录沿用 CycleGAN 的 MIT 许可，HARBOR 目录许可见其 `LICENSE`。
