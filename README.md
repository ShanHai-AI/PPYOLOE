# PPYOLOE
PPYOLOE目标检测训练框架

# PPYOLOE
PP-YOLOE是基于PP-YOLOv2的卓越的单阶段Anchor-free模型，超越了多种流行的YOLO模型。PP-YOLOE有一系列的模型，即s/m/l/x，可以通过width multiplier和depth multiplier配置。PP-YOLOE避免了使用诸如Deformable Convolution或者Matrix NMS之类的特殊算子，以使其能轻松地部署在多种多样的硬件上。

根据PaddleDetection给出的云端模型性能对比，各模型结构和骨干网络的代表模型在COCO数据集上精度mAP和单卡Tesla V100上预测速度(FPS)对比图如下：

![https://gitee.com/paddlepaddle/PaddleDetection/raw/develop/docs/images/fps_map.png](https://gitee.com/paddlepaddle/PaddleDetection/raw/develop/docs/images/fps_map.png)

可以看出，PP-YOLOE真可谓是【又快又好】的典型！

这还不够，PaddleDetection团队还提供了基于PP-YOLOE的各种垂类检测模型的配置文件和权重，供用户下载进行使用：

| 场景 | 相关数据集 | 链接 |
| :-: | :-: | :-: |
| 行人检测 | CrowdHuman | [pphuman](https://gitee.com/paddlepaddle/PaddleDetection/blob/develop/configs/pphuman) |
| 车辆检测 | BDD100K、UA-DETRAC | [ppvehicle](https://gitee.com/paddlepaddle/PaddleDetection/blob/develop/configs/ppvehicle) |
| 小目标检测 | VisDrone | [visdrone](https://gitee.com/paddlepaddle/PaddleDetection/blob/develop/configs/visdrone) |

本项目参考了**PPYOLOE：又快又好的小目标检测训练与部署实现**

https://aistudio.baidu.com/aistudio/projectdetail/4435291?channelType=0&channel=0

# 1 环境准备
## 1.1 数据集准备
数据集的分析和准备过程可参考博客
[使用PPYOLOE训练目标检测]([https://blog.csdn.net/qq_41251963/article/details/129667684])

1.**导入所需要的第三方库**
```
# 调用一些需要的第三方库
import numpy as np
import pandas as pd
import shutil
import json
import os
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from matplotlib.font_manager import FontProperties
from PIL import Image
import random
myfont = FontProperties(fname=r"NotoSansCJKsc-Medium.otf", size=12)
plt.rcParams['figure.figsize'] = (12, 12)
plt.rcParams['font.family']= myfont.get_family()
plt.rcParams['font.sans-serif'] = myfont.get_name()
plt.rcParams['axes.unicode_minus'] = False
```
2.**安装paddlex**
```
# 引入PaddleX
!pip install paddlex
```
3.**创建数据集目录** 

将标注的图像数据上传到 **MyDataset/JPEGImages** 目录下。

将coco格式数据标签**annotations.json**放到**MyDataset**目录下。

```
# 组织数据目录
!mkdir MyDataset
!mkdir MyDataset/JPEGImages
```

4.**按比例切分数据集**
```
# 按比例切分数据集
!paddlex --split_dataset --format COCO --dataset_dir /home/aistudio/MyDataset --val_value 0.1 --test_value 0.0
```

## 1.2 训练环境准备

由于PP-YOLOE还在快速迭代中，因此，对框架的稳定性有一定的要求，PaddlePaddle的框架不要选择最新版。本文使用的单卡训练环境如下：

- 框架版本：PaddlePaddle 2.2.2
- CUDA Version: 11.2
- 模型库版本：PaddleDetection(develop分支)

选择PaddleDetection(develop分支)的原因是，PP-YOLOE的垂类模型迭代更快些，选择空间更大。


5. **git PaddleDetection 2.5版本**代码
   ```
   !git clone -b release/2.5 https://github.com/PaddlePaddle/PaddleDetection.git
   ```
6. **进入PaddleDetection目录**
   ```
   %cd PaddleDetection
   ```

# 2 模型训练
## 2.1 模型选型

PaddleDetection团队提供了针对VisDrone-DET小目标数航拍场景的基于PP-YOLOE的检测模型，本项目使用这些模型作为预训练模型。

本项目以`PP-YOLOE-Alpha-largesize-l`为例，展示训练和部署过程。
## 2.2 模型训练
7. **根据需求修改配置文件，比如检测的目标类别数**

进入/home/aistudio/config_file/目录下，修改visdrone_detection.yml中**num_classes**参数

```
# 训练配置文件覆盖
!cd configs
!mkdir configs/visdrone/
!cd ../
!cp ../config_file/visdrone/ppyoloe_crn_l_80e_visdrone.yml configs/visdrone/ppyoloe_crn_l_80e_visdrone.yml
!cp ../config_file/visdrone/ppyoloe_crn_l_alpha_largesize_80e_visdrone.yml configs/visdrone/ppyoloe_crn_l_alpha_largesize_80e_visdrone.yml
!cp ../config_file/visdrone_detection.yml configs/datasets/visdrone_detection.yml
!cp ../config_file/optimizer_300e.yml configs/ppyoloe/_base_/optimizer_300e.yml
!cp ../config_file/ppyoloe_crn.yml configs/ppyoloe/_base_/ppyoloe_crn.yml
!cp ../config_file/ppyoloe_reader.yml configs/ppyoloe/_base_/ppyoloe_reader.yml
```
8. **开始训练**

mAP 不再增长时，即可停止训练。

```
# 开始训练
!python tools/train.py -c configs/visdrone/ppyoloe_crn_l_alpha_largesize_80e_visdrone.yml --use_vdl=True --vdl_log_dir=./visdrone/ --eval
```

9. **训练完成后评估模型**
 ```
# 模型评估
!python tools/eval.py -c configs/visdrone/ppyoloe_crn_l_alpha_largesize_80e_visdrone.yml -o weights=output/ppyoloe_crn_l_alpha_largesize_80e_visdrone/best_model.pdparams
```
## 2.3 预测推理
```
# 挑一张验证集的图片展示预测效果
!python tools/infer.py -c configs/visdrone/ppyoloe_crn_l_alpha_largesize_80e_visdrone.yml -o weights=output/ppyoloe_crn_l_alpha_largesize_80e_visdrone/best_model --infer_img=/home/aistudio/MyDataset/JPEGImages/1679146167842.jpg --save_results=True
```
# 3 模型部署
同样是小目标检测，相比于`SNIPER: Efficient Multi-Scale Training`暂不支持部署，PP-YOLOE是可以直接导出部署模型，并在多端高性能部署的：
- Paddle Inference [Python](https://gitee.com/paddlepaddle/PaddleDetection/blob/develop/deploy/python) & [C++](https://gitee.com/paddlepaddle/PaddleDetection/blob/develop/deploy/cpp)
- [Paddle-TensorRT](https://gitee.com/paddlepaddle/PaddleDetection/blob/develop/deploy/TENSOR_RT.md)
- [PaddleServing](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2FPaddlePaddle%2FServing)
- [PaddleSlim模型量化](https://gitee.com/paddlepaddle/PaddleDetection/blob/develop/configs/slim)

接下来，我们将介绍PP-YOLOE如何使用Paddle Inference进行部署。
## 3.1 导出模型
10. **导出模型，即可使用FastDeploy进行快速推理**

https://github.com/PaddlePaddle/FastDeploy

```
!python tools/export_model.py -c configs/visdrone/ppyoloe_crn_l_alpha_largesize_80e_visdrone.yml -o weights=output/ppyoloe_crn_l_alpha_largesize_80e_visdrone/best_model.pdparams
```
