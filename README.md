# HintU

[![TGRS](https://img.shields.io/badge/IEEE%20TGRS-2025-blue.svg)](https://ieeexplore.ieee.org/document/10764792) [![ArXiv](https://img.shields.io/badge/ArXiv-2024-red.svg)](https://arxiv.org/abs/2406.13445)
> **Lost in UNet: Improving Infrared Small Target Detection by Underappreciated Local Features**  
> Wuzhou Quan, Wei Zhao, Weiming Wang, Haoran Xie, Fu Lee Wang, Mingqiang Wei

This repository contains the official implementation of the paper "**Lost in UNet: Improving Infrared Small Target Detection by Underappreciated Local Features**".
Besides, it is also a simple and integrated framework for infrared small target detection, which is easy to use and extend.

If our work is helpful to you, please cite it as follows:

```
@ARTICLE{quan2024lost,
  author={Quan, Wuzhou and Zhao, Wei and Wang, Weiming and Xie, Haoran and Lee Wang, Fu and Wei, Mingqiang},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Lost in UNet: Improving Infrared Small Target Detection by Underappreciated Local Features}, 
  year={2025},
  volume={63},
  number={},
  pages={1-15},
  keywords={Feature extraction;Semantics;Object detection;Geoscience and remote sensing;Filtering theory;Convolution;Clutter;Weight measurement;Visualization;Transformers;HintU;infrared small target detection (ISTD);UNet},
  doi={10.1109/TGRS.2024.3504594}
}
```

**_Thanks for your attention!_**

## Prerequisites

### Environment

```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

pip install einops mmengine click clearml rich scikit-image
```

For MiM+, [state-spaces/mamba](https://github.com/state-spaces/mamba) is also required, please refer to [txchen-USTC/MiM-ISTD](https://github.com/txchen-USTC/MiM-ISTD).

### Datasets

#### 1. File system architecture

Please ensure that the [IRSTD1K](https://github.com/RuiZhang97/ISNet), [NUDT-SIRST](https://github.com/YeRen123455/Infrared-Small-Target-Detection), and [SIRSTv2](https://github.com/YimianDai/open-sirst-v2) datasets are properly downloaded and organized as follows:

```
HINTU (Root folder)
└── data
    ├── IRSTD1K
    │   ├── IRSTD1k_Img
    │   └── IRSTD1k_Label
    ├── sirstv2
    │   ├── images
    │   │   ├── targets
    │   │   └── backgrounds
    │   └── annotations
    │       └── masks
    └── NUDT-SIRST
        ├── images
        └── masks
```

If you need a custom folder structure, modify the `folder_arch` dictionary in the `gsettings.py` file to fit your folder architecture.
Additionally, ensure these folders have read and write permissions.

#### 2. Preprocess SIRSTv2 dataset

Execute `python gen_sirstv2_mask_file.py` to rename the mask files.

#### 3. Generate filter file

Execute `python gen_filter_files.py` to split the dataset into two mutually exclusive parts for distinguishing the training and testing sets.
You can modify the value of 0.5 to set the proportion of the test set (the default is 0.5, meaning a 1:1 split).

## Quick Start

### Training

If you want to train a model on IRSTD1k with batch size of 4, you can run the following command:

```bash
python train.py -m HintHCFNet -t irstd1k_train -v irstd1k_test -b 4 --max_epoches 300
```

**ClearML** is used for logging and visualization, for more details, please refer to their [official doc](https://clear.ml/docs/latest/docs/).

`--model_arch_name` or `-m` specifies the model architecture (located in the path of [`deployments/models`](deployments/models)). You can refer to the existing files to create your own model architecture (b.t.w. _it's quite simple_), or refer to the [official MMEngine config file documentation](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html) for more details.


`--train_dataset_name` or `-t` and `--val_dataset_name` or `-v` decide the training and validation datasets (located in the path of [`deployments/datasets`](deployments/datasets)), respectively.

`--batch_size` or `-b` specifies the batch size.

`--max_epoches` specifies the maximum number of training epochs.

### Testing

_Todo_

### Evaluation

_Todo_

## Checkpoints

All released pre-trained weights are available in [Google Drive](https://drive.google.com/drive/folders/1KSclFKKv6Kx0eVOSzTeJusxA9GZfJLX3?usp=sharing).

## Acknowledgement

A large part of the code is borrowed from [ShawnBIT/UNet-family](https://github.com/ShawnBIT/UNet-family), [SuGuilin/UIUNet_mod](https://github.com/SuGuilin/UIUNet_mod), [txchen-USTC/MiM-ISTD](https://github.com/txchen-USTC/MiM-ISTD), [zhengshuchen/HCFNet](https://github.com/zhengshuchen/HCFNet), and so on.
Thanks for their wonderful works.
