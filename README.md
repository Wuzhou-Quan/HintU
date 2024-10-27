# HintU

> **Lost in UNet: Improving Infrared Small Target Detection by Underappreciated Local Features**  
> Wuzhou Quan, Wei Zhao, Weiming Wang, Haoran Xie, Fu Lee Wang, Mingqiang Wei   
> [Paper(ArXiv)](https://arxiv.org/abs/2406.13445)

We have submitted our paper to IEEE TGRS for review and will make the related code accessible after publication.

## Prerequisites

### Environment

```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

pip install einops mmengine click clearml rich scikit-image
```

For MiM+, [state-spaces/mamba](https://github.com/state-spaces/mamba) is also required, please refer to [txchen-USTC/MiM-ISTD](https://github.com/txchen-USTC/MiM-ISTD).

### Datasets

#### 1. File system architecture
Please ensure that the IRSTD1K, NUDT-SIRST, and SIRSTv2 datasets are properly downloaded and organized as follows:

```
root
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

```bash
python train.py --model_arch_name UNet --train_dataset_name irstd1k_train --val_dataset_name irstd1k_test --max_epoches 300
```

**ClearML** is used for logging and visualization, for more details, please refer to their [official doc](https://clear.ml/docs/latest/docs/).

## Checkpoints

All released pre-trained weights are available in [Google Drive](https://drive.google.com/drive/folders/1KSclFKKv6Kx0eVOSzTeJusxA9GZfJLX3?usp=sharing).

## Acknowledgement

A large part of the code is borrowed from [ShawnBIT/UNet-family](https://github.com/ShawnBIT/UNet-family), [SuGuilin/UIUNet_mod](https://github.com/SuGuilin/UIUNet_mod), [txchen-USTC/MiM-ISTD](https://github.com/txchen-USTC/MiM-ISTD), [zhengshuchen/HCFNet](https://github.com/zhengshuchen/HCFNet), and so on.
Thanks for their wonderful works.

## Citation

```
@misc{quan2024lost,
      title={Lost in UNet: Improving Infrared Small Target Detection by Underappreciated Local Features}, 
      author={Wuzhou Quan and Wei Zhao and Weiming Wang and Haoran Xie and Fu Lee Wang and Mingqiang Wei},
      year={2024},
      eprint={2406.13445},
      archivePrefix={arXiv}
}
```
