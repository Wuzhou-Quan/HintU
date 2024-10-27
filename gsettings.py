project_name = "HintU"
print("=" * 80)
print(
    r"""This project is authored by: Quan Wuzhou.
 ________  ___  ___  ________  ________                              
|\   __  \|\  \|\  \|\   __  \|\   ___  \                            
\ \  \|\  \ \  \\\  \ \  \|\  \ \  \\ \  \                           
 \ \  \\\  \ \  \\\  \ \   __  \ \  \\ \  \                          
  \ \  \\\  \ \  \\\  \ \  \ \  \ \  \\ \  \                         
   \ \_____  \ \_______\ \__\ \__\ \__\\ \__\                        
    \|___| \__\|_______|\|__|\|__|\|__| \|__|                        
 ___      \|__| ___  ___  ________  ___  ___  ________  ___  ___     
|\  \     |\  \|\  \|\  \|\_____  \|\  \|\  \|\   __  \|\  \|\  \    
\ \  \    \ \  \ \  \\\  \\|___/  /\ \  \\\  \ \  \|\  \ \  \\\  \   
 \ \  \  __\ \  \ \  \\\  \   /  / /\ \   __  \ \  \\\  \ \  \\\  \  
  \ \  \|\__\_\  \ \  \\\  \ /  /_/__\ \  \ \  \ \  \\\  \ \  \\\  \ 
   \ \____________\ \_______|\________\ \__\ \__\ \_______\ \_______\
    \|____________|\|_______|\|_______|\|__|\|__|\|_______|\|_______|
"""
)
print("=" * 80)

import time
from log import logger

logger.info(f"Hello!")

torch_timer = time.time()
import torch

logger.info(f"Pytorch loaded in {time.time() - torch_timer:.2f}s.")
del torch_timer

logger.info(f"PyTorch Version: {torch.__version__}")
if torch.cuda.is_available():
    logger.info(f"PyTorch CUDA Version: {torch.version.cuda}")
    logger.info(f"PyTorch CUDA Current Device: {torch.cuda.current_device()+1} ({torch.cuda.get_device_name(torch.cuda.current_device())}) / {torch.cuda.device_count()}")
    logger.info(f"PyTorch CUDNN Version: {torch.backends.cudnn.version()}")

import sys, os, random, numpy as np, torch, time, shutil, pathlib

# 随机种子
# Random seed
seed = 3407
if seed is not None:
    logger.info(f"[Settings Initiated] Global seed fixed: {seed}")
else:
    seed = random.randint(0, 100000)
    logger.warning(f"[Settings Initiated] Global seed is randomly set as: {seed}")
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
del seed

# 根目录路径
# Root Path
root_path = pathlib.Path(__file__).parent.absolute()
logger.info("[Settings Initiated] Root path: %s" % os.fspath(root_path))

exps_workdir_path = root_path / "exps"
if not exps_workdir_path.exists():
    exps_workdir_path.mkdir()
logger.info("[Settings Initiated] Model save path: %s" % os.fspath(exps_workdir_path))

# 输出路径
# Output Path
output_path = root_path / "outputs"
logger.info("[Settings Initiated] Output path: %s" % os.fspath(output_path))

# mmengine设置路径
# MMengine Deployments Path
deployment_path = root_path / "deployments"
logger.info("[Settings Initiated] MMengine deployments path: %s" % os.fspath(deployment_path))

# Dataset 文件夹路径
# Dataset Folder Path
datasets_folder_path = root_path / "data"
if not datasets_folder_path.is_dir():
    logger.critical("Dataset folder not found: %s" % os.fspath(datasets_folder_path))
    exit(-1)
logger.info("[Settings Initiated] Dataset folder path: %s" % os.fspath(datasets_folder_path))
datasets_folder_path_str = os.fspath(datasets_folder_path)

folder_arch = {
    "irstd1k": {"root_folder": "IRSTD1K", "img_folder": "IRSTD1k_Img", "mask_folder": "IRSTD1k_Label"},
    "nudtsirst": {"root_folder": "NUDT-SIRST", "img_folder": "images", "mask_folder": "masks"},
    "sirstv2tar": {"root_folder": "sirstv2", "img_folder": "images/targets", "mask_folder": "annotations/masks"},
    "sirstv2bg": {"root_folder": "sirstv2", "img_folder": "images/backgrounds", "mask_folder": "annotations/masks"},
}

for name in folder_arch:
    exec('%s_img_folder = datasets_folder_path / folder_arch[name]["root_folder"] / folder_arch[name]["img_folder"]' % name)
    exec('%s_mask_folder = datasets_folder_path / folder_arch[name]["root_folder"] / folder_arch[name]["mask_folder"]' % name)
    exec('%s_train_filter = datasets_folder_path / folder_arch[name]["root_folder"] / "%s_train.filter"' % (name, name))
    exec('%s_test_filter = datasets_folder_path / folder_arch[name]["root_folder"] / "%s_test.filter"' % (name, name))
