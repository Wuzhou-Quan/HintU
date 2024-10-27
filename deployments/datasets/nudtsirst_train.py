from inad_toolbox.datasets import InadIstdJointDataset_SingleFrame
import gsettings

datasets_used = ["irstd1k", "nudtsirst", "sirstv2tar", "sirstv2bg"]

# !!!WARNING: Please modify the following content carefully!!!
dataset_name = []
datasets_img_path = []
datasets_mask_path = []
filters_path = []
for name in datasets_used:
    dataset_name.append(name)
    datasets_img_path.append(gsettings.__dict__[f"{name}_img_folder"])
    datasets_mask_path.append(gsettings.__dict__[f"{name}_mask_folder"])
    filters_path.append(gsettings.__dict__[f"{name}_train_filter"])
dataset = dict(
    type=InadIstdJointDataset_SingleFrame,
    datasets_name=dataset_name,
    datasets_img_path=datasets_img_path,
    datasets_mask_path=datasets_mask_path,
    filters_path=filters_path,
    use_augment=True,
)
