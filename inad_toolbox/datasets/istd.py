from rich.progress import track
from collections import OrderedDict
from pathlib import Path
from ..utils import load_img_as_gray
import random, math, cv2, numpy as np, os, einops, torch


class InadIstdJointDataset_SingleFrame:
    def __init__(self, datasets_name: list, datasets_img_path: list, datasets_mask_path: list, filters_path: list = None, output_size_wh: tuple = (256, 256), use_augment=False):
        assert len(datasets_img_path) == len(datasets_mask_path) == len(datasets_name) and len(datasets_name)
        self.single_dataset = OrderedDict()
        for dataset_name, img_path, mask_path, filter_file_path in zip(datasets_name, datasets_img_path, datasets_mask_path, filters_path):
            self.single_dataset[dataset_name] = InadIstdDataset_SingleFrame(dataset_name, img_path, mask_path, filter_file_path, output_size_wh, use_augment)
        self.dataset_size = [len(d) for d in self.single_dataset.values()]
        self.dataset_begin_map = []
        for i in range(len(self.dataset_size)):
            index_begin = sum(self.dataset_size[:i])
            index_end = sum(self.dataset_size[: i + 1])
            self.dataset_begin_map.append([index_begin, index_end, datasets_name[i]])
        self.total_sample_num = sum(self.dataset_size)

    def __getitem__(self, ind):
        assert ind < self.total_sample_num
        for b, e, n in self.dataset_begin_map:
            if ind < e:
                name = n
                begin = b
                break
        return self.single_dataset[name][ind - begin]

    def __len__(self):
        return self.total_sample_num


class InadIstdDataset_SingleFrame:
    @staticmethod
    def random_affine(img, mask, degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-2, 2), borderValue=0):
        # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
        # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
        assert img.shape == mask.shape
        a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
        s = random.random() * (scale[1] - scale[0]) + scale[0]
        t1 = (random.random() * 2 - 1) * translate[0]
        t2 = (random.random() * 2 - 1) * translate[1]
        s1 = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)
        s2 = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)

        # Shear
        S = np.eye(3)
        S[0, 1] = s1
        S[1, 0] = s2
        height = img.shape[0]
        width = img.shape[1]

        # Rotation and Scale
        R = np.eye(3)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

        # Translation
        T = np.eye(3)
        T[0, 2] = t1 * img.shape[0]
        T[1, 2] = t2 * img.shape[1]

        M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!

        img = cv2.warpPerspective(img, M, dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=borderValue)  # BGR order borderValue
        mask = cv2.warpPerspective(mask, M, dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=borderValue)  # BGR order borderValue

        return img, mask

    @staticmethod
    def letterbox(img, height=608, width=1088, color=0):
        shape = img.shape[:2]  # shape = [height, width]
        ratio = min(float(height) / shape[0], float(width) / shape[1])
        new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
        dw = (width - new_shape[0]) / 2  # width padding
        dh = (height - new_shape[1]) / 2  # height padding
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
        return img

    @staticmethod
    def get_pairs(img_path: Path, mask_path: Path, use_augment=False, size_wh=[256, 256]):
        img = load_img_as_gray(img_path)
        mask = load_img_as_gray(mask_path)

        img = InadIstdDataset_SingleFrame.letterbox(img, height=size_wh[1], width=size_wh[0])
        mask = InadIstdDataset_SingleFrame.letterbox(mask, height=size_wh[1], width=size_wh[0])
        if use_augment:
            img, mask = InadIstdDataset_SingleFrame.random_affine(img, mask, degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.50, 1.20))
        img = torch.tensor(img)
        mask = torch.tensor(mask)
        img = einops.rearrange(img, "H W -> 1 H W")
        mask = einops.rearrange(mask, "H W -> 1 H W")
        return {"x": img, "gt": mask}

    def __init__(self, dataset_name: Path, dataset_img_path: Path, dataset_mask_path: Path, filter_file_path: Path = None, output_size=(256, 256), use_augment=False):
        assert dataset_img_path.is_dir(), f"Given dataset img folder path ({dataset_img_path}) is not a directory or does not exist."
        assert dataset_mask_path.is_dir(), f"Given dataset mask folder path ({dataset_mask_path}) is not a directory or does not exist."
        if filter_file_path is not None:
            assert filter_file_path.is_file(), f"Given filter file path ({filter_file_path}) is not a file or does not exist."
        self.width = output_size[0]
        self.height = output_size[1]
        self.use_augment = use_augment
        self.sample_num = 0
        self.name = dataset_name
        self.file_path_pairs = []

        filter = None
        if filter_file_path is not None:
            with open(os.fspath(filter_file_path), "r", encoding="utf-8") as f:
                filter = f.readlines()
            filter = [str(f).strip() for f in filter]

        for img_path in track(list(dataset_img_path.iterdir()), f"Loading dataset {dataset_name}..."):
            if filter is not None:
                if img_path.stem not in filter:
                    continue
            mask_path: Path = dataset_mask_path / img_path.name
            if mask_path.is_file():
                self.file_path_pairs.append((img_path, mask_path))

    def __getitem__(self, ind):
        path_pairs = self.file_path_pairs[ind]
        return self.get_pairs(path_pairs[0], path_pairs[1], self.use_augment, size_wh=[self.width, self.height])

    def __len__(self):
        return len(self.file_path_pairs)
