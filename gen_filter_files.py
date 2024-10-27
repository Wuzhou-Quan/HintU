import pathlib, gsettings
from rich.progress import track


def gen_sliced_filter_files(dataset_name: str, folder_arch: dict, ratio: float = 0.2):
    iteration = 1 / ratio - 1

    assert dataset_name in folder_arch.keys()
    root_path = pathlib.Path("data") / folder_arch[dataset_name]["root_folder"]
    img_folder = root_path / folder_arch[dataset_name]["img_folder"]
    mask_folder = root_path / folder_arch[dataset_name]["mask_folder"]

    all_list = []
    for item in track(list(img_folder.iterdir())):
        mask_path: pathlib.Path = mask_folder / item.name
        if mask_path.is_file():
            all_list.append(item.stem)

    all_list = sorted(all_list)
    tar_train_file = root_path / f"{dataset_name}_train.filter"
    tar_test_file = root_path / f"{dataset_name}_test.filter"

    counter = 0
    with open(tar_train_file, "w+", encoding="utf-8") as train_filter, open(tar_test_file, "w+", encoding="utf-8") as test_filter:
        for item in all_list:
            if counter < (iteration):
                train_filter.write(item + "\n")
                counter += 1
            else:
                test_filter.write(item + "\n")
                counter = 0


if __name__ == "__main__":
    for name in gsettings.folder_arch:
        gen_sliced_filter_files(name, gsettings.folder_arch, 0.5)
