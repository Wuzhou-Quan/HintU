import pathlib, gsettings, shutil, os
from rich.progress import track

mask_folder = gsettings.sirstv2tar_mask_folder

for f in track(list(mask_folder.iterdir())):
    if f.suffix != ".png":
        continue
    if "_pixels0" not in f.stem:
        continue
    new_path = pathlib.Path(os.fspath(f).replace("_pixels0", ""))
    if not new_path.exists():
        shutil.copy(f, new_path)
