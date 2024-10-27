import cv2, os, numpy


def load_img_as_gray(img_path):
    img = cv2.imread(os.fspath(img_path))  # BGR
    if len(img.shape) != 2:
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    if img is None:
        raise ValueError("File corrupted {}".format(img_path))
    return img.astype(numpy.float32) / 255
