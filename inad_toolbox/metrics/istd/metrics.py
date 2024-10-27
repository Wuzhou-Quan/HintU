import numpy as np, torch
from skimage import measure


class Metric:
    def __init__(self):
        self.total_frame = 0
        self.total_inter = np.array([])
        self.total_union = np.array([])

        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.PD = 0
        self.target = 0

    def get(self):
        """Gets the current evaluation result."""

        IoU = self.total_inter.sum() / self.total_union.sum()
        nIoU = (self.total_inter / self.total_union).mean()
        Fa = self.dismatch_pixel / self.all_pixel
        Pd = self.PD / self.target
        return [self.total_frame, IoU, nIoU, Fa, Pd]

    def update(self, pred, gt):
        if isinstance(pred, torch.Tensor):
            pred = pred.clone().cpu().detach().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.clone().cpu().detach().numpy()
        assert pred.shape == gt.shape

        mini = 1
        maxi = 1
        nbins = 1
        tp = pred == gt
        intersection = pred * tp

        batch_num = intersection.shape[0]
        inter_arr = []
        union_arr = []
        frame = 0
        for b in range(batch_num):

            lab, _ = np.histogram(gt[b], bins=nbins, range=(mini, maxi))
            if lab.sum() == 0:
                continue
            frame += 1
            inter, _ = np.histogram(intersection[b], bins=nbins, range=(mini, maxi))
            pred_, _ = np.histogram(pred[b], bins=nbins, range=(mini, maxi))
            union = pred_ + lab - inter
            union_arr.append(union)
            inter_arr.append(inter)

            size = gt.shape[2:4]
            coord_image = measure.regionprops(measure.label(pred[b, 0, ...], connectivity=2))
            coord_label = measure.regionprops(measure.label(gt[b, 0, ...], connectivity=2))
            self.target += len(coord_label)
            image_area_total = []
            image_area_match = []
            distance_match = []
            dismatch = []

            for K in range(len(coord_image)):
                area_image = np.array(coord_image[K].area)
                image_area_total.append(area_image)

            for i in range(len(coord_label)):
                centroid_label = np.array(list(coord_label[i].centroid))
                for m in range(len(coord_image)):
                    centroid_image = np.array(list(coord_image[m].centroid))
                    distance = np.linalg.norm(centroid_image - centroid_label)
                    area_image = np.array(coord_image[m].area)
                    if distance < 3:
                        distance_match.append(distance)
                        image_area_match.append(area_image)
                        del coord_image[m]
                        break

            dismatch = [x for x in image_area_total if x not in image_area_match]
            self.dismatch_pixel += np.sum(dismatch)
            self.all_pixel += size[0] * size[1]
            self.PD += len(distance_match)

        self.total_frame += frame
        self.total_inter = np.append(self.total_inter, np.array(inter_arr))
        self.total_union = np.append(self.total_union, np.array(union_arr))
