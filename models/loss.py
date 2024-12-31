import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

SUPPORTED_WEIGHTING = ["default", "GDL"]


class GeneralizedWassersteinDiceLoss(_Loss):
    """
    Generalized Wasserstein Dice Loss [1] in PyTorch.
    Optionally, one can use a weighting method for the
    class-specific sum of errors similar to the one used
    in the generalized Dice Loss [2].
    For this behaviour, please use weighting_mode='GDL'.
    The exact formula of the Wasserstein Dice loss in this case
    can be found in the Appendix of [3].
    References:
    ===========
    [1] "Generalised Wasserstein Dice Score for Imbalanced Multi-class
        Segmentation using Holistic Convolutional Networks."
        Fidon L. et al. MICCAI BrainLes (2017).
    [2] "Generalised dice overlap as a deep learning loss function
        for highly unbalanced segmentations."
        Sudre C., et al. MICCAI DLMIA (2017).
    [3] "Comparative study of deep learning methods for the automatic
        segmentation of lung, lesion and lesion type in CT scans of
        COVID-19 patients."
        Tilborghs, S. et al. arXiv preprint arXiv:2007.15546 (2020).
    """

    def __init__(
        self,
        dist_matrix=np.array([[0.0, 1.0], [1.0, 0.0]]),
        weighting_mode="GDL",
        reduction="mean",
    ):
        """
        :param dist_matrix: 2d tensor or 2d numpy array; matrix of distances
        between the classes.
        It must have dimension C x C where C is the number of classes.
        :param: weighting_mode: str; indicates how to weight the class-specific
        sum of errors.
        'default' corresponds to the GWDL used in the original paper [1],
        'GDL' corresponds to the GWDL used in [2].
        :param reduction: str; reduction mode.
        References:
        ===========
        [1] "Generalised Wasserstein Dice Score for Imbalanced Multi-class
            Segmentation using Holistic Convolutional Networks."
            Fidon L. et al. MICCAI BrainLes (2017).
        [2] "Comparative study of deep learning methods for the automatic
            segmentation of lung, lesion and lesion type in CT scans of
            COVID-19 patients."
            Tilborghs, S. et al. arXiv preprint arXiv:2007.15546 (2020).
        """
        super(GeneralizedWassersteinDiceLoss, self).__init__(reduction=reduction)
        self.M = dist_matrix
        if isinstance(self.M, np.ndarray):
            self.M = torch.from_numpy(self.M)
        if torch.cuda.is_available():
            self.M = self.M.cuda()
        if torch.max(self.M) != 1:
            print(
                "Normalize the maximum of the distance matrix "
                "used in the Generalized Wasserstein Dice Loss to 1."
            )
            self.M = self.M / torch.max(self.M)
        self.num_classes = self.M.size(0)
        self.alpha_mode = weighting_mode
        assert (
            weighting_mode in SUPPORTED_WEIGHTING
        ), "weighting_mode must be in %s" % str(SUPPORTED_WEIGHTING)
        self.reduction = reduction

    def forward(self, input, target):
        """
        Compute the Generalized Wasserstein Dice loss
        between input and target tensors.
        :param input: tensor. input is the scores maps (before softmax).
        The expected shape of input is (N, C, H, W, D) in 3d
        and (N, C, H, W) in 2d.
        :param target: target is the target segmentation.
        The expected shape of target is (N, H, W, D) or (N, 1, H, W, D) in 3d
        and (N, H, W) or (N, 1, H, W) in 2d.
        :return: scalar tensor. Loss function value.
        """
        input = input.cuda()
        target = target.cuda()
        epsilon = np.spacing(1)  # smallest number available
        # Convert the target segmentation to long if needed
        target = target.long()
        # Aggregate spatial dimensions
        flat_input = input.view(input.size(0), input.size(1), -1)  # b,c,s
        flat_target = target.view(target.size(0), -1)  # b,s
        # Apply the softmax to the input scores map
        probs = F.softmax(flat_input, dim=1)  # b,c,s
        # probs = torch.sigmoid(flat_input)
        # Compute the Wasserstein distance map
        wass_dist_map = self.wasserstein_distance_map(probs, flat_target)
        # Compute the generalised number of true positives
        alpha = self.compute_alpha_generalized_true_positives(flat_target)

        # Compute the Generalized Wasserstein Dice loss
        if self.alpha_mode == "GDL":
            # use GDL-style alpha weights (i.e. normalize by the volume of each class)
            # contrary to [1] we also use alpha in the "generalized all error".
            true_pos = self.compute_generalized_true_positive(
                alpha, flat_target, wass_dist_map
            )
            denom = self.compute_denominator(alpha, flat_target, wass_dist_map)
        else:  # default: as in [1]
            # (i.e. alpha=1 for all foreground classes and 0 for the background).
            # Compute the generalised number of true positives
            true_pos = self.compute_generalized_true_positive(
                alpha, flat_target, wass_dist_map
            )
            all_error = torch.sum(wass_dist_map, dim=1)
            denom = 2 * true_pos + all_error
        wass_dice = (2.0 * true_pos + epsilon) / (denom + epsilon)
        wass_dice_loss = 1.0 - wass_dice

        if self.reduction == "sum":
            return wass_dice_loss.sum()
        elif self.reduction == "none":
            return wass_dice_loss
        else:  # default is mean reduction
            return wass_dice_loss.mean()

    def wasserstein_distance_map(self, flat_proba, flat_target):
        """
        Compute the voxel-wise Wasserstein distance (eq. 6 in [1]) for
        the flattened prediction and the flattened labels (ground_truth)
        with respect to the distance matrix on the label space M.
        References:
        ===========
        [1] "Generalised Wasserstein Dice Score for Imbalanced Multi-class
        Segmentation using Holistic Convolutional Networks",
        Fidon L. et al. MICCAI BrainLes 2017
        """
        # Turn the distance matrix to a map of identical matrix
        M_extended = torch.unsqueeze(self.M, dim=0)  # C,C -> 1,C,C
        M_extended = torch.unsqueeze(M_extended, dim=3)  # 1,C,C -> 1,C,C,1
        M_extended = M_extended.expand(
            (
                flat_proba.size(0),
                M_extended.size(1),
                M_extended.size(2),
                flat_proba.size(2),
            )
        )
        # Expand the feature dimensions of the target
        flat_target_extended = torch.unsqueeze(flat_target, dim=1)  # b,s -> b,1,s
        flat_target_extended = flat_target_extended.expand(
            (flat_target.size(0), M_extended.size(1), flat_target.size(1))
        )  # b,1,s -> b,C,s
        flat_target_extended = torch.unsqueeze(
            flat_target_extended, dim=1
        )  # b,C,s -> b,1,C,s
        # Extract the vector of class distances for the ground-truth label at each voxel
        M_extended = torch.gather(
            M_extended, dim=1, index=flat_target_extended
        )  # b,C,C,s -> b,1,C,s
        M_extended = torch.squeeze(M_extended, dim=1)  # b,1,C,s -> b,C,s
        # Compute the wasserstein distance map
        wasserstein_map = M_extended * flat_proba
        # Sum over the classes
        wasserstein_map = torch.sum(wasserstein_map, dim=1)  # b,C,s -> b,s
        return wasserstein_map

    def compute_generalized_true_positive(
        self, alpha, flat_target, wasserstein_distance_map
    ):
        # Extend alpha to a map and select value at each voxel according to flat_target
        alpha_extended = torch.unsqueeze(alpha, dim=2)  # b,C -> b,C,1
        alpha_extended = alpha_extended.expand(
            (flat_target.size(0), self.num_classes, flat_target.size(1))
        )  # b,C,1 -> b,C,s
        flat_target_extended = torch.unsqueeze(flat_target, dim=1)  # b,s -> b,1,s
        alpha_extended = torch.gather(
            alpha_extended, index=flat_target_extended, dim=1
        )  # b,C,s -> b,1,s

        # Compute the generalized true positive as in eq. 9 of [1]
        generalized_true_pos = torch.sum(
            alpha_extended * (1.0 - wasserstein_distance_map),
            dim=[1, 2],
        )
        return generalized_true_pos

    def compute_denominator(self, alpha, flat_target, wasserstein_distance_map):
        # Extend alpha to a map and select value at each voxel according to flat_target
        alpha_extended = torch.unsqueeze(alpha, dim=2)  # b,C -> b,C,1
        alpha_extended = alpha_extended.expand(
            (flat_target.size(0), self.num_classes, flat_target.size(1))
        )  # b,C,1 -> b,C,s
        flat_target_extended = torch.unsqueeze(flat_target, dim=1)  # b,s -> b,1,s
        alpha_extended = torch.gather(
            alpha_extended, index=flat_target_extended, dim=1
        )  # b,C,s -> b,1,s
        # Compute the generalized true positive as in eq. 9
        generalized_true_pos = torch.sum(
            alpha_extended * (2.0 - wasserstein_distance_map),
            dim=[1, 2],
        )
        return generalized_true_pos

    def compute_alpha_generalized_true_positives(self, flat_target):
        """
        Compute the weights \alpha_l of eq. 9 in [1].
        References:
        ===========
        [1] "Generalised Wasserstein Dice Score for Imbalanced Multi-class
        Segmentation using Holistic Convolutional Networks",
        Fidon L. et al. MICCAI BrainLes 2017.
        """
        if self.alpha_mode == "GDL":  # GDL style
            # Define alpha like in the generalized dice loss
            # i.e. the inverse of the volume of each class.
            # Convert target to one-hot class encoding.
            one_hot = (
                F.one_hot(flat_target, num_classes=self.num_classes)
                .permute(0, 2, 1)
                .float()
            )  # shape: b,c,s
            volumes = torch.sum(one_hot, dim=2)  # b,c
            alpha = 1.0 / (volumes + 1.0)
        else:  # default, i.e. as in [1]
            # alpha weights are 0 for the background and 1 otherwise
            alpha_np = np.ones((flat_target.size(0), self.num_classes))  # b,c
            alpha_np[:, 0] = 0.0
            alpha = torch.from_numpy(alpha_np).float()
            if torch.cuda.is_available():
                alpha = alpha.cuda()
        return alpha


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.

    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result


class BinaryDiceLoss(torch.nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1, p=2, reduction="mean"):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert (
            predict.shape[0] == target.shape[0]
        ), "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise Exception("Unexpected reduction {}".format(self.reduction))


class SoftLoULoss(torch.nn.Module):
    def __init__(self, batch=32):
        super(SoftLoULoss, self).__init__()
        self.batch = batch
        self.bce_loss = torch.nn.BCELoss()

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        smooth = 0.00

        intersection = pred * target

        intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
        pred_sum = torch.sum(pred, dim=(1, 2, 3))
        target_sum = torch.sum(target, dim=(1, 2, 3))
        loss = (intersection_sum + smooth) / (
            pred_sum + target_sum - intersection_sum + smooth
        )

        loss = 1 - torch.mean(loss)
        loss1 = self.bce_loss(pred, target)
        return loss + loss1


class SoftLoULoss1(torch.nn.Module):
    def __init__(self, batch=32):
        super(SoftLoULoss1, self).__init__()
        self.batch = batch
        self.bce_loss = torch.nn.BCELoss()

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        smooth = 0.00

        intersection = pred * target

        intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
        pred_sum = torch.sum(pred, dim=(1, 2, 3))
        target_sum = torch.sum(target, dim=(1, 2, 3))
        loss = (intersection_sum + smooth) / (
            pred_sum + target_sum - intersection_sum + smooth
        )

        loss = 1 - torch.mean(loss)
        loss1 = self.bce_loss(pred, target)
        return loss
