import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from scipy.stats.stats import pearsonr
import cv2
from scipy.signal import find_peaks, peak_widths
import math

from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)



############## add symmetric loss function ####################

@weighted_loss
def symmetric_loss_1(pred, target):
    pred_numpy = pred.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    target_numpy = target.data.squeeze().float().cpu().clamp_(0, 1).numpy() 

    loss_list = []
    for i in range(pred_numpy.shape[0]):
        pred_img = np.transpose(pred_numpy[i, :, :, :][[2, 1, 0], :, :], (1, 2, 0))
        pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2GRAY)
        target_img = np.transpose(target_numpy[i, :, :, :][[2, 1, 0], :, :], (1, 2, 0))
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        # pred_img = (pred_img * 255.0).astype(np.float32)
        # target_img = (target_img * 255.0).astype(np.float32)

        t = pred_img[4:128, :].flatten()
        t = t.astype('float32') 
        b = np.flipud(abs(pred_img[129:-3, :])).flatten()
        b = b.astype('float32')

        # find peaks in the spectra
        peaks1, _ = find_peaks(t, height=0.001)
        peaks2, _ = find_peaks(b, height=0.001)

        # create a binary mask for the peaks
        mask1 = np.isin(range(len(t)), peaks1)
        mask2 = np.isin(range(len(b)), peaks2)

        # apply a small constant to avoid dividing by zero
        epsilon = 1e-9
        weights1 = 1 / (t + epsilon)
        weights2 = 1 / (b + epsilon)

        # apply the mask to the weights (this sets the weight to 0 for non-peak regions)
        weights1 = weights1 * mask1
        weights2 = weights2 * mask2

        # normalize weights so they sum up to 1
        weights1 /= weights1.sum()
        weights2 /= weights2.sum()

        # calculate common weights (average of weights1 and weights2)
        weights_common = (weights1 + weights2) / 2
        # print(weights_common)

        # calculate weighted means
        mean1 = np.average(t, weights=weights_common)
        mean2 = np.average(b, weights=weights_common)

        # calculate weighted covariance
        numerator = np.sum(weights_common * (t - mean1) * (b - mean2))

        # calculate weighted standard deviations
        denom1 = np.sqrt(np.sum(weights_common * (t - mean1)**2))
        denom2 = np.sqrt(np.sum(weights_common * (b - mean2)**2))

        # calculate weighted Pearson correlation
        weighted_corr = numerator / (denom1 * denom2)

        # # top_half = np.log10(t + 1)
        # # but_half = np.log10(b + 1)

        # # assign weights correlation
        # epsilon = 1e-9
        # weights = 1 / (top_half + but_half + epsilon)

        # # normalize weights so they sum up to 1
        # weights /= weights.sum()
        # # print(weights)
        # # calculate weighted means
        # mean1 = np.average(top_half, weights=weights)
        # mean2 = np.average(but_half, weights=weights)

        # # calculate weighted covariance
        # numerator = np.sum(weights * (top_half - mean1) * (but_half - mean2))

        # # calculate weighted standard deviations
        # denom1 = np.sqrt(np.sum(weights * (top_half - mean1)**2))
        # denom2 = np.sqrt(np.sum(weights * (but_half - mean2)**2))

        # # calculate weighted Pearson correlation
        # weighted_corr = numerator / (denom1 * denom2)
        weighted_corr_loss = 1 - weighted_corr


        # # apply log transformation
        # top_half = np.log10(t + 1)
        # but_half = np.log10(b + 1)
        # corr = pearsonr(top_half, but_half)[0] ** 2
        # corr_loss = 1 - corr

        if weighted_corr:
            loss_list.append(weighted_corr_loss)

    loss_tensor = torch.tensor(loss_list, dtype=torch.float32, device='cuda:0', requires_grad=True)
    return loss_tensor 

@LOSS_REGISTRY.register()
class SYMLoss1(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(SYMLoss1, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
    
    def forward(self, pred, target):
        return self.loss_weight * symmetric_loss_1(pred, target)


@weighted_loss
def symmetric_loss_2(pred, target):
    # print("!!!!!!!!!!!!!!!! test symmetric !!!!!!!!!!!!!!: ") 
    # print(pred.shape, target.shape)

    pred_numpy = pred.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    target_numpy = target.data.squeeze().float().cpu().clamp_(0, 1).numpy() 

    loss_list = []
    for i in range(pred_numpy.shape[0]):
        pred_img = np.transpose(pred_numpy[i, :, :, :][[2, 1, 0], :, :], (1, 2, 0))
        pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2GRAY)
        target_img = np.transpose(target_numpy[i, :, :, :][[2, 1, 0], :, :], (1, 2, 0))
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        pred_img = (pred_img * 255.0).astype(np.float32)
        target_img = (target_img * 255.0).astype(np.float32)

        # hr_pjres_1d = np.amax(target_img[4:-3, :], axis=0)
        # peaks, _ = find_peaks(hr_pjres_1d, height=0)
        # widths, _, _, _ = peak_widths(hr_pjres_1d, peaks, rel_height=0.5)

        # corr_list = []
        # for peak, width in zip(peaks, widths):
        #     s_idx = max(math.ceil(peak)-math.ceil(width), 0)
        #     e_idx = math.ceil(peak)+math.ceil(width)+1
        #     temp_img = pred_img[:, s_idx:e_idx]
        #     # print("************** temp_img shape: ", temp_img.shape[1], peak, width)
        #     t = temp_img[4:128, :].flatten()
        #     t = t.astype('float32') 
        #     b = np.flipud(abs(temp_img[129:-3, :])).flatten()
        #     b = b.astype('float32') 

        #     top_half = np.log10(t + 1)
        #     but_half = np.log10(b + 1)
        #     corr = pearsonr(top_half, but_half)[0] ** 2
        #     corr_list.append(corr)


        corr_list = []
        for j in range(target_img.shape[1]):
            peaks, _ = find_peaks(target_img[4:-3, j])
            if peaks.size != 0:
                # print(i, peaks)
                temp_t = pred_img[4:128, j]
                temp_t = temp_t.astype('float32')
                temp_b = np.flipud(pred_img[129:-3, j])
                temp_b = temp_b.astype('float32')
                temp_corr = pearsonr(np.log10(temp_t + 1), np.log10(temp_b + 1))[0]**2
                corr_list.append(temp_corr)
        
        sym_loss = 1 - np.nanmean(corr_list)
        loss_list.append(sym_loss)

    loss_tensor = torch.tensor(loss_list, dtype=torch.float32, device='cuda:0', requires_grad=True)
    return loss_tensor

# corr_list = []
#         for j in range(pred_img.shape[1]):
#             temp_t = pred_img[4:128, j]
#             temp_t = temp_t.astype('float64')
#             temp_b = np.flipud(pred_img[129:-3, j])
#             temp_b = temp_b.astype('float64')
#             temp_corr = pearsonr(np.log10(temp_t + 1), np.log10(temp_b + 1))[0] ** 2
#             corr_list.append(temp_corr)

# t = pred_img[4:128, :].flatten()
#         t = t.astype('float64') 
#         b = np.flipud(abs(pred_img[129:-3, :])).flatten()
#         b = b.astype('float64') 

#         top_half = np.log10(t + 1)
#         but_half = np.log10(b + 1)
#         # print("********************** top half, but_half: ", top_half.shape, but_half.shape)
#         corr = pearsonr(top_half, but_half)[0] ** 2
#         corr_loss = 1 - corr
#         # print("****************************** corr: ", corr)
#         if corr:
#             total_corr.append(corr_loss)
     
#     mean_corr = np.nanmean(total_corr)
# pearsonr(pred[4:128, :].flatten(), np.flipud(abs(pred[129:-3, :])).flatten())[0] - 1 
# pearsonr(pred[4:128, :].flatten(), np.flipud(abs(pred[129:-3, :])).flatten())[0] - 1


@LOSS_REGISTRY.register()
class SYMLoss2(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(SYMLoss2, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
    
    def forward(self, pred, target):
        return self.loss_weight * symmetric_loss_2(pred, target)
    

@weighted_loss
def symmetric_loss(pred, target): 
    pred_numpy = pred.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    target_numpy = target.data.squeeze().float().cpu().clamp_(0, 1).numpy() 

    loss_list = []
    for i in range(pred_numpy.shape[0]):
        pred_img = np.transpose(pred_numpy[i, :, :, :][[2, 1, 0], :, :], (1, 2, 0))
        pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2GRAY)
        target_img = np.transpose(target_numpy[i, :, :, :][[2, 1, 0], :, :], (1, 2, 0))
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

    # # if the number of gpu is 1
    # pred_img = np.transpose(pred_numpy[[2, 1, 0], :, :], (1, 2, 0))
    # pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2GRAY)
    # target_img = np.transpose(target_numpy[[2, 1, 0], :, :], (1, 2, 0))
    # target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    # # print("!!!!!! target_img size: ", target_img.shape)

        t = pred_img[4:128, :].flatten()
        b = np.flipud(abs(pred_img[129:-3, :])).flatten()

        top_half = np.log10(t + 1)
        but_half = np.log10(b + 1)
        corr = pearsonr(top_half, but_half)[0]
        corr_loss = 1 - corr
        if corr:
            loss_list.append(corr_loss)
     
    loss_tensor = torch.tensor(loss_list, dtype=torch.float32, device='cuda:0', requires_grad=True)
    # sym_loss = np.nanmean(loss_list)
    return loss_tensor 

@LOSS_REGISTRY.register()
class SYMLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(SYMLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
    
    def forward(self, pred, target):
        return self.loss_weight * symmetric_loss(pred, target)




@weighted_loss
def pearson_loss(pred, target):

    pred_numpy = pred.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    target_numpy = target.data.squeeze().float().cpu().clamp_(0, 1).numpy() 

    loss_list = []
    for i in range(pred_numpy.shape[0]):
        pred_img = np.transpose(pred_numpy[i, :, :, :][[2, 1, 0], :, :], (1, 2, 0))
        pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2GRAY)
        target_img = np.transpose(target_numpy[i, :, :, :][[2, 1, 0], :, :], (1, 2, 0))
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        pred_img = (pred_img * 255.0).astype(np.float32)
        target_img = (target_img * 255.0).astype(np.float32)

        corr_list = []
        for k in range(target_img.shape[1]):
            peaks, _ = find_peaks(target_img[4:-3, k])
            if peaks.size != 0:
                temp_corr = pearsonr(np.log10(target_img[4:-3, k]+1), np.log10(pred_img[4:-3, k]+1))[0]**2
                corr_list.append(temp_corr)
        
        pear_loss = 1 - np.nanmean(corr_list)
        loss_list.append(pear_loss)

    loss_tensor = torch.tensor(loss_list, dtype=torch.float32, device='cuda:0', requires_grad=True)
    return loss_tensor


@LOSS_REGISTRY.register()
class PearsonLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(PearsonLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
    
    def forward(self, pred, target):
        return self.loss_weight * pearson_loss(pred, target)


# # symmetry score based on peak detection
# def peak_match_sliced_profile(temp_sliced_profile, h_threshold, p_threshold, center_idx):
#     height_threshold = h_threshold
#     prominence_threshold = p_threshold
#     # center_idx = 128

#     peaks_1, properties_1 = find_peaks(temp_sliced_profile, height=height_threshold, prominence=prominence_threshold)
#     peak_pairs = []
#     peaks_copy = peaks_1.copy()

#     # Compute full widths at half maximum
#     full_widths_at_half_max, _, _, _ = peak_widths(temp_sliced_profile, peaks_1, rel_height=0.5)

#     if center_idx in peaks_1:
#         temp_pair = (center_idx, center_idx, properties_1['peak_heights'][np.where(peaks_1 == center_idx)[0][0]],
#                            properties_1['peak_heights'][np.where(peaks_1 == center_idx)[0][0]],
#                            full_widths_at_half_max[np.where(peaks_1 == center_idx)[0][0]] / 2,
#                            full_widths_at_half_max[np.where(peaks_1 == center_idx)[0][0]] / 2)
#         peak_pairs.append(temp_pair)
#         peaks_copy = np.delete(peaks_copy, np.where(peaks_copy == center_idx))

#     peaks_c = peaks_copy.copy()
#     for peak in peaks_c:
#         symmetric_pos = 2 * center_idx - peak
#         if peaks_copy.size > 0:
#             # Find the peak that's closest to the symmetric position
#             closest_peak = peaks_copy[np.argmin(np.abs(peaks_copy - symmetric_pos))]
#             # print(closest_peak, symmetric_pos)
#             # (peak_1_pos, peak_2_pos, peak_1_height, peak_2_height, peak_1_hwhh, peak_2_hwhh)
#             temp_pair = (peak, closest_peak, properties_1['peak_heights'][np.where(peaks_1 == peak)[0][0]],
#                                properties_1['peak_heights'][np.where(peaks_1 == closest_peak)[0][0]],
#                                full_widths_at_half_max[np.where(peaks_1 == peak)[0][0]] / 2,
#                                full_widths_at_half_max[np.where(peaks_1 == closest_peak)[0][0]] / 2)
#             peak_pairs.append(temp_pair)
#             peaks_copy = peaks_copy[peaks_copy != peak]
#             peaks_copy = peaks_copy[peaks_copy != closest_peak]

#     # print(peak_pairs)
#     # Remove duplicates (pairs that were added twice)
#     peak_pairs = list(set(tuple(pair) for pair in peak_pairs))

#     # Second round of peak finding with updated threshold
#     for pair in peak_pairs:
#         # print(pair)
#         if pair[0] == pair[1] and pair[0] != center_idx:
#             peaks_2, properties_2 = find_peaks(temp_sliced_profile, height=height_threshold)
#             symmetric_pos = 2 * center_idx - pair[0]
#             if peaks_2.size > 0:
#                 # Find the peak that's closest to the symmetric position
#                 closest_peak = peaks_2[np.argmin(np.abs(peaks_2 - symmetric_pos))]
#                 # print(closest_peak, symmetric_pos)
#                 if np.abs(closest_peak - symmetric_pos) <= 1:
#                     full_widths_at_half_max, _, _, _ = peak_widths(temp_sliced_profile, peaks_2, rel_height=0.5)
#                     peak_pairs.remove(pair)
#                     temp_pair = (pair[0], closest_peak,
#                                        properties_2['peak_heights'][np.where(peaks_2 == pair[0])[0][0]],
#                                        properties_2['peak_heights'][np.where(peaks_2 == closest_peak)[0][0]],
#                                        full_widths_at_half_max[np.where(peaks_2 == pair[0])[0][0]] / 2,
#                                        full_widths_at_half_max[np.where(peaks_2 == closest_peak)[0][0]] / 2)
#                     peak_pairs.append(temp_pair)

#     # Remove duplicates again (pairs that were added twice)
#     # peak_pairs = list(set(tuple(sorted(pair)) for pair in peak_pairs))
#     return peak_pairs


# def get_symmetry_score_sliced_profile(peak_pairs, center_idx):
#     pos_sym_score_list = []
#     height_sym_score_list = []
#     shape_sym_score_list = []

#     # center_idx = 128
#     for pair in peak_pairs:
#         idx_1, idx_2 = pair[0], pair[1]
#         height_1, height_2 = pair[2], pair[3]
#         width_1, width_2 = pair[4], pair[5]

#         if idx_1 == idx_2 and idx_1 == center_idx:
#             position_symmetry_score = 0
#             height_symmetry_score = 0
#             shape_symmetry_score = 0
#         elif idx_1 == idx_2 and idx_1 != center_idx:
#             position_symmetry_score = 1
#             height_symmetry_score = 1
#             shape_symmetry_score = 1
#         else:
#             if abs(idx_1 - center_idx) != 0 or abs(idx_2 - center_idx) != 0:
#                 position_symmetry_score = abs(abs(idx_1 - center_idx) - abs(idx_2 - center_idx)) / \
#                                           max(abs(idx_1 - center_idx), abs(idx_2 - center_idx))
#             else:
#                 position_symmetry_score = 0

#             height_symmetry_score = abs(height_1 - height_2) / max(height_1, height_2)
#             shape_symmetry_score = abs(width_1 - width_2) / max(width_1, width_2)

#         pos_sym_score_list.append(position_symmetry_score)
#         height_sym_score_list.append(height_symmetry_score)
#         shape_sym_score_list.append(shape_symmetry_score)

#     pos_sym_mean = np.mean(pos_sym_score_list)
#     height_sym_mean = np.mean(height_sym_score_list)
#     shape_sym_mean = np.mean(shape_sym_score_list)
#     return pos_sym_mean, height_sym_mean, shape_sym_mean


# def measure_symmetry(im, center_idx):
#     pos_sym_list = []
#     height_sym_list = []
#     shape_sym_list = []

#     for i in range(im.shape[1]):
#         temp_sliced_profile = im[:, i]
#         t_peaks, _ = find_peaks(temp_sliced_profile, height=0.072731346, prominence=1)
#         if t_peaks.size > 0:
#             # print(i)
#             temp_peak_pairs = peak_match_sliced_profile(temp_sliced_profile, 0.072731346, 1, center_idx)
#             pos_sym, height_sym, shape_sym = get_symmetry_score_sliced_profile(temp_peak_pairs, center_idx)
#             pos_sym_list.append(pos_sym)
#             height_sym_list.append(height_sym)
#             shape_sym_list.append(shape_sym)
    
#     # print("************** symmetry peak list: ", pos_sym_list, height_sym_list, shape_sym_list)

#     return np.mean(pos_sym_list), np.mean(height_sym_list), np.mean(shape_sym_list)


# @weighted_loss
# def symmetric_loss_based_on_peaks(pred, target):
#     pred_numpy = pred.data.squeeze().float().cpu().clamp_(0, 1).numpy()
#     target_numpy = target.data.squeeze().float().cpu().clamp_(0, 1).numpy() 

#     # total_corr = []
#     final_pos_sym_list = []
#     final_height_sym_list = []
#     final_shape_sym_list = []
#     for i in range(pred_numpy.shape[0]):
#         pred_img = np.transpose(pred_numpy[i, :, :, :][[2, 1, 0], :, :], (1, 2, 0))
#         pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2GRAY)
#         target_img = np.transpose(target_numpy[i, :, :, :][[2, 1, 0], :, :], (1, 2, 0))
#         target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
#         pred_img = (pred_img * 255.0).astype(np.float32)
#         target_img = (target_img * 255.0).astype(np.float32)
#         print("************* symmetry peak shape: ", pred_img.shape, target_img.shape, np.max(pred_img), np.max(target_img))

#         pos_sym_loss, height_sym_loss, shape_sym_loss = measure_symmetry(pred_img, 128)
#         final_pos_sym_list.append(pos_sym_loss)
#         final_height_sym_list.append(height_sym_loss)
#         final_shape_sym_list.append(shape_sym_loss) 
    
#     # np.nanmean(final_pos_sym_list), np.nanmean(final_height_sym_list), 
#     return np.nanmean(final_shape_sym_list)


# @LOSS_REGISTRY.register()
# class SYMLoss_peak_detection(nn.Module):
#     def __init__(self, loss_weight=1.0, reduction="mean"):
#         super(SYMLoss_peak_detection, self).__init__()
#         if reduction not in ['none', 'mean', 'sum']:
#             raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

#         self.loss_weight = loss_weight
#         self.reduction = reduction 
    
#     def forward(self, pred, target):
#         # pos_sym_l, height_sym_l, 
#         shape_sym_l = symmetric_loss_based_on_peaks(pred, target)
#         print("***************** symmetry peak loss: ", shape_sym_l)
#         # pos_sym_l *= self.loss_weight
#         # height_sym_l *= self.loss_weight
#         shape_sym_l *= self.loss_weight

        # return shape_sym_l

#################################################################      


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero. Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)


@LOSS_REGISTRY.register()
class WeightedTVLoss(L1Loss):
    """Weighted TV loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        if reduction not in ['mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: mean | sum')
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight, reduction=reduction)

    def forward(self, pred, weight=None):
        if weight is None:
            y_weight = None
            x_weight = None
        else:
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]

        y_diff = super().forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        x_diff = super().forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)

        loss = x_diff + y_diff

        return loss


@LOSS_REGISTRY.register()
class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram
