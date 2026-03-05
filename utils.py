'''
utility functions
'''
import numpy as np
import torch
import torch.nn as nn
import os
import random
import pdb
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# ====================================================
# Phase 1: Effective Number of Samples Class Weighting
# Based on Cui et al., "Class-Balanced Loss Based on
# Effective Number of Samples", CVPR 2019
# ====================================================
def compute_effective_weights(class_counts, beta=0.9999):
    """Compute class weights using effective number of samples.

    Args:
        class_counts: array-like of per-class sample counts
        beta: hyperparameter in [0, 1). Higher = more rebalancing.
    Returns:
        numpy array of normalized class weights
    """
    class_counts = np.array(class_counts, dtype=np.float64)
    class_counts = np.maximum(class_counts, 1.0)
    effective_num = 1.0 - np.power(beta, class_counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / np.sum(weights) * len(weights)
    return weights


# ====================================================
# Phase 1: Focal Loss
# Based on Lin et al., "Focal Loss for Dense Object
# Detection", ICCV 2017
# ====================================================
def focal_loss(pred, gold, trg_pad_idx, class_weights=None, gamma=2.0, label_smoothing=0.0):
    """Focal Loss: down-weights easy (well-classified) examples.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        pred: [N, C] logits
        gold: [N] class indices
        trg_pad_idx: padding index to ignore
        class_weights: optional per-class alpha weights
        gamma: focusing parameter (0 = standard CE, 2 = default)
        label_smoothing: label smoothing factor
    """
    non_pad_mask = gold.ne(trg_pad_idx)
    if non_pad_mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    pred_valid = pred[non_pad_mask]
    gold_valid = gold[non_pad_mask]

    # Standard CE per-sample (no reduction)
    ce_loss = F.cross_entropy(
        pred_valid, gold_valid, weight=class_weights,
        reduction='none', label_smoothing=label_smoothing
    )

    # p_t = probability of the correct class
    log_pt = -ce_loss
    pt = torch.exp(log_pt)

    # Focal modulating factor
    focal_weight = (1.0 - pt) ** gamma
    loss = focal_weight * ce_loss

    return loss.mean()


def normalize_offset(input, mask, max_frames):
    output = (input/max_frames)*mask
    output = torch.exp(output)*mask
    return output

def cal_performance(pred, gold, trg_pad_idx, loss_func="CE", class_weights = None, calc_loss=True, actionness=False,
                    focal_gamma=2.0, label_smoothing=0.0):
    # Modifid version of https://github.com/jadore801120/attention-is-all-you-need-pytorch
    '''Apply label smoothing if needed'''
    if calc_loss:   loss = cal_loss(pred, gold.long(), trg_pad_idx, loss_func=loss_func, class_weights=class_weights,
                                    focal_gamma=focal_gamma, label_smoothing=label_smoothing)
    else:           loss = 0
    C = pred.shape[1]
    pred = pred.max(1)[1]           # Technically this is just argmax, because the [1] of max is the indices
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()
    # Calculate class-wise TP, FP, FN, TN
    class_wise = {}
    for i in range(C):
        class_stats = {}
        class_mask = gold.eq(i)
        class_stats["TP"] = pred.eq(gold).masked_select(class_mask).sum().item()
        class_stats["FN"] = class_mask.sum().item() - class_stats["TP"]
        # To not consider padded values
        class_stats["FP"] = (gold.ne(i) & pred.eq(i)).masked_select(non_pad_mask).sum().item()
        class_stats["TN"] = n_word - class_stats["TP"] - class_stats["FP"] - class_stats["FN"]
        # To consider padded values
        class_wise[i + 1*actionness] = class_stats

    return loss, n_correct, n_word, class_wise

def cal_loss(pred, gold, trg_pad_idx, loss_func="CE", class_weights=None,
             focal_gamma=2.0, label_smoothing=0.0):
    # Modified version of https://github.com/jadore801120/attention-is-all-you-need-pytorch
    '''Calculate cross entropy, focal, or binary cross entropy loss'''

    # Weighting not implemented with smoothing
    if loss_func == "smoothing":
        eps = 0.1
        n_class = pred.size(1) + 1
        B = pred.size(0)

        one_hot = torch.zeros((B, n_class)).to(pred.device).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class -1)
        one_hot = one_hot[:, :-1]
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
        loss = loss / non_pad_mask.sum()
    elif loss_func == "CE":
        if class_weights is None:
            loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx,
                                   label_smoothing=label_smoothing)
        else:
            loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, weight=class_weights,
                                   label_smoothing=label_smoothing)
    elif loss_func == "focal":
        loss = focal_loss(pred, gold, trg_pad_idx, class_weights=class_weights,
                          gamma=focal_gamma, label_smoothing=label_smoothing)
    elif loss_func == "BCE":
        B, C = pred.shape
        non_pad_mask = gold.ne(trg_pad_idx)
        # Create one hot encoding of ground truth while skipping padding
        one_hot = torch.zeros_like(pred)
        for n, i in enumerate(gold):
            if i != trg_pad_idx:
                one_hot[n, i] = 1
        if class_weights is None:
            loss = F.binary_cross_entropy_with_logits(pred[non_pad_mask], one_hot[non_pad_mask])
        else:
            loss = F.binary_cross_entropy_with_logits(pred[non_pad_mask], one_hot[non_pad_mask], pos_weight=class_weights)
    else:
        raise NotImplementedError(f"Loss function '{loss_func}' not implemented")
    return loss

# Calculate prediction performance when using actionness and a specific detection threshold
def cal_actionness_performance(pred, gold, threshold=0.5):
    loss = F.binary_cross_entropy_with_logits(pred, gold)
    stats = {}
    stats["TP"] = ((pred > threshold) & (gold == 1)).sum().item()
    stats["FN"] = ((pred <= threshold) & (gold == 1)).sum().item()
    stats["FP"] = ((pred > threshold) & (gold == 0)).sum().item()
    stats["TN"] = ((pred <= threshold) & (gold == 0)).sum().item()
    return loss, stats

# CALF matching using offsets
def CALF_matching(output, target, output_off, target_off, pad_idx, use_actionness=False, output_actionness=None, target_actionness=None):
    # Expected input shape is (B, T, C) for output and (B, T) for target and both offsets
    # Work one batch at a time
    # Create a cost matrix containing the offset differences between predictions and GTs
    # Note which row corresponds to the GT with pad_idx and remove it and rows after it from cost matrix
    # Perform Hungarian algorithm (linear sum assignment) to find the best match for each GT
    # Shuffle predictions to match the order of GTs
    # Note: There is never more GT actions than queries due to how the dataset works (It cuts out actions from GT to match query length)
    B, T, C = output.size()
    matched_index = np.array([range(T)], dtype=int).repeat(B, axis=0)
    rearranged_output = torch.zeros_like(output)
    rearranged_output_off = torch.zeros_like(output_off)
    if use_actionness and output_actionness is not None and target_actionness is not None: 
        rearranged_actionness = torch.zeros_like(output_actionness)
    else:
        rearranged_actionness = None
    for b in range(B):
        cost_matrix = np.full((T, T), np.inf)
        last_index = T
        for t in range(T):
            if target[b, t] == pad_idx:
                last_index = t
                break
            for i in range(T):
                cost_matrix[t, i] = abs(output_off[b, i] - target_off[b, t])
        row_ind, col_ind = linear_sum_assignment(cost_matrix[:last_index])
        for i in row_ind:
            matched_index[b, i] = col_ind[i]
        rearranged_output[b] = output[b, matched_index[b]]
        rearranged_output_off[b] = output_off[b, matched_index[b]]
        if use_actionness and output_actionness is not None and target_actionness is not None:
            rearranged_actionness[b] = output_actionness[b, matched_index[b]]
    return rearranged_output, rearranged_output_off, rearranged_actionness

# CALF matching using probabilities
def CALF_matching2(output, target, output_off, target_off, pad_idx, use_actionness=False, output_actionness=None, target_actionness=None):
    # Expected input shape is (B, T, C) for output and (B, T) for target and both offsets
    # Multiply prediction scores by actionness
    # Work one batch at a time
    # Create a cost matrix containing the new prediction scores corresponding to the class of each ground truth
    # Note which row corresponds to the GT with pad_idx and remove it and rows after it from cost matrix
    # Perform Hungarian algorithm (linear sum assignment) to find the best match for each GT
    # Shuffle predictions to match the order of GTs
    # Note: There is never more GT actions than queries due to how the dataset works (It cuts out actions from GT to match query length)
    B, T, C = output.size()
    matched_index = np.array([range(T)], dtype=int).repeat(B, axis=0)
    rearranged_output = torch.zeros_like(output)
    rearranged_output_off = torch.zeros_like(output_off)
    temp_out = output.clone()
    if use_actionness and output_actionness is not None and target_actionness is not None: 
        rearranged_actionness = torch.zeros_like(output_actionness)
        for i in range(C):
            temp_out[..., i] *= output_actionness
    else:
        rearranged_actionness = None
    for b in range(B):
        cost_matrix = np.zeros((T, T))
        last_index = T
        for t in range(T):
            if target[b, t] == pad_idx:
                last_index = t+1
                break
            for i in range(T):
                cost_matrix[t, i] = -temp_out[b, i, int(target[b,t] - 1*use_actionness)]
        row_ind, col_ind = linear_sum_assignment(cost_matrix[:last_index])
        for i in row_ind:
            matched_index[b, i] = col_ind[i]
        rearranged_output[b] = output[b, matched_index[b]]
        rearranged_output_off[b] = output_off[b, matched_index[b]]
        if use_actionness and output_actionness is not None and target_actionness is not None:
            rearranged_actionness[b] = output_actionness[b, matched_index[b]]
    return rearranged_output, rearranged_output_off, rearranged_actionness

# Taken and modified from pytorch_lightning
max_seed_value = np.iinfo(np.uint32).max
min_seed_value = np.iinfo(np.uint32).min

def seed_everything(seed: int = None) -> int:
    """Function that sets seed for pseudo-random number generators in: pytorch, numpy, python.random.

    Args:
        seed: the integer value seed for global random state in Lightning.
            If `None`, will select it randomly.
    """
    if seed is None:
        seed = _select_seed_randomly(min_seed_value, max_seed_value)
    elif not isinstance(seed, int):
        seed = int(seed)
    if not (min_seed_value <= seed <= max_seed_value):
        print(f"Warning: {seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}")
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    # using `log.info` instead of `rank_zero_info`,
    # so users can verify the seed is properly set in distributed training.
    print(f"Global seed set to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed

def _select_seed_randomly(min_seed_value: int = min_seed_value, max_seed_value: int = max_seed_value) -> int:
    return random.randint(min_seed_value, max_seed_value)