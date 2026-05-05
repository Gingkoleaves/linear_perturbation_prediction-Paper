"""
Oversampling utilities for handling class imbalance in CFY training.

Copyright (C) 2025 Contributors
Licensed under the Apache License, Version 2.0
"""

import torch
import numpy as np
from torch.utils.data import WeightedRandomSampler
import logging

logger = logging.getLogger(__name__)


def compute_class_balanced_weights(class_counts, beta=0.9999):
    """
    Compute class-balanced weights using effective number of samples.

    Reference: "Class-Balanced Loss Based on Effective Number of Samples" (Cui et al., 2019)

    Args:
        class_counts: dict or array of sample counts per class
        beta: balancing parameter (0.9999 for large datasets, 0.99 for small)

    Returns:
        dict: class_id -> weight
    """
    if isinstance(class_counts, dict):
        class_ids = sorted(class_counts.keys())
        counts = np.array([class_counts[cid] for cid in class_ids])
    else:
        class_ids = list(range(len(class_counts)))
        counts = np.array(class_counts)

    # Effective number of samples
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / effective_num

    # Normalize weights
    weights = weights / weights.sum() * len(weights)

    return {cid: float(w) for cid, w in zip(class_ids, weights)}


def create_weighted_sampler_for_dual_perturbations(
    traindata,
    oversampling_strategy='balanced',
    beta=0.9999,
    min_weight=0.1,
    max_weight=10.0,
    single_sample_weight=1.0,
):
    """
    Create a WeightedRandomSampler that oversamples minority classes in dual perturbations.

    Args:
        traindata: training dataset with co_effect_type_id column
        oversampling_strategy:
            - 'balanced': use class-balanced weights (Cui et al., 2019)
            - 'inverse': simple inverse frequency
            - 'sqrt_inverse': square root of inverse frequency
        beta: parameter for class-balanced weighting
        min_weight: minimum sample weight (prevents extreme downweighting)
        max_weight: maximum sample weight (prevents extreme upweighting)
        single_sample_weight: sampling weight assigned to single perturbations
            (label == -1). Set to 0.0 to sample only dual perturbations.

    Returns:
        WeightedRandomSampler or None if no co_effect_type_id column
    """
    if not hasattr(traindata, '_data') or traindata._data is None:
        logger.warning("traindata has no _data attribute, skipping oversampling")
        return None

    if 'co_effect_type_id' not in traindata._data.columns:
        logger.warning("co_effect_type_id not found in traindata, skipping oversampling")
        return None

    # Get class labels
    labels = traindata._data['co_effect_type_id'].to_numpy()

    logger.info(f"Total samples in traindata: {len(labels)}")
    logger.info(f"Label value counts: {np.unique(labels, return_counts=True)}")

    # Count samples per class (only for dual perturbations, label >= 0)
    dual_mask = labels >= 0
    dual_labels = labels[dual_mask]

    logger.info(f"Dual perturbation samples (label >= 0): {len(dual_labels)}")
    logger.info(f"Single perturbation samples (label == -1): {(labels == -1).sum()}")

    if len(dual_labels) == 0:
        logger.warning("No dual perturbation samples found, skipping oversampling")
        return None

    unique_classes, class_counts = np.unique(dual_labels, return_counts=True)
    class_count_dict = {int(c): int(cnt) for c, cnt in zip(unique_classes, class_counts)}

    logger.info("Class distribution in dual perturbations:")
    class_names = ['Additive', 'Synergy', 'Buffering', 'Opposite', 'Other']
    for cid, cnt in class_count_dict.items():
        if cid < len(class_names):
            logger.info(f"  {class_names[cid]:10s}: {cnt:7,} samples")

    # Compute class weights
    if oversampling_strategy == 'balanced':
        class_weights = compute_class_balanced_weights(class_count_dict, beta=beta)
    elif oversampling_strategy == 'inverse':
        total = sum(class_count_dict.values())
        class_weights = {cid: total / cnt for cid, cnt in class_count_dict.items()}
    elif oversampling_strategy == 'sqrt_inverse':
        total = sum(class_count_dict.values())
        class_weights = {cid: np.sqrt(total / cnt) for cid, cnt in class_count_dict.items()}
    else:
        raise ValueError(f"Unknown oversampling_strategy: {oversampling_strategy}")

    # Normalize weights
    weight_values = np.array(list(class_weights.values()))
    weight_mean = weight_values.mean()
    class_weights = {cid: w / weight_mean for cid, w in class_weights.items()}

    logger.info("Class weights after normalization:")
    for cid, w in class_weights.items():
        if cid < len(class_names):
            logger.info(f"  {class_names[cid]:10s}: {w:.3f}")

    # Assign sample weights
    sample_weights = np.ones(len(labels), dtype=np.float32)

    for cid, weight in class_weights.items():
        # Clamp weight to prevent extreme values
        clamped_weight = np.clip(weight, min_weight, max_weight)
        mask = labels == cid
        sample_weights[mask] = clamped_weight

    # Single perturbations can be downweighted or excluded entirely in plugin-only training.
    sample_weights[labels == -1] = float(single_sample_weight)
    logger.info(f"Sample weight stats: min={sample_weights.min():.3f}, "
                f"mean={sample_weights.mean():.3f}, max={sample_weights.max():.3f}")

    total_weight = float(sample_weights.sum())
    if total_weight > 0:
        logger.info("Expected sampling mass after weighting:")
        single_mass = float(sample_weights[labels == -1].sum()) / total_weight
        logger.info(f"  {'Single':10s}: {100.0 * single_mass:6.2f}%")
        for cid, cnt in class_count_dict.items():
            class_mass = float(sample_weights[labels == cid].sum()) / total_weight
            if cid < len(class_names):
                logger.info(
                    f"  {class_names[cid]:10s}: {100.0 * class_mass:6.2f}% "
                    f"(raw_count={cnt:,}, weight={class_weights[cid]:.3f})"
                )

    # Create sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True  # Allow oversampling
    )

    return sampler


def estimate_effective_batch_composition(
    traindata,
    sampler,
    batch_size=1000,
    num_batches=100
):
    """
    Estimate the effective class distribution after oversampling.

    Args:
        traindata: training dataset
        sampler: WeightedRandomSampler
        batch_size: batch size
        num_batches: number of batches to sample for estimation

    Returns:
        dict: estimated class distribution
    """
    if sampler is None:
        return {}

    labels = traindata._data['co_effect_type_id'].to_numpy()

    # Sample indices
    sampled_indices = []
    for _ in range(num_batches * batch_size):
        idx = next(iter(sampler))
        sampled_indices.append(idx)

    sampled_labels = labels[sampled_indices]
    dual_mask = sampled_labels >= 0
    dual_labels = sampled_labels[dual_mask]

    unique, counts = np.unique(dual_labels, return_counts=True)
    distribution = {int(c): int(cnt) for c, cnt in zip(unique, counts)}

    logger.info(f"Estimated class distribution after oversampling ({num_batches} batches):")
    class_names = ['Additive', 'Synergy', 'Buffering', 'Opposite', 'Other']
    for cid, cnt in distribution.items():
        if cid < len(class_names):
            pct = 100.0 * cnt / len(dual_labels)
            logger.info(f"  {class_names[cid]:10s}: {cnt:6,} ({pct:5.2f}%)")

    return distribution
