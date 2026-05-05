"""
Training Comparison Script: Original LPM_CFY vs CFY Plugin Enhanced LPM

This script follows the tutorial workflow to compare training and prediction
performance between the original LPM_CFY model and a CFY plugin enhanced LPM model.

Copyright (C) 2025 Contributors
Licensed under the Apache License, Version 2.0
"""

import os
import sys
import time
import logging
import math
import re
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import polars as pl
import numpy as np
import pandas as pd

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import perturb_lib as plib
from Pdata_Description import Check_Perturbation

# Import our CFY plugin and models
from perturb_lib.cfy_plugin import LPMWithCFY
from perturb_lib.models.collection.lpm import LPM
from perturb_lib.models.collection.lpm_classify import LPM_CFY

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEFAULT_CFY_HEAD_TRAINING_PARAMS: Dict[str, Any] = {
    # 增加整体分类损失权重，让模型更关注分类准确性
    "cfy_cls_loss_weight": 0.40,  # 从 0.35 提高到 0.40

    # 增加 focal loss gamma，更关注难分类样本
    "cfy_cls_focal_gamma": 2.5,  # 从 1.0 提高到 2.5

    # 分类损失权重 - 大幅提高 Synergy 和 Buffering
    "cfy_cls_weight_additive": 1.0,
    "cfy_cls_weight_synergy": 3.0,      # 从 1.5 提高到 3.0 ⭐
    "cfy_cls_weight_buffering": 2.5,    # 从 1.0 提高到 2.5 ⭐
    "cfy_cls_weight_opposite": 2.0,
    "cfy_cls_weight_other": 1.5,        # 新增 Other 类权重

    "cfy_cls_fast_boost": 1.2,
    "cfy_cls_fast_epochs": 1,
    "cfy_cls_misclassified_boost": 1.2,

    # 回归损失权重 - 大幅提高 Synergy 和 Buffering
    "cfy_reg_weight_additive": 1.6,
    "cfy_reg_weight_synergy": 3.5,      # 从 2.2 提高到 3.5 ⭐
    "cfy_reg_weight_buffering": 2.8,    # 从 1.3 提高到 2.8 ⭐
    "cfy_reg_weight_opposite": 3.0,
    "cfy_reg_weight_other": 1.5,        # 新增 Other 类权重

    "cfy_dual_reg_loss_weight": 0.35,
    "cfy_additive_supervision_weight": 0.30,
    "cfy_additive_anchor_weight": 0.50,
    "cfy_non_additive_gate_power": 2.0,
    "cfy_plugin_hidden_dim": 128,
    "cfy_plugin_residual_hidden_dim": 48,
    "cfy_entropy_reg_weight": 0.005,
    "cfy_aux_warmup_epochs": 2,
    "cfy_relation_loss_weight": 0.01,
    "cfy_semantic_consistency_weight": 0.08,
    "cfy_semantic_margin": 0.06,
    "cfy_regression_separation_weight": 0.18,
    "cfy_regression_separation_margin": 0.03,
    "cfy_distribution_nll_weight": 0.004,
    "cfy_distribution_mmd_weight": 0.002,
    "cfy_expert_supervision_weight": 0.30,
    "cfy_expert_supervision_warmup_epochs": 2,
    "cfy_head_ramp_start_epoch": 3,
    "cfy_head_ramp_end_epoch": 8,
    "cfy_program_mix_target": 0.03,
    "cfy_disentangle_mix_target": 0.08,
    "cfy_distribution_mix_target": 0.02,
    "cfy_structured_mix_target": 0.05,
    "cfy_opposite_margin_weight": 0.07,
    "cfy_opposite_margin": 0.03,
    "cfy_router_temperature": 1.0,
    "cfy_router_temperature_min": 0.35,
    "cfy_router_temperature_warmup_epochs": 3,
    "cfy_monitor_semantics": True,
    "cfy_monitor_interval": 50,
}

# Override the legacy wide tuning surface with a compact user-facing parameter set.
DEFAULT_CFY_HEAD_TRAINING_PARAMS = {
    "cfy_cls_loss_weight": 0.40,
    "cfy_additive_anchor_weight": 0.50,
}

INTERNAL_CFY_PLUGIN_DEFAULTS: Dict[str, Any] = {
    "cfy_non_additive_gate_power": 2.0,
    "cfy_plugin_hidden_dim": 128,
    "cfy_plugin_residual_hidden_dim": 48,
    "cfy_router_temperature": 1.0,
    "cfy_router_temperature_min": 0.35,
    "cfy_router_temperature_warmup_epochs": 3,
    "cfy_monitor_semantics": True,
    "cfy_monitor_interval": 50,
}


class CFYEnhancedLPM(LPMWithCFY, LPM):
    """
    CFY Plugin Enhanced LPM for training comparison.
    Simplified version that avoids complex multiple inheritance issues.
    """

    def __init__(self, *args, **kwargs):
        """Initialize CFY Enhanced LPM."""
        # Extract CFY-specific parameters
        cfy_enabled = kwargs.pop('cfy_enabled', True)
        cfy_train_plugin_only = kwargs.pop('cfy_train_plugin_only', False)
        num_classes = kwargs.pop('num_classes', 5)  # Pop from kwargs to avoid passing to LPM
        freeze_backbone_for_dual = kwargs.pop('freeze_backbone_for_dual', True)
        cfy_cls_loss_weight = kwargs.pop('cfy_cls_loss_weight', 0.2)
        cfy_cls_focal_gamma = kwargs.pop('cfy_cls_focal_gamma', 1.5)
        cfy_cls_weight_additive = kwargs.pop('cfy_cls_weight_additive', 1.0)
        cfy_cls_weight_synergy = kwargs.pop('cfy_cls_weight_synergy', 1.6)
        cfy_cls_weight_buffering = kwargs.pop('cfy_cls_weight_buffering', 1.0)
        cfy_cls_weight_opposite = kwargs.pop('cfy_cls_weight_opposite', 2.2)
        cfy_cls_weight_other = kwargs.pop('cfy_cls_weight_other', 1.5)
        cfy_cls_fast_boost = kwargs.pop('cfy_cls_fast_boost', 2.0)
        cfy_cls_fast_epochs = kwargs.pop('cfy_cls_fast_epochs', 4)
        cfy_cls_misclassified_boost = kwargs.pop('cfy_cls_misclassified_boost', 2.0)
        cfy_reg_weight_additive = kwargs.pop('cfy_reg_weight_additive', 1.0)
        cfy_reg_weight_synergy = kwargs.pop('cfy_reg_weight_synergy', 1.8)
        cfy_reg_weight_buffering = kwargs.pop('cfy_reg_weight_buffering', 1.0)
        cfy_reg_weight_opposite = kwargs.pop('cfy_reg_weight_opposite', 2.4)
        cfy_reg_weight_other = kwargs.pop('cfy_reg_weight_other', 1.5)
        cfy_dual_reg_loss_weight = kwargs.pop('cfy_dual_reg_loss_weight', 0.35)
        cfy_additive_supervision_weight = kwargs.pop('cfy_additive_supervision_weight', 0.25)
        cfy_additive_anchor_weight = kwargs.pop('cfy_additive_anchor_weight', 0.0)
        cfy_non_additive_gate_power = kwargs.pop(
            'cfy_non_additive_gate_power',
            INTERNAL_CFY_PLUGIN_DEFAULTS["cfy_non_additive_gate_power"],
        )
        cfy_plugin_hidden_dim = kwargs.pop(
            'cfy_plugin_hidden_dim',
            INTERNAL_CFY_PLUGIN_DEFAULTS["cfy_plugin_hidden_dim"],
        )
        cfy_plugin_residual_hidden_dim = kwargs.pop(
            'cfy_plugin_residual_hidden_dim',
            INTERNAL_CFY_PLUGIN_DEFAULTS["cfy_plugin_residual_hidden_dim"],
        )
        cfy_entropy_reg_weight = kwargs.pop('cfy_entropy_reg_weight', 0.01)
        cfy_aux_warmup_epochs = kwargs.pop('cfy_aux_warmup_epochs', 3)
        cfy_relation_loss_weight = kwargs.pop('cfy_relation_loss_weight', 0.08)
        cfy_semantic_consistency_weight = kwargs.pop('cfy_semantic_consistency_weight', 0.06)
        cfy_semantic_margin = kwargs.pop('cfy_semantic_margin', 0.05)
        cfy_regression_separation_weight = kwargs.pop('cfy_regression_separation_weight', 0.15)
        cfy_regression_separation_margin = kwargs.pop('cfy_regression_separation_margin', 0.02)
        cfy_distribution_nll_weight = kwargs.pop('cfy_distribution_nll_weight', 0.03)
        cfy_distribution_mmd_weight = kwargs.pop('cfy_distribution_mmd_weight', 0.02)
        cfy_expert_supervision_weight = kwargs.pop('cfy_expert_supervision_weight', 0.20)
        cfy_expert_supervision_warmup_epochs = kwargs.pop('cfy_expert_supervision_warmup_epochs', 2)
        cfy_head_ramp_start_epoch = kwargs.pop('cfy_head_ramp_start_epoch', 3)
        cfy_head_ramp_end_epoch = kwargs.pop('cfy_head_ramp_end_epoch', 8)
        cfy_program_mix_target = kwargs.pop('cfy_program_mix_target', 0.15)
        cfy_disentangle_mix_target = kwargs.pop('cfy_disentangle_mix_target', 0.20)
        cfy_distribution_mix_target = kwargs.pop('cfy_distribution_mix_target', 0.10)
        cfy_structured_mix_target = kwargs.pop('cfy_structured_mix_target', 0.18)
        cfy_opposite_margin_weight = kwargs.pop('cfy_opposite_margin_weight', 0.06)
        cfy_opposite_margin = kwargs.pop('cfy_opposite_margin', 0.03)
        cfy_router_temperature = kwargs.pop(
            'cfy_router_temperature',
            INTERNAL_CFY_PLUGIN_DEFAULTS["cfy_router_temperature"],
        )
        cfy_router_temperature_min = kwargs.pop(
            'cfy_router_temperature_min',
            INTERNAL_CFY_PLUGIN_DEFAULTS["cfy_router_temperature_min"],
        )
        cfy_router_temperature_warmup_epochs = kwargs.pop(
            'cfy_router_temperature_warmup_epochs',
            INTERNAL_CFY_PLUGIN_DEFAULTS["cfy_router_temperature_warmup_epochs"],
        )
        cfy_monitor_semantics = kwargs.pop(
            'cfy_monitor_semantics',
            INTERNAL_CFY_PLUGIN_DEFAULTS["cfy_monitor_semantics"],
        )
        cfy_monitor_interval = kwargs.pop(
            'cfy_monitor_interval',
            INTERNAL_CFY_PLUGIN_DEFAULTS["cfy_monitor_interval"],
        )

        # Oversampling parameters
        cfy_enable_oversampling = kwargs.pop('cfy_enable_oversampling', False)
        cfy_oversampling_strategy = kwargs.pop('cfy_oversampling_strategy', 'balanced')
        cfy_oversampling_beta = kwargs.pop('cfy_oversampling_beta', 0.9999)
        default_single_sample_weight = 0.0 if cfy_train_plugin_only else 1.0
        cfy_oversampling_single_weight = kwargs.pop(
            'cfy_oversampling_single_weight',
            default_single_sample_weight,
        )

        # Initialize LPM first
        LPM.__init__(self, *args, **kwargs)

        if cfy_train_plugin_only:
            original_aux_warmup = int(cfy_aux_warmup_epochs)
            original_expert_warmup = int(cfy_expert_supervision_warmup_epochs)
            cfy_aux_warmup_epochs = 0
            cfy_expert_supervision_warmup_epochs = min(original_expert_warmup, 1)
            if (
                original_aux_warmup != cfy_aux_warmup_epochs
                or original_expert_warmup != cfy_expert_supervision_warmup_epochs
            ):
                logger.info(
                    "Plugin-only mode: shorten aux warmups | cls/reg aux %s->%s | expert %s->%s",
                    original_aux_warmup,
                    cfy_aux_warmup_epochs,
                    original_expert_warmup,
                    cfy_expert_supervision_warmup_epochs,
                )

            cfy_cls_focal_gamma = 0.0
            cfy_cls_fast_boost = 1.0
            cfy_cls_fast_epochs = 0
            cfy_cls_misclassified_boost = 1.0
            logger.info("Plugin-only mode: disable focal/misclassification boosting to reduce tuning reliance.")

            disabled_aux_losses = []
            for loss_name, loss_value in (
                ("entropy", cfy_entropy_reg_weight),
                ("relation", cfy_relation_loss_weight),
                ("semantic_consistency", cfy_semantic_consistency_weight),
                ("regression_separation", cfy_regression_separation_weight),
                ("distribution_nll", cfy_distribution_nll_weight),
                ("distribution_mmd", cfy_distribution_mmd_weight),
                ("expert_supervision", cfy_expert_supervision_weight),
                ("additive_supervision", cfy_additive_supervision_weight),
                ("opposite_margin", cfy_opposite_margin_weight),
            ):
                if float(loss_value) > 0:
                    disabled_aux_losses.append(loss_name)

            cfy_entropy_reg_weight = 0.0
            cfy_relation_loss_weight = 0.0
            cfy_semantic_consistency_weight = 0.0
            cfy_regression_separation_weight = 0.0
            cfy_distribution_nll_weight = 0.0
            cfy_distribution_mmd_weight = 0.0
            cfy_expert_supervision_weight = 0.0
            cfy_additive_supervision_weight = 0.0
            cfy_opposite_margin_weight = 0.0

            if disabled_aux_losses:
                logger.info(
                    "Plugin-only mode: disable high-variance structural aux losses: %s",
                    ", ".join(disabled_aux_losses),
                )

            if cfy_enable_oversampling:
                logger.info(
                    "Plugin-only mode: oversampling is active, neutralize class weights to avoid double rebalancing."
                )
                cfy_cls_weight_additive = 1.0
                cfy_cls_weight_synergy = 1.0
                cfy_cls_weight_buffering = 1.0
                cfy_cls_weight_opposite = 1.0
                cfy_cls_weight_other = 1.0
                cfy_reg_weight_additive = 1.0
                cfy_reg_weight_synergy = 1.0
                cfy_reg_weight_buffering = 1.0
                cfy_reg_weight_opposite = 1.0
                cfy_reg_weight_other = 1.0

        # Multi-objective training knobs to improve minority interaction types.
        self.cfy_cls_loss_weight = float(cfy_cls_loss_weight)
        self.cfy_cls_focal_gamma = float(cfy_cls_focal_gamma)
        self.cfy_cls_fast_boost = float(cfy_cls_fast_boost)
        self.cfy_cls_fast_epochs = max(int(cfy_cls_fast_epochs), 0)
        self.cfy_cls_misclassified_boost = max(float(cfy_cls_misclassified_boost), 1.0)
        self.cfy_regression_class_weights = {
            0: float(cfy_reg_weight_additive),
            1: float(cfy_reg_weight_synergy),
            2: float(cfy_reg_weight_buffering),
            3: float(cfy_reg_weight_opposite),
            4: float(cfy_reg_weight_other),
        }
        self.cfy_dual_reg_loss_weight = float(cfy_dual_reg_loss_weight)
        self.cfy_additive_supervision_weight = float(cfy_additive_supervision_weight)
        self.cfy_additive_anchor_weight = float(cfy_additive_anchor_weight)
        self.cfy_non_additive_gate_power = float(cfy_non_additive_gate_power)
        self.cfy_plugin_hidden_dim = int(cfy_plugin_hidden_dim)
        self.cfy_plugin_residual_hidden_dim = int(cfy_plugin_residual_hidden_dim)
        self.cfy_entropy_reg_weight = float(cfy_entropy_reg_weight)
        self.cfy_aux_warmup_epochs = int(cfy_aux_warmup_epochs)
        self.cfy_relation_loss_weight = float(cfy_relation_loss_weight)
        self.cfy_semantic_consistency_weight = float(cfy_semantic_consistency_weight)
        self.cfy_semantic_margin = float(cfy_semantic_margin)
        self.cfy_regression_separation_weight = float(cfy_regression_separation_weight)
        self.cfy_regression_separation_margin = float(cfy_regression_separation_margin)
        self.cfy_distribution_nll_weight = float(cfy_distribution_nll_weight)
        self.cfy_distribution_mmd_weight = float(cfy_distribution_mmd_weight)
        self.cfy_expert_supervision_weight = float(cfy_expert_supervision_weight)
        self.cfy_expert_supervision_warmup_epochs = int(cfy_expert_supervision_warmup_epochs)
        self.cfy_head_ramp_start_epoch = int(cfy_head_ramp_start_epoch)
        self.cfy_head_ramp_end_epoch = int(cfy_head_ramp_end_epoch)
        self.cfy_program_mix_target = float(cfy_program_mix_target)
        self.cfy_disentangle_mix_target = float(cfy_disentangle_mix_target)
        self.cfy_distribution_mix_target = float(cfy_distribution_mix_target)
        self.cfy_structured_mix_target = float(cfy_structured_mix_target)
        self.cfy_opposite_margin_weight = float(cfy_opposite_margin_weight)
        self.cfy_opposite_margin = float(cfy_opposite_margin)
        self.cfy_router_temperature = float(cfy_router_temperature)
        self.cfy_router_temperature_min = float(cfy_router_temperature_min)
        self.cfy_router_temperature_warmup_epochs = int(cfy_router_temperature_warmup_epochs)
        self.cfy_monitor_semantics = bool(cfy_monitor_semantics)
        self.cfy_monitor_interval = max(int(cfy_monitor_interval), 1)
        self.cfy_classification_class_weights = {
            0: float(cfy_cls_weight_additive),
            1: float(cfy_cls_weight_synergy),
            2: float(cfy_cls_weight_buffering),
            3: float(cfy_cls_weight_opposite),
            4: float(cfy_cls_weight_other),
        }
        self.cfy_train_plugin_only = bool(cfy_train_plugin_only)
        self.cfy_enable_oversampling = bool(cfy_enable_oversampling)
        self.cfy_oversampling_strategy = str(cfy_oversampling_strategy)
        self.cfy_oversampling_beta = float(cfy_oversampling_beta)
        self.cfy_oversampling_single_weight = float(cfy_oversampling_single_weight)

        # Store reference to original LPM forward method BEFORE LPMWithCFY initialization
        # This is critical for routing single perturbations through the original path
        self._original_forward_method = LPM.forward
        # !!! LPM实际上没有提供forward，而是包装pytorch——light的trainner.fit到LPM的fit函数中

        if cfy_enabled:
            # Initialize CFY capabilities
            cfy_config = {
                'num_classes': num_classes,
                'freeze_backbone_for_dual': freeze_backbone_for_dual,
                'dropout': kwargs.get('dropout', 0.0),
                'hidden_dim': kwargs.get('hidden_dim', 256),
                'cfy_plugin_hidden_dim': self.cfy_plugin_hidden_dim,
                'cfy_plugin_residual_hidden_dim': self.cfy_plugin_residual_hidden_dim,
            }

            LPMWithCFY.__init__(self, cfy_config=cfy_config)

            # Setup CFY after initialization
            try:
                classifier_input_dim = self.hidden_dim if self.num_layers > 0 else (3 * self.embedding_dim)
                self.setup_cfy(
                    input_dim=classifier_input_dim,
                    hidden_dim=kwargs.get('hidden_dim', 256),
                )
                logger.info(f"CFY Enhanced LPM initialized with {self.num_classes}-class semantic interaction features")
                self.cfy_enabled = True
                if self.cfy_train_plugin_only:
                    self._freeze_backbone_for_plugin_only_training()
            except Exception as e:
                logger.warning(f"CFY setup failed: {e}. Falling back to standard LPM.")
                self.cfy_enabled = False
        else:
            self.cfy_enabled = False

    def _freeze_backbone_for_plugin_only_training(self):
        """Freeze backbone parameters and keep only CFY parameters trainable."""
        cfy_param_ids = {id(p) for p in self.get_cfy_parameters()}
        trainable_count = 0
        frozen_count = 0
        for param in self.parameters():
            if id(param) in cfy_param_ids:
                param.requires_grad = True
                trainable_count += param.numel()
            else:
                param.requires_grad = False
                frozen_count += param.numel()

        logger.info(
            "Plugin-only training enabled: trainable(CFY)=%s, frozen(backbone)=%s",
            f"{trainable_count:,}",
            f"{frozen_count:,}",
        )

    def configure_optimizers(self):  # noqa: D102
        if not self.cfy_train_plugin_only:
            return super().configure_optimizers()

        trainable_params = [p for p in self.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters found in plugin-only mode")

        optimizer_dict = {
            "Adam": torch.optim.Adam(trainable_params, lr=self.learning_rate),
            "AdamW": torch.optim.AdamW(trainable_params, lr=self.learning_rate),
            "Adagrad": torch.optim.Adagrad(trainable_params, lr=self.learning_rate),
            "Adadelta": torch.optim.Adadelta(trainable_params, lr=self.learning_rate),
            "Adamax": torch.optim.Adamax(trainable_params, lr=self.learning_rate),
            "SGD": torch.optim.SGD(trainable_params, lr=self.learning_rate, momentum=0.9),
            "RMSprop": torch.optim.RMSprop(trainable_params, lr=self.learning_rate),
        }
        if self.optimizer_name not in optimizer_dict.keys():
            ValueError(f"Unrecognized optimizer {self.optimizer_name}")
        optimizer = optimizer_dict[self.optimizer_name]
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.learning_rate_decay)
        return [optimizer], [scheduler]

    @staticmethod
    def _prob_to_logit(prob: float) -> float:
        p = min(max(prob, 1e-4), 1.0 - 1e-4)
        return math.log(p / (1.0 - p))

    def _schedule_plugin_mixes(self):
        """Ramp plugin head mixing coefficients to protect early-stage regression learning."""
        if not self.cfy_enabled:
            return

        span = max(self.cfy_head_ramp_end_epoch - self.cfy_head_ramp_start_epoch, 1)
        progress = (self.current_epoch - self.cfy_head_ramp_start_epoch) / span
        progress = min(max(progress, 0.0), 1.0)

        with torch.no_grad():
            if hasattr(self, 'program_mix_logit'):
                self.program_mix_logit.data.fill_(self._prob_to_logit(self.cfy_program_mix_target * progress))
            if hasattr(self, 'disentangle_mix_logit'):
                self.disentangle_mix_logit.data.fill_(self._prob_to_logit(self.cfy_disentangle_mix_target * progress))
            if hasattr(self, 'distribution_mix_logit'):
                self.distribution_mix_logit.data.fill_(self._prob_to_logit(self.cfy_distribution_mix_target * progress))
            if hasattr(self, 'structured_mix_logit'):
                self.structured_mix_logit.data.fill_(self._prob_to_logit(self.cfy_structured_mix_target * progress))

    def get_model_type(self):
        """Return model type string."""
        return f"CFY Enhanced LPM (CFY {'enabled' if self.cfy_enabled else 'disabled'})"

    def fit(self, traindata, valdata=None):
        """Override fit to support oversampling for minority classes."""
        import pytorch_lightning as pyl
        from perturb_lib.data.access import encode_data
        from perturb_lib.models.base import to_tensor_dict
        from torch.utils.data import DataLoader

        self._initialize_vocabularies_and_embeddings(traindata)
        assert self.vocab is not None

        # Create sampler BEFORE encoding (need access to original traindata._data)
        sampler = None
        if self.cfy_enable_oversampling:
            from perturb_lib.cfy_plugin.oversampling_utils import (
                create_weighted_sampler_for_dual_perturbations,
            )

            logger.info("Creating weighted sampler for oversampling minority classes...")
            sampler = create_weighted_sampler_for_dual_perturbations(
                traindata,
                oversampling_strategy=self.cfy_oversampling_strategy,
                beta=self.cfy_oversampling_beta,
                min_weight=0.1,
                max_weight=10.0,
                single_sample_weight=self.cfy_oversampling_single_weight,
            )

            if sampler is not None:
                logger.info(
                    "Oversampling enabled | strategy=%s beta=%.4f single_weight=%.3f",
                    self.cfy_oversampling_strategy,
                    self.cfy_oversampling_beta,
                    self.cfy_oversampling_single_weight,
                )

        # Now encode data
        traindata_tensors = to_tensor_dict(encode_data(traindata, self.vocab))

        # Create train loader with optional oversampling
        if sampler is not None:
            from torch.utils.data import BatchSampler
            batch_sampler = BatchSampler(sampler, batch_size=self.batch_size, drop_last=False)

            train_loader = DataLoader(
                dataset=traindata_tensors,
                sampler=batch_sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=getattr(traindata_tensors, '_collate_fn', None),
                persistent_workers=True if self.num_workers > 0 else False,
                multiprocessing_context="spawn" if self.num_workers > 0 else None,
            )
            logger.info("Using WeightedRandomSampler for oversampling")
        else:
            train_loader = traindata_tensors.get_data_loader(
                self.batch_size, self.num_workers, self.pin_memory, shuffle=True
            )

        if valdata is not None:
            valdata_tensors = to_tensor_dict(encode_data(valdata, self.vocab))
            val_loader = valdata_tensors.get_data_loader(None, self.num_workers, self.pin_memory, shuffle=False)
        else:
            val_loader = None

        trainer = pyl.Trainer(**self.lightning_trainer_pars)
        logger.info(f"Fitting {self.__class__.__name__}..")
        trainer.fit(model=self, train_dataloaders=train_loader, val_dataloaders=val_loader)
        if len(self.throughput_outputs) > 1:
            avg_thr = sum(self.throughput_outputs[1:]) / len(self.throughput_outputs[1:])
            logger.info(f"Average throughput: {int(avg_thr)} samples/sec == {int(avg_thr / self.batch_size)} it/sec")
        logger.info("Model fitting completed")

    def get_model_type(self):
        """Return model type string."""
        return f"CFY Enhanced LPM (CFY {'enabled' if self.cfy_enabled else 'disabled'})"

    def _initialize_vocabularies_and_embeddings(self, data):
        """Reuse loaded backbone vocab/embeddings in plugin-only mode instead of reinitializing."""
        if (
            self.cfy_train_plugin_only
            and self.vocab is not None
            and self.context_embedding_layer is not None
            and self.perturb_embedding_layer is not None
            and self.readout_embedding_layer is not None
        ):
            logger.info("Plugin-only mode: keep loaded backbone vocab/embeddings (skip re-initialization).")
            return
        return super()._initialize_vocabularies_and_embeddings(data)

    def _extract_dual_class_labels(self, batch: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Get dual-sample class labels if co_effect_type labels are present in the batch."""
        label_key = None
        if 'co_effect_type_id' in batch:
            label_key = 'co_effect_type_id'
        elif 'co_effect_type' in batch:
            label_key = 'co_effect_type'
        if label_key is None:
            return None
        if not hasattr(self, '_last_dual_mask') or self._last_dual_mask is None:
            return None

        labels = batch[label_key]
        dual_mask = self._last_dual_mask
        if labels.shape[0] != dual_mask.shape[0]:
            return None

        dual_labels = labels[dual_mask]
        if dual_labels.numel() == 0:
            return None
        return dual_labels.long()

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):  # noqa: D102
        self._schedule_plugin_mixes()
        pred: torch.Tensor = self(batch)
        unreduced_loss: torch.Tensor = self.loss(pred, batch['value'].unsqueeze(-1)).view(-1)
        enable_aux = self.current_epoch >= self.cfy_aux_warmup_epochs
        dual_labels = self._extract_dual_class_labels(batch)

        # Weighted MSE for regression: upweight selected interaction classes.
        regression_weights = torch.ones_like(unreduced_loss)
        if (
            dual_labels is not None
            and hasattr(self, '_last_dual_mask')
            and self._last_dual_mask is not None
        ):
            dual_indices = torch.where(self._last_dual_mask)[0]
            if dual_indices.shape[0] == dual_labels.shape[0]:
                dual_w = torch.ones_like(dual_labels, dtype=unreduced_loss.dtype, device=unreduced_loss.device)
                valid_mask = (dual_labels >= 0) & (dual_labels < self.num_classes)
                for class_id, class_weight in self.cfy_regression_class_weights.items():
                    class_mask = (dual_labels == class_id) & valid_mask
                    if class_mask.any():
                        dual_w[class_mask] = float(class_weight)
                regression_weights[dual_indices] = dual_w

        total_loss = (unreduced_loss * regression_weights).sum() / regression_weights.sum().clamp_min(1e-8)

        # When dual route is detached from backbone, avoid diluting single-task learning
        # by separating single and dual regression objectives.
        if (
            getattr(self, 'freeze_backbone_for_dual', False)
            and hasattr(self, '_last_dual_mask')
            and self._last_dual_mask is not None
            and self._last_dual_mask.numel() == unreduced_loss.numel()
        ):
            single_mask = ~self._last_dual_mask
            dual_mask = self._last_dual_mask

            if single_mask.any():
                single_loss = unreduced_loss[single_mask].mean()
            elif self.cfy_train_plugin_only and dual_mask.any():
                single_loss = unreduced_loss.new_tensor(0.0)
            else:
                single_loss = unreduced_loss.mean()

            if dual_mask.any():
                dual_w = regression_weights[dual_mask]
                dual_loss = (unreduced_loss[dual_mask] * dual_w).sum() / dual_w.sum().clamp_min(1e-8)
            else:
                dual_loss = unreduced_loss.new_tensor(0.0)

            dual_loss_weight = self.cfy_dual_reg_loss_weight
            if self.cfy_train_plugin_only:
                # In plugin-only mode the singles route is frozen and contributes no
                # gradient, so dual regression must remain the primary objective.
                dual_loss_weight = 1.0

            total_loss = single_loss + dual_loss_weight * dual_loss
            self.log('Regression SingleMSE', single_loss.detach(), prog_bar=False)
            self.log('Regression DualWMSE', dual_loss.detach(), prog_bar=False)
            self.log(
                'Regression DualLossWeight',
                torch.tensor(dual_loss_weight, device=unreduced_loss.device),
                prog_bar=False,
            )

        self.log('Regression WMSE', total_loss.detach(), prog_bar=False)
        self.log('Regression MeanWeight', regression_weights.mean().detach(), prog_bar=False)
        self.log('Regression MSE', unreduced_loss.mean().detach(), prog_bar=False)
        self.log('Monitor/DualLabelsAvailable', 1.0 if dual_labels is not None else 0.0, prog_bar=False)

        # Runtime checks for monitoring validity.
        should_check_monitor = self.cfy_monitor_semantics and (batch_idx % self.cfy_monitor_interval == 0)
        if should_check_monitor:
            if dual_labels is None:
                logger.warning(
                    'Monitor check e=%s b=%s | dual labels are missing; per-class WMSE/classification checks are skipped.',
                    int(self.current_epoch),
                    int(batch_idx),
                )
            else:
                dual_count = int(dual_labels.shape[0])
                valid_dual_count = int(((dual_labels >= 0) & (dual_labels < self.num_classes)).sum().item())
                self.log('Monitor/DualCount', float(dual_count), prog_bar=False)
                self.log('Monitor/ValidDualCount', float(valid_dual_count), prog_bar=False)

                configured_reg_weights = [
                    float(self.cfy_regression_class_weights.get(i, 1.0))
                    for i in range(self.num_classes)
                ]
                expect_weighting = any(abs(w - 1.0) > 1e-6 for w in configured_reg_weights)
                mean_weight = float(regression_weights.mean().item())
                if expect_weighting and dual_count > 0 and abs(mean_weight - 1.0) < 1e-6:
                    logger.warning(
                        'Monitor check e=%s b=%s | WMSE weighting looks ineffective (mean weight=%.4f). '
                        'Check dual label coverage and class weight mapping.',
                        int(self.current_epoch),
                        int(batch_idx),
                        mean_weight,
                    )

        # Per-class regression diagnostics on dual perturbations.
        if (
            dual_labels is not None
            and hasattr(self, '_last_dual_mask')
            and self._last_dual_mask is not None
        ):
            dual_indices = torch.where(self._last_dual_mask)[0]
            if dual_indices.shape[0] == dual_labels.shape[0]:
                class_names = ['Additive', 'Synergy', 'Buffering', 'Opposite', 'Other'][:self.num_classes]
                valid_mask = (dual_labels >= 0) & (dual_labels < self.num_classes)
                for class_id, class_name in enumerate(class_names):
                    class_mask = (dual_labels == class_id) & valid_mask
                    if not class_mask.any():
                        continue

                    class_indices = dual_indices[class_mask]
                    class_mse = unreduced_loss[class_indices].mean()
                    class_w = regression_weights[class_indices]
                    class_wmse = (unreduced_loss[class_indices] * class_w).sum() / class_w.sum().clamp_min(1e-8)

                    self.log(f'RegressionByClass/{class_name}_MSE', class_mse.detach(), prog_bar=False)
                    self.log(f'RegressionByClass/{class_name}_WMSE', class_wmse.detach(), prog_bar=False)
                    self.log(
                        f'RegressionByClass/{class_name}_Count',
                        class_mask.float().sum().detach(),
                        prog_bar=False,
                    )

        if (
            self.cfy_enabled
            and self.cfy_additive_anchor_weight > 0
            and dual_labels is not None
            and hasattr(self, '_last_dual_mask')
            and self._last_dual_mask is not None
            and hasattr(self, '_last_dual_base_predictions')
            and self._last_dual_base_predictions is not None
        ):
            dual_indices = torch.where(self._last_dual_mask)[0]
            dual_base_preds = self._last_dual_base_predictions.view(-1)
            if dual_indices.shape[0] == dual_labels.shape[0] == dual_base_preds.shape[0]:
                valid_mask = (dual_labels >= 0) & (dual_labels < self.num_classes)
                if valid_mask.any():
                    labels = dual_labels[valid_mask]
                    is_add = labels == 0
                    if is_add.any():
                        dual_pred = pred.view(-1)[dual_indices][valid_mask]
                        base_pred = dual_base_preds[valid_mask]
                        additive_anchor_loss = ((dual_pred[is_add] - base_pred[is_add]) ** 2).mean()
                        total_loss = total_loss + self.cfy_additive_anchor_weight * additive_anchor_loss
                        self.log('Additive Anchor Loss', additive_anchor_loss.detach(), prog_bar=False)
                        self.log(
                            'Additive Anchor Weight',
                            torch.tensor(self.cfy_additive_anchor_weight, device=dual_pred.device),
                            prog_bar=False,
                        )

        # Auxiliary classifier supervision with focal modulation.
        if (
            self.cfy_enabled
            and enable_aux
            and self.cfy_cls_loss_weight > 0
            and hasattr(self, '_last_class_logits')
            and self._last_class_logits is not None
            and dual_labels is not None
        ):
            valid_mask = (dual_labels >= 0) & (dual_labels < self.num_classes)
            if valid_mask.any():
                logits = self._last_class_logits[valid_mask]
                labels = dual_labels[valid_mask]

                class_weights = torch.tensor(
                    [
                        self.cfy_classification_class_weights.get(i, 1.0)
                        for i in range(self.num_classes)
                    ],
                    device=logits.device,
                    dtype=logits.dtype,
                )

                ce_each = nn.CrossEntropyLoss(weight=class_weights, reduction='none')(logits, labels)
                pred_labels = torch.argmax(logits, dim=1)
                misclassified_mask = pred_labels != labels
                misclassified_boost = torch.where(
                    misclassified_mask,
                    torch.full_like(ce_each, self.cfy_cls_misclassified_boost),
                    torch.ones_like(ce_each),
                )

                if self.cfy_cls_focal_gamma > 0:
                    probs = torch.softmax(logits, dim=1)
                    pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1).clamp_min(1e-6)
                    focal_factor = (1.0 - pt) ** self.cfy_cls_focal_gamma
                    cls_loss = (ce_each * focal_factor * misclassified_boost).mean()
                else:
                    cls_loss = (ce_each * misclassified_boost).mean()

                cls_weight = self.cfy_cls_loss_weight
                if self.current_epoch < self.cfy_cls_fast_epochs:
                    cls_weight = cls_weight * self.cfy_cls_fast_boost

                total_loss = total_loss + cls_weight * cls_loss
                self.log('Classification Loss', cls_loss.detach(), prog_bar=False)
                self.log('Classification Weight', torch.tensor(cls_weight, device=logits.device), prog_bar=False)
                self.log('Classification MisRate', misclassified_mask.float().mean().detach(), prog_bar=False)

        # Encourage more diverse routing probabilities on dual samples.
        if (
            self.cfy_enabled
            and enable_aux
            and self.cfy_entropy_reg_weight > 0
            and hasattr(self, '_last_class_probs')
            and self._last_class_probs is not None
        ):
            probs = self._last_class_probs.clamp_min(1e-8)
            entropy = -(probs * probs.log()).sum(dim=1).mean()
            total_loss = total_loss - self.cfy_entropy_reg_weight * entropy
            self.log('Routing Entropy', entropy.detach(), prog_bar=False)

        if (
            self.cfy_enabled
            and enable_aux
            and self.cfy_relation_loss_weight > 0
            and dual_labels is not None
            and hasattr(self, '_last_expert_outputs')
            and self._last_expert_outputs is not None
            and self._last_expert_outputs.shape[1] >= self.num_classes
        ):
            valid_mask = (dual_labels >= 0) & (dual_labels < self.num_classes)
            if valid_mask.any():
                labels = dual_labels[valid_mask]
                experts = self._last_expert_outputs[valid_mask]

                additive = experts[:, 0]
                synergy = experts[:, 1]
                buffering = experts[:, 2]
                opposite = experts[:, 3]

                relation_terms = []

                is_syn = labels == 1
                if is_syn.any():
                    relation_terms.append(F.relu(additive[is_syn] - synergy[is_syn] + 0.02).mean())

                is_buf = labels == 2
                if is_buf.any():
                    relation_terms.append(F.relu(buffering[is_buf] - additive[is_buf] + 0.02).mean())

                is_opp = labels == 3
                if is_opp.any():
                    relation_terms.append(F.relu(opposite[is_opp] + additive[is_opp] + 0.02).mean())

                if relation_terms:
                    relation_loss = torch.stack(relation_terms).mean()
                    total_loss = total_loss + self.cfy_relation_loss_weight * relation_loss
                    self.log('Relation Loss', relation_loss.detach(), prog_bar=False)

                if self.cfy_opposite_margin_weight > 0:
                    is_opp = labels == 3
                    if is_opp.any():
                        opposite_margin_loss = F.relu(
                            opposite[is_opp] + additive[is_opp] + self.cfy_opposite_margin
                        ).mean()
                        total_loss = total_loss + self.cfy_opposite_margin_weight * opposite_margin_loss
                        self.log('Opposite Margin Loss', opposite_margin_loss.detach(), prog_bar=False)

        # Strengthen semantics across all classes with label-vs-others margin.
        if (
            self.cfy_enabled
            and enable_aux
            and self.cfy_semantic_consistency_weight > 0
            and dual_labels is not None
            and hasattr(self, '_last_expert_outputs')
            and self._last_expert_outputs is not None
            and self._last_expert_outputs.shape[1] >= self.num_classes
        ):
            valid_mask = (dual_labels >= 0) & (dual_labels < self.num_classes)
            if valid_mask.any():
                labels = dual_labels[valid_mask]
                experts = self._last_expert_outputs[valid_mask]

                chosen = experts.gather(1, labels.unsqueeze(1)).squeeze(1)
                others = experts.clone()
                others.scatter_(1, labels.unsqueeze(1), float('-inf'))
                max_other = others.max(dim=1).values
                semantic_consistency = F.relu(max_other - chosen + self.cfy_semantic_margin).mean()

                total_loss = total_loss + self.cfy_semantic_consistency_weight * semantic_consistency
                self.log('Semantic Consistency Loss', semantic_consistency.detach(), prog_bar=False)
                self.log('Semantic Margin', torch.tensor(self.cfy_semantic_margin, device=experts.device), prog_bar=False)

        # Enforce regression-level separation: label-matched expert should fit target
        # better than other semantic experts.
        if (
            self.cfy_enabled
            and enable_aux
            and self.cfy_regression_separation_weight > 0
            and dual_labels is not None
            and hasattr(self, '_last_expert_outputs')
            and self._last_expert_outputs is not None
            and hasattr(self, '_last_dual_mask')
            and self._last_dual_mask is not None
            and self._last_expert_outputs.shape[1] >= self.num_classes
        ):
            valid_mask = (dual_labels >= 0) & (dual_labels < self.num_classes)
            if valid_mask.any():
                labels = dual_labels[valid_mask]
                experts = self._last_expert_outputs[valid_mask]
                dual_targets = batch['value'][self._last_dual_mask][valid_mask].unsqueeze(1)

                per_expert_mse = (experts - dual_targets) ** 2
                target_mse = per_expert_mse.gather(1, labels.unsqueeze(1))
                others_mse = per_expert_mse.clone()
                others_mse.scatter_(1, labels.unsqueeze(1), float('inf'))

                margin_mat = self.cfy_regression_separation_margin + target_mse - others_mse
                separation_loss = F.relu(margin_mat).mean()

                total_loss = total_loss + self.cfy_regression_separation_weight * separation_loss
                self.log('Regression Separation Loss', separation_loss.detach(), prog_bar=False)
                self.log(
                    'Regression Separation Margin',
                    torch.tensor(self.cfy_regression_separation_margin, device=experts.device),
                    prog_bar=False,
                )

        if (
            self.cfy_enabled
            and self.current_epoch >= self.cfy_expert_supervision_warmup_epochs
            and self.cfy_expert_supervision_weight > 0
            and dual_labels is not None
            and hasattr(self, '_last_expert_outputs')
            and self._last_expert_outputs is not None
            and hasattr(self, '_last_dual_mask')
            and self._last_dual_mask is not None
        ):
            valid_mask = (dual_labels >= 0) & (dual_labels < self.num_classes)
            if valid_mask.any():
                labels = dual_labels[valid_mask]
                experts = self._last_expert_outputs[valid_mask]
                dual_targets = batch['value'][self._last_dual_mask][valid_mask]

                expert_pred = experts.gather(1, labels.unsqueeze(1)).squeeze(1)
                expert_mse = (expert_pred - dual_targets) ** 2

                expert_loss = expert_mse.mean()

                total_loss = total_loss + self.cfy_expert_supervision_weight * expert_loss
                self.log('Expert Supervision Loss', expert_loss.detach(), prog_bar=False)

        if (
            self.cfy_enabled
            and self.cfy_additive_supervision_weight > 0
            and dual_labels is not None
            and hasattr(self, '_last_expert_outputs')
            and self._last_expert_outputs is not None
            and hasattr(self, '_last_dual_mask')
            and self._last_dual_mask is not None
        ):
            valid_mask = (dual_labels >= 0) & (dual_labels < self.num_classes)
            if valid_mask.any():
                labels = dual_labels[valid_mask]
                experts = self._last_expert_outputs[valid_mask]
                dual_targets = batch['value'][self._last_dual_mask][valid_mask]

                is_add = labels == 0
                add_count = is_add.float().sum()
                self.log('RegressionByClass/Additive_BatchCount', add_count.detach(), prog_bar=False)
                if is_add.any():
                    additive_pred = experts[is_add, 0]
                    additive_target = dual_targets[is_add]
                    additive_expert_loss = ((additive_pred - additive_target) ** 2).mean()
                    total_loss = total_loss + self.cfy_additive_supervision_weight * additive_expert_loss
                    self.log('Additive Expert Loss', additive_expert_loss.detach(), prog_bar=False)

        if (
            self.cfy_enabled
            and enable_aux
            and hasattr(self, '_last_distribution_mean')
            and self._last_distribution_mean is not None
            and hasattr(self, '_last_distribution_logvar')
            and self._last_distribution_logvar is not None
            and dual_labels is not None
            and hasattr(self, '_last_dual_mask')
            and self._last_dual_mask is not None
        ):
            dual_targets = batch['value'].unsqueeze(-1)[self._last_dual_mask]
            dual_mu = self._last_distribution_mean
            dual_logvar = self._last_distribution_logvar

            if dual_targets.shape[0] == dual_mu.shape[0]:
                if self.cfy_distribution_nll_weight > 0:
                    nll = 0.5 * (dual_logvar + ((dual_targets - dual_mu) ** 2) * torch.exp(-dual_logvar))
                    dist_nll = nll.mean()
                    total_loss = total_loss + self.cfy_distribution_nll_weight * dist_nll
                    self.log('Distribution NLL', dist_nll.detach(), prog_bar=False)

                if self.cfy_distribution_mmd_weight > 0:
                    mmd = self.compute_mmd_loss(dual_mu, dual_targets.detach())
                    total_loss = total_loss + self.cfy_distribution_mmd_weight * mmd
                    self.log('Distribution MMD', mmd.detach(), prog_bar=False)

        self.training_step_outputs.append(unreduced_loss.detach())

        # Lightweight observability for semantic fairness diagnostics.
        if (
            self.cfy_enabled
            and self.cfy_monitor_semantics
            and (batch_idx % self.cfy_monitor_interval == 0)
            and hasattr(self, '_last_class_probs')
            and self._last_class_probs is not None
            and hasattr(self, '_last_semantic_outputs')
            and self._last_semantic_outputs is not None
            and self._last_semantic_outputs.numel() > 0
        ):
            with torch.no_grad():
                semantic = self._last_semantic_outputs
                probs = self._last_class_probs

                sem_mean = semantic.mean(dim=0)
                sem_std = semantic.std(dim=0, unbiased=False)
                route_mass = probs.mean(dim=0)
                route_argmax = torch.argmax(probs, dim=1)
                route_temp = float(self._get_router_temperature()) if hasattr(self, '_get_router_temperature') else float('nan')
                argmax_ratio = torch.stack([
                    (route_argmax == cid).float().mean()
                    for cid in range(self.num_classes)
                ])

                names = ['Additive', 'Synergy', 'Buffering', 'Opposite', 'Other'][:self.num_classes]
                for cid, cname in enumerate(names):
                    self.log(f'Semantic/{cname}_Mean', sem_mean[cid], prog_bar=False)
                    self.log(f'Semantic/{cname}_Std', sem_std[cid], prog_bar=False)
                    self.log(f'Routing/{cname}_MeanProb', route_mass[cid], prog_bar=False)
                    self.log(f'Routing/{cname}_ArgmaxRatio', argmax_ratio[cid], prog_bar=False)

                # Per-true-label routing diagnostics to detect class collapse.
                if dual_labels is not None and dual_labels.shape[0] == probs.shape[0]:
                    valid_label_mask = (dual_labels >= 0) & (dual_labels < self.num_classes)
                    if valid_label_mask.any():
                        valid_labels = dual_labels[valid_label_mask]
                        valid_probs = probs[valid_label_mask]
                        valid_argmax = torch.argmax(valid_probs, dim=1)
                        for true_cid, true_name in enumerate(names):
                            group_mask = valid_labels == true_cid
                            group_count = float(group_mask.float().sum().item())
                            self.log(
                                f'RoutingByLabel/{true_name}_Count',
                                torch.tensor(group_count, device=valid_probs.device),
                                prog_bar=False,
                            )
                            if not group_mask.any():
                                zero = torch.tensor(0.0, device=valid_probs.device)
                                self.log(f'RoutingByLabel/{true_name}_SelfMeanProb', zero, prog_bar=False)
                                self.log(f'RoutingByLabel/{true_name}_SelfArgmaxRatio', zero, prog_bar=False)
                                self.log(f'RoutingByLabel/{true_name}_ToBufferingMeanProb', zero, prog_bar=False)
                                self.log(f'RoutingByLabel/{true_name}_ToBufferingArgmaxRatio', zero, prog_bar=False)
                                continue

                            group_probs = valid_probs[group_mask]
                            group_mean_prob = group_probs.mean(dim=0)
                            group_argmax = valid_argmax[group_mask]
                            group_argmax_ratio = torch.stack([
                                (group_argmax == pred_cid).float().mean()
                                for pred_cid in range(self.num_classes)
                            ])

                            self.log(
                                f'RoutingByLabel/{true_name}_SelfMeanProb',
                                group_mean_prob[true_cid],
                                prog_bar=False,
                            )
                            self.log(
                                f'RoutingByLabel/{true_name}_SelfArgmaxRatio',
                                group_argmax_ratio[true_cid],
                                prog_bar=False,
                            )
                            self.log(
                                f'RoutingByLabel/{true_name}_ToBufferingMeanProb',
                                group_mean_prob[2],
                                prog_bar=False,
                            )
                            self.log(
                                f'RoutingByLabel/{true_name}_ToBufferingArgmaxRatio',
                                group_argmax_ratio[2],
                                prog_bar=False,
                            )

                            if self.num_classes == 5:
                                logger.info(
                                    'Routing by label e=%s b=%s true=%s n=%s | mean_prob[A,S,B,O,Ot]=[%.3f, %.3f, %.3f, %.3f, %.3f] '
                                    '| argmax[A,S,B,O,Ot]=[%.3f, %.3f, %.3f, %.3f, %.3f]',
                                    int(self.current_epoch),
                                    int(batch_idx),
                                    true_name,
                                    int(group_mask.sum().item()),
                                    float(group_mean_prob[0].item()),
                                    float(group_mean_prob[1].item()),
                                    float(group_mean_prob[2].item()),
                                    float(group_mean_prob[3].item()),
                                    float(group_mean_prob[4].item()),
                                    float(group_argmax_ratio[0].item()),
                                    float(group_argmax_ratio[1].item()),
                                    float(group_argmax_ratio[2].item()),
                                    float(group_argmax_ratio[3].item()),
                                    float(group_argmax_ratio[4].item()),
                                )
                            else:
                                logger.info(
                                    'Routing by label e=%s b=%s true=%s n=%s | mean_prob[A,S,B,O]=[%.3f, %.3f, %.3f, %.3f] '
                                    '| argmax[A,S,B,O]=[%.3f, %.3f, %.3f, %.3f]',
                                    int(self.current_epoch),
                                    int(batch_idx),
                                    true_name,
                                    int(group_mask.sum().item()),
                                    float(group_mean_prob[0].item()),
                                    float(group_mean_prob[1].item()),
                                    float(group_mean_prob[2].item()),
                                    float(group_mean_prob[3].item()),
                                    float(group_argmax_ratio[0].item()),
                                    float(group_argmax_ratio[1].item()),
                                    float(group_argmax_ratio[2].item()),
                                    float(group_argmax_ratio[3].item()),
                                )

                logger.info(
                    'Semantic monitor e=%s b=%s | temp=%.3f | mean[A,S,B,O,Ot]=[%.4f, %.4f, %.4f, %.4f, %.4f] '
                    '| route_prob=[%.3f, %.3f, %.3f, %.3f, %.3f] | argmax=[%.3f, %.3f, %.3f, %.3f, %.3f]',
                    int(self.current_epoch),
                    int(batch_idx),
                    route_temp,
                    float(sem_mean[0].item()),
                    float(sem_mean[1].item()),
                    float(sem_mean[2].item()),
                    float(sem_mean[3].item()),
                    float(sem_mean[4].item()),
                    float(route_mass[0].item()),
                    float(route_mass[1].item()),
                    float(route_mass[2].item()),
                    float(route_mass[3].item()),
                    float(route_mass[4].item()),
                    float(argmax_ratio[0].item()),
                    float(argmax_ratio[1].item()),
                    float(argmax_ratio[2].item()),
                    float(argmax_ratio[3].item()),
                    float(argmax_ratio[4].item()),
                )

                collapse_class = int(torch.argmax(argmax_ratio).item())
                collapse_ratio = float(argmax_ratio[collapse_class].item())
                is_route_collapse = collapse_ratio > 0.90
                self.log('Monitor/RoutingCollapse', 1.0 if is_route_collapse else 0.0, prog_bar=False)
                if is_route_collapse:
                    collapse_name = names[collapse_class] if collapse_class < len(names) else str(collapse_class)
                    logger.warning(
                        'Monitor alert e=%s b=%s | routing collapse detected on %s (ratio=%.3f): '
                        'argmax[A,S,B,O,Ot]=[%.3f, %.3f, %.3f, %.3f, %.3f]',
                        int(self.current_epoch),
                        int(batch_idx),
                        collapse_name,
                        collapse_ratio,
                        float(argmax_ratio[0].item()),
                        float(argmax_ratio[1].item()),
                        float(argmax_ratio[2].item()),
                        float(argmax_ratio[3].item()),
                        float(argmax_ratio[4].item()),
                    )

        return total_loss


class TrainingComparison:
    """Class to manage training comparison between different model types."""

    def __init__(self,
                 context: Sequence[str],
                 max_epochs: int = 5,
                 batch_size: int = 1000,
                 test_size: float = 0.1,
                 learning_rate: float = 0.002,
                 param_load_mode: Optional[str] = None,
                 save_param_cache: Optional[bool] = None,
                 force_retrain_baseline: Optional[bool] = None,
                 force_retrain_head: Optional[bool] = None,
                 cfy_enable_oversampling: bool = False,
                 cfy_oversampling_strategy: str = 'balanced',
                 cfy_oversampling_beta: float = 0.9999):
        """
        Initialize training comparison.

        Args:
            context: Context(s) to use for training data
            max_epochs: Maximum training epochs
            batch_size: Training batch size
            test_size: Fraction of data to use for testing
            learning_rate: Learning rate for optimizer
            cfy_enable_oversampling: Enable oversampling for minority classes
            cfy_oversampling_strategy: 'balanced', 'inverse', or 'sqrt_inverse'
            cfy_oversampling_beta: Beta parameter for class-balanced weighting
        """
        self.context = context
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.test_size = test_size
        self.learning_rate = learning_rate
        self.cfy_enable_oversampling = cfy_enable_oversampling
        self.cfy_oversampling_strategy = cfy_oversampling_strategy
        self.cfy_oversampling_beta = cfy_oversampling_beta

        # Parameter loading/saving controls.
        # Modes: none | baseline | head | all
        mode_from_env = os.getenv("PLIB_PARAM_LOAD_MODE", "all")
        self.param_load_mode = (param_load_mode or mode_from_env).strip().lower()
        if self.param_load_mode not in {"none", "baseline", "head", "all"}:
            logger.warning("Unknown param_load_mode=%s, fallback to 'all'", self.param_load_mode)
            self.param_load_mode = "all"

        if save_param_cache is None:
            save_param_cache = os.getenv("PLIB_SAVE_PARAM_CACHE", "1").strip() != "0"
        self.save_param_cache = bool(save_param_cache)

        if force_retrain_baseline is None:
            force_retrain_baseline = os.getenv("PLIB_FORCE_RETRAIN_BASELINE", "0").strip() == "1"
        if force_retrain_head is None:
            force_retrain_head = os.getenv("PLIB_FORCE_RETRAIN_HEAD", "0").strip() == "1"
        self.force_retrain_baseline = bool(force_retrain_baseline)
        self.force_retrain_head = bool(force_retrain_head)

        # Initialize random seed for reproducibility
        plib.set_all_seeds(24)
        plib.set_path_to_cache("perturb_lib/my_cache")

        # Set PyTorch precision
        torch.set_float32_matmul_precision('high')

        self.results = {}
        self._baseline_backbone_result: Optional[Dict[str, Any]] = None
        self.dataset_tag: Optional[str] = None
        self.cfy_head_training_params: Dict[str, Any] = dict(DEFAULT_CFY_HEAD_TRAINING_PARAMS)

    @staticmethod
    def _sanitize_tag(text: str) -> str:
        text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
        return text.strip("_") or "dataset"

    def _build_dataset_tag(self) -> str:
        context_tag = "-".join(sorted(map(str, self.context)))
        n_train = len(self.traindata) if hasattr(self, 'traindata') else -1
        n_val = len(self.valdata) if hasattr(self, 'valdata') and self.valdata is not None else 0
        n_test = len(self.testdata) if hasattr(self, 'testdata') else -1
        raw = f"ctx={context_tag}__ntr={n_train}__nva={n_val}__nte={n_test}"
        return self._sanitize_tag(raw)

    def _get_param_cache_dir(self) -> Path:
        cache_dir = Path("perturb_lib") / "checks" / "model_param_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _context_param_tag(self) -> str:
        context_tag = "__".join(sorted(map(str, self.context)))
        return self._sanitize_tag(context_tag)

    def _get_cfy_training_param_json_path(self) -> Path:
        params_dir = Path("perturb_lib") / "checks" / "cfy_head_training_params"
        params_dir.mkdir(parents=True, exist_ok=True)
        return params_dir / f"{self._context_param_tag()}.json"

    @staticmethod
    def _coerce_param_value(default_value: Any, new_value: Any) -> Any:
        if isinstance(default_value, bool):
            return bool(new_value)
        if isinstance(default_value, int) and not isinstance(default_value, bool):
            return int(new_value)
        if isinstance(default_value, float):
            return float(new_value)
        return new_value

    def _load_cfy_head_training_params_from_json(self) -> Dict[str, Any]:
        params = dict(DEFAULT_CFY_HEAD_TRAINING_PARAMS)
        json_path = self._get_cfy_training_param_json_path()
        if not json_path.exists():
            logger.info("CFY head training param json not found, using defaults: %s", json_path)
            return params

        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            loaded = payload.get("params", payload) if isinstance(payload, dict) else {}
            if not isinstance(loaded, dict):
                logger.warning("Invalid json format in %s, using defaults", json_path)
                return params

            for key, default_value in DEFAULT_CFY_HEAD_TRAINING_PARAMS.items():
                if key in loaded:
                    params[key] = self._coerce_param_value(default_value, loaded[key])

            logger.info("Loaded CFY head training params from %s", json_path)
            return params
        except Exception as e:
            logger.warning("Failed loading CFY head training params from %s: %s", json_path, e)
            return params

    def _save_cfy_head_training_params_to_json(self, params: Dict[str, Any]):
        json_path = self._get_cfy_training_param_json_path()
        payload = {
            "context": list(self.context),
            "saved_at": time.time(),
            "params": params,
        }
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Saved CFY head training params to %s", json_path)

    def _get_param_cache_paths(self) -> Dict[str, Path]:
        if not self.dataset_tag:
            self.dataset_tag = self._build_dataset_tag()
        cache_dir = self._get_param_cache_dir()
        return {
            "baseline": cache_dir / f"baseline_lpm__{self.dataset_tag}.pt",
            "cfy_head": cache_dir / f"cfy_head__{self.dataset_tag}.pt",
        }

    def _can_load_baseline(self) -> bool:
        return self.param_load_mode in {"baseline", "all"}

    def _can_load_head(self) -> bool:
        return self.param_load_mode in {"head", "all"}

    @staticmethod
    def _extract_cfy_head_state_dict(model: CFYEnhancedLPM) -> Dict[str, torch.Tensor]:
        prefixes = (
            "shared_encoder",
            "classifier",
            "experts",
            "pair_stat",
            "semantic",
            "relation",
            "program",
            "distribution",
            "disentangle",
            "structured",
            "opposite_margin",
        )
        key_tokens = (
            "expert",
            "pair_stat",
            "mix_logit",
            "semantic",
            "relation",
            "program",
            "distribution",
            "disentangle",
            "structured",
            "opposite_margin",
        )
        head_state: Dict[str, torch.Tensor] = {}
        for k, v in model.state_dict().items():
            if (k.startswith(prefixes) or any(tok in k for tok in key_tokens)) and isinstance(v, torch.Tensor):
                head_state[k] = v.detach().cpu().clone()
        return head_state

    @staticmethod
    def _extract_cfy_head_signature(model: CFYEnhancedLPM) -> Dict[str, Any]:
        """Return a lightweight compatibility signature for CFY head caches."""
        head_state = TrainingComparison._extract_cfy_head_state_dict(model)
        tensor_shapes = {
            key: list(value.shape)
            for key, value in sorted(head_state.items())
        }
        return {
            "format_version": 2,
            "tensor_shapes": tensor_shapes,
            "training_recipe": {
                "plugin_only": bool(getattr(model, "cfy_train_plugin_only", False)),
                "enable_oversampling": bool(getattr(model, "cfy_enable_oversampling", False)),
                "oversampling_strategy": str(getattr(model, "cfy_oversampling_strategy", "")),
                "oversampling_beta": float(getattr(model, "cfy_oversampling_beta", 0.0)),
            },
        }

    @staticmethod
    def _extract_tensor_only_state_dict(model: nn.Module) -> Dict[str, Any]:
        tensor_state: Dict[str, Any] = {}
        skipped = 0
        preserved_metadata_keys = {"context_symbols", "perturb_symbols", "readout_symbols"}
        for k, v in model.state_dict().items():
            if isinstance(v, torch.Tensor):
                tensor_state[k] = v.detach().cpu().clone()
            elif k in preserved_metadata_keys:
                # LPM uses these symbol lists to rebuild vocab on load.
                tensor_state[k] = v
            else:
                skipped += 1
        if skipped > 0:
            logger.warning("Skipped %s non-tensor entries while saving state_dict cache.", skipped)
        return tensor_state

    @staticmethod
    def _load_partial_state_dict(model: nn.Module, partial_state: Dict[str, torch.Tensor]) -> int:
        if not partial_state:
            return 0
        current = model.state_dict()
        loaded = 0
        for k, v in partial_state.items():
            if k in current and current[k].shape == v.shape:
                current[k] = v.to(dtype=current[k].dtype)
                loaded += 1
        model.load_state_dict(current, strict=False)
        return loaded

    @staticmethod
    def _load_local_torch_cache(cache_path: Path) -> Dict[str, Any]:
        """Load local cache payload, compatible with PyTorch 2.6 weights_only default."""
        try:
            return torch.load(cache_path, map_location="cpu")
        except pickle.UnpicklingError as exc:
            if "Weights only load failed" not in str(exc):
                raise
            logger.warning(
                "Retrying cache load with weights_only=False for trusted local cache: %s",
                cache_path,
            )
            try:
                return torch.load(cache_path, map_location="cpu", weights_only=False)
            except TypeError:
                return torch.load(cache_path, map_location="cpu")

    def load_and_prepare_data(self):
        """Load and prepare training data following tutorial workflow."""
        logger.info(f"Loading data for context: {self.context}")

        # Load data
        pdata = plib.load_plibdata(self.context)
        logger.info(f"Loaded {len(pdata)} total samples")

        # Check perturbation data and add co-effect types
        checked_data = Check_Perturbation(pdata)
        checked_data.check_multi_perturbation_coeffect()
        checked_data.check_perturbation_types()
        pdata = checked_data.pdata

        # Split data
        traindata, valdata, testdata = plib.split_plibdata_3fold(
            pdata, context_ids=self.context
        )

        # Add numeric co-effect labels for training supervision while keeping
        # original string labels for reporting/evaluation.
        self._encode_coeffect_type_ids(traindata)
        if valdata is not None:
            self._encode_coeffect_type_ids(valdata)
        self._encode_coeffect_type_ids(testdata)

        logger.info(f"Data split - Train: {len(traindata)}, Val: {len(valdata) if valdata else 0}, Test: {len(testdata)}")

        # Store data splits
        self.traindata = traindata
        self.valdata = valdata
        self.testdata = testdata
        self.pdata = pdata
        self.dataset_tag = self._build_dataset_tag()
        self.cfy_head_training_params = self._load_cfy_head_training_params_from_json()
        self._save_cfy_head_training_params_to_json(self.cfy_head_training_params)
        logger.info(
            "Exposed CFY head params: %s",
            ", ".join(sorted(self.cfy_head_training_params.keys())),
        )

        logger.info(
            "Dataset tag for parameter cache: %s | load_mode=%s save_cache=%s",
            self.dataset_tag,
            self.param_load_mode,
            self.save_param_cache,
        )

        return traindata, valdata, testdata

    @staticmethod
    def _encode_coeffect_type_ids(data):
        """Create numeric co_effect_type_id from co_effect_type strings."""
        if not hasattr(data, '_data') or data._data is None:
            return
        if 'co_effect_type' not in data._data.columns:
            return

        data._data = data._data.with_columns(
            pl.col('co_effect_type')
            .replace_strict(
                {
                    'Additive': 0,
                    'Synergy': 1,
                    'Buffering': 2,
                    'Opposite': 3,
                    'Other': 4,
                },
                default=-1,
            )
            .cast(pl.Int64)
            .alias('co_effect_type_id')
        )

        if hasattr(data, '_data_pd'):
            data._data_pd = None

    def get_base_model_args(self, include_cfy_params: bool = False) -> Dict[str, Any]:
        """Get base model arguments common to all model types.

        Args:
            include_cfy_params: If True, include CFY-specific parameters (for CFYEnhancedLPM)
        """
        args = {
            "lightning_trainer_pars": {
                "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
                "max_epochs": self.max_epochs,
                "enable_checkpointing": False,
                "enable_progress_bar": True,
            },
            "optimizer_name": "AdamW",
            "learning_rate": self.learning_rate,
            "learning_rate_decay": 0.98,
            "num_layers": 2,
            "hidden_dim": 256,
            "dropout": 0.0,
            "batch_size": self.batch_size,
            "embedding_dim": 64,
            "num_workers": 1,
            "early_stopping_patience": 0,  # Disable for faster testing
        }

        # Only add oversampling parameters for CFYEnhancedLPM
        if include_cfy_params:
            args.update({
                "cfy_enable_oversampling": self.cfy_enable_oversampling,
                "cfy_oversampling_strategy": self.cfy_oversampling_strategy,
                "cfy_oversampling_beta": self.cfy_oversampling_beta,
            })

        return args

    def train_original_lpm_cfy(self) -> Dict[str, Any]:
        """Train the original LPM_CFY model."""
        logger.info("\\n" + "="*50)
        logger.info("Training Original LPM_CFY Model")
        logger.info("="*50)

        start_time = time.time()

        try:
            # Get model arguments
            model_args = self.get_base_model_args()
            model_args['cls_loss_weight'] = 0.0  # keep RMSE objective comparable to plugin run

            # Create model
            model = LPM_CFY(**model_args)
            logger.info(f"Created {type(model).__name__}")

            # Count parameters
            param_count = sum(p.numel() for p in model.parameters())
            logger.info(f"Model parameters: {param_count:,}")

            # Train model
            logger.info("Starting training...")
            train_start = time.time()
            model.fit(self.traindata, self.valdata)
            train_time = time.time() - train_start

            # Test model
            logger.info("Running predictions...")
            pred_start = time.time()
            predictions = model.predict(self.testdata)
            pred_time = time.time() - pred_start

            # Evaluate
            evaluator = plib.load_evaluator("RMSE")
            overall_rmse = evaluator.evaluate(predictions, self.testdata)

            total_time = time.time() - start_time

            results = {
                'model_type': 'Original LPM_CFY',
                'model': model,
                'predictions': predictions,
                'overall_rmse': overall_rmse,
                'param_count': param_count,
                'train_time': train_time,
                'pred_time': pred_time,
                'total_time': total_time,
                'success': True
            }

            logger.info(f"Original LPM_CFY Results:")
            logger.info(f"  RMSE: {overall_rmse:.4f}")
            logger.info(f"  Parameters: {param_count:,}")
            logger.info(f"  Train time: {train_time:.2f}s")
            logger.info(f"  Pred time: {pred_time:.2f}s")

            return results

        except Exception as e:
            logger.error(f"Original LPM_CFY training failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'model_type': 'Original LPM_CFY',
                'success': False,
                'error': str(e),
                'total_time': time.time() - start_time
            }

    def train_cfy_enhanced_lpm(self) -> Dict[str, Any]:
        """Train the CFY Plugin Enhanced LPM model."""
        logger.info("\\n" + "="*50)
        logger.info("Training CFY Enhanced LPM Model")
        logger.info("="*50)

        start_time = time.time()

        try:
            # Get model arguments
            model_args = self.get_base_model_args(include_cfy_params=True)

            # Add CFY-specific arguments
            model_args.update({
                'cfy_enabled': True,
                'num_classes': 5,
                'freeze_backbone_for_dual': True,
                **self.cfy_head_training_params,
            })

            # Create model
            model = CFYEnhancedLPM(**model_args)
            logger.info(f"Created {model.get_model_type()}")

            # Count parameters
            param_count = sum(p.numel() for p in model.parameters())
            logger.info(f"Model parameters: {param_count:,}")

            # Train model
            logger.info("Starting training...")
            train_start = time.time()
            model.fit(self.traindata, self.valdata)
            train_time = time.time() - train_start

            # Test model
            logger.info("Running predictions...")
            pred_start = time.time()
            predictions = model.predict(self.testdata)
            pred_time = time.time() - pred_start

            # Evaluate
            evaluator = plib.load_evaluator("RMSE")
            overall_rmse = evaluator.evaluate(predictions, self.testdata)

            total_time = time.time() - start_time

            results = {
                'model_type': 'CFY Enhanced LPM',
                'model': model,
                'predictions': predictions,
                'overall_rmse': overall_rmse,
                'param_count': param_count,
                'train_time': train_time,
                'pred_time': pred_time,
                'total_time': total_time,
                'cfy_enabled': getattr(model, 'cfy_enabled', False),
                'success': True
            }

            logger.info(f"CFY Enhanced LPM Results:")
            logger.info(f"  RMSE: {overall_rmse:.4f}")
            logger.info(f"  Parameters: {param_count:,}")
            logger.info(f"  Train time: {train_time:.2f}s")
            logger.info(f"  Pred time: {pred_time:.2f}s")
            logger.info(f"  CFY enabled: {getattr(model, 'cfy_enabled', False)}")

            return results

        except Exception as e:
            logger.error(f"CFY Enhanced LPM training failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'model_type': 'CFY Enhanced LPM',
                'success': False,
                'error': str(e),
                'total_time': time.time() - start_time
            }

    def train_cfy_plugin_only_from_backbone(self) -> Dict[str, Any]:
        """Load a trained baseline backbone and train only CFY plugin parameters."""
        logger.info("\n" + "="*50)
        logger.info("Training CFY Plugin-Only from Backbone")
        logger.info("="*50)

        start_time = time.time()

        try:
            cache_paths = self._get_param_cache_paths()

            # 1) Get baseline backbone source (reuse in-memory baseline result when available).
            if self._baseline_backbone_result is not None and self._baseline_backbone_result.get('success', False):
                backbone_result = self._baseline_backbone_result
                logger.info("Using in-memory cached baseline backbone for plugin-only training")
            else:
                backbone_result = self.train_baseline_lpm()
            if not backbone_result.get('success', False):
                raise RuntimeError(f"Backbone training failed: {backbone_result.get('error', 'unknown error')}")
            backbone_model = backbone_result['model']

            # 2) Initialize CFY model in plugin-only mode.
            model_args = self.get_base_model_args(include_cfy_params=True)
            model_args.update({
                'cfy_enabled': True,
                'cfy_train_plugin_only': True,
                'num_classes': 5,
                'freeze_backbone_for_dual': True,
                **self.cfy_head_training_params,
            })
            model = CFYEnhancedLPM(**model_args)

            # 3) Copy backbone parameters from trained baseline.
            load_info = model.load_state_dict(backbone_model.state_dict(), strict=False)
            if load_info is not None and hasattr(load_info, 'missing_keys') and hasattr(load_info, 'unexpected_keys'):
                logger.info(
                    "Loaded baseline backbone into plugin model | missing=%s unexpected=%s",
                    len(load_info.missing_keys),
                    len(load_info.unexpected_keys),
                )
            else:
                logger.info("Loaded baseline backbone into plugin model (state_dict return info unavailable)")

            # Keep plugin-only freezing after loading.
            model._freeze_backbone_for_plugin_only_training()

            # 3.5) Optionally preload cached CFY head params for this dataset.
            if self._can_load_head() and not self.force_retrain_head and cache_paths["cfy_head"].exists():
                head_payload = self._load_local_torch_cache(cache_paths["cfy_head"])
                cached_params = head_payload.get("cfy_head_training_params")
                cached_signature = head_payload.get("head_signature")
                current_signature = self._extract_cfy_head_signature(model)
                params_match = isinstance(cached_params, dict) and cached_params == self.cfy_head_training_params
                signature_match = cached_signature == current_signature
                if params_match and signature_match:
                    loaded = self._load_partial_state_dict(model, head_payload.get("head_state_dict", {}))
                    logger.info(
                        "Loaded cached CFY head params from %s | tensors=%s",
                        cache_paths["cfy_head"],
                        loaded,
                    )
                elif params_match and not signature_match:
                    logger.info(
                        "Skip loading cached CFY head from %s because head signature changed.",
                        cache_paths["cfy_head"],
                    )
                else:
                    logger.info(
                        "Skip loading cached CFY head from %s because tuning params changed.",
                        cache_paths["cfy_head"],
                    )

            # Count parameters with detailed breakdown
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            # Count CFY-specific parameters
            cfy_params = sum(p.numel() for p in model.get_cfy_parameters())

            # Count embedding parameters
            embedding_params = 0
            if hasattr(model, 'context_embedding_layer') and model.context_embedding_layer is not None:
                embedding_params += sum(p.numel() for p in model.context_embedding_layer.parameters())
            if hasattr(model, 'perturb_embedding_layer') and model.perturb_embedding_layer is not None:
                embedding_params += sum(p.numel() for p in model.perturb_embedding_layer.parameters())
            if hasattr(model, 'readout_embedding_layer') and model.readout_embedding_layer is not None:
                embedding_params += sum(p.numel() for p in model.readout_embedding_layer.parameters())

            # Count predictor parameters
            predictor_params = 0
            if hasattr(model, 'predictor'):
                predictor_params = sum(p.numel() for p in model.predictor.parameters())

            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"  - Embedding parameters: {embedding_params:,} (frozen)")
            logger.info(f"  - Predictor parameters: {predictor_params:,} (frozen)")
            logger.info(f"  - CFY plugin parameters: {cfy_params:,} (trainable)")
            logger.info(f"  - Other parameters: {total_params - embedding_params - predictor_params - cfy_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")

            # 4) Train plugin-only model.
            logger.info("Starting plugin-only training...")
            train_start = time.time()
            model.fit(self.traindata, self.valdata)
            train_time = time.time() - train_start

            logger.info("Running predictions...")
            pred_start = time.time()
            predictions = model.predict(self.testdata)
            pred_time = time.time() - pred_start

            evaluator = plib.load_evaluator("RMSE")
            overall_rmse = evaluator.evaluate(predictions, self.testdata)
            total_time = time.time() - start_time

            results = {
                'model_type': 'CFY Plugin-Only (Loaded Backbone)',
                'model': model,
                'predictions': predictions,
                'overall_rmse': overall_rmse,
                'param_count': total_params,
                'trainable_param_count': trainable_params,
                'cfy_param_count': cfy_params,
                'embedding_params': embedding_params,
                'predictor_params': predictor_params,
                'train_time': train_time,
                'pred_time': pred_time,
                'total_time': total_time,
                'success': True,
            }

            logger.info("CFY Plugin-Only Results:")
            logger.info(f"  RMSE: {overall_rmse:.4f}")
            logger.info(f"  Total Parameters: {total_params:,}")
            logger.info(f"  Trainable (CFY) Parameters: {trainable_params:,}")
            logger.info(f"  Train time: {train_time:.2f}s")
            logger.info(f"  Pred time: {pred_time:.2f}s")

            if self.save_param_cache:
                head_payload = {
                    "dataset_tag": self.dataset_tag,
                    "context": list(self.context),
                    "saved_at": time.time(),
                    "cfy_head_training_params": self.cfy_head_training_params,
                    "head_signature": self._extract_cfy_head_signature(model),
                    "head_state_dict": self._extract_cfy_head_state_dict(model),
                }
                torch.save(head_payload, cache_paths["cfy_head"])
                logger.info("Saved CFY head cache to %s", cache_paths["cfy_head"])

            return results

        except Exception as e:
            logger.error(f"CFY plugin-only training failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'model_type': 'CFY Plugin-Only (Loaded Backbone)',
                'success': False,
                'error': str(e),
                'total_time': time.time() - start_time
            }

    def train_baseline_lpm(self) -> Dict[str, Any]:
        """Train a baseline LPM model for comparison."""
        logger.info("\\n" + "="*50)
        logger.info("Training Baseline LPM Model")
        logger.info("="*50)

        start_time = time.time()

        try:
            cache_paths = self._get_param_cache_paths()

            # Get model arguments
            model_args = self.get_base_model_args()

            # Create standard LPM model
            model = LPM(**model_args)
            logger.info(f"Created {type(model).__name__}")

            # Count parameters with detailed breakdown
            total_params = sum(p.numel() for p in model.parameters())

            # Count embedding parameters
            embedding_params = 0
            if hasattr(model, 'context_embedding_layer') and model.context_embedding_layer is not None:
                embedding_params += sum(p.numel() for p in model.context_embedding_layer.parameters())
            if hasattr(model, 'perturb_embedding_layer') and model.perturb_embedding_layer is not None:
                embedding_params += sum(p.numel() for p in model.perturb_embedding_layer.parameters())
            if hasattr(model, 'readout_embedding_layer') and model.readout_embedding_layer is not None:
                embedding_params += sum(p.numel() for p in model.readout_embedding_layer.parameters())

            # Count predictor parameters
            predictor_params = 0
            if hasattr(model, 'predictor'):
                predictor_params = sum(p.numel() for p in model.predictor.parameters())

            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"  - Embedding parameters: {embedding_params:,}")
            logger.info(f"  - Predictor parameters: {predictor_params:,}")
            logger.info(f"  - Other parameters: {total_params - embedding_params - predictor_params:,}")

            loaded_from_cache = False
            if self._can_load_baseline() and not self.force_retrain_baseline and cache_paths["baseline"].exists():
                try:
                    payload = self._load_local_torch_cache(cache_paths["baseline"])
                    load_info = model.load_state_dict(payload.get("state_dict", {}), strict=False)
                    if getattr(model, "vocab", None) is None:
                        logger.warning(
                            "Cached baseline at %s is missing vocabulary metadata; retraining baseline.",
                            cache_paths["baseline"],
                        )
                        model = LPM(**model_args)
                        # Recalculate parameters after recreating model
                        total_params = sum(p.numel() for p in model.parameters())
                        embedding_params = 0
                        if hasattr(model, 'context_embedding_layer') and model.context_embedding_layer is not None:
                            embedding_params += sum(p.numel() for p in model.context_embedding_layer.parameters())
                        if hasattr(model, 'perturb_embedding_layer') and model.perturb_embedding_layer is not None:
                            embedding_params += sum(p.numel() for p in model.perturb_embedding_layer.parameters())
                        if hasattr(model, 'readout_embedding_layer') and model.readout_embedding_layer is not None:
                            embedding_params += sum(p.numel() for p in model.readout_embedding_layer.parameters())
                        predictor_params = 0
                        if hasattr(model, 'predictor'):
                            predictor_params = sum(p.numel() for p in model.predictor.parameters())
                    else:
                        loaded_from_cache = True
                        logger.info(
                            "Loaded cached baseline params from %s | missing=%s unexpected=%s",
                            cache_paths["baseline"],
                            len(getattr(load_info, "missing_keys", [])),
                            len(getattr(load_info, "unexpected_keys", [])),
                        )
                except Exception as exc:
                    logger.warning(
                        "Failed loading baseline cache from %s, retraining from scratch: %s",
                        cache_paths["baseline"],
                        exc,
                    )
                    model = LPM(**model_args)
                    # Recalculate parameters after recreating model
                    total_params = sum(p.numel() for p in model.parameters())
                    embedding_params = 0
                    if hasattr(model, 'context_embedding_layer') and model.context_embedding_layer is not None:
                        embedding_params += sum(p.numel() for p in model.context_embedding_layer.parameters())
                    if hasattr(model, 'perturb_embedding_layer') and model.perturb_embedding_layer is not None:
                        embedding_params += sum(p.numel() for p in model.perturb_embedding_layer.parameters())
                    if hasattr(model, 'readout_embedding_layer') and model.readout_embedding_layer is not None:
                        embedding_params += sum(p.numel() for p in model.readout_embedding_layer.parameters())
                    predictor_params = 0
                    if hasattr(model, 'predictor'):
                        predictor_params = sum(p.numel() for p in model.predictor.parameters())

            # Train model
            train_start = time.time()
            if loaded_from_cache:
                logger.info("Skipping baseline training because cached params were loaded")
                # Recalculate parameters after loading from cache
                total_params = sum(p.numel() for p in model.parameters())
                embedding_params = 0
                if hasattr(model, 'context_embedding_layer') and model.context_embedding_layer is not None:
                    embedding_params += sum(p.numel() for p in model.context_embedding_layer.parameters())
                if hasattr(model, 'perturb_embedding_layer') and model.perturb_embedding_layer is not None:
                    embedding_params += sum(p.numel() for p in model.perturb_embedding_layer.parameters())
                if hasattr(model, 'readout_embedding_layer') and model.readout_embedding_layer is not None:
                    embedding_params += sum(p.numel() for p in model.readout_embedding_layer.parameters())
                predictor_params = 0
                if hasattr(model, 'predictor'):
                    predictor_params = sum(p.numel() for p in model.predictor.parameters())
            else:
                logger.info("Starting training...")
                model.fit(self.traindata, self.valdata)
            train_time = time.time() - train_start

            # Test model
            logger.info("Running predictions...")
            pred_start = time.time()
            predictions = model.predict(self.testdata)
            pred_time = time.time() - pred_start

            # Evaluate
            evaluator = plib.load_evaluator("RMSE")
            overall_rmse = evaluator.evaluate(predictions, self.testdata)

            total_time = time.time() - start_time

            results = {
                'model_type': 'Baseline LPM',
                'model': model,
                'predictions': predictions,
                'overall_rmse': overall_rmse,
                'param_count': total_params,
                'embedding_params': embedding_params,
                'predictor_params': predictor_params,
                'train_time': train_time,
                'pred_time': pred_time,
                'total_time': total_time,
                'success': True
            }

            logger.info(f"Baseline LPM Results:")
            logger.info(f"  RMSE: {overall_rmse:.4f}")
            logger.info(f"  Total Parameters: {total_params:,}")
            logger.info(f"  Train time: {train_time:.2f}s")
            logger.info(f"  Pred time: {pred_time:.2f}s")
            logger.info(f"  Loaded from cache: {loaded_from_cache}")

            if self.save_param_cache and not loaded_from_cache:
                payload = {
                    "dataset_tag": self.dataset_tag,
                    "context": list(self.context),
                    "saved_at": time.time(),
                    "model_args": model_args,
                    "state_dict": self._extract_tensor_only_state_dict(model),
                }
                torch.save(payload, cache_paths["baseline"])
                logger.info("Saved baseline cache to %s", cache_paths["baseline"])

            self._baseline_backbone_result = results

            return results

        except Exception as e:
            logger.error(f"Baseline LPM training failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'model_type': 'Baseline LPM',
                'success': False,
                'error': str(e),
                'total_time': time.time() - start_time
            }

    def evaluate_per_co_effect_type(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate model performance per co-effect type."""
        if not results['success'] or 'predictions' not in results:
            return {}

        predictions = results['predictions']

        # Check if co_effect_type column exists
        if "co_effect_type" not in self.testdata._data.columns:
            return {}

        logger.info(f"\\nPer Co-Effect Type Evaluation - {results['model_type']}:")

        evaluator = plib.load_evaluator("RMSE")
        test_df = self.testdata._data
        co_effect_results = {}

        co_effect_types = ["Additive", "Synergy", "Buffering", "Opposite", "Other"]

        for ctype in co_effect_types:
            mask = test_df["co_effect_type"] == ctype
            count = mask.sum()

            if count == 0:
                logger.info(f"  {ctype:10s}: no samples in test set")
                co_effect_results[ctype] = {'count': 0, 'rmse': None}
                continue

            # Filter predictions and testdata by co-effect type
            subset_preds = predictions[mask.to_numpy().astype(bool)]

            import copy
            subset_data = copy.copy(self.testdata)
            subset_data._data = test_df.filter(mask)
            if hasattr(subset_data, "_data_pd"):
                subset_data._data_pd = None

            rmse = evaluator.evaluate(subset_preds, subset_data)
            logger.info(f"  {ctype:10s}: RMSE = {rmse:.4f}  (n={count})")

            co_effect_results[ctype] = {'count': count, 'rmse': rmse}

        # Also evaluate single perturbations (no co_effect_type label)
        mask_single = test_df["co_effect_type"].is_null()
        count_single = mask_single.sum()
        if count_single > 0:
            subset_preds = predictions[mask_single.to_numpy().astype(bool)]

            import copy
            subset_data = copy.copy(self.testdata)
            subset_data._data = test_df.filter(mask_single)
            if hasattr(subset_data, "_data_pd"):
                subset_data._data_pd = None

            rmse = evaluator.evaluate(subset_preds, subset_data)
            logger.info(f"  {'Single':10s}: RMSE = {rmse:.4f}  (n={count_single})")

            co_effect_results['Single'] = {'count': count_single, 'rmse': rmse}

        return co_effect_results

    def diagnose_single_alignment(
        self,
        baseline_result: Dict[str, Any],
        plugin_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Diagnose whether single samples stay on baseline path and quantify prediction drift."""
        diagnosis: Dict[str, Any] = {}

        if not baseline_result.get("success") or not plugin_result.get("success"):
            logger.info("Single-route diagnosis skipped because one of baseline/plugin results failed.")
            return diagnosis

        if self.testdata is None or self.testdata._data is None:
            logger.info("Single-route diagnosis skipped because test data is unavailable.")
            return diagnosis

        test_df = self.testdata._data
        required_cols = {"perturbation", "co_effect_type"}
        if not required_cols.issubset(set(test_df.columns)):
            logger.info("Single-route diagnosis skipped because required columns are missing: %s", required_cols)
            return diagnosis

        single_mask = test_df["co_effect_type"].is_null().to_numpy().astype(bool)

        perturb_counts = None
        perturb_series = test_df["perturbation"]
        try:
            # Encoded/list format path.
            perturb_counts = perturb_series.list.len().to_numpy()
        except Exception:
            # String format path (e.g., "A+B" / "A").
            try:
                perturb_as_str = [str(x) for x in perturb_series.to_list()]
                perturb_counts = np.asarray([len(s.split("+")) for s in perturb_as_str], dtype=np.int64)
            except Exception as exc:
                logger.warning(
                    "Single-route perturbation count parsing failed; routing overlap stats skipped: %s",
                    exc,
                )

        if perturb_counts is not None:
            is_dual = perturb_counts == 2
            is_other = ~is_dual
        else:
            # Keep prediction-diff diagnostics even if routing inference fails.
            is_dual = np.zeros_like(single_mask, dtype=bool)
            is_other = np.zeros_like(single_mask, dtype=bool)

        single_total = int(single_mask.sum())
        single_dual = int((single_mask & is_dual).sum())
        single_other = int((single_mask & is_other).sum())

        diagnosis.update(
            {
                "single_total": single_total,
                "single_routed_other": single_other,
                "single_routed_dual": single_dual,
            }
        )

        logger.info("\nSingle Route Diagnosis:")
        logger.info("  single(total): %s", single_total)
        if perturb_counts is not None:
            logger.info("  single routed as non-dual(other): %s", single_other)
            logger.info("  single routed as dual: %s", single_dual)
        else:
            logger.info("  single routing overlap unavailable (could not infer perturbation counts).")

        baseline_preds = baseline_result.get("predictions")
        plugin_preds = plugin_result.get("predictions")
        if baseline_preds is None or plugin_preds is None:
            logger.info("  prediction alignment check skipped (missing predictions).")
            return diagnosis

        baseline_preds = np.asarray(baseline_preds)
        plugin_preds = np.asarray(plugin_preds)
        if baseline_preds.shape != plugin_preds.shape:
            logger.info(
                "  prediction alignment check skipped (shape mismatch): baseline=%s plugin=%s",
                baseline_preds.shape,
                plugin_preds.shape,
            )
            return diagnosis

        if single_total == 0:
            logger.info("  prediction alignment check skipped (no single samples in test set).")
            return diagnosis

        diff = np.abs(plugin_preds[single_mask] - baseline_preds[single_mask])
        mean_abs = float(np.mean(diff))
        max_abs = float(np.max(diff))
        q95_abs = float(np.quantile(diff, 0.95))

        diagnosis.update(
            {
                "single_pred_abs_diff_mean": mean_abs,
                "single_pred_abs_diff_p95": q95_abs,
                "single_pred_abs_diff_max": max_abs,
            }
        )

        logger.info("  single |plugin - baseline| mean: %.6f", mean_abs)
        logger.info("  single |plugin - baseline| p95 : %.6f", q95_abs)
        logger.info("  single |plugin - baseline| max : %.6f", max_abs)

        return diagnosis

    def run_comparison(self) -> Dict[str, Any]:
        """Run the full training comparison."""
        logger.info("="*60)
        logger.info("Starting Training Comparison")
        logger.info("="*60)

        # Load and prepare data
        self.load_and_prepare_data()

        if len(self.testdata) == 0:
            logger.error("No test data available. Cannot proceed with comparison.")
            return {'error': 'No test data available'}

        # Train all models
        models_to_test = [
            ('baseline_lmp', self.train_baseline_lpm),
            # ('original_lpm_cfy', self.train_original_lpm_cfy),
            # ('cfy_enhanced_lpm', self.train_cfy_enhanced_lpm),
            ('cfy_plugin_only_from_backbone', self.train_cfy_plugin_only_from_backbone),
        ]

        comparison_results = {}

        for model_name, train_func in models_to_test:
            logger.info(f"\\nTesting {model_name}...")
            try:
                results = train_func()

                # Add per co-effect type evaluation
                if results['success']:
                    co_effect_results = self.evaluate_per_co_effect_type(results)
                    results['co_effect_results'] = co_effect_results

                comparison_results[model_name] = results

            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                comparison_results[model_name] = {
                    'model_type': model_name,
                    'success': False,
                    'error': str(e)
                }

        baseline_key = 'baseline_lmp'
        plugin_key = 'cfy_plugin_only_from_backbone'
        if baseline_key in comparison_results and plugin_key in comparison_results:
            diagnosis = self.diagnose_single_alignment(
                comparison_results[baseline_key],
                comparison_results[plugin_key],
            )
            if diagnosis:
                comparison_results[plugin_key]['single_route_diagnosis'] = diagnosis

        # Generate summary
        self.generate_comparison_summary(comparison_results)

        return comparison_results

    def generate_comparison_summary(self, results: Dict[str, Any]):
        """Generate and display comparison summary."""
        logger.info("\\n" + "="*60)
        logger.info("TRAINING COMPARISON SUMMARY")
        logger.info("="*60)

        # Create comparison table
        successful_results = {k: v for k, v in results.items() if v.get('success', False)}

        if not successful_results:
            logger.info("No models trained successfully.")
            return

        # Performance comparison
        logger.info("\\nPerformance Comparison:")
        logger.info(f"{'Model':<25} {'RMSE':<8} {'Params':<10} {'Train(s)':<10} {'Pred(s)':<8}")
        logger.info("-" * 65)

        for model_name, result in successful_results.items():
            logger.info(
                f"{result['model_type']:<25} "
                f"{result['overall_rmse']:<8.4f} "
                f"{result['param_count']:<10,} "
                f"{result['train_time']:<10.2f} "
                f"{result['pred_time']:<8.3f}"
            )

        # Find best model
        best_rmse = min(r['overall_rmse'] for r in successful_results.values())
        best_model = next(r for r in successful_results.values() if r['overall_rmse'] == best_rmse)
        logger.info(f"\\nBest overall RMSE: {best_model['model_type']} ({best_rmse:.4f})")

        # Co-effect type performance
        logger.info("\\nPer Co-Effect Type Performance:")

        co_effect_types = ["Single", "Additive", "Synergy", "Buffering", "Opposite", "Other"]
        header = f"{'Model':<25}"
        for ctype in co_effect_types:
            header += f" {ctype:<10}"
        logger.info(header)
        logger.info("-" * (25 + 11 * len(co_effect_types)))

        for model_name, result in successful_results.items():
            if 'co_effect_results' not in result:
                continue

            line = f"{result['model_type']:<25}"
            for ctype in co_effect_types:
                if ctype in result['co_effect_results']:
                    rmse = result['co_effect_results'][ctype]['rmse']
                    if rmse is not None:
                        line += f" {rmse:<10.4f}"
                    else:
                        line += f" {'N/A':<10}"
                else:
                    line += f" {'-':<10}"
            logger.info(line)

        # Additional insights
        logger.info("\\nKey Insights:")

        # Compare parameter counts
        if len(successful_results) >= 2:
            param_counts = [(r['model_type'], r['param_count']) for r in successful_results.values()]
            param_counts.sort(key=lambda x: x[1])

            logger.info(f"  Parameter efficiency: {param_counts[0][0]} is most parameter-efficient")

            if any('CFY' in r['model_type'] for r in successful_results.values()):
                cfy_results = [r for r in successful_results.values() if 'CFY' in r['model_type']]
                baseline_results = [r for r in successful_results.values() if 'CFY' not in r['model_type']]

                if cfy_results and baseline_results:
                    cfy_rmse = cfy_results[0]['overall_rmse']
                    baseline_rmse = baseline_results[0]['overall_rmse']
                    improvement = ((baseline_rmse - cfy_rmse) / baseline_rmse) * 100

                    if improvement > 0:
                        logger.info(f"  CFY models show {improvement:.1f}% improvement over baseline")
                    else:
                        logger.info(f"  CFY models show {-improvement:.1f}% performance decrease vs baseline")


def main():
    """Main function to run the training comparison."""
    # Configuration
    # context = ["HumanCellLine_K562_GrowthScreen_Horlbeck18"]  # Start with single context
    # context = ["HumanCellLine_Jurkat_GrowthScreen_Horlbeck18"]  # Start with single context
    context = ["HumanCellLine_K562_OverexpressionPerturbSeq_Norman19"]
    # context = ["HumanCellLine_HepG2_PerturbSeq_Nadig25"]
    # context = ["HumanCellLine_Jurkat_PerturbSeq_Nadig25"]
    # context = ["HumanCellLine_A172_ChemGeneticPerturb_HPRT1_MMR_McFaline23"]
    # context = ["HumanCellLine_Glioblastoma_CRISPRiKinome_McFaline23"]

    # Training parameters optimized for class imbalance
    max_epochs = 5  # Increased from 10 to allow more learning
    batch_size = 1000  # Reduced from 5000 to better handle class imbalance
    learning_rate = 0.005  # Increased from 0.002 for smaller batches

    logger.info("Starting CFY Plugin Training Comparison")
    logger.info(f"Context: {context}")
    logger.info(f"Max epochs: {max_epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    cfy_enable_oversampling = os.getenv("PLIB_CFY_OVERSAMPLING", "0").strip() == "1"
    cfy_oversampling_strategy = os.getenv("PLIB_CFY_OVERSAMPLING_STRATEGY", "inverse").strip()
    cfy_oversampling_beta = float(os.getenv("PLIB_CFY_OVERSAMPLING_BETA", "0.9999").strip())

    logger.info(
        "Oversampling: %s (strategy=%s, beta=%.4f)",
        "ENABLED" if cfy_enable_oversampling else "DISABLED",
        cfy_oversampling_strategy,
        cfy_oversampling_beta,
    )

    # Create comparison instance
    comparison = TrainingComparison(
        context=context,
        max_epochs=max_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        cfy_enable_oversampling=cfy_enable_oversampling,
        cfy_oversampling_strategy=cfy_oversampling_strategy,
        cfy_oversampling_beta=cfy_oversampling_beta,
    )

    # Run comparison
    results = comparison.run_comparison()

    # Save results if needed
    results_file = Path("training_comparison_results.txt")
    with open(results_file, 'w') as f:
        f.write("Training Comparison Results\n")
        f.write("="*50 + "\n")
        for model_name, result in results.items():
            f.write(f"\n{model_name}: {result}\n")

    logger.info(f"\nResults saved to {results_file}")
    logger.info("Training comparison completed!")

    return results


if __name__ == "__main__":
    results = main()



