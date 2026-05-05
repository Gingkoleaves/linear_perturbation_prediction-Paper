"""
CFY (Classify then Forward Yield) Plugin Base Classes

This module provides a pluggable architecture for enhancing any model with
classify-forward-yield capabilities, allowing adaptive MLP dimensions and
seamless integration with existing model forward/backward processes.

Key Design Principles:
1. Model-agnostic: Works with any base model architecture
2. Adaptive MLP: Automatically adjusts layer dimensions based on input parameters
3. Non-invasive integration: Preserves original model's forward/backward flow
4. Gradient-friendly: Ensures no deep iteration issues during backpropagation

Copyright (C) 2025 Contributors
Licensed under the Apache License, Version 2.0
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import logging

logger = logging.getLogger(__name__)

def temperature_softmax(logits, temperature=0.5):
    # temperature < 1 会拉大差距
    return F.softmax(logits / temperature, dim=-1)

class CFYMixinBase(ABC, nn.Module):
    """
    Abstract base class for CFY (Classify then Forward Yield) functionality.

    This mixin can be applied to any existing model to add dual-perturbation
    processing capabilities with adaptive MLP dimensions.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int = 4,
        dropout: float = 0.0,
        activation: str = "relu",
        **kwargs
    ):
        """
        Initialize CFY plugin components.

        Args:
            input_dim: Dimension of input features
            hidden_dim: Base hidden layer dimension
            num_classes: Number of interaction classes (4 or 5 for additive/synergy/buffering/opposite[/other])
            dropout: Dropout probability
            activation: Activation function name
        """
        # Only call super().__init__() if this is the first Module initialization
        # This prevents overriding existing model attributes in multiple inheritance
        if not hasattr(self, '_modules'):
            # This is the first nn.Module initialization
            super().__init__()
        else:
            # Module already initialized, just call nn.Module.__init__ directly
            # to avoid disrupting the module registration system
            pass

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        if num_classes not in [4, 5]:
            raise ValueError("CFY plugin supports 4 or 5 interaction classes: additive/synergy/buffering/opposite[/other]")
        self.num_classes = num_classes
        self.dropout = dropout
        self.plugin_hidden_dim = int(kwargs.get('cfy_plugin_hidden_dim', min(hidden_dim, 128)))
        self.plugin_residual_hidden_dim = int(
            kwargs.get(
                'cfy_plugin_residual_hidden_dim',
                max(min(self.plugin_hidden_dim // 2, 64), 32),
            )
        )
        self.pair_stat_dim = 10

        # Preserve the backbone freeze behavior used in training script.
        self.freeze_backbone_for_dual = bool(kwargs.get('freeze_backbone_for_dual', False))

        # Activation function
        self.activation_name = str(activation)
        self.activation = self._get_activation(activation)

        # Initialize CFY components
        self._build_cfy_components()

        # Class names for interaction types
        self.class_names = ["additive", "synergy", "buffering", "opposite", "other"][:self.num_classes]
        residual_scale_logits = torch.full((self.num_classes,), -2.0)
        residual_scale_logits[0] = -4.0
        if self.num_classes >= 5:
            residual_scale_logits[4] = -2.5
        self.expert_residual_log_scale = nn.Parameter(residual_scale_logits)

        # Optional affine calibration for semantic experts computed from perturbation pairs.
        self.semantic_expert_scale = nn.Parameter(torch.ones(self.num_classes))
        self.semantic_expert_bias = nn.Parameter(torch.zeros(self.num_classes))

        # Cached tensors from the latest dual-perturbation route for optional training losses.
        self._last_dual_mask: Optional[Tensor] = None
        self._last_dual_base_predictions: Optional[Tensor] = None
        self._last_class_logits: Optional[Tensor] = None
        self._last_class_probs: Optional[Tensor] = None
        self._last_expert_outputs: Optional[Tensor] = None
        self._last_semantic_outputs: Optional[Tensor] = None
        self._last_distribution_mean: Optional[Tensor] = None
        self._last_distribution_logvar: Optional[Tensor] = None

    def _get_activation(self, activation_name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.01)
        }
        return activations.get(activation_name.lower(), nn.ReLU())

    def _make_activation_layer(self) -> nn.Module:
        """Create a fresh activation module for Sequential blocks."""
        return self._get_activation(self.activation_name)

    @staticmethod
    def _zero_init_last_linear(module: nn.Module):
        """Zero-init the output layer so the plugin starts from the backbone baseline."""
        last_linear = None
        for child in module.modules():
            if isinstance(child, nn.Linear):
                last_linear = child
        if last_linear is not None:
            nn.init.zeros_(last_linear.weight)
            if last_linear.bias is not None:
                nn.init.zeros_(last_linear.bias)

    def _build_cfy_components(self):
        """Build a baseline-anchored residual MoE for dual perturbations."""
        dropout = self.dropout if self.dropout > 0 else 0.1
        plugin_hidden = max(int(self.plugin_hidden_dim), 32)
        residual_hidden = max(int(self.plugin_residual_hidden_dim), 16)

        self.shared_encoder = nn.Sequential(
            nn.Linear(self.input_dim, plugin_hidden),
            nn.LayerNorm(plugin_hidden),
            self._make_activation_layer(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(plugin_hidden, plugin_hidden),
            nn.LayerNorm(plugin_hidden),
            self._make_activation_layer(),
            nn.Dropout(dropout),
            nn.Linear(plugin_hidden, self.num_classes),
        )
        semantic_hidden = max(self.num_classes * 4, 16)
        self.pair_stat_norm = nn.LayerNorm(self.pair_stat_dim)
        self.semantic_stat_proj = nn.Sequential(
            nn.Linear(self.pair_stat_dim, semantic_hidden),
            nn.LayerNorm(semantic_hidden),
            self._make_activation_layer(),
            nn.Dropout(dropout),
            nn.Linear(semantic_hidden, self.num_classes),
        )
        self.semantic_feature_norm = nn.LayerNorm(self.num_classes)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(plugin_hidden + self.num_classes, residual_hidden),
                    nn.LayerNorm(residual_hidden),
                    self._make_activation_layer(),
                    nn.Dropout(dropout),
                    nn.Linear(residual_hidden, 1),
                )
                for _ in range(self.num_classes)
            ]
        )

        self.shared_encoder.apply(self._init_weights)
        self.classifier.apply(self._init_weights)
        self.semantic_stat_proj.apply(self._init_weights)
        self.experts.apply(self._init_weights)

        for expert in self.experts:
            self._zero_init_last_linear(expert)

        logger.info(
            "CFY compact head dims | input=%s plugin_hidden=%s residual_hidden=%s classes=%s",
            self.input_dim,
            plugin_hidden,
            residual_hidden,
            self.num_classes,
        )

    @staticmethod
    def _init_weights(module):
        """Initialize module weights with Xavier uniform initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @abstractmethod
    def extract_interaction_features(self, batch_data: Any) -> Tensor:
        """
        Extract interaction features from batch data for dual perturbations.

        This method must be implemented by each model adaptor to define how
        to construct the interaction-aware features from the model's input format.

        Args:
            batch_data: Model-specific batch data

        Returns:
            Tensor: Extracted interaction features for CFY processing
        """
        pass

    @abstractmethod
    def identify_dual_perturbations(self, batch_data: Any) -> Tuple[Tensor, Tensor]:
        """
        Identify which samples have dual perturbations vs other types.

        Args:
            batch_data: Model-specific batch data

        Returns:
            Tuple[Tensor, Tensor]: (is_dual_mask, is_other_mask) boolean tensors
        """
        pass

    def get_perturbation_pair_embeddings(
        self,
        batch_data: Any,
    ) -> Optional[Tuple[Tensor, Tensor]]:
        """Return (p1, p2) perturbation pair embeddings if available."""
        return None

    def _compute_semantic_expert_outputs(self, p1_embeddings: Tensor, p2_embeddings: Tensor) -> Tensor:
        """
        Learn compact semantic cues from raw perturbation-pair statistics.

        This replaces hand-weighted class formulas with a small learned projection,
        reducing dependence on manually tuned coefficients while keeping the cue
        extraction lightweight and architecture-driven.
        """
        effect_p1 = (p1_embeddings ** 2).mean(dim=1)
        effect_p2 = (p2_embeddings ** 2).mean(dim=1)
        additive = effect_p1 + effect_p2

        pair_product = p1_embeddings * p2_embeddings
        pos_product = F.relu(pair_product).mean(dim=1)
        neg_product = F.relu(-pair_product).mean(dim=1)
        cross_var = pair_product.var(dim=1, unbiased=False)
        cos_sim = F.cosine_similarity(p1_embeddings, p2_embeddings, dim=1)
        l1_distance = torch.abs(p1_embeddings - p2_embeddings).mean(dim=1)
        euclidean_dist = torch.norm(p1_embeddings - p2_embeddings, dim=1)
        effect_gap = torch.abs(effect_p1 - effect_p2)

        pair_stats = torch.stack(
            [
                additive,
                effect_p1,
                effect_p2,
                pos_product,
                neg_product,
                cross_var,
                cos_sim,
                l1_distance,
                euclidean_dist,
                effect_gap,
            ],
            dim=1,
        )
        semantic_outputs = self.semantic_stat_proj(self.pair_stat_norm(pair_stats))

        scales = self.semantic_expert_scale.unsqueeze(0)
        biases = self.semantic_expert_bias.unsqueeze(0)
        return semantic_outputs * scales + biases

    def _get_router_temperature(self) -> float:
        """Use a soft router early, then anneal toward sharper routing later."""
        start = max(float(getattr(self, 'cfy_router_temperature', 1.0)), 1e-3)
        end = max(float(getattr(self, 'cfy_router_temperature_min', start)), 1e-3)
        warmup_epochs = max(int(getattr(self, 'cfy_router_temperature_warmup_epochs', 0)), 0)

        if warmup_epochs <= 0:
            return end

        current_epoch = max(int(getattr(self, 'current_epoch', 0)), 0)
        progress = min(float(current_epoch) / float(warmup_epochs), 1.0)
        return start + (end - start) * progress

    @staticmethod
    def compute_mmd_loss(
        source: Tensor,
        target: Tensor,
        kernel_scale: float = 1.0,
    ) -> Tensor:
        """Compute unbiased MMD loss for distribution alignment."""
        if source.numel() == 0 or target.numel() == 0:
            return source.new_tensor(0.0)

        x = source.view(source.shape[0], -1)
        y = target.view(target.shape[0], -1)

        xx = torch.cdist(x, x, p=2).pow(2)
        yy = torch.cdist(y, y, p=2).pow(2)
        xy = torch.cdist(x, y, p=2).pow(2)

        gamma = 1.0 / max(kernel_scale * x.shape[1], 1e-6)
        k_xx = torch.exp(-gamma * xx)
        k_yy = torch.exp(-gamma * yy)
        k_xy = torch.exp(-gamma * xy)

        return k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()

    def cfy_forward(
        self,
        classifier_features: Tensor,
        perturb_pair: Optional[Tuple[Tensor, Tensor]] = None,
        base_predictions: Optional[Tensor] = None,
        return_intermediate: bool = False
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Forward pass through the CFY architecture.

        Args:
            classifier_features: Tensor of shape (batch_size, input_dim)
            return_intermediate: Whether to return intermediate outputs for analysis

        Returns:
            Tensor or Dict: Final predictions, optionally with intermediate outputs
        """
        if perturb_pair is None:
            raise RuntimeError("CFY forward requires perturbation pair embeddings (p1, p2)")

        classifier_input = classifier_features
        p1_embeddings, p2_embeddings = perturb_pair
        classifier_input = torch.cat([classifier_features, p1_embeddings, p2_embeddings], dim=1)

        # Enforce initialization-time dimension alignment.
        # The concatenated dual features must match the shared encoder input width;
        # then the shared hidden width must match the classifier input width.
        feature_dim = int(classifier_input.shape[1])
        shared_input_dim = (
            self.shared_encoder[0].in_features
            if isinstance(self.shared_encoder, nn.Sequential)
            else self.shared_encoder.in_features
        )
        if shared_input_dim != feature_dim:
            raise RuntimeError(
                "Shared encoder input dim mismatch: "
                f"shared_encoder expects {shared_input_dim}, got {feature_dim}. "
                "Please ensure setup_cfy inferred predictor pre-regression width correctly."
            )

        shared_hidden = self.shared_encoder(classifier_input)
        hidden_dim = int(shared_hidden.shape[1])
        classifier_input_dim = (
            self.classifier[0].in_features
            if isinstance(self.classifier, nn.Sequential)
            else self.classifier.in_features
        )
        if classifier_input_dim != hidden_dim:
            raise RuntimeError(
                "Classifier hidden dim mismatch: "
                f"classifier expects {classifier_input_dim}, got {hidden_dim}. "
                "Please ensure CFY shared encoder and classifier widths stay aligned."
            )

        # Stage 1: Classification from backbone pre-regression embeddings + p1/p2.
        class_logits = self.classifier(shared_hidden)
        
        """ Optional: use adaptive temperature
        confidence = class_logits.max(dim=1).values
        adaptive_temp = 0.1 + 0.4 * torch.sigmoid(-confidence)  # range ~0.1-0.5, higher temp for less confident samples
        class_probs = temperature_softmax(class_logits, temperature=adaptive_temp)
        """
        
        class_probs = temperature_softmax(class_logits, temperature=self._get_router_temperature())

        # Stage 2: Pair semantics are treated as residual priors rather than hard-coded outputs.
        semantic_outputs = self._compute_semantic_expert_outputs(p1_embeddings, p2_embeddings)
        semantic_features = self.semantic_feature_norm(semantic_outputs)

        if base_predictions is not None:
            additive_base = base_predictions.view(-1)
        else:
            additive_base = semantic_outputs[:, 0]
        expert_input = torch.cat([shared_hidden, semantic_features], dim=1)
        raw_residuals = torch.stack(
            [expert(expert_input).squeeze(-1) for expert in self.experts],
            dim=1,
        )
        residual_scales = F.softplus(self.expert_residual_log_scale).unsqueeze(0)
        bounded_residuals = torch.tanh(raw_residuals) * residual_scales
        expert_outputs = additive_base.unsqueeze(1).repeat(1, self.num_classes) + bounded_residuals
        # Preserve the backbone prediction on the dominant additive route.
        expert_outputs[:, 0] = additive_base

        if self.num_classes > 1:
            additive_prob = class_probs[:, 0].unsqueeze(1)
            non_additive_mass = (1.0 - additive_prob).clamp_min(0.0)
            gate_power = max(float(getattr(self, 'cfy_non_additive_gate_power', 1.0)), 1.0)
            correction_gate = non_additive_mass.pow(gate_power)

            normalized_non_additive_probs = class_probs[:, 1:] / non_additive_mass.clamp_min(1e-8)
            non_additive_residual = (
                normalized_non_additive_probs * bounded_residuals[:, 1:]
            ).sum(dim=1, keepdim=True)
            final_predictions = additive_base.unsqueeze(1) + correction_gate * non_additive_residual
        else:
            final_predictions = additive_base.unsqueeze(1)

        if return_intermediate:
            return {
                'predictions': final_predictions,
                'class_logits': class_logits,
                'class_probs': class_probs,
                'expert_outputs': expert_outputs,
                'semantic_outputs': semantic_outputs,
            }

        return final_predictions

    def get_cfy_parameters(self) -> List[nn.Parameter]:
        """Get all CFY-related parameters for optimization."""
        params = []
        params.extend(list(self.shared_encoder.parameters()))
        params.extend(list(self.classifier.parameters()))
        params.extend(list(self.pair_stat_norm.parameters()))
        params.extend(list(self.semantic_stat_proj.parameters()))
        params.extend(list(self.semantic_feature_norm.parameters()))
        params.extend(list(self.experts.parameters()))
        params.extend([
            self.expert_residual_log_scale,
            self.semantic_expert_scale,
            self.semantic_expert_bias,
        ])
        return params

    def cfy_state_dict(self) -> Dict[str, Any]:
        """Get CFY component state dict for saving/loading."""
        return {
            'shared_encoder': self.shared_encoder.state_dict(),
            'classifier': self.classifier.state_dict(),
            'pair_stat_norm': self.pair_stat_norm.state_dict(),
            'semantic_stat_proj': self.semantic_stat_proj.state_dict(),
            'semantic_feature_norm': self.semantic_feature_norm.state_dict(),
            'experts': self.experts.state_dict(),
            'expert_residual_log_scale': self.expert_residual_log_scale.detach().cpu(),
            'semantic_expert_scale': self.semantic_expert_scale.detach().cpu(),
            'semantic_expert_bias': self.semantic_expert_bias.detach().cpu(),
            'config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'plugin_hidden_dim': self.plugin_hidden_dim,
                'plugin_residual_hidden_dim': self.plugin_residual_hidden_dim,
                'num_classes': self.num_classes,
                'dropout': self.dropout
            }
        }

    def load_cfy_state_dict(self, state_dict: Dict[str, Any]):
        """Load CFY component state dict."""
        if 'shared_encoder' in state_dict:
            self.shared_encoder.load_state_dict(state_dict['shared_encoder'])
        self.classifier.load_state_dict(state_dict['classifier'])
        if 'pair_stat_norm' in state_dict:
            self.pair_stat_norm.load_state_dict(state_dict['pair_stat_norm'])
        if 'semantic_stat_proj' in state_dict:
            self.semantic_stat_proj.load_state_dict(state_dict['semantic_stat_proj'])
        if 'semantic_feature_norm' in state_dict:
            self.semantic_feature_norm.load_state_dict(state_dict['semantic_feature_norm'])
        if 'experts' in state_dict:
            self.experts.load_state_dict(state_dict['experts'])
        if 'expert_residual_log_scale' in state_dict:
            scale = state_dict['expert_residual_log_scale'].to(self.expert_residual_log_scale.device)
            self.expert_residual_log_scale.data.copy_(scale)
        if 'semantic_expert_scale' in state_dict:
            scale = state_dict['semantic_expert_scale'].to(self.semantic_expert_scale.device)
            self.semantic_expert_scale.data.copy_(scale)
        if 'semantic_expert_bias' in state_dict:
            bias = state_dict['semantic_expert_bias'].to(self.semantic_expert_bias.device)
            self.semantic_expert_bias.data.copy_(bias)


class CFYPluginMixin(CFYMixinBase):
    """
    Concrete mixin class that can be added to any existing model to provide CFY capabilities.

    This is designed to be multiply-inherited alongside existing model classes.
    """

    def __init__(self, *args, cfy_config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize CFY plugin. Should be called after the base model's __init__.

        Args:
            cfy_config: Configuration dict for CFY components
        """
        # Extract CFY-specific args from kwargs
        cfy_args = {}
        if cfy_config:
            cfy_args.update(cfy_config)

        # Provide default values for required parameters
        cfy_args.setdefault('hidden_dim', 128)  # Default hidden_dim
        cfy_args.setdefault('dropout', 0.0)
        cfy_args.setdefault('activation', 'relu')
        cfy_args.setdefault('num_classes', 5)  # Default to 5 classes (including Other)

        # Initialize CFY components with a temporary non-zero input dim;
        # setup_cfy will rebuild modules with the real feature dimensionality.
        super().__init__(input_dim=1, **cfy_args)

        self._cfy_initialized = False

    def setup_cfy(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        **cfy_kwargs
    ):
        """
        Setup CFY components after base model initialization.

        Args:
            input_dim: Input feature dimension for CFY processing
            hidden_dim: Hidden dimension (uses base model's if not specified)
        """
        if self._cfy_initialized:
            logger.warning("CFY already initialized, skipping setup")
            return

        # Use base model's hidden_dim if not specified
        if hidden_dim is None and hasattr(self, 'hidden_dim'):
            hidden_dim = getattr(self, 'hidden_dim')
        elif hidden_dim is None:
            raise ValueError("hidden_dim must be specified or available from base model")

        # Prefer predictor pre-regression width as classifier input for better
        # cross-backbone generalization.
        inferred_dim = self._infer_classifier_input_dim_from_predictor()
        if inferred_dim is not None and inferred_dim > 0:
            if input_dim != inferred_dim:
                logger.info(
                    "Overriding CFY classifier input_dim from %s to predictor pre-reg dim %s",
                    input_dim,
                    inferred_dim,
                )
            input_dim = inferred_dim

        # Update dimensions and rebuild components
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self._build_cfy_components()

        self._cfy_initialized = True
        logger.info(f"CFY plugin initialized with input_dim={input_dim}, hidden_dim={hidden_dim}")

    def _infer_classifier_input_dim_from_predictor(self) -> Optional[int]:
        """Infer classifier input width using [pre-regression features, p1, p2]."""
        predictor = getattr(self, 'predictor', None)

        base_dim: Optional[int] = None

        if isinstance(predictor, nn.Linear):
            if predictor.out_features == 1:
                base_dim = int(predictor.in_features)
            else:
                return None

        if isinstance(predictor, nn.Sequential):
            for module in reversed(list(predictor)):
                if isinstance(module, nn.Linear) and module.out_features == 1:
                    base_dim = int(module.in_features)
                    break

        if base_dim is None:
            return None

        embedding_dim = getattr(self, 'embedding_dim', None)
        if embedding_dim is None:
            return base_dim

        return base_dim + (2 * int(embedding_dim))

    def hybrid_forward(self, batch_data: Any, use_original: bool = True) -> Tensor:
        """
        Hybrid forward pass that routes inputs through original model or CFY based on perturbation type.

        For dual perturbations:
        - Uses the original model's embedding and intermediate layers
        - Replaces the final regression head (hidden_dim->1) with classification head

        For non-dual perturbations:
        - Uses the original model's full forward pass

        Args:
            batch_data: Input batch data
            use_original: Whether to use original model for non-dual perturbations

        Returns:
            Tensor: Model predictions
        """
        if not self._cfy_initialized:
            raise RuntimeError("CFY plugin not initialized. Call setup_cfy() first.")

        if not use_original:
            raise RuntimeError("CFY hybrid forward only supports use_original=True")
        if not isinstance(batch_data, dict):
            raise RuntimeError("CFY hybrid forward expects dict-like batch data")
        required_keys = ['embedded_contexts', 'embedded_perturbs', 'embedded_readouts', 'perturbation_flat', 'perturbation_offset']
        missing_keys = [k for k in required_keys if k not in batch_data]
        if missing_keys:
            raise RuntimeError(f"CFY hybrid forward missing required keys: {missing_keys}")
        if not isinstance(batch_data['embedded_perturbs'], Tensor):
            raise RuntimeError("CFY hybrid forward requires embedded_perturbs to be a Tensor")
        if not hasattr(self, 'predictor'):
            raise RuntimeError("CFY hybrid forward requires backbone predictor")

        # Identify perturbation types
        is_dual_mask, is_other_mask = self.identify_dual_perturbations(batch_data)

        # Prepare output tensor
        batch_size = self._get_batch_size(batch_data)
        device = self._get_device(batch_data)
        predictions = torch.zeros(batch_size, 1, device=device)

        # Route non-dual perturbations through original model
        if is_other_mask.any():
            other_features = torch.cat([
                batch_data['embedded_contexts'][is_other_mask],
                batch_data['embedded_perturbs'][is_other_mask],
                batch_data['embedded_readouts'][is_other_mask],
            ], dim=1)
            with torch.no_grad():  # 关键修改：完全隔离梯度
                predictions[is_other_mask] = self.predictor(other_features)

        # Route dual perturbations through CFY (using original model's intermediate features)
        if is_dual_mask.any():
            dual_backbone_features = torch.cat([
                batch_data['embedded_contexts'][is_dual_mask],
                batch_data['embedded_perturbs'][is_dual_mask],
                batch_data['embedded_readouts'][is_dual_mask],
            ], dim=1)
            dual_base_preds, dual_classifier_features = self._run_predictor_with_pre_reg_features(dual_backbone_features)

            # Preserve single-route quality by preventing dual auxiliary heads from
            # updating backbone features when backbone freezing is requested.
            if self.freeze_backbone_for_dual:
                dual_classifier_features = dual_classifier_features.detach()

            # Meta baseline for dual samples: original model prediction.
            if dual_base_preds.dim() > 1:
                dual_base_preds = dual_base_preds.view(-1)
            if self.freeze_backbone_for_dual:
                dual_base_preds = dual_base_preds.detach()
            self._last_dual_base_predictions = dual_base_preds

            perturb_pair = self.get_perturbation_pair_embeddings(batch_data)
            if perturb_pair is None:
                raise RuntimeError("CFY dual route requires perturbation pair embeddings")
            p1_embeddings, p2_embeddings = perturb_pair
            dual_perturb_pair = (p1_embeddings[is_dual_mask], p2_embeddings[is_dual_mask])
            if self.freeze_backbone_for_dual:
                dual_perturb_pair = (dual_perturb_pair[0].detach(), dual_perturb_pair[1].detach())
            
            # Use CFY forward with classification head instead of regression head
            dual_out = self.cfy_forward(
                dual_classifier_features,
                perturb_pair=dual_perturb_pair,
                base_predictions=dual_base_preds,
                return_intermediate=self.training,
            )
            if isinstance(dual_out, dict):
                dual_preds = dual_out['predictions']
                self._last_dual_mask = is_dual_mask.detach()
                self._last_class_logits = dual_out['class_logits']
                self._last_class_probs = dual_out['class_probs']
                self._last_expert_outputs = dual_out['expert_outputs']
                self._last_semantic_outputs = dual_out.get('semantic_outputs')
                self._last_distribution_mean = dual_out.get('distribution_mean')
                self._last_distribution_logvar = dual_out.get('distribution_logvar')
            else:
                dual_preds = dual_out
                self._last_dual_mask = None
                self._last_dual_base_predictions = None
                self._last_class_logits = None
                self._last_class_probs = None
                self._last_expert_outputs = None
                self._last_semantic_outputs = None
                self._last_distribution_mean = None
                self._last_distribution_logvar = None

            predictions[is_dual_mask] = dual_preds
        else:
            self._last_dual_mask = None
            self._last_dual_base_predictions = None
            self._last_class_logits = None
            self._last_class_probs = None
            self._last_expert_outputs = None
            self._last_semantic_outputs = None
            self._last_distribution_mean = None
            self._last_distribution_logvar = None

        return predictions

    def _run_predictor_with_pre_reg_features(
        self,
        feature_batch: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Run predictor on provided features and capture the tensor right before
        the final regression layer.
        """
        predictor = getattr(self, 'predictor', None)
        if predictor is None:
            raise RuntimeError("Backbone predictor is required for CFY routing")

        pre_reg: Optional[Tensor] = None

        # Try to hook the input to the final Linear(out_features=1) layer.
        final_reg_layer: Optional[nn.Linear] = None
        if isinstance(predictor, nn.Linear) and predictor.out_features == 1:
            final_reg_layer = predictor
        elif isinstance(predictor, nn.Sequential):
            for module in reversed(list(predictor)):
                if isinstance(module, nn.Linear) and module.out_features == 1:
                    final_reg_layer = module
                    break
        if final_reg_layer is None:
            raise RuntimeError("Could not locate final regression layer in predictor")

        hook_handle = None
        captured: Dict[str, Tensor] = {}
        if final_reg_layer is not None:
            def _capture_pre_reg(_module: nn.Module, inputs: Tuple[Tensor, ...], _output: Tensor):
                if len(inputs) > 0 and isinstance(inputs[0], Tensor):
                    captured['pre_reg'] = inputs[0]

            hook_handle = final_reg_layer.register_forward_hook(_capture_pre_reg)

        try:
            preds = predictor(feature_batch)
            if 'pre_reg' in captured:
                pre_reg = captured['pre_reg']
            if pre_reg is None:
                raise RuntimeError("Failed to capture pre-regression features from predictor")
            return preds, pre_reg
        finally:
            if hook_handle is not None:
                hook_handle.remove()

    @abstractmethod
    def _get_batch_size(self, batch_data: Any) -> int:
        """Get batch size from batch data (model-specific)."""
        pass

    @abstractmethod
    def _get_device(self, batch_data: Any) -> torch.device:
        """Get device from batch data (model-specific)."""
        pass


"""
Norman19

temp = 0.2:
INFO:__main__:Model                             Single     Additive   Synergy    Buffering  Opposite   Other     
INFO:__main__:-------------------------------------------------------------------------------------------
INFO:__main__:Baseline LPM                      0.0384     0.0250     0.1124     0.0756     0.0425     0.0771    
INFO:__main__:CFY Plugin-Only (Loaded Backbone) 0.0384     0.0312     0.1171     0.0849     0.0515     0.0850    

temp = 0.05:
INFO:__main__:Model                             Single     Additive   Synergy    Buffering  Opposite   Other     
INFO:__main__:-------------------------------------------------------------------------------------------
INFO:__main__:Baseline LPM                      0.0384     0.0250     0.1124     0.0756     0.0425     0.0771    
INFO:__main__:CFY Plugin-Only (Loaded Backbone) 0.0384     0.0305     0.1168     0.0845     0.0500     0.0841    

temp = 0.05, stronger classifier, more similarity signal:
INFO:__main__:Model                             Single     Additive   Synergy    Buffering  Opposite   Other     
INFO:__main__:Baseline LPM                      0.0384     0.0250     0.1124     0.0756     0.0425     0.0771    
INFO:__main__:CFY Plugin-Only (Loaded Backbone) 0.0384     0.0263     0.1138     0.0839     0.0448     0.0808    

+ 过采样 15轮
INFO:__main__:Model                             Single     Additive   Synergy    Buffering  Opposite   Other     
INFO:__main__:-------------------------------------------------------------------------------------------
INFO:__main__:Baseline LPM                      0.0384     0.0250     0.1124     0.0756     0.0425     0.0771    
INFO:__main__:CFY Plugin-Only (Loaded Backbone) 0.0384     0.0269     0.1129     0.0777     0.0451     0.0790   

+ 过采样 8轮
INFO:__main__:Model                             Single     Additive   Synergy    Buffering  Opposite   Other     
INFO:__main__:-------------------------------------------------------------------------------------------
INFO:__main__:Baseline LPM                      0.0384     0.0250     0.1124     0.0756     0.0425     0.0771    
INFO:__main__:CFY Plugin-Only (Loaded Backbone) 0.0384     0.0242     0.1122     0.0764     0.0425     0.0778  

+ 过采样 5轮
INFO:__main__:Model                             Single     Additive   Synergy    Buffering  Opposite   Other     
INFO:__main__:-------------------------------------------------------------------------------------------
INFO:__main__:Baseline LPM                      0.0384     0.0250     0.1124     0.0756     0.0425     0.0771    
INFO:__main__:CFY Plugin-Only (Loaded Backbone) 0.0384     0.0240     0.1122     0.0753     0.0419     0.0767    

+ 过采样 1轮
INFO:__main__:Model                             Single     Additive   Synergy    Buffering  Opposite   Other     
INFO:__main__:-------------------------------------------------------------------------------------------
INFO:__main__:Baseline LPM                      0.0384     0.0250     0.1124     0.0756     0.0425     0.0771    
INFO:__main__:CFY Plugin-Only (Loaded Backbone) 0.0384     0.0250     0.1124     0.0756     0.0425     0.0771 

temp = 0.01:
INFO:__main__:Model                             Single     Additive   Synergy    Buffering  Opposite   Other     
INFO:__main__:-------------------------------------------------------------------------------------------
INFO:__main__:Baseline LPM                      0.0384     0.0250     0.1124     0.0756     0.0425     0.0771    
INFO:__main__:CFY Plugin-Only (Loaded Backbone) 0.0384     0.0298     0.1173     0.0876     0.0512     0.0858    
"""
