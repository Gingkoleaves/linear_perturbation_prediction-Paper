"""
LPM Model Adaptors for CFY Plugin

This module provides concrete adaptors that enable CFY (Classify then Forward Yield)
functionality for LPM models. These adaptors implement the abstract methods from
CFYMixinBase to handle LPM-specific data formats and model integration.

Copyright (C) 2025 Contributors
Licensed under the Apache License, Version 2.0
"""

from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
import logging

from .base import CFYMixinBase, CFYPluginMixin


class BaselineEmbeddingCFYAdaptor(CFYMixinBase):
    """CFY adaptor for baseline-output embeddings.

    Use when you already have:
    - backbone/baseline features (pre-regression embeddings) per sample
    - perturbation pair embeddings (p1, p2)

    This adaptor produces a scalar prediction via CFYPluginMixin.cfy_forward()
    without changing cfy_plugin internals.

    Expected batch_data keys:
    - 'base_features': Tensor[batch, base_dim]
    - 'p1_emb': Tensor[batch, emb_dim]
    - 'p2_emb': Tensor[batch, emb_dim]

    Optional keys:
    - 'base_predictions': Tensor[batch] or Tensor[batch,1]
    - 'is_dual_mask': BoolTensor[batch] (defaults to all True)
    """

    def __init__(
        self,
        base_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int = 5,
        dropout: float = 0.0,
        activation: str = "relu",
        **kwargs,
    ):
        # For this adaptor, the CFY classifier consumes [base_features, p1, p2]
        input_dim = int(base_dim) + (2 * int(embedding_dim))
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            activation=activation,
            **kwargs,
        )
        self.base_dim = int(base_dim)
        self.embedding_dim = int(embedding_dim)

    def extract_interaction_features(self, batch_data: Dict[str, Tensor]) -> Tensor:
        if 'base_features' not in batch_data:
            raise ValueError("Expected 'base_features' in batch_data")
        base_features = batch_data['base_features']
        if base_features.dim() != 2 or base_features.shape[1] != self.base_dim:
            raise ValueError(f"base_features must be [B,{self.base_dim}] but got {tuple(base_features.shape)}")

        if 'p1_emb' not in batch_data or 'p2_emb' not in batch_data:
            raise ValueError("Expected 'p1_emb' and 'p2_emb' in batch_data")
        p1 = batch_data['p1_emb']
        p2 = batch_data['p2_emb']
        if p1.dim() != 2 or p2.dim() != 2:
            raise ValueError("p1_emb/p2_emb must be 2D tensors")
        if p1.shape != p2.shape:
            raise ValueError("p1_emb and p2_emb must have the same shape")
        if p1.shape[1] != self.embedding_dim:
            raise ValueError(f"p1_emb dim must be {self.embedding_dim}")

        return torch.cat([base_features, p1, p2], dim=1)

    def identify_dual_perturbations(self, batch_data: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        if 'is_dual_mask' in batch_data:
            is_dual = batch_data['is_dual_mask']
            if is_dual.dtype != torch.bool:
                raise ValueError("is_dual_mask must be bool")
        else:
            is_dual = torch.ones(self._get_batch_size(batch_data), device=self._get_device(batch_data), dtype=torch.bool)
        is_other = ~is_dual
        return is_dual, is_other

    def get_perturbation_pair_embeddings(self, batch_data: Dict[str, Tensor]) -> Optional[Tuple[Tensor, Tensor]]:
        if 'p1_emb' in batch_data and 'p2_emb' in batch_data:
            return batch_data['p1_emb'], batch_data['p2_emb']
        return None

    def _get_batch_size(self, batch_data: Dict[str, Tensor]) -> int:
        if 'base_features' in batch_data:
            return int(batch_data['base_features'].shape[0])
        if 'p1_emb' in batch_data:
            return int(batch_data['p1_emb'].shape[0])
        raise ValueError("Could not determine batch size")

    def _get_device(self, batch_data: Dict[str, Tensor]) -> torch.device:
        if 'base_features' in batch_data:
            return batch_data['base_features'].device
        if 'p1_emb' in batch_data:
            return batch_data['p1_emb'].device
        raise ValueError("Could not determine device")

    def forward(self, batch_data: Dict[str, Tensor]) -> Tensor:
        features = self.extract_interaction_features(batch_data)
        pair = self.get_perturbation_pair_embeddings(batch_data)
        if pair is None:
            raise RuntimeError("Missing perturbation pair embeddings")

        base_preds = batch_data.get('base_predictions', None)
        if base_preds is None:
            # If not provided, default to zeros (lets experts learn absolute output)
            base_preds = torch.zeros(self._get_batch_size(batch_data), device=self._get_device(batch_data))
        if isinstance(base_preds, Tensor) and base_preds.dim() == 2 and base_preds.shape[1] == 1:
            base_preds = base_preds.view(-1)

        # cfy_forward internally concatenates p1/p2 again, so we pass ONLY the base_features here.
        out = self.cfy_forward(
            batch_data['base_features'],
            perturb_pair=pair,
            base_predictions=base_preds,
            return_intermediate=self.training,
        )
        if isinstance(out, dict):
            return out['predictions']
        return out


logger = logging.getLogger(__name__)


class LPMCFYAdaptor(CFYMixinBase):
    """
    CFY adaptor specifically designed for LPM (Large Perturbation Model) architectures.

    This adaptor handles the LPM-specific data formats, embedding structures,
    and forward/backward integration patterns.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int = 5,
        dropout: float = 0.0,
        activation: str = "relu",
        **kwargs
    ):
        """
        Initialize LPM CFY adaptor.

        Args:
            embedding_dim: Dimension of embeddings from LPM
            hidden_dim: Hidden layer dimension
            num_classes: Number of interaction classes (4 or 5)
            dropout: Dropout probability
            activation: Activation function name
        """
        # Fixed semantic feature layout: [context, p1, p2, p1*p2, readout]
        input_dim = 5 * embedding_dim

        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            activation=activation,
            **kwargs
        )

        self.embedding_dim = embedding_dim

        logger.info(f"LPM CFY Adaptor initialized: embedding_dim={embedding_dim}, input_dim={input_dim}")

    def extract_interaction_features(self, batch_data: Dict[str, Tensor]) -> Tensor:
        """
        Extract interaction features from LPM batch data for dual perturbations.

        Args:
            batch_data: LPM batch data containing tensors for context, perturbation, readout

        Returns:
            Tensor: Interaction features of shape (batch_size, input_dim)
        """
        # Get embeddings - assumes these are already computed by the base LPM model
        if 'embedded_contexts' in batch_data:
            embedded_contexts = batch_data['embedded_contexts']
            embedded_readouts = batch_data['embedded_readouts']
        else:
            raise ValueError("Expected embedded contexts and readouts in batch_data")

        # For dual perturbations, we need individual perturbation embeddings
        if 'perturbation_flat' in batch_data and 'perturbation_offset' in batch_data:
            p1_embeddings, p2_embeddings = self._extract_dual_perturbation_embeddings(batch_data)
        else:
            raise ValueError("Expected perturbation_flat and perturbation_offset for dual perturbations")

        interaction = p1_embeddings * p2_embeddings
        features = torch.cat([
            embedded_contexts,
            p1_embeddings,
            p2_embeddings,
            interaction,
            embedded_readouts
        ], dim=1)

        return features

    def _extract_dual_perturbation_embeddings(
        self,
        batch_data: Dict[str, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """
        Extract individual perturbation embeddings for dual perturbations from flat/offset format.

        Args:
            batch_data: Batch data containing perturbation_flat and perturbation_offset

        Returns:
            Tuple[Tensor, Tensor]: (p1_embeddings, p2_embeddings)
        """
        perturbation_flat = batch_data['perturbation_flat']
        perturbation_offset = batch_data['perturbation_offset']

        # Get perturbation embedding layer from batch_data or model.
        if 'perturb_embedding_layer' in batch_data:
            perturb_embedding_layer = batch_data['perturb_embedding_layer']
        elif hasattr(self, 'perturb_embedding_layer'):
            perturb_embedding_layer = self.perturb_embedding_layer
        else:
            raise ValueError("Perturbation embedding layer not available")

        # Identify dual perturbations
        counts = self._get_perturbation_counts(batch_data)
        is_dual = counts == 2
        dual_indices = torch.where(is_dual)[0]

        if len(dual_indices) == 0:
            # No dual perturbations in this batch
            batch_size = len(perturbation_offset)
            device = perturbation_flat.device
            dummy = torch.zeros(batch_size, self.embedding_dim, device=device)
            return dummy, dummy

        # Extract dual perturbation indices
        dual_offsets = perturbation_offset[dual_indices]
        p1_indices = perturbation_flat[dual_offsets]
        p2_indices = perturbation_flat[dual_offsets + 1]

        # Get embeddings through the same perturbation representation path used
        # by the backbone (supports both plain EmbeddingBag and custom wrappers
        # such as LPM_API attention/LLM embedding modules).
        p1_emb = self._lookup_perturb_embeddings(
            perturb_embedding_layer=perturb_embedding_layer,
            perturb_indices=p1_indices,
            batch_data=batch_data,
        )
        p2_emb = self._lookup_perturb_embeddings(
            perturb_embedding_layer=perturb_embedding_layer,
            perturb_indices=p2_indices,
            batch_data=batch_data,
        )

        # Expand to full batch size (fill non-dual with zeros)
        batch_size = len(perturbation_offset)
        device = p1_emb.device

        p1_full = torch.zeros(batch_size, self.embedding_dim, device=device)
        p2_full = torch.zeros(batch_size, self.embedding_dim, device=device)

        p1_full[dual_indices] = p1_emb
        p2_full[dual_indices] = p2_emb

        return p1_full, p2_full

    def _lookup_perturb_embeddings(
        self,
        perturb_embedding_layer: Any,
        perturb_indices: Tensor,
        batch_data: Dict[str, Tensor],
    ) -> Tensor:
        """Lookup per-perturb embeddings using the backbone's embedding stack."""
        if hasattr(perturb_embedding_layer, 'weight'):
            # nn.Embedding / nn.EmbeddingBag-like path.
            embeds = perturb_embedding_layer.weight[perturb_indices]
        elif hasattr(perturb_embedding_layer, 'embedding_bag') and hasattr(perturb_embedding_layer.embedding_bag, 'weight'):
            # Wrapper modules (e.g., LPM_API FastPerturbationEmbeddingWithAttention).
            embeds = perturb_embedding_layer.embedding_bag.weight[perturb_indices]
        else:
            raise ValueError("Unsupported perturb_embedding_layer type for individual perturb lookup")

        # Apply the same projection used by the backbone if available.
        projection = None
        if 'perturb_projection' in batch_data:
            projection = batch_data['perturb_projection']
        elif hasattr(self, 'perturb_projection'):
            projection = self.perturb_projection

        if projection is not None:
            embeds = projection(embeds)

        return embeds

    def _get_perturbation_counts(self, batch_data: Dict[str, Tensor]) -> Tensor:
        """Compute number of perturbations per sample from flat/offset tensors."""
        offsets = batch_data["perturbation_offset"]
        total_len = batch_data["perturbation_flat"].shape[0]
        next_offsets = torch.cat([offsets[1:], torch.tensor([total_len], device=offsets.device)])
        return next_offsets - offsets

    def identify_dual_perturbations(self, batch_data: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Identify which samples have dual perturbations vs other types.

        Args:
            batch_data: LPM batch data

        Returns:
            Tuple[Tensor, Tensor]: (is_dual_mask, is_other_mask) boolean tensors
        """
        counts = self._get_perturbation_counts(batch_data)
        is_dual = counts == 2
        is_other = ~is_dual
        return is_dual, is_other

    def get_perturbation_pair_embeddings(
        self,
        batch_data: Dict[str, Tensor],
    ) -> Optional[Tuple[Tensor, Tensor]]:
        """Return per-sample perturbation pair embeddings for semantic experts."""
        if 'embedded_perturbs' in batch_data:
            embedded_perturbs = batch_data['embedded_perturbs']
            if isinstance(embedded_perturbs, tuple) and len(embedded_perturbs) == 2:
                return embedded_perturbs

        if 'perturbation_flat' in batch_data and 'perturbation_offset' in batch_data:
            return self._extract_dual_perturbation_embeddings(batch_data)

        return None


class LPMWithCFY(CFYPluginMixin):
    """
    Mixin class that can be added to any LPM model to provide CFY capabilities.

    Usage:
        class MyLPMWithCFY(LPMWithCFY, LPM):
            def __init__(self, *args, **kwargs):
                cfy_config = kwargs.pop('cfy_config', {})
                super().__init__(*args, **kwargs)
                
                # Setup CFY after LPM initialization
                self.setup_cfy(
                    input_dim=5 * self.embedding_dim,  # LPM interaction features
                    hidden_dim=self.hidden_dim
                )
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize LPM with CFY capabilities.

        Note: This should be called AFTER the base LPM model's __init__.
        In multiple inheritance, this class should be the first base class
        to ensure proper initialization order.
        """
        # First initialize CFYPluginMixin
        CFYPluginMixin.__init__(self)

        # Initialize LPM adaptor as None (will be created in setup_cfy)
        self._lpm_adaptor = None

    def setup_cfy(
        self,
        input_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        **cfy_kwargs
    ):
        """
        Setup CFY components for LPM models.

        Args:
            input_dim: Input feature dimension (auto-calculated if None)
            hidden_dim: Hidden dimension (uses LPM's if None)
            cfy_kwargs: Additional CFY options
        """
        # Auto-calculate input_dim for LPM if not specified
        if input_dim is None:
            if hasattr(self, 'embedding_dim'):
                input_dim = 5 * self.embedding_dim  # Standard LPM CFY input
            else:
                raise ValueError("embedding_dim not available, input_dim must be specified")

        if hidden_dim is None:
            hidden_dim = getattr(self, 'hidden_dim', 256)

        # Initialize LPM adaptor WITHOUT creating classifier (it only provides feature extraction)
        # We'll use the classifier from CFYPluginMixin instead to avoid duplication
        self._lpm_adaptor = LPMCFYAdaptor(
            embedding_dim=getattr(self, 'embedding_dim', input_dim // 5),
            hidden_dim=hidden_dim,
            **cfy_kwargs
        )

        # Remove the duplicate classifier from _lpm_adaptor to save parameters
        # The classifier will be created by super().setup_cfy() below
        if hasattr(self._lpm_adaptor, 'classifier'):
            delattr(self._lpm_adaptor, 'classifier')

        # Call parent setup - this creates the actual classifier we'll use
        super().setup_cfy(input_dim=input_dim, hidden_dim=hidden_dim, **cfy_kwargs)

    def extract_interaction_features(self, batch_data: Any) -> Tensor:
        """Extract interaction features using LPM adaptor."""
        if self._lpm_adaptor is None:
            raise RuntimeError("CFY not initialized. Call setup_cfy() first.")
        return self._lpm_adaptor.extract_interaction_features(batch_data)

    def identify_dual_perturbations(self, batch_data: Any) -> Tuple[Tensor, Tensor]:
        """Identify dual perturbations using LPM adaptor."""
        if self._lpm_adaptor is None:
            raise RuntimeError("CFY not initialized. Call setup_cfy() first.")
        return self._lpm_adaptor.identify_dual_perturbations(batch_data)

    def get_perturbation_pair_embeddings(
        self,
        batch_data: Any,
    ) -> Optional[Tuple[Tensor, Tensor]]:
        """Get perturbation pair embeddings for semantic experts via LPM adaptor."""
        if self._lpm_adaptor is None:
            raise RuntimeError("CFY not initialized. Call setup_cfy() first.")
        return self._lpm_adaptor.get_perturbation_pair_embeddings(batch_data)

    def _get_batch_size(self, batch_data: Dict[str, Tensor]) -> int:
        """Get batch size from LPM batch data."""
        # Try common LPM tensor keys
        for key in ['context', 'perturbation_offset', 'readout']:
            if key in batch_data:
                return batch_data[key].shape[0]
        raise ValueError("Could not determine batch size from batch_data")

    def _get_device(self, batch_data: Dict[str, Tensor]) -> torch.device:
        """Get device from LPM batch data."""
        for key in ['context', 'perturbation_offset', 'readout']:
            if key in batch_data:
                return batch_data[key].device
        raise ValueError("Could not determine device from batch_data")

    def forward(self, batch_data: Dict[str, Tensor]) -> Tensor:
        """
        Enhanced forward pass with CFY routing.

        Routes dual perturbations through CFY and others through original LPM.
        For dual perturbations, uses the original model's forward process but
        replaces the final regression head (hidden_dim->1) with a classification head.
        """
        if not hasattr(self, '_cfy_initialized') or not self._cfy_initialized:
            raise RuntimeError("CFY is not initialized. Call setup_cfy() before forward().")
        if not hasattr(self, 'embed_tensor_dict'):
            raise RuntimeError("CFY LPM adaptor requires embed_tensor_dict from backbone model")

        embedded_contexts, embedded_perturbs, embedded_readouts = self.embed_tensor_dict(batch_data)
        if not isinstance(embedded_perturbs, Tensor):
            raise RuntimeError("CFY LPM adaptor requires tensor embedded_perturbs")
        if not hasattr(self, 'perturb_embedding_layer'):
            raise RuntimeError("CFY LPM adaptor requires perturb_embedding_layer")
        if 'perturbation_flat' not in batch_data or 'perturbation_offset' not in batch_data:
            raise RuntimeError("CFY LPM adaptor requires perturbation_flat and perturbation_offset")

        enhanced_batch = batch_data.copy()
        enhanced_batch.update({
            'embedded_contexts': embedded_contexts,
            'embedded_perturbs': embedded_perturbs,
            'embedded_readouts': embedded_readouts,
            'perturb_embedding_layer': self.perturb_embedding_layer,
            'perturbation_flat': batch_data['perturbation_flat'],
            'perturbation_offset': batch_data['perturbation_offset'],
        })
        if hasattr(self, 'perturb_projection'):
            enhanced_batch['perturb_projection'] = self.perturb_projection

        return self.hybrid_forward(enhanced_batch, use_original=True)