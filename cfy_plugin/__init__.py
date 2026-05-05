"""
CFY (Classify then Forward Yield) Plugin for Perturblib

This plugin provides a model-agnostic framework for adding classify-forward-yield
capabilities to any model, with specific adaptors for LPM models.

Key Features:
- Flexible 4 or 5-class semantic interaction modeling (additive/synergy/buffering/opposite[/other])
- Seamless integration with existing model forward/backward processes
- Model-agnostic base classes
- LPM-specific adaptors and mixins
- Gradient-safe implementation to prevent deep iteration issues

Usage Examples:

1. Using the adaptor directly:
    ```python
    from perturb_lib.cfy_plugin import LPMCFYAdaptor

    adaptor = LPMCFYAdaptor(
        embedding_dim=128,
        hidden_dim=256
    )
    ```

2. Creating an LPM model with CFY:
    ```python
    from perturb_lib.cfy_plugin import LPMWithCFY
    from perturb_lib.models.collection.lpm import LPM

    class MyLPMWithCFY(LPMWithCFY, LPM):
        def __init__(self, *args, **kwargs):
            cfy_config = kwargs.pop('cfy_config', {})
            super().__init__(*args, cfy_config=cfy_config, **kwargs)

            # Setup CFY after initialization
            self.setup_cfy()
    ```

3. Using for other model types:
    ```python
    from perturb_lib.cfy_plugin import CFYPluginMixin

    class MyModelWithCFY(CFYPluginMixin, MyBaseModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.setup_cfy(input_dim=my_input_dim, hidden_dim=my_hidden_dim)

        def extract_interaction_features(self, batch_data):
            # Implement for your model's data format
            pass

        def identify_dual_perturbations(self, batch_data):
            # Implement for your model's data format
            pass
    ```

Copyright (C) 2025 Contributors
Licensed under the Apache License, Version 2.0
"""

from .base import CFYMixinBase, CFYPluginMixin
from .adaptors import BaselineEmbeddingCFYAdaptor, LPMCFYAdaptor, LPMWithCFY

__all__ = [
    # Base classes
    'CFYMixinBase',
    'CFYPluginMixin',

    # Adaptors
    'BaselineEmbeddingCFYAdaptor',
    'LPMCFYAdaptor',
    'LPMWithCFY',
]

__version__ = '1.0.0'