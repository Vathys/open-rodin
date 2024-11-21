from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch.nn as nn

from omegaconf import MISSING


class BaseModel(nn.Module):
    """
    What the child model is expect to declare:
        default_conf: dictionary of the default configuration of the model.
        It recursively updates the default_conf of all parent classes, and
        it is updated by the user-provided configuration passed to __init__.
        Configurations can be nested.

        strict_conf (optional): boolean. If false, BaseModel does not raise
        an error when the user provides an unknown configuration entry.

        _init(self, conf): initialization method, where conf is the final
        configuration object (also accessible with `self.conf`). Accessing
        unknown configuration entries will raise an error.

        _forward(self, data): method that returns a dictionary of batched
        prediction tensors based on a dictionary of batched input data tensors.
    """

    @dataclass
    class BaseModelConfig:
        name: str = MISSING
        trainable: bool = True
        timeit: Optional[bool] = False

    strict_conf = False
    required_keys = []

    are_weights_initialized = False

    def __init__(self, cfg: BaseModelConfig):
        """Perform some logic and call the _init method of the child model."""
        super().__init__()
        self.cfg = cfg

        self._init(cfg)

        if not cfg.trainable:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, data):
        # check if all required keys are present in data

        return self._forward(data)


    @abstractmethod
    def _init(self, cfg: BaseModelConfig):
        raise NotImplementedError("Init method must be implemented")

    @abstractmethod
    def _forward(self, data):
        raise NotImplementedError("Forward method must be implemented")
