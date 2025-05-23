from typing import Iterable
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer


class Mapper:
    _algo_to_func = {
        "embedding": nn.Embedding,
        "linear": nn.Linear,
        "flatten": nn.Flatten,
        "batchnorm1d": nn.BatchNorm1d,
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "softmax": nn.Softmax,
        "tanh": nn.Tanh,
        "dropout": nn.Dropout,
    }

    _init_to_func = {
        "xavier_uniform": nn.init.xavier_uniform_,
        "kaiming_uniform": nn.init.kaiming_uniform_,
    }

    _optim_to_func = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "sgd": optim.SGD,
    }

    def __init__(self, layers: list[dict], optimizer: dict):
        self.layers = layers
        self.optimizer = optimizer

    @staticmethod
    def _unpack_func_and_args(k_to_args: dict, k_to_func: dict) -> tuple:
        return next(((k_to_func[k], v) for k, v in k_to_args.items() if k in k_to_func), (None, None))

    @classmethod
    def _to_layer(cls, layer: dict) -> nn.Module:
        layer_func, layer_args = cls._unpack_func_and_args(layer, cls._algo_to_func)
        if layer_func:
            nn_layer: nn.Module = layer_func(**layer_args)

            init_func, init_args = cls._unpack_func_and_args(layer, cls._init_to_func)
            if init_func and nn_layer.weight is not None:
                init_func(nn_layer.weight, **init_args)

            confidence: float = layer.get("confidence")
            if confidence is not None:
                with torch.no_grad():
                    nn_layer.weight *= confidence

            return nn_layer
        else:
            raise ValueError(f"Unsupported layer: {layer}")


    def to_layers(self) -> list[nn.Module]:
        return [self._to_layer(l) for l in self.layers]

    def to_optimizer(self, params: Iterable[Tensor]) -> Optimizer:
        optim_func, optim_args = self._unpack_func_and_args(self.optimizer, self._optim_to_func)
        if optim_func:
            if "betas" in optim_args:
                optim_args |= {"betas": tuple(optim_args["betas"])}
            return optim_func(params, **optim_args)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")
