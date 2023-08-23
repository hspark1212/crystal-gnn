import torch


class Normalizer(object):
    """
    normalize for regression
    """

    def __init__(self, mean: float, std: float) -> None:
        if mean is not None and std is not None:
            self._norm_func = lambda tensor: (tensor - mean) / std
            self._denorm_func = lambda tensor: tensor * std + mean
        else:
            self._norm_func = lambda tensor: tensor
            self._denorm_func = lambda tensor: tensor

        self.mean = mean
        self.std = std

    def encode(self, tensor) -> torch.Tensor:
        return self._norm_func(tensor)

    def decode(self, tensor) -> torch.Tensor:
        return self._denorm_func(tensor)
