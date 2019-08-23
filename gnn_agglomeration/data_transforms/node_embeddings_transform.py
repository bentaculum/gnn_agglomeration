import numbers
from itertools import repeat
import torch


class NodeEmbeddingsTransform:
    """Translates node attributes (data.x) by randomly sampled translation values
    within a given interval.
    Inspired by https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/random_translate.html#RandomTranslate

    Args:
        translate (sequence or float or int): Maximum translation in each dimension

    """

    def __init__(self, translate):
        self.translate = translate

    def __call__(self, data):
        (n, dim), t = data.x.size(), self.translate
        if isinstance(t, numbers.Number):
            t = list(repeat(t, times=dim))
        assert len(t) == dim

        ts = []
        for d in range(dim):
            ts.append(data.x.new_empty(n).uniform_(-abs(t[d]), abs(t[d])))

        data.x = data.x + torch.stack(ts, dim=-1)
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}'
