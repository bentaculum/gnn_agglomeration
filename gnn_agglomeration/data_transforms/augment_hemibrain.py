import torch_geometric.transforms as T


class AugmentHemibrain():
    """
    Combines multiple pyg transforms to form the full data augmentation for hemibrain data

    Args:
        config (namespace): global configuration namespace
    """

    def __init__(self, config):
        rotations = [T.RandomRotate(180, axis=i) for i in range(3)]
        translation = T.RandomTranslate(config.augment_translate_limit)
        self.transform = T.Compose([*rotations, translation])

    def __call__(self, data):
        return self.transform(data)

    def __repr__(self):
        return self.transform.__repr__()


