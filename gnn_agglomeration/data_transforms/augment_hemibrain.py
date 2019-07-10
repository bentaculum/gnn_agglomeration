import torch_geometric.transforms as T


def augment_hemibrain(config):
    rotations = [T.RandomRotate(180, axis=i) for i in range(3)]
    translation = T.RandomTranslate(config.augment_translate_limit)
    return T.Compose([*rotations, translation])
