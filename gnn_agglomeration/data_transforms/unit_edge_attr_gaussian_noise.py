import torch


class UnitEdgeAttrGaussianNoise:
    """
    Adds noise sampled from isotropic Gaussian to an edge variable in interval (0,1)

    Args:
        mu (float): Mean of noise distribution
        sigma (float): Standard deviation of noise distribution
        dims (list or None): Which edge attrs should be transformed.
            If None, all edge attrs will be used.
    """

    def __init__(self, mu=0.0, sigma=1.0, dims=None):
        self.dims = dims
        self.normal = torch.distributions.normal.Normal(mu, sigma)

    def __call__(self, data):
        if self.dims is not None:
            unit_attr = data.edge_attr[:, self.dims]
        else:
            unit_attr = data.edge_attr

        num_attrs = [unit_attr.size(-1)]

        logits = torch.log(unit_attr / (1 - unit_attr))
        logits += self.normal.sample(num_attrs)
        noisy_attr = 1 / (1 + torch.exp(-logits))

        if self.dims is not None:
            data.edge_attr[:, self.dims] = noisy_attr
        else:
            data.edge_attr = noisy_attr

        return data

    def __repr__(self):
        return f'{self.__class__.__name__}'
