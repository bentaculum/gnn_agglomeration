import logging
import torch
from torch.nn import Parameter
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax

from .attention_mlp import AttentionMLP

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class OurConv(MessagePassing):
    # TODO adapt docu
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 dim,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True,
                 normalize_with_softmax=True,
                 local_layers=1,
                 local_hidden_dims=None,
                 non_linearity='leaky_relu',
                 att_use_node_features=False,
                 attention_nn_params=None):
        super(OurConv, self).__init__('add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.normalize_with_softmax = normalize_with_softmax

        self.non_linearity = non_linearity
        self.local_layers = local_layers
        self.att_use_node_features = att_use_node_features
        self.weight_list = torch.nn.ParameterList()

        # avoid empty list as default value
        if not local_hidden_dims:
            weight_dims = []
        else:
            weight_dims = local_hidden_dims.copy()

        # TODO possibly use the same weight matrix for all attention heads?
        #  --> also adapt in forward, message
        weight_dims.insert(0, in_channels)
        weight_dims.append(heads * out_channels)

        for i in range(local_layers):
            w = Parameter(torch.Tensor(
                weight_dims[i], weight_dims[i + 1]))
            self.weight_list.append(w)
            logger.debug(f'node layer ({weight_dims[i]}, {weight_dims[i+1]})')

        if att_use_node_features:
            att_in_features = 2 * out_channels + dim
        else:
            att_in_features = dim

        self.att = AttentionMLP(
            heads=self.heads,
            in_features=att_in_features,
            layers=attention_nn_params['layers'],
            layer_dims=attention_nn_params['layer_dims'],
            bias=attention_nn_params['bias'],
            non_linearity=attention_nn_params['non_linearity'],
            batch_norm=attention_nn_params['batch_norm'],
            dropout_probs=attention_nn_params['dropout_probs'],
        )

        # TODO bias per local matmul
        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        for w in self.weight_list:
            torch.nn.init.kaiming_uniform_(w, nonlinearity='leaky_relu')
        torch_geometric.nn.inits.zeros(self.bias)
        self.att.reset_parameters()

    def forward(self, x, edge_index, pseudo):
        """"""
        # add second dimensionality in case pseudo is 1D
        pseudo = pseudo.unsqueeze(-1) if pseudo.dim() == 1 else pseudo
        # add third dimensionality for attention head dimension, at penultimate
        # position
        pseudo = pseudo.unsqueeze(-2)

        # TODO: debatable whether the last dimension should be replicated as
        # well, e.g. to self.out_channels
        pseudo = pseudo.expand(-1, self.heads, -1)

        # TODO can I use torch.nn.Linear here?
        # TODO separate after the first layer
        # TODO plot the inner happenings
        for i in range(self.local_layers):
            x = torch.mm(x, self.weight_list[i])
            x = getattr(F, self.non_linearity)(x)

        x = x.view(-1, self.heads, self.out_channels)

        return self.propagate(
            edge_index,
            x=x,
            num_nodes=x.size(0),
            pseudo=pseudo)

    def message(self, edge_index_i, x_i, x_j, num_nodes, pseudo):
        # Compute attention coefficients
        # TODO we should be able to speed this up if we don't pass x_i and x_j to this function
        if self.att_use_node_features:
            alpha = torch.cat([x_i, x_j, pseudo], dim=-1)
        else:
            alpha = pseudo

        alpha = self.att(alpha)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        if self.normalize_with_softmax:
            alpha = softmax(alpha, edge_index_i, num_nodes)

        # Dropout on attention vector
        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
