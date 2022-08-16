import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class Swish(nn.Module):
    def __init__(self, beta=1):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class GNN_Layer(MessagePassing):
    """
    Parameters
    ----------
    in_features : int
        Dimensionality of input features.
    out_features : int
        Dimensionality of output features.
    """

    def __init__(self, in_features, out_features, hidden_features):
        super(GNN_Layer, self).__init__(node_dim=-2, aggr="mean")

        self.message_net = nn.Sequential(
            nn.Linear(2 * in_features, hidden_features),
            Swish(),
            nn.BatchNorm1d(hidden_features),
            nn.Linear(hidden_features, out_features),
            Swish()
        )
        self.update_net = nn.Sequential(
            nn.Linear(in_features + hidden_features, hidden_features),
            Swish(),
            nn.Linear(hidden_features, out_features),
            Swish()
        )

    def forward(self, x, edge_index, batch):
        """Propagate messages along edges."""
        x = self.propagate(edge_index, x=x)
        # x = self.norm(x, batch)
        return x

    def message(self, x_i, x_j):
        """Message update."""
        message = self.message_net(torch.cat((x_i, x_j), dim=-1))
        return message

    def update(self, message, x):
        """Node update."""
        x += self.update_net(torch.cat((x, message), dim=-1))
        return x


class CongestioNN(torch.nn.Module):
    def __init__(
        self,
        in_features=4,
        out_features=32,
        hidden_features=32,
        hidden_layer=1
    ):

        super(CongestioNN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.hidden_layer = hidden_layer

        # in_features have to be of the same size as out_features for the time being
        modules = []
        for _ in range(self.hidden_layer):
            modules.append(
                GNN_Layer(self.out_features, self.out_features, self.hidden_features)
            )

        self.cgnn = torch.nn.ModuleList(modules=modules)

        self.head_pre_pool = nn.Sequential(
            nn.Linear(self.out_features, self.hidden_features),
            Swish(),
            nn.Linear(self.hidden_features, self.hidden_features)
        )

        self.head_post_pool = nn.Sequential(
            nn.Linear(self.hidden_features, self.hidden_features),
            Swish(),
            nn.Linear(hidden_features, 1)
        )

        self.embedding_mlp = nn.Linear(self.in_features, self.out_features)

    def forward(self, data):
        batch = data.batch
        x = data.x
        edge_index = data.edge_index

        x = self.embedding_mlp(x)
        for i in range(self.hidden_layer):
            x = self.cgnn[i](x, edge_index, batch)

        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.swish = Swish()

        self.dropout = dropout

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = self.swish(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)

        return x
