
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch.conv import SAGEConv

'''
Contains the actual neural network architectures.
Supports GraphSAGE with either the pool,mean,gcn, or lstm aggregator as well as GAT.
The input, output, and intermediate layer sizes can all be specified.
Typically will call init_graph_net and pass along the desired model and hyperparameters.

Also contains the CNN Refinement net which is a very simple 2 layer 3D convolutional neural network.
As input, it expects 8 channels, which are the concatenated 4 input modalities and 4 output logits of the GNN predictions.
'''

class GAT(nn.Module):
    def __init__(self,in_feats,layer_sizes,n_classes,heads,residuals,
                activation=F.elu,feat_drops=0,attn_drops=0,negative_slopes=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.layers.append(GATConv(
            in_feats, layer_sizes[0], heads[0],
            feat_drops, attn_drops, negative_slopes, False, self.activation))
        # hidden layers
        for i in range(1, len(layer_sizes)):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.layers.append(GATConv(
                layer_sizes[i-1] * heads[i-1], layer_sizes[i], heads[i],
                feat_drops, attn_drops, negative_slopes, residuals[i], self.activation))
        # output projection
        self.layers.append(GATConv(
            layer_sizes[-1] * heads[-1], n_classes, 1,
            feat_drops, attn_drops, negative_slopes, False, None))

    def forward(self,G, inputs):
        i = inputs
        for l in range(len(self.layers)-1):
            i = self.layers[l](G, i).flatten(1)
        # output projection
        logits = self.layers[-1](G, i).mean(1)
        return logits

def init_graph_net(model_type,hp):

    if(model_type=='GAT'):
        net = GAT(in_feats=hp.in_feats,layer_sizes=hp.layer_sizes,n_classes=hp.out_classes,
                                heads=hp.gat_heads,residuals=hp.gat_residuals)
    else:
        raise Exception(f"Unknown model type: {model_type}")
    return net




