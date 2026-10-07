import torch
import dgl.nn.pytorch as dglnn
import torch.nn as nn


def conv_map(conv_type, input_dim, hidden_dim):
    if conv_type == 'SAGEConv':
        conv_layer = dglnn.SAGEConv(input_dim, hidden_dim, aggregator_type='mean')
    elif conv_type == 'SAGEv2Conv':
        conv_layer = dglnn.SAGEConv(input_dim, hidden_dim, aggregator_type='pool')
    elif conv_type == 'GCNConv':
        conv_layer = dglnn.SAGEConv(input_dim, hidden_dim, aggregator_type='gcn')
    elif conv_type == 'GATConv':
        heads = 2
        conv_layer = dglnn.GATConv(input_dim, hidden_dim, num_heads=heads, allow_zero_in_degree=True)
    elif conv_type == 'GINConv':
        apply_func = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        conv_layer = dglnn.GINConv(apply_func)
    elif conv_type == 'GraphConv':
        conv_layer = dglnn.GraphConv(input_dim, hidden_dim, allow_zero_in_degree=True)
    elif conv_type == 'GATv2Conv':
        heads = 2
        conv_layer = dglnn.GATv2Conv(input_dim, hidden_dim, num_heads=heads, allow_zero_in_degree=True)
    elif conv_type == 'TAGConv':
        conv_layer = dglnn.TAGConv(input_dim, hidden_dim)
    elif conv_type == 'EdgeConv':
        conv_layer = dglnn.EdgeConv(input_dim, hidden_dim, allow_zero_in_degree=True)
    else:
        raise Exception('Wrong convolution function!')
    return conv_layer


def bi_conv_map(bi_conv_type, input_dim, hidden_dim):
    if bi_conv_type == 'BiSAGEConv':
        bi_conv_layer = dglnn.HeteroGraphConv({
            ('gene', 'have', 'cell'): dglnn.SAGEConv(input_dim, hidden_dim * 2, aggregator_type='mean'),
            ('cell', 'have', 'gene'): dglnn.SAGEConv(input_dim, hidden_dim, aggregator_type='mean'),
        })
    elif bi_conv_type == 'BiSAGEv2Conv':
        bi_conv_layer = dglnn.HeteroGraphConv({
            ('gene', 'have', 'cell'): dglnn.SAGEConv(input_dim, hidden_dim * 2, aggregator_type='pool'),
            ('cell', 'have', 'gene'): dglnn.SAGEConv(input_dim, hidden_dim, aggregator_type='pool'),
        })
    elif bi_conv_type == 'BiGCNConv':
        bi_conv_layer = dglnn.HeteroGraphConv({
            ('gene', 'have', 'cell'): dglnn.SAGEConv(input_dim, hidden_dim * 2, aggregator_type='gcn'),
            ('cell', 'have', 'gene'): dglnn.SAGEConv(input_dim, hidden_dim, aggregator_type='gcn'),
        })
    elif bi_conv_type == 'BiGATConv':
        heads = 2
        bi_conv_layer = dglnn.HeteroGraphConv({
            ('gene', 'have', 'cell'): dglnn.GATConv(input_dim, hidden_dim * 2, num_heads=heads, allow_zero_in_degree=True),
            ('cell', 'have', 'gene'): dglnn.GATConv(input_dim, hidden_dim, num_heads=heads, allow_zero_in_degree=True),
        })
    elif bi_conv_type == 'BiGINConv':
        apply_func_gc = nn.Sequential(nn.Linear(input_dim, hidden_dim * 2), nn.ReLU(), nn.Linear(hidden_dim * 2, hidden_dim * 2))
        apply_func_cg = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        bi_conv_layer = dglnn.HeteroGraphConv({
            ('gene', 'have', 'cell'): dglnn.GINConv(apply_func_gc),
            ('cell', 'have', 'gene'): dglnn.GINConv(apply_func_cg),
        })
    elif bi_conv_type == 'BiGraphConv':
        bi_conv_layer = dglnn.HeteroGraphConv({
            ('gene', 'have', 'cell'): dglnn.GraphConv(input_dim, hidden_dim * 2, allow_zero_in_degree=True),
            ('cell', 'have', 'gene'): dglnn.GraphConv(input_dim, hidden_dim, allow_zero_in_degree=True),
        })
    elif bi_conv_type == 'BiNoneConv':
        bi_conv_layer = None
    else:
        raise Exception('Wrong bipartite convolution function!')
    return bi_conv_layer


def act_map(name):
    assert name == 'relu'
    return torch.nn.functional.relu


class fusion_map(nn.Module):
    def __init__(self, fusion_type, hidden_dim, out_dim):
        super().__init__()
        assert fusion_type == 'concat'
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, src_emb, dst_emb):
        return self.mlp(torch.cat([src_emb, dst_emb], dim=1))

