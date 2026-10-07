"""TFGNAS supernet with gene-graph and gene-cell operation candidates."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from operations import act_map, bi_conv_map, conv_map, fusion_map


CONV_OPS = [
    "SAGEConv", "SAGEv2Conv", "GCNConv", "GATConv", "GINConv",
    "GraphConv", "GATv2Conv", "TAGConv", "EdgeConv",
]
BI_OPS = [
    "BiSAGEConv", "BiSAGEv2Conv", "BiGCNConv", "BiGATConv",
    "BiGINConv", "BiGraphConv", "BiNoneConv",
]


def _collapse_heads(value: torch.Tensor) -> torch.Tensor:
    return value.mean(dim=1) if value.ndim == 3 else value


class BiNoneMLP(nn.Module):
    """Feature-only bipartite alternative with the same inputs as other bi-convs."""

    def __init__(self, input_dim: int, out_gene: int, out_cell: int):
        super().__init__()
        self.gene_mlp = nn.Sequential(
            nn.Linear(input_dim, out_gene), nn.ReLU(), nn.Linear(out_gene, out_gene)
        )
        self.cell_mlp = nn.Sequential(
            nn.Linear(input_dim, out_cell), nn.ReLU(), nn.Linear(out_cell, out_cell)
        )

    def forward(self, graph: dgl.DGLHeteroGraph, inputs: Dict[str, torch.Tensor]):
        del graph
        return {
            "gene": self.gene_mlp(inputs["gene"]),
            "cell": self.cell_mlp(inputs["cell"]),
        }


class MixedCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, binone_impl: str,
                 activation: str, op_normalization: str):
        super().__init__()
        half = hidden_dim // 2
        self.conv_ops = nn.ModuleList([conv_map(name, input_dim, half) for name in CONV_OPS])
        bi_modules = []
        for name in BI_OPS:
            if name != "BiNoneConv":
                bi_modules.append(bi_conv_map(name, input_dim, half))
            else:
                bi_modules.append(BiNoneMLP(input_dim, out_gene=half, out_cell=hidden_dim))
        self.bi_ops = nn.ModuleList(bi_modules)
        self.activation = act_map(activation)
        self.op_normalization = op_normalization

    def _normalize_operation(self, value: torch.Tensor) -> torch.Tensor:
        value = _collapse_heads(value)
        return F.layer_norm(value, (value.shape[-1],))

    def forward(
        self,
        graph: dgl.DGLGraph,
        gene_x: torch.Tensor,
        bipartite_graph: dgl.DGLHeteroGraph,
        bi_x: Dict[str, torch.Tensor],
        conv_weights: torch.Tensor,
        bi_gene_weights: torch.Tensor,
        bi_cell_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        conv_outputs = torch.stack(
            [self._normalize_operation(op(graph, gene_x)) for op in self.conv_ops], dim=1
        )
        conv_mixed = (conv_outputs * conv_weights.unsqueeze(-1)).sum(dim=1)
        conv_mixed = self.activation(conv_mixed)

        bi_outputs = [op(bipartite_graph, bi_x) for op in self.bi_ops]
        bi_gene = torch.stack([self._normalize_operation(out["gene"]) for out in bi_outputs], dim=1)
        bi_cell = torch.stack([self._normalize_operation(out["cell"]) for out in bi_outputs], dim=1)
        bi_gene_mixed = (bi_gene * bi_gene_weights.unsqueeze(-1)).sum(dim=1)
        bi_cell_mixed = (bi_cell * bi_cell_weights.unsqueeze(-1)).sum(dim=1)

        gene_out = torch.cat([conv_mixed, bi_gene_mixed], dim=1)
        return gene_out, {"gene": gene_out, "cell": bi_cell_mixed}


class TFGNASModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_genes: int,
        tf_node_ids: Iterable[int],
        structural_features: torch.Tensor,
    ):
        super().__init__()
        hidden_dim, layers, prototype_dim = 256, 3, 128
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.prototype_dim = prototype_dim
        self.temperature = 1.0
        self.register_buffer("tf_mask", torch.zeros(num_genes, dtype=torch.bool))
        self.tf_mask[list(tf_node_ids)] = True
        self.register_buffer("structural_features", structural_features.float())

        self.base_encoder = nn.Sequential(
            nn.Linear(input_dim, prototype_dim), nn.ReLU(), nn.Linear(prototype_dim, prototype_dim)
        )
        self.structure_encoder = nn.Sequential(
            nn.Linear(structural_features.shape[1], 32), nn.ReLU(), nn.Linear(32, 32)
        )
        self.structure_fusion = nn.Sequential(
            nn.Linear(prototype_dim + 32, prototype_dim), nn.ReLU(),
            nn.Linear(prototype_dim, prototype_dim),
        )
        self.conv_prototypes = nn.Parameter(torch.randn(len(CONV_OPS), prototype_dim) * 0.02)
        self.bi_prototypes = nn.Parameter(torch.randn(len(BI_OPS), prototype_dim) * 0.02)

        cells = []
        dim = input_dim
        for _ in range(layers):
            cells.append(MixedCell(
                dim, hidden_dim, "mlp", "relu", "layernorm"
            ))
            dim = hidden_dim
        self.cells = nn.ModuleList(cells)

        self.src_process = nn.Sequential(
            nn.Linear(hidden_dim * layers, hidden_dim), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dst_process = nn.Sequential(
            nn.Linear(hidden_dim * layers, hidden_dim), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fusion = fusion_map("concat", hidden_dim, 1)

    def tf_representation(self, gene_features: torch.Tensor) -> torch.Tensor:
        base = self.base_encoder(gene_features)
        structure = self.structure_encoder(self.structural_features)
        return self.structure_fusion(torch.cat([base, structure], dim=1))

    def _architecture_probabilities(self, representation: torch.Tensor):
        conv = F.softmax(representation @ self.conv_prototypes.t(), dim=1)
        bi = F.softmax(representation @ self.bi_prototypes.t(), dim=1)
        return conv, bi

    def architecture_probabilities(self, gene_features: torch.Tensor):
        rep = self.tf_representation(gene_features)
        return (*self._architecture_probabilities(rep), rep)

    def forward(self, graph, bipartite_graph, samples, return_arch=False):
        initial_gene = graph.ndata["feat"]
        conv_tf, bi_tf, tf_rep = self.architecture_probabilities(initial_gene)

        # Only TF nodes receive customized weights. Other genes/cells use the
        # mean architecture of the TFs, keeping customization tied to TFs.
        tf_conv_mean = conv_tf[self.tf_mask].mean(dim=0, keepdim=True)
        tf_bi_mean = bi_tf[self.tf_mask].mean(dim=0, keepdim=True)
        conv_weights = tf_conv_mean.expand(initial_gene.shape[0], -1).clone()
        bi_gene_weights = tf_bi_mean.expand(initial_gene.shape[0], -1).clone()
        conv_weights[self.tf_mask] = conv_tf[self.tf_mask]
        bi_gene_weights[self.tf_mask] = bi_tf[self.tf_mask]
        bi_cell_weights = tf_bi_mean.expand(bipartite_graph.num_nodes("cell"), -1)

        gene_x = initial_gene
        bi_x = bipartite_graph.ndata["feat"]
        representations = []
        for cell in self.cells:
            gene_x, bi_x = cell(
                graph, gene_x, bipartite_graph, bi_x,
                conv_weights, bi_gene_weights, bi_cell_weights,
            )
            representations.append(gene_x)
        gene_repr = torch.cat(representations, dim=1)
        src = self.src_process(gene_repr[samples[:, 0]])
        dst = self.dst_process(gene_repr[samples[:, 1]])
        logits = self.fusion(src, dst)
        if return_arch:
            return logits, conv_tf, bi_tf, tf_rep
        return logits
