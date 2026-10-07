"""Training utilities for the final nine-dimensional TFGNAS model."""
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

def _scatter_sum(values: torch.Tensor, indices: torch.Tensor, size: int) -> torch.Tensor:
    output = values.new_zeros(size)
    output.scatter_add_(0, indices, values)
    return output


def structural_features(graph, bipartite_graph, ) -> torch.Tensor:
    """Build selectable, label-free structural descriptors for every gene."""
    groups = ("one_hop", "two_hop", "gene_cell")
    out_degree = graph.out_degrees().float()
    in_degree = graph.in_degrees().float()
    total_degree = out_degree + in_degree
    columns = []

    if "one_hop" in groups:
        columns.extend([out_degree, in_degree, total_degree])

    if "two_hop" in groups:
        source, target = graph.edges()
        # Directed two-hop path counts, computed sparsely without materializing A^2.
        outgoing_two_hop = _scatter_sum(out_degree[target], source, graph.num_nodes())
        incoming_two_hop = _scatter_sum(in_degree[source], target, graph.num_nodes())
        columns.extend([
            outgoing_two_hop,
            incoming_two_hop,
            outgoing_two_hop + incoming_two_hop,
        ])

    if "gene_cell" in groups:
        etype = ("cell", "have", "gene")
        cell_ids, gene_ids = bipartite_graph.edges(etype=etype)
        cell_degree = bipartite_graph.out_degrees(etype=etype).float()
        gene_cell_degree = bipartite_graph.in_degrees(etype=etype).float()
        incident_sum = _scatter_sum(cell_degree[cell_ids], gene_ids, graph.num_nodes())
        incident_sq_sum = _scatter_sum(cell_degree[cell_ids].square(), gene_ids, graph.num_nodes())
        denominator = gene_cell_degree.clamp_min(1.0)
        incident_mean = incident_sum / denominator
        incident_variance = (incident_sq_sum / denominator - incident_mean.square()).clamp_min(0.0)
        columns.extend([gene_cell_degree, incident_mean, incident_variance.sqrt()])

    features = torch.stack([torch.log1p(column) for column in columns], dim=1)
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (features - mean) / std


def tf_node_ids(data_e) -> list:
    ids = []
    for tf_edges in data_e.gene_key_datas:
        if len(tf_edges) == 0:
            continue
        tf_name = str(tf_edges[0][0]).upper()
        if data_e.gene_to_name:
            tf_name = data_e.gene_to_name[tf_name]
        ids.append(data_e.gene_to_idx[tf_name])
    return sorted(set(ids))


@torch.no_grad()
def evaluate(model, graph, bipartite_graph, samples, labels):
    model.eval()
    logits = model(graph, bipartite_graph, samples)
    loss = F.binary_cross_entropy_with_logits(logits, labels).item()
    scores = logits.cpu().numpy()
    truth = labels.cpu().numpy()
    auc_value = roc_auc_score(truth, scores)
    precision, recall, _ = precision_recall_curve(truth, scores, pos_label=1)
    ap_value = auc(recall, precision)
    accuracy = ((logits > 0.5) == labels).float().mean().item()
    return accuracy, auc_value, ap_value, loss, logits.detach()


def _architecture_parameters(model):
    """Parameters that generate TF-specific operation probabilities."""
    modules = [
        model.base_encoder, model.structure_encoder,
        model.structure_fusion,
    ]
    params = [
        model.conv_prototypes, model.bi_prototypes,
    ]
    for module in modules:
        params.extend(module.parameters())
    return params


def optimization_step(model, optimizer, graph, bipartite_graph, samples, labels):
    model.train()
    optimizer.zero_grad()
    logits = model(graph, bipartite_graph, samples)
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    optimizer.step()
    return loss
