"""
Accurate Graph-based Multi-Positive Unlabeled Learning via Disentangled Multi-view Feature Propagation
Authors:
- Junghun Kim (bandalg97@snu.ac.kr), Seoul National University
- Ka Hyun Park (kahyunpark@snu.ac.kr), Seoul National University
- Hoyoung Yoon (crazy8597@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University
"""

"""
This file includes functions and structure for the model.
"""


import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric import nn as gnn
from torch_geometric.utils import add_self_loops, remove_self_loops
import networkx as nx

from .utils import *


def to_adj_matrix(edge_index, num_nodes):
    return torch.sparse_coo_tensor(
        edge_index, torch.ones(edge_index.size(1)), (num_nodes, num_nodes))

def make_structural_features(edge_index, num_nodes, num_features, device):
    adj = to_adj_matrix(edge_index.cpu(), num_nodes).to(device)

    f_norm = (torch.norm(adj) ** 2).cpu().numpy()
    u, s, _ = torch.svd_lowrank(adj, int(num_features/2))

    if np.sum(s.cpu().numpy() ** 2) / f_norm < 0.9:
        return u * s.unsqueeze(0)
    else:
        u, s, _ = torch.svd_lowrank(adj, int(adj.shape[0] * 0.9))
        sarr = s.cpu().numpy() ** 2
        idx = np.where(np.cumsum(sarr) / f_norm >= 0.9)[0][0]
        return u[:, :idx] * s[:idx].unsqueeze(0)

def make_static_features(edge_index, num_nodes, device):
    edges = edge_index.cpu().numpy().T
    graph = nx.Graph()
    graph.add_edges_from(edges)
    graph.add_nodes_from(range(num_nodes))

    # Initialize feature matrix
    features = []

    # Degree (normalized)
    degree = np.array([graph.degree(n) for n in range(num_nodes)])
    normalized_degree = degree / max(degree.max(), 1)
    features.append(normalized_degree)

    # Clustering Coefficient
    clustering_coeff = np.array([nx.clustering(graph, n) for n in range(num_nodes)])
    features.append(clustering_coeff)

    # Betweenness Centrality
    betweenness = nx.betweenness_centrality(graph, normalized=True)
    betweenness_centrality = np.array([betweenness.get(n, 0) for n in range(num_nodes)])
    features.append(betweenness_centrality)

    # Closeness Centrality
    closeness = nx.closeness_centrality(graph)
    closeness_centrality = np.array([closeness.get(n, 0) for n in range(num_nodes)])
    features.append(closeness_centrality)

    # Eigenvector Centrality
    eigenvector = nx.eigenvector_centrality(graph, max_iter=1000)
    eigenvector_centrality = np.array([eigenvector.get(n, 0) for n in range(num_nodes)])
    features.append(eigenvector_centrality)

    # Pagerank
    pagerank = nx.pagerank(graph)
    pagerank_values = np.array([pagerank.get(n, 0) for n in range(num_nodes)])
    features.append(pagerank_values)

    # Shortest Path Length (Average)
    shortest_path_length = []
    for node in range(num_nodes):
        lengths = nx.single_source_shortest_path_length(graph, node)
        avg_length = sum(lengths.values()) / len(lengths) if lengths else 0
        shortest_path_length.append(avg_length)
    features.append(np.array(shortest_path_length))

    # K-core Number
    k_core = nx.core_number(graph)
    k_core_values = np.array([k_core.get(n, 0) for n in range(num_nodes)])
    features.append(k_core_values)

    # Eccentricity
    if nx.is_connected(graph):
        eccentricity = nx.eccentricity(graph)
        eccentricity_values = np.array([eccentricity.get(n, 0) for n in range(num_nodes)])
    else:
        eccentricity_values = np.zeros(num_nodes)
    features.append(eccentricity_values)

    # Concatenate features and convert to tensor
    feature_matrix = np.stack(features, axis=1)
    feature_tensor = torch.tensor(feature_matrix, dtype=torch.float32, device=device)

    return feature_tensor

def make_neighbor_features(edge_index, features, num_nodes, device):
    edge_index, _ = remove_self_loops(edge_index)

    # Compute degree of each node
    row, col = edge_index
    deg = torch.bincount(row, minlength=num_nodes).float()

    # Normalize adjacency matrix (row-normalized)
    deg_inv = 1.0 / deg
    deg_inv[deg == 0] = 0.0
    norm = deg_inv[row]

    # Aggregate features
    propagated_features = torch.zeros_like(features)
    propagated_features.index_add_(0, row, features[col] * norm.unsqueeze(1))

    return propagated_features.to(device)

def generate_edge_weights(model, features, edges):
    with torch.no_grad():
        struct_out, static_out, neighbor_out, neighbor2_out, _ = model.compute_hidden(features, edges)

        struct_edge_weights = compute_edge_weights(struct_out, edges)
        static_edge_weights = compute_edge_weights(static_out, edges)
        neighbor_edge_weights = compute_edge_weights(neighbor_out, edges)
        neighbor2_edge_weights = compute_edge_weights(neighbor2_out, edges)

        struct_edge_weights, static_edge_weights, neighbor_edge_weights, neighbor2_edge_weights = normalize_edge_weights(
            struct_edge_weights, static_edge_weights, neighbor_edge_weights, neighbor2_edge_weights)
        edge_weights = [struct_edge_weights, static_edge_weights, neighbor_edge_weights, neighbor2_edge_weights]

    return edge_weights

class HOPE(nn.Module):
    def __init__(self, num_features, num_class, num_hidden=16, num_layers=1):
        super().__init__()
        self.struct_x = None
        self.static_x = None
        self.neighbor_x = None

        self.feature_layers = []
        self.struct_layers = []
        self.neighbor_layers = []
        self.neighbor2_layers = []
        self.static_layers = []
        self.linear = nn.Linear(num_hidden*5, num_class)

        self.num_features = num_features
        self.num_class = num_class
        self.num_hidden = num_hidden
        self.num_layers = num_layers

    def preprocess(self, x, edge_index, device):
        num_nodes = x.shape[0]
        struct_x = make_structural_features(edge_index, num_nodes, self.num_features, device)
        static_x = make_static_features(edge_index, num_nodes, device)
        neighbor_x = make_neighbor_features(edge_index, x, num_nodes, device)

        self.struct_x = F.normalize(struct_x, p=2, dim=1)
        self.static_x = F.normalize(static_x, p=2, dim=1)
        self.neighbor_x = F.normalize(neighbor_x, p=2, dim=1)
        # self.neighbor_x = neighbor_x

        for i in range(self.num_layers):
            num_inputs_struct = self.struct_x.shape[1] if i == 0 else self.num_hidden
            num_inputs_neighbor = self.neighbor_x.shape[1] if i == 0 else self.num_hidden
            num_inputs_static = self.static_x.shape[1] if i == 0 else self.num_hidden
            num_inputs = self.num_features if i == 0 else self.num_hidden
            self.struct_layers.append(gnn.GCNConv(num_inputs_struct, self.num_hidden, cached=True))
            self.static_layers.append(gnn.GCNConv(num_inputs_static, self.num_hidden, cached=True))
            self.neighbor_layers.append(gnn.GCNConv(num_inputs, self.num_hidden, cached=True))
            self.neighbor2_layers.append(gnn.GCNConv(num_inputs_neighbor, self.num_hidden, cached=True))
            self.feature_layers.append(nn.Linear(num_inputs, self.num_hidden))
        self.struct_layers = nn.ModuleList(self.struct_layers).to(device)
        self.static_layers = nn.ModuleList(self.static_layers).to(device)
        self.neighbor_layers = nn.ModuleList(self.neighbor_layers).to(device)
        self.neighbor2_layers = nn.ModuleList(self.neighbor2_layers).to(device)
        self.feature_layers = nn.ModuleList(self.feature_layers).to(device)

    def compute_hidden(self, x, edge_index, edge_weights=None):
        struct_out = self.struct_x
        static_out = self.static_x
        neighbor_out = x
        neighbor2_out = self.neighbor_x
        feature_out = x

        for i in range(len(self.struct_layers)):
            if edge_weights is not None:
                struct_edge_weights = edge_weights[0]
                static_edge_weights = edge_weights[1]
                neighbor_edge_weights = edge_weights[2]
                neighbor2_edge_weights = edge_weights[3]
            else:
                struct_edge_weights, static_edge_weights, neighbor_edge_weights, neighbor2_edge_weights = (
                    None, None, None, None)
            struct_out = torch.relu(self.struct_layers[i](struct_out, edge_index, struct_edge_weights))
            static_out = torch.relu(self.static_layers[i](static_out, edge_index, static_edge_weights))
            neighbor_out = torch.relu(self.neighbor_layers[i](neighbor_out, edge_index, neighbor_edge_weights))
            neighbor2_out = torch.relu(self.neighbor2_layers[i](neighbor2_out, edge_index, neighbor2_edge_weights))
            feature_out = torch.relu(self.feature_layers[i](feature_out))

        return struct_out, static_out, neighbor_out, neighbor2_out, feature_out

    def embedding(self, x, edge_index, edge_weights=None):
        struct_out, static_out, neighbor_out, neighbor2_out, feature_out = (
            self.compute_hidden(x, edge_index, edge_weights))

        out = torch.cat((struct_out, static_out, neighbor_out, neighbor2_out, feature_out), 1)
        return out

    def forward(self, x, edge_index, edge_weights=None):
        struct_out, static_out, neighbor_out, neighbor2_out, feature_out = (
            self.compute_hidden(x, edge_index, edge_weights))

        out = torch.cat((struct_out, static_out, neighbor_out, neighbor2_out, feature_out), 1)
        return self.linear(out)