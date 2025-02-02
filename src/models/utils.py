"""
Accurate Graph-based Multi-Positive Unlabeled Learning via Disentangled Multi-view Feature Propagation
Authors:
- Junghun Kim (bandalg97@snu.ac.kr), Seoul National University
- Ka Hyun Park (kahyunpark@snu.ac.kr), Seoul National University
- Hoyoung Yoon (crazy8597@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University
"""

"""
This file includes utility functions.
"""


import numpy as np
import torch


def compute_edge_weights(features, edge_index, normalize=True):
    src, tgt = edge_index

    if normalize:
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True)
        features = (features - mean) / (std + 1e-6)

    edge_weights = torch.sum(features[src] * features[tgt], dim=1)

    return edge_weights

def normalize_edge_weights(*edge_weights):
    stacked_weights = torch.stack(edge_weights, dim=0)
    normalized_edge_weights = torch.softmax(stacked_weights, dim=0)
    normalized_edge_weights = [weight for weight in normalized_edge_weights]

    return normalized_edge_weights

def compute_prior(trn_labels, num_class=2):
    prior = []
    count = np.bincount(trn_labels, minlength=int(num_class))
    for i in range(len(count)):
        prior.append(count[i] / len(trn_labels))
    return prior

def estimate_prior(trn_nodes, predictions, num_class, trn_labels):
    expected_trn_labels = torch.argmax(predictions, dim=1).float()
    expected_trn_labels[trn_labels != 0] = trn_labels[trn_labels != 0]

    prior_labels = expected_trn_labels.cpu().numpy()
    prior_labels = torch.from_numpy(np.delete(prior_labels, trn_nodes))
    prior = compute_prior(prior_labels, num_class=num_class)

    # expected_trn_labels = torch.argmax(predictions, dim=1)
    return prior, expected_trn_labels