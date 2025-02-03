"""
Accurate Graph-based Multi-Positive Unlabeled Learning via Disentangled Multi-view Feature Propagation
"""

"""
This file includes loss function for the model.
"""


from torch import nn, sparse
import torch
import numpy as np


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, predictions, labels):
        labels = labels.type(torch.long)
        loss = self.loss(predictions, labels.to(predictions.device))
        return loss

class InferenceModel(nn.Module):
    def __init__(self, edges, potential=0.95, threshold=1e-6, max_iters=100, num_class=2):
        super().__init__()

        if isinstance(edges, np.ndarray):
            values = torch.ones(edges.shape[0])
            edges = sparse.FloatTensor(torch.from_numpy(edges).t(), values)
        self.threshold = threshold
        self.max_iters = max_iters
        self.softmax = nn.Softmax(dim=1)

        indices = edges.coalesce().indices()
        self.src_nodes = nn.Parameter(indices[0, :], requires_grad=False)
        self.dst_nodes = nn.Parameter(indices[1, :], requires_grad=False)
        self.num_nodes = edges.size(0)

        # noinspection PyProtectedMember
        self.num_edges = edges._nnz() // 2
        self.rev_edges = nn.Parameter(self.set_rev_edges(edges), requires_grad=False)
        self.potential = nn.Parameter(torch.full([num_class, num_class],
                                                 fill_value=(1 - potential) / num_class * (num_class - 1)),
                                      requires_grad=False)
        for i in range(num_class):
            self.potential[i, i] = potential / num_class

    def set_rev_edges(self, edges):
        degrees = sparse.mm(edges, torch.ones([self.num_nodes, 1])).view(-1).int()
        zero = torch.zeros(1, dtype=torch.int64)
        indices = torch.cat([zero, degrees.cumsum(dim=0)[:-1]])
        counts = torch.zeros(self.num_nodes, dtype=torch.int64)
        rev_edges = torch.zeros(2 * self.num_edges, dtype=torch.int64)
        edge_idx = 0
        for dst, degree in enumerate(degrees):
            for _ in range(degree):
                src = self.dst_nodes[edge_idx]
                rev_edges[indices[src] + counts[src]] = edge_idx
                edge_idx += 1
                counts[src] += 1
        return rev_edges

    def update_messages(self, messages, beliefs):
        new_beliefs = beliefs[self.src_nodes]
        rev_messages = messages[self.rev_edges]
        new_msgs = torch.mm(new_beliefs / rev_messages, self.potential)
        return new_msgs / new_msgs.sum(dim=1, keepdim=True)

    def compute_beliefs(self, priors, messages):
        beliefs = priors.log()
        beliefs.index_add_(0, self.dst_nodes, messages.log())
        return self.softmax(beliefs)

    def forward(self, priors, num_class=2):
        beliefs = priors
        messages = torch.full([self.num_edges * num_class, num_class], fill_value=0.5, device=priors.device)
        for _ in range(self.max_iters):
            old_beliefs = beliefs
            messages = self.update_messages(messages, beliefs)
            beliefs = self.compute_beliefs(priors, messages)
            diff = (beliefs - old_beliefs).abs().max()
            if diff < self.threshold:
                break
        return beliefs

class Hope_loss(nn.Module):
    def __init__(self, edges, priors, potential=0.9, recompute=False, exp_labels=None, trn_labels=None, num_class=4):
        super().__init__()
        if isinstance(priors, float):
            self.pi = priors
            assert exp_labels is not None

        if recompute:
            priors = self.to_priors(exp_labels, trn_labels, priors, num_class)
            model = InferenceModel(edges, potential, num_class=num_class)
            self.marginals = nn.Parameter(model(priors, num_class), requires_grad=False)
        else:
            priors = self.to_initial_priors(trn_labels, num_class)
            self.marginals = priors

        self.loss = CrossEntropyLoss()
        self.priors = priors

    @staticmethod
    def to_priors(exp_labels, trn_labels, prior, num_class):
        num_nodes = exp_labels.size(0)
        priors = torch.zeros(num_nodes, num_class, device=exp_labels.device)
        priors[exp_labels == 0, :] = (1-prior[0])/(num_class-1)
        priors[exp_labels == 0, 0] = prior[0]

        for i in range(1, num_class):
            priors[exp_labels == i, :] = prior[i]/(num_class-1)
            priors[exp_labels == i, i] = 1 - prior[i]
            priors[trn_labels == i, :] = 0
            priors[trn_labels == i, i] = 1
        return priors

    @staticmethod
    def to_initial_priors(labels, num_class):
        num_nodes = labels.size(0)
        priors = torch.zeros(num_nodes, num_class, device=labels.device)
        for i in range(num_class):
            priors[labels == i, :] = 0
            priors[labels == i, i] = 1
        return priors

    def forward(self, predictions, labels):
        all_nodes = torch.arange(predictions.size(0), device=predictions.device)
        pos_nodes = all_nodes[labels != 0]
        unl_nodes = all_nodes[labels == 0].cpu()

        r_hat_p = self.loss(predictions[pos_nodes], labels[pos_nodes]).mean()
        predictions = nn.functional.log_softmax(predictions, dim=1)
        r_hat_u = -(predictions[unl_nodes] * self.marginals[unl_nodes].to(predictions.device)).mean()

        return r_hat_p + r_hat_u

class ContrastiveLoss(nn.Module):
    def __init__(self, priors, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.priors = priors
        self.margin = margin

    def forward(self, embeddings, labels, edges):
        edges = edges.to(labels.device)

        # Define positive and negative classes
        positive_classes = torch.tensor([1, 2, 3], device=labels.device)
        negative_class = 0

        # Positive pairs: both nodes in positive classes
        pos_mask = (labels[edges[0]].unsqueeze(1) == positive_classes).any(dim=1) & \
                   (labels[edges[1]].unsqueeze(1) == positive_classes).any(dim=1)

        # Negative pairs: one node in positive classes, the other in negative class
        neg_mask = ((labels[edges[0]].unsqueeze(1) == positive_classes).any(dim=1) & (
                    labels[edges[1]] == negative_class)) | \
                   ((labels[edges[1]].unsqueeze(1) == positive_classes).any(dim=1) & (
                               labels[edges[0]] == negative_class))

        pos_pairs = edges[:, pos_mask]
        neg_pairs = edges[:, neg_mask]

        # Balance positive and negative pairs
        num_pos_pairs = pos_pairs.size(1)
        num_neg_pairs = neg_pairs.size(1)

        perm = torch.randperm(num_neg_pairs)[:num_pos_pairs]
        neg_pairs = neg_pairs[:, perm]

        pos_distances = torch.norm(embeddings[pos_pairs[0]] - embeddings[pos_pairs[1]], dim=1)
        neg_distances = torch.norm(embeddings[neg_pairs[0]] - embeddings[neg_pairs[1]], dim=1)

        pos_loss = torch.mean(pos_distances ** 2)
        neg_loss = torch.mean(torch.clamp(self.margin - neg_distances, min=0) ** 2)

        return 0.5*(pos_loss + neg_loss)

class Hope_loss_with_contrastive(Hope_loss):
    def __init__(self, edges, priors, potential=0.9, recompute=False, exp_labels=None, trn_labels=None, num_class=4, margin=1.0):
        super().__init__(edges, priors, potential, recompute, exp_labels, trn_labels, num_class)
        self.contrastive_loss = ContrastiveLoss(priors, margin=margin)
        self.edges = torch.tensor(edges.T)

    def forward(self, predictions, labels, embeddings):
        all_nodes = torch.arange(predictions.size(0), device=predictions.device)
        pos_nodes = all_nodes[labels != 0]
        unl_nodes = all_nodes[labels == 0].cpu()

        r_hat_p = self.loss(predictions[pos_nodes], labels[pos_nodes]).mean()
        predictions = nn.functional.log_softmax(predictions, dim=1)
        r_hat_u = -(predictions[unl_nodes] * self.marginals[unl_nodes].to(predictions.device)).mean()

        # Contrastive loss
        contrastive_loss = self.contrastive_loss(embeddings, labels, self.edges)

        return r_hat_p + r_hat_u + contrastive_loss
