"""
Accurate Graph-based Multi-Positive Unlabeled Learning via Disentangled Multi-view Feature Propagation
Authors:
- Junghun Kim (bandalg97@snu.ac.kr), Seoul National University
- Ka Hyun Park (kahyunpark@snu.ac.kr), Seoul National University
- Hoyoung Yoon (crazy8597@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University
"""

"""
This file includes functions for training.
"""

import io

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

import torch


def train_model(model, features, edges, labels, test_nodes, loss_func, optimizer, trn_labels, epochs, edge_weights=None, contrastive=True):
    logs = []
    saved_model, best_epoch, best_f1 = io.BytesIO(), -1, -1
    torch.save(model.state_dict(), saved_model)
    best_loss = np.inf

    for epoch in range(epochs + 1):
        model.train()
        if edge_weights == None:
            out = model(features, edges)
        else:
            out = model(features, edges, edge_weights)

        if contrastive:
            embeddings = model.embedding(features, edges, edge_weights)
            loss = loss_func(out, trn_labels, embeddings)
        else:
            loss = loss_func(out, trn_labels)

        if epoch > 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        test_f1, test_acc = evaluate_model(model, features, edges, labels, test_nodes, edge_weights)

        logs.append((epoch, np.round(loss.item(), 4), np.round(test_f1, 4), np.round(test_acc, 4)))
        if loss.item() < best_loss:
            best_epoch = epoch
            best_loss = loss.item()
            saved_model.seek(0)
            torch.save(model.state_dict(), saved_model)

    saved_model.seek(0)
    model.load_state_dict(torch.load(saved_model))

    columns = ['epoch', 'trn_loss', 'test_f1', 'test_acc']
    print(f'Epochs: {logs[best_epoch][0]}, Loss: {logs[best_epoch][1]}, '
          f'Test F1: {logs[best_epoch][2]}, Test Accuracy: {logs[best_epoch][3]}')
    return best_epoch, pd.DataFrame(logs, columns=columns), best_loss, model

def evaluate_model(model, features, edges, labels, test_nodes, edge_weights=None):
    model.eval()
    with torch.no_grad():
        if edge_weights == None:
            out = model(features, edges).cpu()
        else:
            out = model(features, edges, edge_weights).cpu()
        out_labels = torch.argmax(out, dim=1)

    test_f1 = f1_score(labels[test_nodes], out_labels[test_nodes], average='macro')
    test_acc = accuracy_score(labels[test_nodes], out_labels[test_nodes])
    return test_f1, test_acc