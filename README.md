# Accurate Graph-based Multi-Positive Unlabeled Learning via Disentangled Multi-view Feature Propagation

This is the code repository for Accurate Graph-based Multi-Positive Unlabeled Learning via Disentangled Multi-view Feature Propagation.
This includes the implementation of D-MVP (**D**istangled **M**ulti-**V**iew Feature **P**ropagation), our novel
approach for graph-based MPU learning.

## Requirements

We recommend using the following versions of packages:
- `python==3.7.13`
- `cuda==11.6`
- `cudnn==8.5.0`
- `pytorch==1.13.1`
- `torch-geometric==2.3.1`

## Code Description
- `models/gnn.py` implements for the D-MVP model.
- `models/loss.py` contains the loss function of D-MVP.
- `models/pgm.py` contains the implementation of conventional MPU loss (GRAB). 
- `models/train.py` contains functions for training the MPU node classifier.
- `models/utils.py` contains utility functions.
- `data.py` contains functions for loading data.
- `main.py` is the main script for training our node classifier graph-based MPU learning.

## Data Overview
|    **Dataset**    |           **Path or Package**            | 
|:-----------------:|:----------------------------------------:| 
|     **Cora**      | `torch_geometric.datasets.Planetoid`     | 
|    **Cora-ML**    | `torch_geometric.datasets.CitationFull`     | 
|   **CiteSeer**    | `torch_goemetric.datasets.Planetoid` | 
| **CiteSeer-full** | `torch_goemetric.datasets.CitationFull` | 
|   **Chameleon**   | `torch_goemetric.datasets.WikipediaNetwork` | 

We load public datasets from the Torch Geometric package.

## How to Run

You can reproduce the experimental results in the paper with the following commands:
```shell
python main.py --data Cora
python main.py --data Cora_ML
python main.py --data CiteSeer
python main.py --data CiteSeer_full
python main.py --data chameleon
```

Hyperparameters for the main script are summarized as follows:
- `gpu`: index of a GPU to use.
- `seed`: a random seed (any integer).
- `data`: name of a dataset.
- `epochs`: number of iterations to train.
- `val-ratio`: ratio of edges to use in validation.
- `test-ratio`: ratio of edges to use in test.
- `verbose`: print details while running the experiment if set to 'y'.
- `early-stop`: patience number for early stop.
- `layers`: number of layers in GCN link predictor.
- `units`: number of units in GCN link predictor.
- `iteration`: number of iterations.