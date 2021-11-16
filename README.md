## Setup
*Purpose*: to compare performance on the `ogbnmag` dataset of R-GCN implementations across PyG and DGL by having identical models and hyperparameters.  Using the [OGB Leaderboard implementation](https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/mag/sampler.py) as a reference, this is the following architecture/settings:

- Minibatch training with a batch size of 1024
- 3 training epochs 
- `Paper` nodes have 128 dimensional word-count features, remaining node types have 128-dimensional learnable embeddings
- Two R-GCN layers
    - Each relation has a dedicated linear projection matrix
    - Self-loops are treated as another relation type (i.e., has its own linear layer)
    - For a given neighbor nodetype, the "average" aggregation function is used
    - The final node embedding is a sum across all the relation-specific embeddings 
- Neighborhood sampling for each layer with values of `[25, 20]`
- Hidden size of 64
- Adam optimizer with `lr=0.01` and `wd=0.0`
- Dropout of 0.5 on both R-GCN layers
- `ReLU` activation on 1st R-GCN layer
- `log_softmax` on final R-GCN layer with `NLLLoss`


## Reference implementations
The PyG implementation from the OGB leaderboard is found here: https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/mag/sampler.py

I made small adjustments to the DGL example found here: https://github.com/dmlc/dgl/blob/master/examples/pytorch/rgcn-hetero/model.py
These modifications, meant to match the Leaderboard architecture/parameters, are in the `rgcn_hetero_dgl.py` file.  Some examples of changes required to match PyG architecture:

- Model/training code needed adapted to handle fact that "paper" nodes have natural features while all other node-types require learnable node embeddings 
  - E.g., RelGraphEmbed needed to exclude having embeddings for "paper" node types, since they have natural features
- The input RelGraphConv layer needed `weight=True`

## Performance 
### PyG
Directly running the `sampler.py` file results in the following performance:
```
Namespace(
    device=0, dropout=0.5, epochs=3, hidden_channels=64, 
    lr=0.01, num_layers=2, runs=10)
    
# Other relevant parameters
# "Relation" linear layers have no bias term
# "Root" linear layers (i.e., self-loops) have bias term 

All runs:
Highest Train: 80.03 ± 0.40
Highest Valid: 47.75 ± 0.43
Final Train: 68.77 ± 6.07
Final Test: 46.85 ± 0.48
```

### DGL

```
{'num_layers': 2, 'fanout': [25, 20], 'batch_size': 1024, 'dropout': 0.5, 
 'n_hidden': 64, 'lr': 0.01, 'n_bases': -1, 'n_epochs': 3, 'runs': 10}

       train_acc valid_acc test_acc
mean   0.937422  0.442171  0.401702
std    0.001838  0.004746  0.006150
```