import os, sys
import time
import argparse
import itertools
from tqdm import tqdm

import pandas as pd
import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

# Add relevant paths to submodules
cwd = os.path.dirname(os.path.realpath(__file__))
ogb_example_dir = f"{cwd}/ogb/examples/nodeproppred/mag"
sys.path.append(ogb_example_dir)
dgl_example_dir = f"{cwd}/dgl/examples/pytorch/rgcn-hetero"
sys.path.append(dgl_example_dir)

# From DGL example code
from model import RelGraphConvLayer 
# From OGB example code
from logger import Logger

def extract_embed(node_embed, input_nodes):
    emb = {}
    for ntype, nid in input_nodes.items():
        nid = input_nodes[ntype]
        if ntype in node_embed:
            emb[ntype] = node_embed[ntype][nid]
    return emb

class RelGraphEmbed(nn.Module):
    # Adding "Exclude" option so don't get embeddings for "paper" nodes
    r"""Embedding layer for featureless heterograph."""
    def __init__(self, g, embed_size, exclude=list(), embed_name='embed',
                 activation=None, dropout=0.0):
        
        super(RelGraphEmbed, self).__init__()
        self.g = g
        self.embed_size = embed_size
        self.embed_name = embed_name
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        # create weight embeddings for each node for each relation
        self.embeds = nn.ParameterDict()
        for ntype in g.ntypes:
            if ntype in exclude:
                continue
            embed = nn.Parameter(th.Tensor(g.number_of_nodes(ntype), self.embed_size))
            nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain('relu'))
            self.embeds[ntype] = embed

    def forward(self, block=None):
        """Forward computation

        Parameters
        ----------
        block : DGLHeteroGraph, optional
            If not specified, directly return the full graph with embeddings stored in
            :attr:`embed_name`. Otherwise, extract and store the embeddings to the block
            graph and return.

        Returns
        -------
        DGLHeteroGraph
            The block graph fed with embeddings.
        """
        return self.embeds


class EntityClassify(nn.Module):
    def __init__(self,
                 g, in_dim,
                 h_dim, out_dim,
                 num_bases,
                 num_hidden_layers=1,
                 dropout=0,
                 use_self_loop=True):
        super(EntityClassify, self).__init__()
        self.g = g
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.rel_names = list(set(g.etypes))
        self.rel_names.sort()
        if num_bases < 0 or num_bases > len(self.rel_names):
            self.num_bases = len(self.rel_names)
        else:
            self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop

        self.layers = nn.ModuleList()
        # i2h
        self.layers.append(RelGraphConvLayer(
            self.in_dim, self.h_dim, self.rel_names,
            self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
            dropout=self.dropout))
        # h2h
        for i in range(self.num_hidden_layers):
            self.layers.append(RelGraphConvLayer(
                self.h_dim, self.h_dim, self.rel_names,
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                dropout=self.dropout))
        # h2o
        self.layers.append(RelGraphConvLayer(
            self.h_dim, self.out_dim, self.rel_names,
            self.num_bases, activation=None,
            self_loop=self.use_self_loop))

    def forward(self, h, blocks):
        for layer, block in zip(self.layers, blocks):
            h = layer(block, h)
        return h

    def inference(self, g, batch_size, device, num_workers, x=None):
        """Minibatch inference of final representation over all node types.

        ***NOTE***
        For node classification, the model is trained to predict on only one node type's
        label.  Therefore, only that type's final representation is meaningful.
        """

        if x is None:
            x = self.embed_layer()

        for l, layer in enumerate(self.layers):
            y = {
                k: th.zeros(
                    g.number_of_nodes(k),
                    self.h_dim if l != len(self.layers) - 1 else self.out_dim)
                for k in g.ntypes}

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                {k: th.arange(g.number_of_nodes(k)) for k in g.ntypes},
                sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=num_workers)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].to(device)

                h = {k: x[k][input_nodes[k]].to(device) for k in input_nodes.keys()}
                h = layer(block, h)

                for k in h.keys():
                    y[k][output_nodes[k]] = h[k].cpu()

            x = y
        return y


def parse_args():
    # DGL
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=64,
            help="number of hidden units")
    parser.add_argument("--lr", type=float, default=0.01,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=-1,
            help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("-e", "--n-epochs", type=int, default=3,
            help="number of training epochs")
    parser.add_argument("--model_path", type=str, default=None,
            help='path for save the model')

    # OGB
    parser.add_argument('--runs', type=int, default=10)

    args = parser.parse_args()
    return args

def prepare_data(args):
    dataset = DglNodePropPredDataset(name="ogbn-mag")
    split_idx = dataset.get_idx_split()
    g, labels = dataset[0] # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
    labels = labels['paper'].flatten()

    def add_reverse_hetero(g):
        relations = {}
        num_nodes_dict = {ntype: g.num_nodes(ntype) for ntype in g.ntypes}
        for metapath in g.canonical_etypes:
            # Original edges
            src, dst = g.all_edges(etype=metapath[1])
            relations[metapath] = (src, dst)

            reverse_metapath = (metapath[2], 'rev-' + metapath[1], metapath[0])
            relations[reverse_metapath] = (dst, src)           # Reverse edges

        new_g = dgl.heterograph(relations, num_nodes_dict=num_nodes_dict)

        # copy_ndata:
        for ntype in g.ntypes:
            for k, v in g.nodes[ntype].data.items():
                new_g.nodes[ntype].data[k] = v.detach().clone()

        return new_g

    g = add_reverse_hetero(g)
    print("Loaded graph: {}".format(g))

    split_idx = dataset.get_idx_split()
    logger = Logger(args['runs'], args)

    # train sampler
    sampler = dgl.dataloading.MultiLayerNeighborSampler(args['fanout'])
    train_loader = dgl.dataloading.NodeDataLoader(
        g, split_idx['train'], sampler,
        batch_size=args['batch_size'], shuffle=True, num_workers=32)

    # validation sampler
    # we do not use full neighbor to save computation resources
    val_sampler = dgl.dataloading.MultiLayerNeighborSampler(args['fanout'])
    val_loader = dgl.dataloading.NodeDataLoader(
        g, split_idx['valid'], val_sampler,
        batch_size=32768, shuffle=True, num_workers=32)
    
    return (g, labels, dataset.num_classes, split_idx,  
        logger, train_loader, val_loader)

def get_model(g, num_classes, args):
    embed_layer = RelGraphEmbed(g, 128, exclude=['paper'])
    
    model = EntityClassify(
        g, 128, args['n_hidden'], num_classes,
        num_bases=args['n_bases'],
        num_hidden_layers=args['num_layers'] - 2,
        dropout=args['dropout'],
        use_self_loop=True,
    )

    return embed_layer, model

@th.no_grad()
def evaluate(g, model, loader, node_embed, labels, category, device):
    model.eval()
    total_loss = 0
    total_acc = 0
    count = 0
    for input_nodes, seeds, blocks in loader:
        blocks = [blk.to(device) for blk in blocks]
        seeds = seeds[category]     # we only predict the nodes with type "category"

        emb = extract_embed(node_embed, input_nodes)
        # Get the batch's raw "paper" features
        emb.update({'paper': g.ndata['feat']['paper'][input_nodes['paper']]})
        lbl = labels[seeds]
        
        if th.cuda.is_available():
            emb = {k : e.cuda() for k, e in emb.items()}
            lbl = lbl.cuda()
        

        logits = model(emb, blocks)[category]
        y_hat = logits.log_softmax(dim=-1)
        loss = F.nll_loss(y_hat, lbl)

        acc = th.sum(y_hat.argmax(dim=1) == lbl).item()
        total_loss += loss.item() * len(seeds)
        total_acc += acc
        count += len(seeds)
     
    return total_loss / count, total_acc / count

def train_dgl(g, model, node_embed, optimizer, train_loader, val_loader, 
              labels, model_path, device, args):
    
    # training loop
    print("start training...")
    category = 'paper'
    

    if th.cuda.is_available():
        model.cuda()
    model.train()

    for epoch in range(args['n_epochs']):
        N = labels.size(0)
        pbar = tqdm(total=N)
        pbar.set_description(f'Epoch {epoch:02d}')
        t0 = time.time()
        
        total_loss = 0

        for i, (input_nodes, seeds, blocks) in enumerate(train_loader):
            blocks = [blk.to(device) for blk in blocks]
            seeds = seeds[category]     # we only predict the nodes with type "category"
            batch_size = seeds.shape[0]

            emb = extract_embed(node_embed, input_nodes)
            # Get the batch's raw "paper" features
            emb.update({'paper': g.ndata['feat']['paper'][input_nodes['paper']]})
            lbl = labels[seeds]
            
            if th.cuda.is_available():
                emb = {k : e.cuda() for k, e in emb.items()}
                lbl = lbl.cuda()
            
            optimizer.zero_grad()
            logits = model(emb, blocks)[category]
            
            y_hat = logits.log_softmax(dim=-1)
            loss = F.nll_loss(y_hat, lbl)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size
            pbar.update(batch_size)

            train_acc = th.sum(y_hat.argmax(dim=1) == lbl).item() / len(seeds)
        
        pbar.close()
        t_delta = time.time() - t0
        loss = total_loss / N
        
        print(f"Epoch {epoch}:  took {t_delta} s")

        val_loss, val_acc = evaluate(g, model, val_loader, node_embed, labels, category, device)
        print("Epoch {:05d} | Valid Acc: {:.4f} | Valid loss: {:.4f}".
              format(epoch, val_acc, val_loss))
    print()
    if model_path is not None:
        th.save(model.state_dict(), model_path)

@th.no_grad()
def test(g, model, node_embed, y_true, device, split_idx, args):
    model.eval()
    category = 'paper'
    evaluator = Evaluator(name='ogbn-mag')

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args['num_layers'])
    loader = dgl.dataloading.NodeDataLoader(
        g, {'paper': th.arange(g.num_nodes('paper'))}, sampler,
        batch_size=16384, shuffle=False, num_workers=32)
    

    N = y_true.size(0)
    pbar = tqdm(total=N)
    pbar.set_description(f'Full Inference')
    t0 = time.time()
    
    y_hats = list()

    for i, (input_nodes, seeds, blocks) in enumerate(loader):
        blocks = [blk.to(device) for blk in blocks]
        seeds = seeds[category]     # we only predict the nodes with type "category"
        batch_size = seeds.shape[0]

        emb = extract_embed(node_embed, input_nodes)
        # Get the batch's raw "paper" features
        emb.update({'paper': g.ndata['feat']['paper'][input_nodes['paper']]})
        
        if th.cuda.is_available():
            emb = {k : e.cuda() for k, e in emb.items()}
        
        logits = model(emb, blocks)[category]
        y_hat = logits.log_softmax(dim=-1).argmax(dim=1, keepdims=True)
        y_hats.append(y_hat.cpu())
        
        pbar.update(batch_size)
    
    pbar.close()
    t_delta = time.time() - t0
    print(f"Took {t_delta}s for full inference")

    y_pred = th.cat(y_hats, dim=0)
    y_true = th.unsqueeze(y_true, 1)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']['paper']],
        'y_pred': y_pred[split_idx['train']['paper']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']['paper']],
        'y_pred': y_pred[split_idx['valid']['paper']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']['paper']],
        'y_pred': y_pred[split_idx['test']['paper']],
    })['acc']

    return train_acc, valid_acc, test_acc

def main(args):
    # Static parameters
    hyperparameters = dict(
        num_layers = 2,
        fanout = [25, 20], #TODO: verify this is in the same layer-order between PyG and DGL
        batch_size=1024,
    )
    hyperparameters.update(vars(args))
    print(hyperparameters)

    device = f'cuda:0' if th.cuda.is_available() else 'cpu'

    (g, labels, num_classes, split_idx, 
        logger, train_loader, val_loader) = prepare_data(hyperparameters)
    
    results = dict()
    for i in range(hyperparameters['runs']):

        embed_layer, model = get_model(g, num_classes, hyperparameters)
        model = model.to(device)

        #print(embed_layer)
        #print(model)

        # optimizer
        all_params = itertools.chain(model.parameters(), embed_layer.parameters())
        optimizer = th.optim.Adam(all_params, lr=hyperparameters['lr'])

        train_dgl(g, model, embed_layer(), optimizer, train_loader, val_loader, 
                labels, 'models/model0.pt', device, hyperparameters)

        
        train_acc, valid_acc, test_acc = test(g, model, embed_layer(), labels, device, split_idx, hyperparameters)
        print(f"Train Acc: {train_acc} | Val Acc: {valid_acc} | Test Acc: {test_acc}")
        results[i] = dict(train_acc=train_acc, valid_acc=valid_acc, test_acc=test_acc)

    print("Final performance: ")
    print(results)

    print(pd.DataFrame.from_dict(results, orient='index').agg(['mean', 'std']))

if __name__ == '__main__':
    args = parse_args()
    main(args)