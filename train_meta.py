from __future__ import division
from __future__ import print_function

import argparse
import time
import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim

from gae.model import GCNModelVAE
from gae.optimizer import loss_function
from gae.utils import load_data, mask_test_edges, preprocess_graph, get_roc_score, get_acc_score
from AttackAdj import AttackEdges
import torch.nn.functional as F

def get_edge_list(adj,name):
    # Create an edge list from the adjacency matrix
    rows, cols = np.where(adj == 1)
    edges = list(zip(rows, cols))

    # Convert edge list to DataFrame
    df_edges = pd.DataFrame(edges, columns=['Source', 'Target'])

    # Optional: Remove duplicates (for undirected graphs)
    df_edges = df_edges.drop_duplicates().reset_index(drop=True)

    # Save to CSV (no index column)
    df_edges.to_csv(name + '.csv', index=False)

def attackModel(args, numOfAttakcEdges):
    is_cuda_available = torch.cuda.is_available()

    # Step 2: Set Device
    device = torch.device("cuda" if is_cuda_available else "cpu")

    print("Using {} dataset".format(args.dataset_str))
    adj, features = load_data(args.dataset_str)
    n_nodes, feat_dim = features.shape

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train
    get_edge_list((adj.toarray()), 'edge_list')
    target_size = len(test_edges) + len(test_edges_false)

    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    # adj_label = sparse_to_tuple(adj_label)
    adj_label = torch.FloatTensor(adj_label.toarray())

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    pos_weight = torch.FloatTensor(np.array(pos_weight))
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    accumulated_gradients = None
    adj_norm = adj_norm.to_dense()
    adj_norm.requires_grad_(True)

    hidden_emb = None
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()

        features = features.to(device)
        adj_norm = adj_norm.to(device)

        recovered, mu, logvar = model(features, adj_norm)

        adj_label = adj_label.to(device)
        pos_weight = pos_weight.to(device)
        n_nodes = torch.tensor(n_nodes, dtype=torch.int)
        norm = torch.tensor(norm, dtype=torch.float)

        loss = loss_function(preds=recovered, labels=adj_label,
                             mu=mu, logvar=logvar, n_nodes=n_nodes,
                             norm=norm, pos_weight=pos_weight)

        pred_tensor = torch.empty(target_size, dtype=torch.float32)
        tmp = torch.mm(mu, mu.t())

        for k, e in enumerate(test_edges):
            pred_tensor[k] = tmp[e[0], e[1]]
        for k, e in enumerate(test_edges_false, start=len(test_edges)):
            pred_tensor[k] = tmp[e[0], e[1]]


        one_tensor = torch.ones(len(test_edges))
        zero_tensor = torch.zeros(len(test_edges_false))
        label_tensor = torch.cat((one_tensor, zero_tensor), dim=0)

        loss1 = F.binary_cross_entropy_with_logits(pred_tensor, label_tensor)

        # Compute gradient of loss with respect to adj_norm
        adj_norm_grad = torch.autograd.grad(loss1, adj_norm, retain_graph=True)[0]

        #Update parameters
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()

        if accumulated_gradients is None:
            accumulated_gradients = adj_norm_grad.clone()
        else:
            accumulated_gradients += adj_norm_grad

        hidden_emb = mu.cpu().data.numpy()
        roc_curr, ap_curr = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
              "val_ap=", "{:.5f}".format(ap_curr),
              "time=", "{:.5f}".format(time.time() - t)
              )

    print("Optimization Finished!")

    final_adj_norm_gradient = accumulated_gradients

    print("Computing accumulated gradient with respect to Adjacent matrix")
    adj_attack = AttackEdges(final_adj_norm_gradient, adj_train, test_edges, test_edges_false, numOfAttakcEdges)

    print("Attack Finished!")

    print("-----Evaluation on original data-----")

    # Testing on testing set
    model.eval()
    features.to(device)
    adj_norm.to(device)
    recovered, mu, logvar = model(features, adj_norm)

    hidden_emb = mu.cpu().data.numpy()

    roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
    print('Test ROC score: ' + str(roc_score))
    print('Test AP score: ' + str(ap_score))

    # Training on ADV ADJ
    print("------Evaluation on poisoning data-----")
    adj_label = adj_attack + sp.eye(adj_attack.shape[0])
    adj_label = torch.FloatTensor(adj_label.toarray())

    adj_attack = adj_attack.toarray()
    adj_attack_final = torch.tensor(adj_attack, dtype=torch.float32)  # numpy to tensor

    pos_weight = float(
        adj_attack_final.shape[0] * adj_attack_final.shape[0] - adj_attack_final.sum()) / adj_attack_final.sum()
    pos_weight = torch.FloatTensor(np.array(pos_weight))
    norm = adj_attack_final.shape[0] * adj_attack_final.shape[0] / float(
        (adj_attack_final.shape[0] * adj_attack_final.shape[0] - adj_attack_final.sum()) * 2)

    model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    adjATK_norm = preprocess_graph(adj_attack_final)

    features = features.to(device)
    adjATK_norm = adjATK_norm.to(device)

    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()

        features = features.to(device)
        adj_norm = adj_norm.to(device)

        recovered, mu, logvar = model(features, adjATK_norm)

        adj_label = adj_label.to(device)
        pos_weight = pos_weight.to(device)
        n_nodes = torch.tensor(n_nodes, dtype=torch.int)
        norm = torch.tensor(norm, dtype=torch.float)

        loss = loss_function(preds=recovered, labels=adj_label,
                             mu=mu, logvar=logvar, n_nodes=n_nodes,
                             norm=norm, pos_weight=pos_weight)

        # Update parameters
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()

        hidden_emb = mu.cpu().data.numpy()
        roc_curr, ap_curr = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)

        print("Epoch:", '%04d' % (epoch + 1), "ADV_train_loss=", "{:.5f}".format(cur_loss),
              "val_ap=", "{:.5f}".format(ap_curr),
              "time=", "{:.5f}".format(time.time() - t)
              )

    # Testing with same testing set
    model.eval()
    recovered, mu, logvar = model(features, adjATK_norm)

    hidden_emb = mu.cpu().data.numpy()

    roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
    print('Test ROC score: ' + str(roc_score))
    print('Test AP score: ' + str(ap_score))


