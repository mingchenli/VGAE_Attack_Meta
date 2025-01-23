import torch
import scipy.sparse as sp

def AttackEdges(grad, adj, test_edges, test_edges_false, NumofAttackEdges):
    print(f"Generating adj_attack")
    grad_adj = (grad + grad.T) / 2

    # Avoid changes to test edges
    for edges in test_edges:
        x, y = edges
        grad_adj[x, y] = 0
        grad_adj[y, x] = 0
    for edges in test_edges_false:
        x, y = edges
        grad_adj[x, y] = 0
        grad_adj[y, x] = 0

    adj_attack = adj.toarray()
    budget = NumofAttackEdges
    count_add = 0
    count_remove = 0

    processed_indices = set()

    num_add = int(budget * 0.5)
    num_delete = budget - num_add

    while budget != 0:
        max_value, max_index = torch.max(grad_adj), torch.argmax(grad_adj)
        min_value, min_index = torch.min(grad_adj), torch.argmin(grad_adj)

        if count_add < num_add and abs(max_value) > abs(min_value):
            x, y = divmod(max_index.item(), grad_adj.size(1))
            if (x, y) in processed_indices or (y, x) in processed_indices:
                grad_adj[x, y] = 0.0
                grad_adj[y, x] = 0.0
                continue
            processed_indices.add((x, y))
            processed_indices.add((y, x))
            if adj_attack[x, y] == 0:
                adj_attack[x, y] = 1
                adj_attack[y, x] = 1
                count_add += 1
                budget -= 1
        elif count_remove < num_delete:
            x, y = divmod(min_index.item(), grad_adj.size(1))
            if (x, y) in processed_indices or (y, x) in processed_indices:
                grad_adj[x, y] = 0.0
                grad_adj[y, x] = 0.0
                continue
            processed_indices.add((x, y))
            processed_indices.add((y, x))
            if adj_attack[x, y] == 1:
                adj_attack[x, y] = 0
                adj_attack[y, x] = 0
                count_remove += 1
                budget -= 1

    adj_attack = sp.csr_matrix(adj_attack)

    return adj_attack

