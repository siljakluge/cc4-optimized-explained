"""
utils.py — Graph Assembly Utilities for Multi-Agent Reinforcement Learning

This module provides helper functions to merge multiple PyTorch Geometric subgraphs
into a single graph representation. This is used for constructing a global observation
from per-agent or per-subnet local graph data.

Functions:
- combine_subgraphs: Concatenates multiple subgraph tuples into a single graph (x, edge_index)
- combine_marl_states: Merges multi-agent observations including global, server, and user nodes
"""

import torch 

def combine_subgraphs(states):
    xs,eis = zip(*states)

    # Compute offsets via cumulative sum of node counts
    sizes = [x.size(0) for x in xs]
    offsets = torch.zeros(len(sizes), dtype=torch.long)
    offsets[1:] = torch.tensor(sizes[:-1]).cumsum(0) if len(sizes) > 1 else offsets[1:]

    # Apply offsets to edge indices
    new_eis = [ei + off.item() for ei, off in zip(eis, offsets)]

    xs = torch.cat(xs, dim=0)
    eis = torch.cat(new_eis, dim=1)

    return xs,eis

def combine_marl_states(s):
    '''
    Combines states given observations of the form:
    x, ei, servers, n_servers, users, n_users, action_edges, is_multi_subnet

    (Note: is_multi_subnet values need to be separated)
    '''
    xs, eis, gvs, srvs, nsrvs, usrs, nusrs, edges, is_multi = [list(element) for element in zip(*s)]

    # Compute offsets via cumulative sum of node counts
    n = len(xs)
    sizes = [xs[i].size(0) for i in range(n)]

    if n > 1:
        cumsum = torch.tensor(sizes[:-1], dtype=torch.long).cumsum(0)
        offsets = torch.zeros(n, dtype=torch.long)
        offsets[1:] = cumsum
    else:
        offsets = torch.zeros(n, dtype=torch.long)

    # Pre-allocate lists and apply offsets
    new_ei = [None] * n
    new_srv = [None] * n
    new_usr = [None] * n
    new_edges = [None] * n

    for i in range(n):
        off = offsets[i]
        new_edges[i] = edges[i] + off
        new_srv[i] = srvs[i] + off
        new_usr[i] = usrs[i] + off
        new_ei[i] = eis[i] + off

    # Concatenate all tensors
    xs = torch.cat(xs, dim=0)
    gvs = torch.cat(gvs, dim=0)
    srvs = torch.cat(new_srv)
    nsrvs = torch.cat(nsrvs)
    usrs = torch.cat(new_usr)
    nusrs = torch.cat(nusrs)
    eis = torch.cat(new_ei, dim=1)
    edges = torch.cat(new_edges, dim=1)

    # Is_Multi should be the same for all elements
    return xs,eis,gvs, srvs,nsrvs, usrs,nusrs, edges, is_multi[0]
