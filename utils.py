import numpy as np
import json
import configure as conf
from collections import OrderedDict
import codecs
import numpy as np
import os
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import torch
import random
import networkx as nx


def read_data(input_path, word_idx, if_increase_dict):
    seqs = []
    graphs = []

    if if_increase_dict:
        word_idx[conf.GO] = 1
        word_idx[conf.EOS] = 2
        word_idx[conf.unknown_word] = 3
    with open(input_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            jo = json.loads(line, object_pairs_hook=OrderedDict)
            seq = jo['seq']
            seqs.append(seq)
            if if_increase_dict:
                for w in seq.split():
                    if w not in word_idx:
                        word_idx[w] = len(word_idx) + 1

                for id in jo['g_ids_features']:
                    features = jo['g_ids_features'][id]
                    for w in features.split():
                        if w not in word_idx:
                            word_idx[w] = len(word_idx) + 1

            graph = {}
            graph['g_ids'] = jo['g_ids']
            graph['g_ids_features'] = jo['g_ids_features']
            graph['g_adj'] = jo['g_adj']
            graphs.append(graph)

    return seqs, graphs

def vectorize_data(word_idx, texts):
    tv = []
    for text in texts:
        stv = []
        for w in text.split():
            if w not in word_idx:
                stv.append(word_idx[conf.unknown_word])
            else:
                stv.append(word_idx[w])
        tv.append(stv)
    return tv
def write_word_idx(word_idx, path):
    dir = path[:path.rfind('/')]
    if not os.path.exists(dir):
        os.makedirs(dir)

    with codecs.open(path, 'w', 'utf-8') as f:
        for word in word_idx:
            f.write(word+" "+str(word_idx[word])+'\n')
def cons_batch_graph(graphs):
    g_ids = {}
    g_ids_features = {}
    g_fw_adj = {}
    g_bw_adj = {}
    g_nodes = []

    for g in graphs:
        ids = g['g_ids']
        id_adj = g['g_adj']
        features = g['g_ids_features']
        nodes = []
        # we first add all nodes into batch_graph and create a mapping from graph id to batch_graph id, this mapping will be
        # used in the creation of fw_adj and bw_adj
        id_gid_map = {}
        offset = len(g_ids.keys())
        for id in ids:
            id = int(id)
            g_ids[offset + id] = len(g_ids.keys())
            g_ids_features[offset + id] = features[str(id)]
            id_gid_map[id] = offset + id
            nodes.append(offset + id)
        g_nodes.append(nodes)

        for id in id_adj:
            adj = id_adj[id]
            id = int(id)
            g_id = id_gid_map[id]
            if g_id not in g_fw_adj:
                g_fw_adj[g_id] = []
            for t in adj:
                t = int(t)
                g_t = id_gid_map[t]
                g_fw_adj[g_id].append(g_t)
                if g_t not in g_bw_adj:
                    g_bw_adj[g_t] = []
                g_bw_adj[g_t].append(g_id)

    node_size = len(g_ids.keys())
    for id in range(node_size):
        if id not in g_fw_adj:
            g_fw_adj[id] = []
        if id not in g_bw_adj:
            g_bw_adj[id] = []

    graph = {}
    graph['g_ids'] = g_ids
    graph['g_ids_features'] = g_ids_features
    graph['g_nodes'] = g_nodes
    graph['g_fw_adj'] = g_fw_adj
    graph['g_bw_adj'] = g_bw_adj

    return graph
def vectorize_batch_graph(graph, word_idx):
    # vectorize the graph feature and normalize the adj info
    id_features = graph['g_ids_features']
    gv = {}
    nv = []
    word_max_len = 0
    for id in id_features:
        feature = id_features[id]
        word_max_len = max(word_max_len, len(feature.split()))
    word_max_len = min(word_max_len, conf.word_size_max)

    for id in graph['g_ids_features']:
        feature = graph['g_ids_features'][id]
        fv = []
        for token in feature.split():
            if len(token) == 0:
                continue
            if token in word_idx:
                fv.append(word_idx[token])
            else:
                fv.append(word_idx[conf.unknown_word])

        for _ in range(word_max_len - len(fv)):
            fv.append(0)
        fv = fv[:word_max_len]
        nv.append(fv)

    nv.append([0 for temp in range(word_max_len)])
    gv['g_ids_features'] = np.array(nv)

    g_fw_adj = graph['g_fw_adj']
    g_fw_adj_v = []

    degree_max_size = 0
    for id in g_fw_adj:
        degree_max_size = max(degree_max_size, len(g_fw_adj[id]))

    g_bw_adj = graph['g_bw_adj']
    for id in g_bw_adj:
        degree_max_size = max(degree_max_size, len(g_bw_adj[id]))

    degree_max_size = min(degree_max_size, conf.sample_size_per_layer)

    for id in g_fw_adj:
        adj = g_fw_adj[id]
        for _ in range(degree_max_size - len(adj)):
            adj.append(len(g_fw_adj.keys()))
        adj = adj[:degree_max_size]
        g_fw_adj_v.append(adj)

    # PAD node directs to the PAD node
    g_fw_adj_v.append([len(g_fw_adj.keys()) for _ in range(degree_max_size)])

    g_bw_adj_v = []
    for id in g_bw_adj:
        adj = g_bw_adj[id]
        for _ in range(degree_max_size - len(adj)):
            adj.append(len(g_bw_adj.keys()))
        adj = adj[:degree_max_size]
        g_bw_adj_v.append(adj)

    # PAD node directs to the PAD node
    g_bw_adj_v.append([len(g_bw_adj.keys()) for _ in range(degree_max_size)])

    gv['g_ids'] = graph['g_ids']
    gv['g_nodes'] =np.array(graph['g_nodes'])
    gv['g_bw_adj'] = np.array(g_bw_adj_v)
    gv['g_fw_adj'] = np.array(g_fw_adj_v)

    return gv

def get_torch_dataset(sequences,graphs,word_idx):
    dataset = []
    for (seq,obj) in zip(sequences,graphs):
        ids = obj['g_ids']
        features = obj['g_ids_features']
        adj_dict = obj['g_adj']
        bwd_n,bwd_i,fwd_n, fwd_i = [],[],[],[]
        for k,nei in adj_dict.items():
            index = int(k)
            for n in nei:
                fwd_i.append(index)
                fwd_n.append(n)
                bwd_n.append(n)
                bwd_i.append(index)
        fwd_edge_index = torch.tensor([fwd_i,fwd_n],dtype=torch.long)
        bwd_edge_index = torch.tensor([bwd_i,bwd_n], dtype=torch.long)
        feature_matrix = np.zeros((len(adj_dict),len(word_idx.keys()) + 1),dtype = float)
        feat = list(word_idx.keys())
        i = 0
        for k,v in features.items():
            index = feat.index(v)
            feature_matrix[i,index] = 1.
            i+=1
        
        x = torch.tensor(feature_matrix,dtype=torch.float32)
        # print(feature_matrix.shape,adj.shape)
        y = torch.zeros(1,(len(list(word_idx.keys()))+1),dtype = torch.float32)

        seq = seq.split(" ")
        keys = list(word_idx.keys())
        for s in seq:
            y[0,keys.index(s)] = 1.0
        # print(y)
        data = Data(x = x, edge_index = fwd_edge_index,bwd_edge_index = bwd_edge_index, y= y)
        # dd = Data(x =x, edge_index=fwd_edge_index,y = y)
        # g= to_networkx(dd)
        # print(nx.info(g))
        dataset.append(data)
    return dataset
def get_max(train,test):
    max_ = 0
    for d in train:
        if d.num_nodes>max_:max_=d.num_nodes
    for d in test:
        if d.num_nodes>max_:max_=d.num_nodes
    return max_






        
        











