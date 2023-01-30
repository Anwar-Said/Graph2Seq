import argparse
import torch
import numpy as np
import networkx as nx
from utils import *
from torch_geometric.utils import to_networkx
from torch_geometric.data import DataLoader
from model import *
import torch.optim as optim
from torch.nn.utils import clip_grad_value_
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()
import pickle as pkl
# device = 'cpu'
def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def train(model, loader, optimizer, device):
    loss_all = []
    model.train()
    hidden = model.init_state()
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        hidden = model.detach_hidden(hidden)
        output,hidden = model(data.x, data.edge_index,data.bwd_edge_index,data.batch,hidden)
        loss = criterion(output, data.y)
        loss.backward()
        clip_grad_value_(model.parameters(), 2.0)
        optimizer.step()
        loss_all.append(loss.item())
    return np.mean(loss_all)


def evaluate(pred, target):
    pred = torch.round(torch.sigmoid(pred))
    #I am computing average word-based accuracy here
    # Note that the authors opted for sample-based accuracy
    acc = total = 0
    for p,t in zip(pred,target):
        matched = [v.item() for i,v in enumerate(p) if t[i].item()==1 and v.item()==t[i].item()]
        ones = (t == 1.).sum(dim=0).item()
        acc +=len(matched)
        total +=ones
    return (acc/total)*100
def test(model, loader, device):
    acc_all = []
    model.eval()
    hidden = model.init_state()
    for data in loader:
        data = data.to(device)
        hidden = model.detach_hidden(hidden)
        output,hidden = model(data.x, data.edge_index,data.bwd_edge_index,data.batch,hidden)
        acc = evaluate(output, data.y)
        acc_all.append(acc)
    return np.mean(acc_all)

def logger(info):
    f = open(os.path.join(res_path, 'log.csv'), 'a')
    print(info, file=f)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='no_cycle')
    parser.add_argument('--runs', type=int, default=1) ## not using - but multiple runs with different seeds are required to get stable results
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--gnn', type=str, default="ResGatedGraphConv", choices=["ResGatedGraphConv","GCNConv","SAGEConv","GraphConv"])
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    params_path = "params/"
    res_path = "results/"
    data_path = "dataset/"
    if not os.path.isdir(params_path):
        os.mkdir(params_path)
    if not os.path.isdir(res_path):
        os.mkdir(res_path)

    set_seed(conf.seed)

    word_idx = {}
    write_word_idx(word_idx, conf.word_idx_file_path)
    print("loading data")
    #read_data function is taken from the authors' implementation
    texts_train, graphs_train = read_data(conf.train_data_path, word_idx, if_increase_dict=True)
    ##convert dataset into torch_geoemetric format
    train_dataset = get_torch_dataset(texts_train,graphs_train,word_idx)
    texts_test, graphs_test = read_data(conf.test_data_path, word_idx, if_increase_dict=False)
    test_dataset = get_torch_dataset(texts_test,graphs_test,word_idx)
    conf.encoder_hidden_dim =  get_max(train_dataset,train_dataset)
    conf.num_features = test_dataset[0].x.shape[1]
    
    train_loader = DataLoader(train_dataset, batch_size=conf.train_batch_size, shuffle=True, worker_init_fn=args.seed)
    test_loader = DataLoader(test_dataset, batch_size=conf.test_batch_size)
    gnn = eval(args.gnn)
    model = Graph2Seq(conf.num_features, conf,gnn,device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=conf.lr, weight_decay=conf.wd, amsgrad=False)
    best_test_acc = 0
    losses, test_accuracy = [],[]
    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, optimizer, device)
        test_acc = test(model, test_loader, device)
        losses.append(train_loss)
        test_accuracy.append(test_acc)
        if test_acc>best_test_acc:
                best_test_acc = test_acc#save best weights
                torch.save(model.state_dict(), params_path + 'base_checkpoint-best-acc.pkl') 
        print("epoch: {}, train loss: {}, test acc:{}".format(epoch, round(train_loss,4),round((test_acc),4)))

    model.load_state_dict(torch.load(params_path + 'base_checkpoint-best-acc.pkl'))
    train_loss = round(train(model, train_loader, optimizer, device),4)
    model.eval()
    acc = test(model, test_loader, device)
    print("optimization finished!")
    print("best test acc:", round((acc),4))
    # with(open(args.gnn+"_loss_curve.pkl","wb")) as f:
    #     pkl.dump(losses,f)
    # with(open(args.gnn+"_test_curve.pkl","wb")) as f:
    #     pkl.dump(test_accuracy,f)
    log = "dataset: {}, GNN: {},#epochs: {}, seed:{}, batch_size: {}, gnn_hidden: {}, rnn_hidden: {}, loss: {}, acc:{}".format(args.dataset,
    args.gnn, args.epochs, args.seed, conf.train_batch_size,conf.gnn_hidden, conf.rnn_hidden,train_loss, round((acc),4))
    print(log)
    logger(log)


