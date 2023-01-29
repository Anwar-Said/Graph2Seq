import time
import copy
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, BatchNorm1d, Conv1d, MaxPool1d
from torch_geometric.nn import MLP,global_mean_pool,ResGatedGraphConv,GraphConv,GCNConv,SAGEConv
import math
import numpy as np


class Graph2Seq(nn.Module):
    def __init__(self, num_feat,config,gnn,device):
        super(Graph2Seq, self).__init__()
        self.hidden = config.gnn_hidden
        self.gnn_layers = config.gnn_layers
        self.rnn_hidden = config.rnn_hidden
        self.rnn_layers = config.rnn_layers
        self.dropout = config.dropout
        self.vocab_size = config.vocab_size
        self.encoder_hidden_dim = config.encoder_hidden_dim
        self.device = device
        self.batch_size = config.train_batch_size
        self.sequence_length = 1
        
        #first layer - features transformation
        self.fc_gnn = Linear(num_feat, self.hidden, bias=False)
        gnn_conv = gnn(self.hidden, self.hidden)
        self.conv_layers = nn.ModuleList([copy.deepcopy(gnn_conv) 
                           for _ in range(self.gnn_layers)]) 
        
        self.mlp = MLP([self.hidden*2,self.hidden*2, self.hidden*2], dropout=0.5)
        ## alignment model - FFN
        self.alm = Linear((self.rnn_hidden*100)+self.rnn_hidden, 100)
        self.rnn = nn.LSTM(self.hidden*2, self.rnn_hidden, self.rnn_layers,dropout=self.dropout,batch_first = False)
        self.dropout = nn.Dropout(self.dropout)
        self.fc = nn.Linear(self.rnn_hidden, self.vocab_size)
    def init_state(self):
        return (torch.zeros(self.rnn_layers, self.sequence_length, self.rnn_hidden),
                torch.zeros(self.rnn_layers, self.sequence_length, self.rnn_hidden))
    
    
    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell
    
    
    def attention(self, states, h):
        h = torch.reshape(h, (self.batch_size,100, -1))
        states = states.to(self.device)
        states_att = []
        for s in states:
            batch_attentions = []
            for b in h:
                flt = torch.cat((s.flatten(),b.flatten()),dim =0)
                alpha = self.alm(flt)
                alpha_ij = torch.reshape(alpha/alpha.sum(),(alpha.shape[0],1))
                alpha_ij = torch.matmul(alpha_ij.t(),b)
                batch_attentions.append(alpha_ij)
            tt = torch.cat(batch_attentions).sum(dim=0)
            states_att.append(tt)
        new_state = torch.cat(states_att)
        new_state = torch.reshape(new_state,(states.shape[0],states.shape[1],states.shape[2]))
        return new_state
         
    def forward(self,x,edge_index,bwd_edge_index,batch,hidden):
        
        #ENCODER

        #initial feature transformation (optional)
        x = self.fc_gnn(x)
        #applying GNN convolutions
        for conv in self.conv_layers:
            fwd_x = conv(x, edge_index).relu()
            #convolution with backward edges
            bwd_x = conv(x,bwd_edge_index).relu()
        #concatenating last layer's hidden representations for both forward and backward pass
        x = torch.cat((fwd_x, bwd_x),dim = 1)
        #Fully connected layers as mentioned in the paper
        x_mlp = self.mlp(x)
        #graph-level mean pooling
        x_g = global_mean_pool(x_mlp,batch)


        #### DECODER


        #reshaping embeddings for LSTM
        x_emb = torch.reshape(x_g,(self.batch_size,self.sequence_length,x_g.shape[1]))
        hid, state_c = hidden
        hid = hid.to(self.device)
        state_c = state_c.to(self.device)
        #updating context vector using attention
        state_c = self.attention(state_c, x)
        output,hidden = self.rnn(x_emb,(hid,state_c))
        logits = self.dropout(output) 
        predictions = self.fc(logits)
        #taking last state's predictions
        predictions = predictions[:,-1,:]
        return predictions,hidden