#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 16:12:08 2022

@author: Guillermo Torres
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ModelWeightsInit import *


## LSTM Intra Ventana

class EpilepsyLSTM(nn.Module):
    """
    Implementation:
        A channel independent generalized seizure detection method for pediatric epileptic seizures
        batch_size 600
        epochs 1000
        lr = 1e-4
        optmizer Adam
    """
    def __init__(self, inputmodule_params,net_params,outmodule_params):
        super().__init__()

        print('Running class: ', self.__class__.__name__)
        
        ### NETWORK PARAMETERS
        n_nodes=inputmodule_params['n_nodes']
    
        Lstacks=net_params['Lstacks']
        dropout=net_params['dropout'] 
        hidden_size=net_params['hidden_size']
       
        n_classes=outmodule_params['n_classes']
        hd=outmodule_params['hd']
        
        self.inputmodule_params=inputmodule_params
        self.net_params=net_params
        self.outmodule_params=outmodule_params

        ### FEATURE EXTRACTOR CNN
        # Bloque 1: Convolución sobre canales y tiempo
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 5), padding=(1, 2))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        # Bloque 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 5), padding=(1, 2))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        # Bloque 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        
        ### NETWORK ARCHITECTURE
        # IF batch_first THEN (batch, timesteps, features), ELSE (timesteps, batch, features)
        self.lstm = nn.LSTM(input_size=n_nodes, # the number of expected features (out of convs)
                                       hidden_size= hidden_size, # the number of features in the hidden state h
                                       num_layers= Lstacks, # number of stacked lstms 
                                       batch_first = True,
                                       bidirectional = False,
                                       dropout=dropout)

        self.fc = nn.Sequential(nn.Linear(hidden_size, hd),
                                nn.ReLU(),
                                nn.Linear(hd, n_classes)
                                ) 

    
    def init_weights(self):
         init_weights_xavier_normal(self)
        
    def forward(self, x):
        # Bloque 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))

        # Bloque 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        # Bloque 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # x: [B, F=128, C', T']
        
        B, Fm, Cp, Tp = x.shape

        ## Reshape input
        # input [batch, features (=n_nodes), sequence_length (T)] ([N, 21, 640])
        x = x.permute(0, 3, 1, 2).contiguous()     # [B, Tp, Fm, Cp]
        x = x.view(B, Tp, Fm * Cp)                 # [B, Tp, features]
        
        ## LSTM Processing
        out, (hn, cn) = self.lstm(x)
        # out is [batch, sequence_length, hidden_size] for last stack output
        # hn and cn are [1, batch, hidden_size]
        out = out[:, -1, :] # hT state of lenght hidden_size

        ## Output Classification (Class Probabilities)
        x = self.fc(out)

        return x
    
def get_default_hyperparameters():
   
    # initialize dictionaries
    inputmodule_params={}
    net_params={}
    outmodule_params={}
    
    # network input parameters
    inputmodule_params['n_nodes'] = 256
    
    # LSTM unit  parameters
    net_params['Lstacks'] = 3  # stacked layers (num_layers)
    net_params['dropout'] = 0.0
    net_params['hidden_size']= 256  #h
   
    # network output parameters
    outmodule_params['n_classes']=2
    outmodule_params['hd']=128
    
    return inputmodule_params, net_params, outmodule_params

## LSTM Inter Ventana

class CNNBackbone(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 1), padding=(1, 2))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 1), padding=(1, 2))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2, 2))

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d((2, 2))

        # Proyección a un "hipervector" de tamaño fijo
        self.gap = nn.AdaptiveAvgPool2d((1, 1))   # -> [B, 128, 1, 1]
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x):
        # x: [B, 1, C, T]
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = self.gap(x).squeeze(-1).squeeze(-1)  # [B, 128]
        x = self.proj(x)                         # [B, out_dim]
        return x


class EpilepsyCNNLSTMSeq(nn.Module):
    def __init__(self, emb_dim=256, hidden_size=128, lstm_layers=1, dropout=0.0, n_classes=2, hd=128):
        super().__init__()
        self.backbone = CNNBackbone(out_dim=emb_dim)

        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hd),
            nn.ReLU(),
            nn.Linear(hd, n_classes)
        )

    def forward(self, x_seq):
        # x_seq: [B, K, 1, C, T]
        B, K, one, C, T = x_seq.shape

        # Convertimos a batch grande para pasar por CNN: [B*K, 1, C, T]
        x = x_seq.view(B * K, one, C, T)

        # Embeddings por ventana: [B*K, emb_dim]
        emb = self.backbone(x)

        # Volvemos a secuencia: [B, K, emb_dim]
        emb = emb.view(B, K, -1)

        # LSTM sobre hipervectores sucesivos
        out, _ = self.lstm(emb)      # [B, K, hidden_size]
        h_last = out.mean(dim=1)     # mean pooling temporal

        logits = self.fc(h_last)
        return logits