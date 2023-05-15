import torch.nn as nn
import numpy as np
import pandas as pd


class NetEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=32):
        super(NetEncoder, self).__init__()
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs):
        out = self.encoder(inputs)
        return out
    

class NetDecoder(nn.Module):
    def __init__(self, output_dim, input_dim=32, hidden_dim=128):
        super(NetDecoder, self).__init__()
        self.output_dim = output_dim
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
    def forward(self, inputs):
        out = self.decoder(inputs)
        return out

    
class NetClassifier(nn.Module):
    def __init__(self, output_dim, input_dim=32):
        super(NetClassifier, self).__init__()
        self.input_dim = 32
        self.output_dim = output_dim
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, inputs):
        out = self.classifier(inputs)
        return out
        
def readfile_csv(Filename):
    dataset = {}
    dataset["expression"] = pd.read_csv(Filename["exp_filename"], header=0, index_col=0)
    dataset["cell_type"] = pd.read_csv(Filename["type_filename"], header=0, index_col=False).x.values
    dataset["expression"] = dataset["expression"].div(dataset["expression"].apply(lambda x: x.sum(), axis=1), axis=0) * 10000
    return dataset

def label_transform_num(label):

    label = np.array(label, dtype="<U128")
    label_set = np.unique(label)
    num_label_set = np.arange(0, len(label_set))

    label_num_transform = dict(zip(label_set, num_label_set))
    num_label_transform = dict(zip(num_label_set, label_set))

    num_label = np.zeros(len(label))

    for s in label_set:
        num_label[np.where(label == s)] = label_num_transform[s]

    return num_label, num_label_transform, label_num_transform

def num_to_label(num_label, num_label_transform):
    
    num_label = np.array(num_label)
    
    label = np.zeros(len(num_label), dtype='<U128')
    
    for s in np.unique(num_label):
        label[np.where(num_label == s)] = num_label_transform[s]
                
    return label

def read_label(Filename):
       
    label = pd.read_csv(Filename["type_filename"], header=0, index_col=False).x.values
    
    return label

def readfile_csv(Filename):
    
    dataset = {}
       
    dataset["expression"] = pd.read_csv(Filename["exp_filename"], header=0, index_col=0)
    dataset["cell_type"] = pd.read_csv(Filename["type_filename"], header=0, index_col=False).x.values

    dataset["expression"] = dataset["expression"].div(dataset["expression"].apply(lambda x: x.sum(), axis=1), axis=0) * 10000

    return dataset