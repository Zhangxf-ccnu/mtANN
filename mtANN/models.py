import torch.nn as nn


class NetEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NetEncoder, self).__init__()
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
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NetDecoder, self).__init__()
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
    def __init__(self, input_dim, output_dim):
        super(NetClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, inputs):
        out = self.classifier(inputs)
        return out
        
        
        