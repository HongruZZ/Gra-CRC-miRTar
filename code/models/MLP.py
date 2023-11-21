from torch import nn as nn

class MLP(nn.Module):
    def __init__(self, inSize, outSize, hiddenList=[512], dropout=0.3, actFunc=nn.ReLU): #inSize = hidden_dim = 64, outSize = n_classes = 4, 这里的hiddenlist往里面填元素可以获得多层感知机, 640*4
        super(MLP, self).__init__()
        layers = nn.Sequential()
        for i,os in enumerate(hiddenList):
            layers.add_module(str(i*2), nn.Linear(inSize, os))
            layers.add_module(str(i*2+1), actFunc())
            inSize = os
        self.hiddenLayers = layers
        self.dropout = nn.Dropout(p=dropout)
        self.out = nn.Linear(inSize, outSize, bias=True)
    def forward(self, x):
        x = self.hiddenLayers(x)
        #print(x.shape)
        return self.out(self.dropout(x))