from torch import nn
from efficientnet_pytorch import EfficientNet

class Model(nn.Module):
    def __init__(self, backbone, num_trainable_layers):
        super(Model, self).__init__()
        self.__model = backbone
        
        if isinstance(backbone, EfficientNet):
            self.__model._fc = nn.Sequential(
                nn.Linear(self.__model._fc.in_features, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 200),
            )
        else:
            self.__model.fc = nn.Sequential(
                nn.Linear(self.__model.fc.in_features, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 200),
            )
        
        layers_count = 0
        for _ in self.__model.parameters():
            layers_count += 1
        
        assert isinstance(num_trainable_layers, int)
        assert 0 < num_trainable_layers <= layers_count or num_trainable_layers == -1
        if num_trainable_layers != -1:
            for i, p in enumerate(self.__model.parameters()):
                if i < layers_count - num_trainable_layers:
                    p.requires_grad = False
        
    def forward(self, x):
        return self.__model(x)
