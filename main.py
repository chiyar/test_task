from experiment import Experiment
from torchvision import models
from efficientnet_pytorch import EfficientNet


if __name__ == "__main__":
    config = {
        'model': {
            'backbone': EfficientNet.from_pretrained('efficientnet-b5'),
            'num_trainable_layers': -1,
        },
        
        'batch_size': 256,
        'learning_rate': 3e-4,
        'weight_decay': 0.,
        'epochs': 100,
        
        'visualize': True,
    }
    
    experiment = Experiment(config=config)
    experiment.run()
