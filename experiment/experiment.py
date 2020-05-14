from models import Model
from utils import DATA_ROOT, RESULTS_ROOT, TENSORBOARD_TAG, train, test
from dataset import TinyImagenetDataset, DatasetItem
from visualization import plot_confusion_matrix

import torch
from torchvision import transforms
from torch.nn.modules import loss
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os

class Experiment:
    def __init__(self, config):
        self.config = config
        
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        train_dataset = TinyImagenetDataset(DATA_ROOT / "train", train_transform)
        self.__train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            collate_fn=DatasetItem.collate,
            num_workers=0,
        )
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        test_dataset = TinyImagenetDataset(DATA_ROOT / "val" / "images", test_transform)
        self.__test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            collate_fn=DatasetItem.collate,
            num_workers=0,
        )
        
        self.__loss_function = loss.CrossEntropyLoss()
        self.DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    def run(self):
        model = Model(backbone=self.config['model']['backbone'],
                      num_trainable_layers=self.config['model']['num_trainable_layers'])
        model.to(self.DEVICE)
        
        optimizer = optim.Adam(model.parameters(),
                               lr=self.config['learning_rate'],
                               weight_decay=self.config['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

        random_id = np.random.randint(10000)
        for epoch in range(self.config['epochs']):
            writer = SummaryWriter(RESULTS_ROOT / TENSORBOARD_TAG)
            
            print('Training...')
            loss = train(model, self.DEVICE, self.__train_loader, optimizer, self.__loss_function, epoch, writer)
            print('Validating...')
            test_loss, test_accuracy = test(model, self.DEVICE, self.__test_loader, self.__loss_function, epoch, writer)
            scheduler.step(test_loss)
            
            writer.close()
            if self.config['visualize']:
                visualization_dir =  RESULTS_ROOT / 'confusion_matrix'
                if not os.path.exists(visualization_dir):
                    os.mkdir(visualization_dir)
                print('Plotting confusion matrix...')
                plot_confusion_matrix(model, self.__test_loader, visualization_dir / '{}_{}.png'.format(random_id, epoch))
