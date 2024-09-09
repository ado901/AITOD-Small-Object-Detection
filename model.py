
import copy
import enum
import time
from sympy import im
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer
from earlystopping import EarlyStopping
import torch
from torch.utils.tensorboard import SummaryWriter
from funcs import add_gradient_hist
# this is the solver class that is used to train the model
import os
class Model():
    def __init__(self,
                 model,
                 train_loader:DataLoader,
                 val_loader:DataLoader,
                 optimizer:Optimizer,
                 device,
                 early_stopping:EarlyStopping,
                 epochs,
                 name,
                 scheduler:LRScheduler,
                 starttime,
                 params:dict=None,
                 writer:SummaryWriter=None,
                 start=0):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.early_stopping = early_stopping
        self.epochs = epochs
        self.name=name
        self.scheduler=scheduler
        self.params=params
        self.writer=writer
        # start time is used for tensorboard logging
        self.starttime=starttime
        self.uuid=params['uuid']
        # start is used to load the model from a certain epoch if you want to continue training
        self.start=start
        if start!=0:
            self.load_model(start-1,'weights')
            # resume training from the last epoch and set the best params to the last epoch
            self.early_stopping.best_params=copy.deepcopy(model.state_dict())
        
    def train_epochs(self):
        self.model.to(self.device)
        iteration=0
        for epoch in range(self.start,self.epochs):
            start=time.time()
            train_loss=self.train(epoch)
            val_loss=self.validate(epoch)
            
            # if the scheduler is ReduceLROnPlateau, it needs the validation loss to adjust the learning rate
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(sum(val_loss)/len(self.val_loader))
            else:
                self.scheduler.step()
            print(f"Epoch #{epoch+1} train loss: {sum(train_loss)/len(self.train_loader):.3f}")
            print(f"validation loss: {sum(val_loss)/len(self.val_loader):.3f}")
            print(f'LR: {self.optimizer.param_groups[0]["lr"]}')
            end=time.time()
            # store the loss in tensorboard
            self.writer.add_scalar('Loss/train', sum(train_loss)/len(self.train_loader), epoch)
            self.writer.add_scalar('Loss/val', sum(val_loss)/len(self.val_loader), epoch)
            self.writer.add_figure('grads', add_gradient_hist(self.model), global_step=epoch)
            print(f"Epoch #{epoch+1} took {((end-start)/60):.3f} minutes")
            # check if the early stopping condition is met
            if self.early_stopping(sum(val_loss)/len(self.val_loader), self.model):
                print("Early stopping")
                break
            # save the model after each epoch, note that this saves the best model until now but only on disk,
            # it continues training with the current weights
            self.save_model(self.early_stopping.best_params,epoch)
            self.writer.flush()
            
    def save_model(self, state_dict,epoch=0):
        # if you want to save the model
        parent_dir = os.path.join('weights', f'{self.name}')
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        # to differentiate between different runs, the uuid is used
        check_path = os.path.join(parent_dir, f'checkpoint_{epoch}_{self.uuid}.pth')
        # saving model weights, params and epoch
        torch.save({
                'epoch': epoch+1,
                'model_state_dict': state_dict,
                'params': self.params,
                }, check_path)
        print(f"Model saved! at {check_path}")
        
    def load_model(self, epoch,path):
        # function to load the model
        check_path = os.path.join(path, f'{self.name}/checkpoint_{epoch}_{self.uuid}.pth')
        checkpoint = torch.load(check_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print("Model loaded!")
    def train(self,epoch):
        train_loss = []
        self.model.train()
        bar=tqdm(self.train_loader)
        for i, data in enumerate(bar):
            images, targets = data
            images=[image.to(self.device) for image in images]
            targets=[{k: (v.to(self.device) if k != 'image_name' else v) for k, v in t.items()} for t in targets]
            self.optimizer.zero_grad()
            loss_dict = self.model(images, targets)
            losses=sum(loss for loss in loss_dict.values())
            # backpropagation
            losses.backward()
            self.optimizer.step()
            train_loss.append(losses.item())
            bar.set_description(f"Loss: {losses.item():.4f}")
            
            
        return train_loss
    
    def validate(self, epoch):
        val_loss = []
        self.model.train()
        # no gradient calculation needed
        with torch.no_grad():
            bar=tqdm(self.val_loader)
            for i, data in enumerate(bar):
                images, targets = data
                images=[image.to(self.device) for image in images]
                targets=[{k: (v.to(self.device) if k != 'image_name' else v) for k, v in t.items()} for t in targets]
                
                loss_dict = self.model(images, targets)
                losses=sum(loss for loss in loss_dict.values())
                val_loss.append(losses.item())
                bar.set_description(f"Loss: {losses.item():.4f}")
                
        return val_loss
        

