import argparse
from os import name
from re import M
from uuid import UUID
from aitod import AITOD
from torch.utils.data import DataLoader
from earlystopping import EarlyStopping
import torch
from configs import params_retinanet,params_faster_rcnn_resnet,params_custom_rpn_faster_rcnn_resnet
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from model import Model
from tqdm import tqdm
import uuid
# these are the arguments that can be passed to the script for loading a model and continuing training
# for example: python run.py --epoch 50 --uuid 64efdbb0-687a-11ef-ac4b-0242ac1c000c --log_path logs/2024-09-01-15-53-56/64efdbb0-687a-11ef-ac4b-0242ac1c000c
# keep in mind that the variable MODEL should point to the same model that was used to train the model with the uuid provided
parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--uuid', type=UUID, default=None, help='uuid of the model to start training from')
parser.add_argument('--log_path', type=str, default=None, help='path to save logs')
epoch=parser.parse_args().epoch
assert (epoch>0 and parser.parse_args().uuid is not None and parser.parse_args().log_path is not None) or epoch==0, 'epoch > 0 and uuid must be provided together'

pathtrain='aitod/train'
pathtest='aitod/test'
pathval='aitod/val'
print(torch.cuda.is_available())

from funcs import get_train_transform,collate_fn,CLASSES,MODELS, get_val_transform
num_workers=8
# choice of model from the enum in funcs.py
MODEL=MODELS.RETINANET_RESNET
if MODEL==MODELS.RETINANET_RESNET:
    params=params_retinanet
elif MODEL==MODELS.FASTER_RCNN_RESNET:
    params=params_faster_rcnn_resnet
elif MODEL==MODELS.FASTER_RCNN_RESNET_RPN:
    params=params_custom_rpn_faster_rcnn_resnet
else:
    raise NotImplementedError

# these are parameters from the configs.py file
epochs=params['epochs']
batchsize=params['batch_size']
from torch.utils.tensorboard import SummaryWriter
import time
# start tensorboard session and log the parameters as text
now = time.time()
uuid=uuid.uuid1() if epoch==0 else parser.parse_args().uuid
params['uuid']=uuid
timestamp=time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(now))
tb=SummaryWriter(log_dir=f'logs/{timestamp}/{uuid}' if parser.parse_args().log_path is None else parser.parse_args().log_path)
tb.add_text('params',str(params))

# load both train and validation datasets
datasetprovatrain=AITOD(pathtrain,transform=get_train_transform())
train_loader = DataLoader(
        datasetprovatrain,
        batch_size=batchsize,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
datasetprovaval=AITOD(pathval,transform=get_val_transform())
val_loader = DataLoader(
        datasetprovaval,
        batch_size=batchsize,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
# load the model and optimizer
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(params)
model=MODEL(len(CLASSES), pretrained=params['model']['params']['pretrained'])
name=MODEL.__repr__().split(' ')[1]
# loading the parameters using the params of configs.py
optimizer=torch.optim.SGD(model.parameters(), **params['optimizer']['params'],) if params['optimizer']['name']=='SGD' else torch.optim.Adam(model.parameters(), **params['optimizer']['params'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **params['scheduler']['params']) if params['scheduler']['name']=='step_lr' else torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=2)
# finally loading the solver and starting the training
solver=Model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    early_stopping=EarlyStopping(),
    epochs=epochs,
    name=name,
    scheduler=scheduler,
    starttime=timestamp,
    params=params,
    writer=tb,
    start=epoch)
solver.train_epochs()

