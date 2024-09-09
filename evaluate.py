
import os
import torch
from funcs import CLASSES,evaluate, get_val_transform,load_model,collate_fn,MODELS
from aitod import AITOD
from torch.utils.data import DataLoader
import pandas as pd





# this is the evaluation script that is run after training the model, it loads the model and evaluates it on the test set
# you need to set the uuid, the date of log, the model choice and the epoch to load to run this script


uuid='2bfd8f84-690e-11ef-9f77-0242ac1c000c'
date='2024-09-02-09-31-46'
modelchoice=MODELS.FASTER_RCNN_RESNET
model= modelchoice(numclasses=len(CLASSES))
name=modelchoice.__repr__().split(' ')[1]
model, params=load_model(54,model,name, uuid=uuid)

test_dataset=AITOD('aitod/test',transform=get_val_transform())
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
outputs=evaluate(model,test_loader,torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
import pandas as pd
#using pandas to save results
dicts=outputs.compute()
dicts.pop('classes')
#convert tensor to value
dicts={k:v.item() if type(v)==torch.Tensor else v for k,v in dicts.items()}
dicts['model']=name
dicts['optimizer']=params['optimizer']['name']
dicts['optim_params']=params['optimizer']['params']
dicts['scheduler']=params['scheduler']['name']
dicts['scheduler_params']=params['scheduler']['params']
dicts['batch_size']=params['batch_size']
dicts['image_repr']=params['image_repr']
dicts['epochs']=params['epochs']
dicts['model_params']=params['model']['params']
dicts['uuid']=uuid
print(dicts)
df=pd.DataFrame([dicts])
df.to_csv('results.csv',index=False,header=False if os.path.exists('results.csv') else True , mode='a' if os.path.exists('results.csv') else 'w')


