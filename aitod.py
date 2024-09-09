# dataset is inside aitod folder.  labels are in json files. 
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from torchvision.transforms import Normalize
import torch
import json
import random
import pandas as pd
import numpy as np

CLASSES=['airplane','bridge','storage-tank','ship','swimming-pool','vehicle','person','wind-mill']

 # json file structure   
'''dict_keys(['categories', 'annotations', 'images'])
dict_keys(['id', 'name', 'supercategory'])
dict_keys(['area', 'bbox', 'category_id', 'id', 'image_id', 'iscrowd', 'segmentation'])
dict_keys(['file_name', 'id', 'width', 'height'])'''
class AITOD(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.imgs = []
        self.targets = []
        # load json file and checks if the boxes are valid
        with open(f'{root}/annotations.json') as f:
            data = json.load(f)
            dftargets = pd.DataFrame(data['annotations'])
            # used pandas for better performance
            for img in data['images']:
                labels= dftargets[dftargets['image_id']==img['id']]['category_id'].values.tolist()
                boxes = dftargets[dftargets['image_id']==img['id']]['bbox']
                if len(boxes)>0:
                    #convert to x1,y1,x2,y2 format
                    boxes = [[box[0],box[1],box[0]+box[2],box[1]+box[3]] for box in boxes if box[2]>0 and box[3]>0]
                    self.imgs.append(img['file_name'])
                
                    self.targets.append({'labels':labels, 'boxes':boxes, 'image_id':img['id']})
                
    # input of rcnn model    
    '''

        The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each image, and should be in 0-1 range. Different images can have different sizes.

        boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.

        labels (Int64Tensor[N]): the class label for each ground-truth box



'''     
# adapt the output to the input of the model
    def __getitem__(self, idx) -> tuple[torch.Tensor, dict]:	
        img = Image.open(os.path.join(self.root,'images', self.imgs[idx])).convert("RGB")
        target = self.targets[idx]
        
        target['boxes'] = torch.tensor(target['boxes'])
        target['labels'] = torch.tensor(target['labels'])
        target['image_id'] = torch.tensor([idx])
        target['image_name'] = self.imgs[idx]
        
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    
    def __len__(self):
        return len(self.imgs)
