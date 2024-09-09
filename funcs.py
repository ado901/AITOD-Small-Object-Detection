from enum import Enum
from functools import partial
import torchvision.transforms as T
import torch,torchvision
import os
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import retinanet as rn
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.retinanet import RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import cv2
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from torchvision import transforms

def get_train_transform():
    '''
    This function returns a composition of transformations to be applied to the training dataset.
    '''	
    return T.Compose([
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float),
        T.RandomHorizontalFlip(p=0.5), 
        
        
    ])
def get_val_transform():
    '''
    This function returns a composition of transformations to be applied to the validation dataset.'''
    return T.Compose([
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float),
        
    ])
def get_datavis_transform():
    '''this function returns a composition of transformations to be applied to the dataset for visualization'''
    return T.Compose([
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float),
        
    ])
    
def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))

# classes in the dataset. CLASSES_MAP is a dictionary that maps the class number to the class name    
CLASSES=['airplane','bridge','storage-tank','ship','swimming-pool','vehicle','person','wind-mill','background']
""" e.g. {"1": "pedestrian"} """
CLASSES_MAP={str(i):CLASSES[i] for i in range(len(CLASSES))}


def faster_rcnn_resnet(numclasses, pretrained=True):
    '''this function returns a faster rcnn model with a resnet backbone, it can be pretrained or not
    Args:
        numclasses: int, number of classes in the dataset
        pretrained: bool, if True the model is pretrained on COCO dataset'''
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # change the number of classes in the model
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features,numclasses)
    return model
def retinanet_resnet(numclasses, pretrained=True):
    '''
    This function returns a retinanet model with a resnet backbone, it can be pretrained or not
    Args:
        numclasses: int, number of classes in the dataset
        pretrained: bool, if True the model is pretrained on COCO dataset
    '''
    model=torchvision.models.detection.retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None)
    num_anchors = model.head.classification_head.num_anchors
    in_features = model.backbone.out_channels
    # need to change the classification head to match the number of classes
    model.head.classification_head = rn.RetinaNetClassificationHead(
        in_channels=in_features,
        num_anchors=num_anchors,
        num_classes=numclasses,
    )
    return model
#tiny objects
def custom_RPN_faster_rcnn_resnet(numclasses, pretrained=True):
    '''
    This function returns a faster rcnn model with a resnet backbone and a custom RPN, it can be pretrained or not
    there are 5 smaller anchor sizes and 3 aspect ratios
    Args:
        numclasses: int, number of classes in the dataset
        pretrained: bool, if True the backbone is pretrained on ImageNet1K dataset
    '''
    
    backbone = torchvision.models.resnet50(pretrained=torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None)
    backbone.out_channels = 256
    backbone=torchvision.models.detection.backbone_utils.resnet_fpn_backbone('resnet50', weights='DEFAULT')
    #anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 0.8, 0.2),))
    anchor_sizes = ((4,), (8,), (16,), (32,), (64,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

    anchor_generator = AnchorGenerator(
                    anchor_sizes, aspect_ratios
                )

    model = FasterRCNN(backbone, num_classes=numclasses, rpn_anchor_generator=anchor_generator)
    return model
    
    
    
    
class MODELS(Enum):
    '''Enum class to choose the model to use
    each model is a function that returns the model with the specified backbone'''
    FASTER_RCNN_RESNET = faster_rcnn_resnet
    RETINANET_RESNET= retinanet_resnet
    FASTER_RCNN_RESNET_RPN=custom_RPN_faster_rcnn_resnet
def load_model(epoch, model,name, uuid):
    '''This function loads the model from the weights folder
    Args:
            epoch: int, the epoch to load the model from
            model: object, the model to load
            name: str, the name of the model
            uuid: str, the uuid of the model'''
    # function to load the model
    check_path = os.path.join('weights',name, f'checkpoint_{epoch}_{uuid}.pth')
    checkpoint = torch.load(check_path,weights_only=False, map_location=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    params=checkpoint['params']
    print("Model loaded!")
    return model, params

'''As input to forward and update the metric accepts the following input:

preds (~List): A list consisting of dictionaries each containing the key-values (each dictionary corresponds to a single image). Parameters that should be provided per dict
boxes (~torch.Tensor): float tensor of shape (num_boxes, 4) containing num_boxes detection boxes of the format specified in the constructor.
By default, this method expects (xmin, ymin, xmax, ymax) in absolute image coordinates, but can be changed
using the box_format parameter. Only required when iou_type="bbox".
scores (~torch.Tensor): float tensor of shape (num_boxes) containing detection scores for the boxes.
labels (~torch.Tensor): integer tensor of shape (num_boxes) containing 0-indexed detection classes for the boxes.
masks (~torch.Tensor): boolean tensor of shape (num_boxes, image_height, image_width) containing boolean masks. Only required when iou_type="segm".'''
def evaluate(model, test_loader, device) -> MeanAveragePrecision:
    '''This function evaluates the model on the test set, it uses a mean average precision metric library from torchmetrics'''
    model.to(device)
    model.eval()
    bar=tqdm(test_loader)
    mAP=MeanAveragePrecision(box_format='xyxy',iou_type='bbox')
    for data in bar:
        images, targets = data
        images=[image.to(device) for image in images]
        targets=[{k: (v.to(device) if k != 'image_name' else v) for k, v in t.items()} for t in targets]
        with torch.no_grad():
            output = model(images, targets)
            preds=[]
            target=[]
            for i in range(len(output)):
                pred={'boxes':output[i]['boxes'],'scores':output[i]['scores'],'labels':output[i]['labels']}
                targ={'boxes':targets[i]['boxes'],'labels':targets[i]['labels']}
                
                
                # FIX: there are cases where the number of labels is different from the number of boxes, this is a workaround
                num_pred_boxes = len(pred['boxes'])
                num_pred_labels = len(pred['labels'])
                num_target_boxes = len(targ['boxes'])
                num_target_labels = len(targ['labels'])

                # Aggiusta le etichette previste se necessario
                if num_pred_labels > num_pred_boxes:
                    pred['labels'] = pred['labels'][:num_pred_boxes]
                elif num_pred_labels < num_pred_boxes:
                    pred['labels'].extend([0] * (num_pred_boxes - num_pred_labels))  # Aggiungi etichette di riempimento (es. 9)

                # Aggiusta le etichette reali se necessario
                if num_target_labels > num_target_boxes:
                    targ['labels'] = targ['labels'][:num_target_boxes]
                elif num_target_labels < num_target_boxes:
                    targ['labels'].extend([0] * (num_target_boxes - num_target_labels))  # Aggiungi etichette di riempimento (es. 9)
                preds.append(pred)
                target.append(targ)
                """ print(preds[0].values())
                print(target[0].values()) """
                
            mAP.update(preds=preds,target=target)


            
    print("Evaluation done!")
    return mAP

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
def show_image_with_bbox(image, boxes, labels,typeim='xywh'):
    '''This function shows the image with the bounding boxes, the boxes are in the format xywh or xyxy'''
    if type(image)==str:
        image = Image.open(image)
    fig, ax = plt.subplots()
    ax.imshow(image)
    for i in range(len(boxes)):
        box = boxes[i]
        label = labels[i]
        # convert xywh, rect requires xywh
        if typeim=='xywh':
            rect = patches.Rectangle((box[0],box[1]),box[2],box[3],linewidth=1,edgecolor='r',facecolor='none')
        else:
            rect = patches.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        plt.text(box[0],box[1],CLASSES[label],color='red')
        
    plt.title('Image with Bounding Boxes')
    return fig
import json

from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt

# get gradients list for each layer in the network (from tfonta github)
def add_gradient_hist(net:torch.nn.Module):
    '''this function returns a histogram of the gradients for each layer in the network'''
    ave_grads = [] 
    layers = []
    for n,p in net.named_parameters():
        if ("bias" not in n):
            layers.append(n)
            if p.requires_grad: 
                ave_grad = np.abs(p.grad.clone().detach().cpu().numpy()).mean()
            else:
                ave_grad = 0
            ave_grads.append(ave_grad)
        
    layers = [layers[i].replace(".weight", "") for i in range(len(layers))]
    
    fig = plt.figure(figsize=(12, 12))
    plt.bar(np.arange(len(ave_grads)), ave_grads, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation=90)
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=np.max(ave_grads) / 2)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    #plt.grid(True)
    plt.legend([Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['mean-gradient', 'zero-gradient'])
    plt.tight_layout()
    #plt.show()
    
    return fig

#example: "aitod/train/images/000000000139.jpg"
# annotation is in "aitod/train/annotations.json"
'''dict_keys(['categories', 'annotations', 'images'])
dict_keys(['id', 'name', 'supercategory'])
dict_keys(['area', 'bbox', 'category_id', 'id', 'image_id', 'iscrowd', 'segmentation'])
dict_keys(['file_name', 'id', 'width', 'height'])'''
def get_bbox_from_image(imagePath:str)-> tuple[list[list[int]],list[int]]:
    '''This function retrieves the bounding boxes and labels from an image path using the annotations.json file'''	
    jsonpath= os.path.join(os.path.dirname(imagePath).replace('images',''),'annotations.json')
    with open (jsonpath) as f:
        data=json.load(f)
        image_name=os.path.basename(imagePath)
        ann=data['annotations']
        bboxes=[]
        labels=[]
        image_id=[img['id'] for img in data['images'] if img['file_name']==image_name][0]
        bboxes=[ann['bbox'] for ann in ann if ann['image_id']==image_id]
        labels=[ann['category_id'] for ann in ann if ann['image_id']==image_id]
        return bboxes,labels
    
def get_count_per_class(data: dict) -> dict:
    """
    Count per class given COCO annotations in dict format.
    """
    id_to_class_name = {x['id']: x['name'] for x in data['categories']} 
    annotations = data['annotations']
    
    counts = {}
    for annotation in annotations:
        class_name = id_to_class_name[annotation['category_id']]
        counts[class_name] = counts.get(class_name, 0) + 1
        
    return counts