

from tensorboard import summary
import torch
from torch.utils.data import DataLoader
from funcs import load_model, faster_rcnn_resnet, CLASSES
import torchvision
from aitod import AITOD
from funcs import get_datavis_transform, show_image_with_bbox, CLASSES_MAP
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
summary_writer = SummaryWriter(log_dir='logs/datavisualization')
model=faster_rcnn_resnet(len(CLASSES),pretrained=True)
model, params=load_model(model=model,epoch=45,name='faster_rcnn_resnet',uuid='d2c42cdc-679d-11ef-a6a6-0242ac1c000c')
model.eval()
test_data=AITOD('aitod/test',transform=get_datavis_transform())
#get random image
test_loader=DataLoader(test_data,batch_size=1,shuffle=True)
for i in range(5):
    images,targets=next(iter(test_loader))
    model=model.to('cuda')
    images=images.to('cuda')
    print(targets)
    model.eval()
    with torch.no_grad():
        prediction=model(images)
        images=[image.cpu() for image in images]
        
        boxes=prediction[0]['boxes']
        labels=prediction[0]['labels']
        scores=prediction[0]['scores']
        boxes=boxes[scores>0.5]
        labels=labels[scores>0.5]
        scores=scores[scores>0.5]
        boxes=boxes.cpu()
        labels=labels.cpu()
        img=show_image_with_bbox(f'aitod/test/images/{targets["image_name"][0]}', boxes, labels,typeim='xyxy')
        
        plt.show()
        img=show_image_with_bbox(f'aitod/test/images/{targets["image_name"][0]}', targets['boxes'][0], targets['labels'][0],typeim='xyxy')
        plt.show()
        l=[CLASSES_MAP[str(label.item())] for label in labels]
        summary_writer.add_image_with_boxes('test sample', images[0], boxes, labels=l, global_step=i)
        summary_writer.add_image_with_boxes('test sample target', images[0], targets['boxes'][0], labels=[CLASSES_MAP[str(label.item())] for label in targets['labels'][0]], global_step=i)