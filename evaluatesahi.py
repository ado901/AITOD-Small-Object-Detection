from torchmetrics.detection.mean_ap import MeanAveragePrecision
from sahi.predict import get_sliced_prediction

from sahi import AutoDetectionModel
from funcs import CLASSES, get_val_transform,load_model,MODELS
from aitod import AITOD
import torch

# like evaluate.py, but with sahi
# is far more slower than evaluate.py because of slicing and not parallelizing the predictions. To test faster_rcnn_resnet
# it can take up to 2 hours on my laptop
modelchoice=MODELS.FASTER_RCNN_RESNET
model= modelchoice(numclasses=len(CLASSES))
name=modelchoice.__repr__().split(' ')[1]
uuid='d2c42cdc-679d-11ef-a6a6-0242ac1c000c'
date='2024-08-31-13-35-01'
model, params=load_model(45,model,name, uuid=uuid)
mAP=MeanAveragePrecision(box_format='xyxy',iou_type='bbox')
test_dataset=AITOD('aitod/test',transform=get_val_transform())
sahi_model = AutoDetectionModel.from_pretrained("torchvision", model=model, device="cuda:0")
import tqdm
bar=tqdm.tqdm(test_dataset)
for image, target in bar:
    result=get_sliced_prediction(image=f"aitod/test/images/{target['image_name']}",detection_model=sahi_model,verbose=0,slice_height=512,slice_width=512,overlap_height_ratio=0.2,overlap_width_ratio=0.2)
    boxes=[[box.bbox.minx,box.bbox.miny,box.bbox.maxx,box.bbox.maxy] for box in result.object_prediction_list]
    labels=[box.category.id for box in result.object_prediction_list]
    scores=[box.score.value for box in result.object_prediction_list]
    num_pred_boxes = len(boxes)
    num_pred_labels = len(labels)
    num_target_boxes = len(target['boxes'])
    num_target_labels = len(target['labels'])

    # Aggiusta le etichette previste se necessario
    if num_pred_labels > num_pred_boxes:
        labels = labels[:num_pred_boxes]
    elif num_pred_labels < num_pred_boxes:
        labels.extend([0] * (num_pred_boxes - num_pred_labels))  # Aggiungi etichette di riempimento (es. 9)

    # Aggiusta le etichette reali se necessario
    if num_target_labels > num_target_boxes:
        target['labels'] = target['labels'][:num_target_boxes]
    elif num_target_labels < num_target_boxes:
        target['labels'].extend([0] * (num_target_boxes - num_target_labels))  # Aggiungi etichette di riempimento (es. 9)
    preds=[{'boxes':torch.tensor(boxes),'labels':torch.tensor(labels),'scores':torch.tensor(scores)}]
    target=[{'boxes':target['boxes'],'labels':target['labels']}]
    mAP.update(preds,target)
a=mAP.compute()
with open('resultsSahi.txt','w') as f:
    f.write(str(a))

