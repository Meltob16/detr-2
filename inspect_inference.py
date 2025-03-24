import torch
import torchvision
import matplotlib.pyplot as plt
import argparse
from datasets import build_dataset
from models import build_model
import torchvision.transforms as T
import PIL

CLASSES = [
    'N/A', 'ship'
]

class Args(argparse.Namespace):
    lr=1e-5
    lr_backbone=1e-6
    batch_size=2
    weight_decay=1e-4
    epochs=2
    lr_drop=200
    clip_max_norm=0.1
    frozen_weights=None
    backbone='resnet50'
    dilation=False
    position_embedding='sine'
    enc_layers=6
    dec_layers=6
    dim_feedforward=2048
    hidden_dim=256
    dropout=0.1
    nheads=8
    num_classes = 2
    num_queries=100
    pre_norm=False
    masks=False
    aux_loss=True
    set_cost_class=1
    set_cost_bbox=5
    set_cost_giou=2
    mask_loss_coef=1
    dice_loss_coef=1
    bbox_loss_coef=5
    giou_loss_coef=2
    eos_coef=0.1
    dataset_file='coco'
    coco_path='c:/datasets/sentinel2_coco'
    coco_panoptic_path=None
    remove_difficult=False
    output_dir=''
    device='cpu'
    seed=42
    resume='C:/repos/detr-untouched/models/out/sentinel2_5_epochs/checkpoint.pth'
    start_epoch=0
    eval=True
    num_workers=2
    world_size=1
    dist_url='env://'
    distributed=False

args = Args()

# coco_idx_to_label = {idx: label for idx, label in enumerate(CLASSES)}

transform_rgb = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        output_tensor = []
        for t, m, s in zip(tensor, self.mean, self.std):
            output_tensor.append(t.mul(s).add(m))
            # t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return torch.stack(output_tensor, dim=0)

unnorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(img, labels, boxes, mask=None, norm = True):
    if (type(img) == PIL.Image.Image):
        h = img.height
        w = img.width
    else:
        h, w = img.shape[1:]

    if mask != None:
        # width
        if torch.where(mask[0])[0].shape[0] > 0:
            mask_w = torch.where(mask[0])[0][0]
            w = min(w, mask_w)
        if torch.where(mask[:, 0])[0].shape[0]:
            mask_h = torch.where(mask[:, 0])[0][0]
            h = min(h, mask_h)
            
    boxes = rescale_bboxes(boxes, (w, h))
    plt.figure(figsize=(16,10))
    if norm:
        img = unnorm(img)
    #image = (unimage*256).to(torch.uint8)
    pil_img = torchvision.transforms.functional.to_pil_image(img)
    plt.imshow(pil_img)
    
    ax = plt.gca()
    colors = COLORS * 100
    for label, (xmin, ymin, xmax, ymax), c in zip(labels, boxes.tolist(), colors):
        print(label)
        print(xmin, ymin, xmax, ymax)
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=1))
        text = f'{CLASSES[label.argmax()]}'
        ax.text(xmin, ymin, text, fontsize=12, color='red')
                # bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

import time
import math
from PIL import Image
import os
def postprocess_img(img_path): 
  im = Image.open(img_path)
    #   mean-std normalize the input image (batch-size: 1)
  img = transform_rgb(im).unsqueeze(0)
#   im = unnorm(im)
  # propagate through the model
  start = time.time()
  outputs = model(img)
  end = time.time()
  print(f'Prediction time per image: {math.ceil(end - start)}s ', )

  probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
  keep = probas.max(-1).values > 0.4

   # convert boxes from [0; 1] to image scales
#   bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
  import numpy as np
#   img = img.squeeze(0)
  to_torch = T.ToTensor()
  im = to_torch(im)
  ## min-max normalization
  min_val = im.min()
  max_val = im.max()
  image_norm = (im - min_val) / (max_val - min_val)
  image_sqrt = np.sqrt(image_norm)
  plot_results(image_sqrt, probas[keep], outputs['pred_boxes'][0, keep], norm = False)

## show samples from the dataset
dataset_train = build_dataset(image_set='val', args=args)
print(len(dataset_train[0]))
print(dataset_train[0][0].shape)
print(dataset_train[1][0].shape)
dataset_train[0][1]

example = dataset_train[0]
# plot_results(example[0], example[1]["labels"], example[1]["boxes"])

## show sample from model
# load the model
model, criterion, postprocessors = build_model(args)
TRAINED_PATH = 'C:/repos/detr-untouched/models/out/sentinel2_5_epochs/checkpoint.pth'
checkpoint = torch.load(TRAINED_PATH, map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['model'], strict=False)
TEST_IMAGE_PATH = 'c:/datasets/sentinel2_coco/valid'

img_format = {'jpg', 'png', 'jpeg', 'tif', 'tiff'}
paths = list()

for obj in os.scandir(TEST_IMAGE_PATH):
    if obj.is_file() and obj.name.split(".")[-1] in img_format:
        paths.append(obj.path)

for img_path in paths[10:15]:
    postprocess_img(img_path)