import torch
import main
import argparse
if __name__ == "__main__":
    pretrained = True

    if pretrained:
        # Get pretrained weights
        checkpoint = torch.hub.load_state_dict_from_url(
                    url='https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth',
                    map_location='cpu',
                    check_hash=True)
        # Remove class weights
        del checkpoint["model"]["class_embed.weight"]
        del checkpoint["model"]["class_embed.bias"]

        # SaveOGH
        torch.save(checkpoint,
                'detr-r50_no-class-head.pth')
        
    resume_model = "detr-r50_no-class-head.pth" if pretrained else "scratch"

    class Args(argparse.Namespace):
        lr=1e-5
        lr_backbone=1e-6
        batch_size=2
        weight_decay=1e-4
        epochs=5
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
        output_dir='models/out/sentinel2_5_epochs'
        device='cuda'
        seed=42
        resume=resume_model
        start_epoch=0
        eval=False
        num_workers=2
        world_size=1
        dist_url='env://'
        distributed=False

    main.main(Args())