Models = {
    'CenterNet':
        {
            'folder': 'centernet_mobilenetv2_fpn_od',
            'ckpt': 'ckpt-301'
        },
    'EfficientDet':
        {
            'folder': 'efficientdet_d2_coco17_tpu-32',
            'ckpt': 'ckpt-0'
        },
    'SSDmobileNet':
        {
            'folder': 'ssd_mobilenet_v2_320x320_coco17_tpu-8',
            'ckpt': 'ckpt-0'
        },
    'SSDresNet50':
        {
            'folder': 'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8',
            'ckpt': 'ckpt-0'
        },
    'FasterRCNN':
        {
            'folder': 'faster_rcnn_resnet50_v1_1024x1024_coco17_tpu-8',
            'ckpt': 'ckpt-0'
        }

}
from .ObjDetector import ObjectDetector
