# import some common detectron2 uCONFIG_FILEtilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2

CONFIG_FILE="./configs/COCO-InstanceSegmentation/mask_cascade_rcnn_ResNeSt_50_FPN_syncBN_1x.yaml"

# get image
# !wget http://images.cocodataset.org/val2017/000000439715.jpg -O input.jpg
im = cv2.imread("./input.jpg")

# Create config
cfg = get_cfg()
cfg.merge_from_file(CONFIG_FILE)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/mask_cascade_rcnn_ResNeSt_50_FPN_syncBN_1x/137851257/model_final_f6e8b1.pkl"

# Create predictor
predictor = DefaultPredictor(cfg)

# Make prediction
outputs = predictor(im)

print(outputs)
