import os
import cv2
import torch
import utils.custom_parser as custom_parser
from matplotlib import pyplot as plt

from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.config import get_cfg
from detectron2 import model_zoo


class MyPredictor():
    def __init__(self,thresh,model_path):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.WEIGHTS = model_path
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.STEPS = []
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh
        self.predictor = DefaultPredictor(cfg)

    def predict(self,img):
        return self.predictor(img)

def instance2dict(instance):
    dict = {}
    dict["pred_boxes"] = instance.pred_boxes
    dict["scores"] = instance.scores
    dict["pred_classes"] = instance.pred_classes
    dict["pred_masks"] = instance.pred_masks
    return dict



if __name__ == "__main__":
    parser = custom_parser.CustomParser(description="Prediction module for Mask MC_CNN with visualization function.")
    parser.add_argument("img",help="Single image or directory with images to run detection on")
    parser.add_argument("model",help="Model weights file path")
    parser.add_argument("--verbose",default=False,action="store_true",help="If set, displays plot with detection"
                                                                           "(single-image only)")
    parser.add_argument("--save_dir",help="If set, saves detection results into this directory")
    parser.add_argument("--save_verbose",default=False,action="store_true",
                        help="If set, saves visualized detection into save directory")
    parser.add_argument("--threshold",default=0.7,type=float,help="Threshold for detection (default is 0.7)")

    args = parser.parse_args()

    predictor = MyPredictor(args.threshold,args.model)
    imgs = []
    if os.path.isdir(args.img):
        imgs_names = os.listdir(args.img)
        for name in imgs_names:
            imgs.append(cv2.imread(os.path.join(args.img,name)))
    elif os.path.isfile(args.img):
        imgs_names = [args.img]
        imgs = [cv2.imread(args.img)]
        single_image = True
    else:
        raise ValueError("Image path provided is not an image file nor a directory with images")

    predictions = []
    for i,img in enumerate(imgs):
        predictions.append(predictor.predict(img))
        print(f"Processed {imgs_names[i]}")

    if len(predictions) == 1:
        img = imgs[0]
        prediction = predictions[0]
        v = Visualizer(img[:, :, ::-1],
                       scale=0.8,
                       instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                       )
        v = v.draw_instance_predictions(prediction["instances"].to("cpu"))
        f, axxs = plt.subplots(2, figsize=(14, 10))
        axxs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axxs[1].imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
        plt.show()

    if not args.save_dir is None:
        for i,prediction in enumerate(predictions):
            img_name = imgs_names[i]
            if args.save_verbose:
                img = imgs[i]
                v = Visualizer(img[:, :, ::-1],
                               scale=0.8,
                               instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                               )
                v = v.draw_instance_predictions(prediction["instances"].to("cpu"))
                cv2.imwrite(os.path.join(args.save_dir,f"{img_name[:-4]}_detection.png"),v.get_image())
            torch.save(instance2dict(prediction["instances"]),
                       os.path.join(args.save_dir,f"{img_name[:-4]}_detection_data.pth"))
