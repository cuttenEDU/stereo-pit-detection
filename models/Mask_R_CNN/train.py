import torch
import utils.custom_parser as custom_parser
import os
import json
from datetime import datetime

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer,HookBase
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
setup_logger()

class PeriodicSaveHook(HookBase):
    def __init__(self,period,save_path):
        super().__init__()
        self.period = period
        self.save_path = save_path

    def after_step(self):
        if self.trainer.iter % self.period == 0 and self.trainer.iter != 0:
            savename = f"model-{datetime.now().strftime('%m:%d:%Y-%H:%M:%S')}-iter#{self.trainer.iter}"
            savepath = os.path.join(self.save_path,savename)
            torch.save(self.trainer.model.state_dict(),savepath)
            print(f"Saved as {savename}")



def train(dataset_path,json_path,save_path,params):

    register_coco_instances("pit_dataset", {}, json_path, dataset_path)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("pit_dataset",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = params["bs"]
    cfg.SOLVER.BASE_LR = params["lr"]
    cfg.SOLVER.MAX_ITER = params["max_iter"]
    cfg.OUTPUT_DIR = save_path
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = params["num_classes"]

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)

    trainer.register_hooks([PeriodicSaveHook(params["save_each"],cfg.OUTPUT_DIR)])
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    params = {
        "max_iter":3000,
        "lr":0.00025,
        "bs":2,
        "num_classes":1,
        "save_each":100
    }

    parser = custom_parser.CustomParser(description="Prediction module for Mask MC_CNN with visualization function.")
    parser.add_argument("dataset",help="Training dataset path")
    parser.add_argument("json",help="JSON file with labels path")
    parser.add_argument("save_dir",help="Where to save models' weights")
    parser.add_argument("--max_iter",default=params["max_iter"],type=int,help=f"Maximum number of training iterations"
                                                                  f"(default is {params['max_iter']})")
    parser.add_argument("--lr",default=params['lr'],type=float,help=f"Learning rate (default is {params['lr']})")
    parser.add_argument("--bs",default=params['bs'],type=int,help=f"Amount of images in 1 iteration "
                                                                  f"(default is {params['bs']})")
    parser.add_argument("--num_classes",default=params['num_classes'],type=int,help=f"Number of classes to classify"
                                                                                    f"default is {params['num_classes']}")
    parser.add_argument("--save_each",default=params['save_each'],type=int,help="Model saing rate in iterations"
                                                                                f"default is {params['save_each']}")

    args = parser.parse_args()

    params["max_iter"] = args.max_iter
    params["lr"] = args.lr
    params["bs"] = args.bs
    params["num_classes"] = args.num_classes
    params["save_each"] = args.save_each

    print(f"Starting training with parameters: {json.dumps(params,indent=2)}")

    train(args.dataset,args.json,args.save_dir,params)





