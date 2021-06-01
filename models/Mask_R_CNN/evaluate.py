import utils.custom_parser as custom_parser

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator,inference_on_dataset
from detectron2.data import build_detection_test_loader


def evaluate(dataset_path,json_path,model_path):
    register_coco_instances("my_dataset", {}, json_path, dataset_path)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("my_dataset",)
    cfg.DATASETS.TEST = ("my_dataset",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_path
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("my_dataset", ("bbox", "segm"), False)
    val_loader = build_detection_test_loader(cfg, "my_dataset")

    res = inference_on_dataset(predictor.model, val_loader, evaluator)

    return {"AP_bbox":res["bbox"]["AP"],"AP_segm":res["segm"]["AP"]}


if __name__ == "__main__":
    parser = custom_parser.CustomParser(description="Evalutaion module for the Mask_R_CNN model.")
    parser.add_argument("dataset",help="Directory with evaluation data")
    parser.add_argument("json",help="JSON labels file path")
    parser.add_argument("model",help="Model weights file path")

    args = parser.parse_args()
    res = evaluate(args.dataset,args.json,args.model)
    print()
    print(f"Average precision in object detection: {res['AP_bbox']}")
    print(f"Average precision in instance segmentation: {res['AP_segm']}")