import cv2

from models.MC_CNN import model as mc_cnn
from models.Mask_R_CNN import predict as object_detection
from detectron2.utils.visualizer import ColorMode, Visualizer
from utils import custom_parser

import os
import torch
import time
from models.MC_CNN.kitti import kitti_utils


def stereo_detect(model,img_l,img_r,mc_cnn_params,threshold, detection_model):
    predictor = object_detection.MyPredictor(threshold, detection_model)
    result = mc_cnn.predict(model,img_l,img_r,mc_cnn_params).permute(1,2,0).cpu().numpy()
    print("Predicted disp")
    result = cv2.resize(result,(640,360))
    img = cv2.cvtColor(result,cv2.COLOR_GRAY2BGR)
    result = predictor.predict(img)
    print("Detected objects")
    return img,result

if __name__ == "__main__":
    mc_cnn_params = mc_cnn.default_mc_cnn_parameters

    parser = custom_parser.CustomParser(description="Prediction script for pit stereo detection system")
    parser.add_argument("stereo_model", help="Path to MC_CNN model's state dict file")
    parser.add_argument("detection_model", help="Path to MC_CNN model's state dict file")
    parser.add_argument("L", help="Path to left image of stereopair")
    parser.add_argument("R", help="Path to right image of stereopair")
    parser.add_argument("save_path",help="Path to save results")
    parser.add_argument("--disp-max", type=int,
                        help=f"Maximum disparity allowed for pixel. (default is {mc_cnn_params['disp_max']}",
                        default=mc_cnn_params["disp_max"])
    parser.add_argument("--no_std", action="store_true", help="Disables standartization of the images")
    parser.add_argument("--sgm_pi1", type=int,
                        help=f"PI1 Hyperparameter for semi-global matching algorythm (default is {mc_cnn_params['pi1']})",
                        default=mc_cnn_params["pi1"])
    parser.add_argument("--sgm_pi2", type=int,
                        help=f"PI2 Hyperparameter for semi-global matching algorythm (default is {mc_cnn_params['pi2']})",
                        default=mc_cnn_params["pi2"])
    parser.add_argument("--sgm_tau_so", type=int,
                        help=f"tau_so Hyperparameter for semi-global matching algorythm (default is {mc_cnn_params['tau_so']})",
                        default=mc_cnn_params['tau_so'])
    parser.add_argument("--sgm_alpha", type=int,
                        help=f"Alpha Hyperparameter for semi-global matching algorythm (default is {mc_cnn_params['alpha']})",
                        default=mc_cnn_params['alpha'])
    parser.add_argument("--sgm_q1", type=int,
                        help=f"Q1 Hyperparameter for semi-global matching algorythm (default is {mc_cnn_params['q1']})",
                        default=mc_cnn_params["q1"])
    parser.add_argument("--sgm_q2", type=int,
                        help=f"Q2 Hyperparameter for semi-global matching algorythm (default is {mc_cnn_params['q2']})",
                        default=mc_cnn_params['q2'])
    parser.add_argument("--threshold", default=0.7, type=float, help="Threshold for detection (default is 0.7)")

    args = parser.parse_args()

    std = not args.no_std
    mc_cnn_params["disp_max"] = args.disp_max
    mc_cnn_params["pi1"] = args.sgm_pi1
    mc_cnn_params["pi2"] = args.sgm_pi2
    mc_cnn_params["alpha"] = args.sgm_alpha
    mc_cnn_params["tau_so"] = args.sgm_tau_so
    mc_cnn_params["q1"] = args.sgm_q1
    mc_cnn_params["q2"] = args.sgm_q2

    filename = os.path.basename(args.L)

    model = mc_cnn.load_net(args.stereo_model).cuda()
    model = mc_cnn.switch_to_prod(model)

    img_l, img_r = kitti_utils.load_stereopair(args.L, args.R, std=std,mode="torch")
    img_l = img_l.permute(2,0,1).cuda()
    img_r = img_r.permute(2,0,1).cuda()
    img,result = stereo_detect(model,img_l,img_r,mc_cnn_params,args.threshold, args.detection_model)

    cv2.imwrite(os.path.join(args.save_path,f"{filename}_disp.png"),img)
    print("Saved disp")
    
    v = Visualizer(img[:,:,::-1],
                   scale=0.8,
                   instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                   )
    v = v.draw_instance_predictions(result["instances"].to("cpu"))
    cv2.imwrite(os.path.join(args.save_path,f"{filename}_detection.png"),cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    print("Saved detection results")






