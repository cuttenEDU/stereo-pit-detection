import utils.custom_parser as custom_parser
import model
import os
import ntpath
import kitti.kitti_utils as kitti_utils
import torch
import cv2 as cv

def load_stereopair(L,R,std):
    img_l, img_r = kitti_utils.load_stereopair(L, R, std=std)
    img_l = torch.from_numpy(img_l).permute(2, 0, 1).cuda()
    img_r = torch.from_numpy(img_r).permute(2, 0, 1).cuda()
    return img_l,img_r

if __name__ == "__main__":

    mc_cnn_params = model.default_mc_cnn_parameters

    parser = custom_parser.CustomParser(description="Prediction script for MC_CNN stereo-vision neural network model. "
                                                    "Can run in single and multiple stereo-pairs modes (check arguments for more)"
                                                    "(in second case images in L and R directories from the same stereo-pair  MUST have same names)")

    parser.add_argument("model",help="Path to MC_CNN model's state dict file")
    parser.add_argument("L",help="Path to left image of stereopair OR directory with left images of stereopairs "
                                 "\n(in second case images in L and R directories from the same stereo-pair  MUST have same names)")
    parser.add_argument("R",help="Path to right image of stereopair OR directory with right images of stereopairs"
                                 "\n((in second case images in L and R directories from the same stereo-pair  MUST have same names)")
    parser.add_argument("--save-path",help="Path for saving disparities. They will be saved as .png images (default is directory with stereo-pair)")
    parser.add_argument("--disp-max",type=int,help=f"Maximum disparity allowed for pixel. (default is {mc_cnn_params['disp_max']}",
                        default=mc_cnn_params["disp_max"])
    parser.add_argument("--no_std",action="store_true",help="Disables standartization of the images")
    parser.add_argument("--visualize",action="store_true",help="Enables visualization of algorythm (available only in single stereopair mode")
    parser.add_argument("--sgm_pi1",type=int,help=f"PI1 Hyperparameter for semi-global matching algorythm (default is {mc_cnn_params['pi1']})",
                        default=mc_cnn_params["pi1"])
    parser.add_argument("--sgm_pi2",type=int,help=f"PI2 Hyperparameter for semi-global matching algorythm (default is {mc_cnn_params['pi2']})",
                        default=mc_cnn_params["pi2"])
    parser.add_argument("--sgm_tau_so",type=int,help=f"tau_so Hyperparameter for semi-global matching algorythm (default is {mc_cnn_params['tau_so']})",
                        default=mc_cnn_params['tau_so'])
    parser.add_argument("--sgm_alpha",type=int,help=f"Alpha Hyperparameter for semi-global matching algorythm (default is {mc_cnn_params['alpha']})",
                        default=mc_cnn_params['alpha'])
    parser.add_argument("--sgm_q1",type=int,help=f"Q1 Hyperparameter for semi-global matching algorythm (default is {mc_cnn_params['q1']})",
                        default=mc_cnn_params["q1"])
    parser.add_argument("--sgm_q2",type=int,help=f"Q2 Hyperparameter for semi-global matching algorythm (default is {mc_cnn_params['q2']})",
                        default=mc_cnn_params['q2'])

    args = parser.parse_args()
    std = not args.no_std
    mc_cnn_params["disp_max"] = args.disp_max
    mc_cnn_params["pi1"] = args.sgm_pi1
    mc_cnn_params["pi2"] = args.sgm_pi2
    mc_cnn_params["alpha"] = args.sgm_alpha
    mc_cnn_params["tau_so"] = args.sgm_tau_so
    mc_cnn_params["q1"] = args.sgm_q1
    mc_cnn_params["q2"] = args.sgm_q2

    mc_cnn = model.load_net(args.model).cuda()
    mc_cnn = model.switch_to_prod(mc_cnn)

    if os.path.isdir(args.L) and os.path.isdir(args.R):
        image_mode = False
    elif os.path.isfile(args.L) and os.path.isfile(args.R):
        image_mode = True
    else:
        raise ValueError("Incorrect L and R arguments. They must be either paths to 2 images or paths to 2 directories with images")

    if image_mode:
        img_l,img_r = load_stereopair(args.L,args.R,std)
        result = model.predict(mc_cnn,img_l,img_r,mc_cnn_params,args.visualize)
        filename = ntpath.basename(args.L)
        save_path = os.path.join(args.L, "..",f"{filename}_disp.png")
        if os.path.isfile(save_path):
            ans = ''
            while ans != 'y' and ans != 'n':
                ans = input("Save file already exists, rewrite? (y/n)")
            if (ans == 'n'):
                exit()

        cv.imwrite(os.path.join(args.L, "..",f"{filename}_disp.png"), result.permute(1, 2, 0).numpy())

    else:
        imgs = os.listdir(args.L)
        save_path = os.path.join(args.L, "..","disps")
        if os.path.isdir(save_path):
            ans = ''
            while ans != 'y' and ans != 'n':
                ans = input("Save dir already exists, rewrite? (y/n)")
            if (ans == 'n'):
                exit()
        os.mkdir(save_path)
        for img_name in imgs:
            print(f"Processing image {img_name}")
            L = os.path.join(args.L, img_name)
            R = os.path.join(args.R, img_name)
            img_l,img_r = load_stereopair(L,R,std)
            result = model.predict(mc_cnn, img_l, img_r, mc_cnn_params)
            cv.imwrite(os.path.join(save_path, f"{img_name}_disp.png"), result.permute(1, 2, 0).numpy())









