import utils.custom_parser as custom_parser
import os
import torch
from model import *
from kitti.kitti_utils import load_stereopair,load_disparity_map

def evaluate(dataset_path,model_path,params,amount,std):

    print("Starting evaluation...")
    cuda_available = torch.cuda.is_available()

    total = 0
    bad_total = 0

    disps_list = os.listdir(os.path.join(dataset_path, "disp"))
    if amount != -1:
        disps_list = disps_list[:amount]
    else:
        amount = len(disps_list)
    dir_L = os.path.join(dataset_path, "image_left")
    dir_R = os.path.join(dataset_path, "image_right")

    network = load_net(model_path, input_planes=3)
    network = switch_to_prod(network)
    if cuda_available:
        network = network.cuda()

    for i,disp in enumerate(disps_list):
        fl = True
        path_L = os.path.join(dir_L, disp)
        path_R = os.path.join(dir_L, disp)
        disp_path = os.path.join(dataset_path, f"disp", disp)
        img_l,img_r = load_stereopair(path_L,path_R,std=std)
        img_l = torch.from_numpy(img_l)
        img_r = torch.from_numpy(img_r)
        height, width = img_l.shape[:2]
        img_l = img_l.permute(2, 0, 1)
        img_r = img_r.permute(2, 0, 1)

        if cuda_available:
            img_l = img_l.cuda()
            img_r = img_r.cuda()

        result = predict(network, img_l, img_r, params)

        ground_truth = load_disparity_map(disp_path,path_L,True)

        img_total,img_bad_total = compare_delta(result,ground_truth)

        total += img_total
        bad_total += img_bad_total

        print(f"Evaluated {i} of {amount} total.\tError:{bad_total / total * 100}%\tAccuracy: {100 - bad_total/total*100}%")

    print(f"Error : {bad_total / total * 100}%")
    print(f"Accuracy : {100 - bad_total/total*100}%")


def compare_delta(result,ground_truth,delta = 2.0):
    result = result.flatten()
    ground_truth = ground_truth.flatten()
    assert result.size() == ground_truth.size()
    result = torch.where(ground_truth < 0.5,ground_truth,result)
    res = torch.abs(result - ground_truth)
    bad_amnt = (res > 2.0).sum()
    return int(result.shape[0]),bad_amnt.item()





if __name__ == '__main__':
    mc_cnn_params = default_mc_cnn_parameters


    parser = custom_parser.CustomParser(description="Evaluation script for MC_CNN model")
    parser.add_argument("test_dataset",type=str,help="Path to test dataset")
    parser.add_argument("model",type=str,help="Path to model that being tested")
    parser.add_argument("--amount",type=int,help="Amount of disps to evaluate",default=20)

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



    args = parser.parse_args()

    std = not args.no_std
    mc_cnn_params["disp_max"] = args.disp_max
    mc_cnn_params["pi1"] = args.sgm_pi1
    mc_cnn_params["pi2"] = args.sgm_pi2
    mc_cnn_params["alpha"] = args.sgm_alpha
    mc_cnn_params["tau_so"] = args.sgm_tau_so
    mc_cnn_params["q1"] = args.sgm_q1
    mc_cnn_params["q2"] = args.sgm_q2

    evaluate(args.test_dataset,args.model,mc_cnn_params,args.amount,std)
