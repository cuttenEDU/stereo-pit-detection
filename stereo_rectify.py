import cv2 as cv
import numpy as np
import os
from utils import custom_parser





def rectify(args):
    path = args.input_path

    output_path = args.input_path if args.output_path is None else args.output_path

    if not os.path.exists(os.path.join(output_path,"L_rect")):
        os.mkdir(os.path.join(output_path,"L_rect"))
    if not os.path.exists(os.path.join(output_path,"R_rect")):
        os.mkdir(os.path.join(output_path,"R_rect"))

    calib_data_path = os.path.join(path, "calib.npz") if args.calib_data is None else args.calib_data

    calib_data = np.load(calib_data_path)

    m_L = calib_data.get("m_L")
    m_R = calib_data.get("m_R")
    d_L = calib_data.get("d_L")
    d_R = calib_data.get("d_R")
    P_L = calib_data.get("P_L")
    P_R = calib_data.get("P_R")
    R_L = calib_data.get("R_L")
    R_R = calib_data.get("R_R")

    img_size = tuple(map(int,args.img_size.split('x')))

    map1_1,map1_2 = cv.initUndistortRectifyMap(m_L,d_L,R_L,P_L,img_size,cv.CV_16SC2)
    map2_1,map2_2 = cv.initUndistortRectifyMap(m_R,d_R,R_R,P_R,img_size,cv.CV_16SC2)



    for i,img_name in enumerate(os.listdir(os.path.join(path, "L"))):
        img = cv.imread(os.path.join(path, "L", img_name))
        img = cv.resize(img,img_size)
        res = cv.remap(img,map1_1,map1_2,cv.INTER_LINEAR)
        cv.imwrite(os.path.join(output_path, "L_rect", f"{i}.png"),cv.resize(res,(960,540)))

    print(f"Processed {i+1} left images")

    for i,img_name in enumerate(os.listdir(os.path.join(path, "R"))):
        img = cv.imread(os.path.join(path, "R", img_name))
        img = cv.resize(img,img_size)
        res = cv.remap(img,map2_1,map2_2,cv.INTER_LINEAR)
        cv.imwrite(os.path.join(output_path, "R_rect", f"{i}.png"),cv.resize(res,(960,540)))

    print(f"Processed {i+1} right images")

if __name__ == "__main__":
    parser = custom_parser.CustomParser()
    parser.add_argument("input_path", type=str,
                        help="Path to directory with /L and /R folders with corresponding left and right imgs")
    parser.add_argument("--calib_data", type=str, help="Path to file with stereo calibration data "
                                                       "(looks in input_path dir for calib.npz file by default)")
    parser.add_argument("--img_size", type=str, help="Size of the output image wxh (960x540 by default)",
                        default="1280x720")
    parser.add_argument("--output_path", type=str,
                        help="Path to output directory, where /L_rect and /R_rect folders will be created OR filled \n"
                             "with rectified imgs")

    args = parser.parse_args()


    rectify(args)