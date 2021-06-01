import cv2 as cv
import numpy as np
import os
from utils import custom_parser
from matplotlib import pyplot as plt



def calibrate(args):

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    n_corners = tuple(map(int,args.grid.split("x")))

    path = args.input_path
    output_path = args.output if not args.output is None else os.path.join(path, "calib.npz")



    v_corners = n_corners[0]
    h_corners = n_corners[1]

    objp = np.zeros((h_corners*v_corners,3), np.float32)
    objp[:,:2] = np.mgrid[0:v_corners,0:h_corners].T.reshape(-1,2)


    imgs_l = []
    imgs_r = []

    paths_l = os.listdir(os.path.join(path, "L_calib"))
    paths_r = os.listdir(os.path.join(path, "R_calib"))

    img_size = (1280,720)

    for img in paths_l:
        i = cv.imread(os.path.join(path, "L_calib", img))
        i = cv.resize(i,img_size)
        imgs_l.append(i)

    for img in paths_r:
        i = cv.imread(os.path.join(path, "R_calib", img))
        i = cv.resize(i,img_size)
        imgs_r.append(i)

    l_checkers_imgs = {}
    l_objpoints = [] # 3d point in real world space
    l_imgpoints = [] # 2d points in image plane.

    r_checkers_imgs = {}
    r_objpoints = [] # 3d point in real world space
    r_imgpoints = [] # 2d points in image plane.

    for i,img in enumerate(imgs_l):
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCornersSB(gray, n_corners,None)
        if ret:
            l_objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            l_imgpoints.append(corners2)

            img = cv.drawChessboardCorners(img, n_corners, corners2,ret)
            l_checkers_imgs[f"{i}.png"] = img

    for i,img in enumerate(imgs_r):
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCornersSB(gray, n_corners,None)
        if ret:
            r_objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            r_imgpoints.append(corners2)

            img = cv.drawChessboardCorners(img, n_corners, corners2,ret)
            r_checkers_imgs[f"{i}.png"] = (img)

    assert len(r_objpoints) == len(l_objpoints)
    print(f"Calibrating on {len(r_objpoints)} checkers stereopairs")

    m_L = cv.initCameraMatrix2D(l_objpoints, l_imgpoints, img_size, 0)
    m_R = cv.initCameraMatrix2D(r_objpoints, r_imgpoints, img_size, 0)
    d_L = np.zeros((1, 5))
    d_R = np.zeros((1, 5))
    ret, m_L, d_L, m_R, d_R, R, T, E, F = cv.stereoCalibrate(l_objpoints, l_imgpoints, r_imgpoints, m_L, d_L, m_R, d_R, img_size,
                                                             flags=cv.CALIB_FIX_INTRINSIC + cv.CALIB_FIX_FOCAL_LENGTH)
    print(f"Reprojection error: {ret}")

    R_L, R_R, P_L, P_R, Q, roi_1, roi_2 = cv.stereoRectify(m_L, d_L, m_R, d_R, img_size, R, T, None, None, None, None, None, None, 1, gray.shape[::-1])

    print(f"Left ROI: {roi_1}")
    print(f"Right ROI: {roi_2}")

    if args.verbose:
        map1_1, map1_2 = cv.initUndistortRectifyMap(m_L, d_L, R_L, P_L, img_size, cv.CV_16SC2)
        map2_1, map2_2 = cv.initUndistortRectifyMap(m_R, d_R, R_R, P_R, img_size, cv.CV_16SC2)
        f, axxs = plt.subplots(2, 2)
        axxs[0,0].imshow(imgs_l[0])
        axxs[0,0].set_title("Left original image")
        axxs[0,1].imshow(imgs_r[0])
        axxs[0,1].set_title("Right original image")
        res_l = cv.remap(imgs_l[0], map1_1, map1_2, cv.INTER_LINEAR)
        res_r = cv.remap(imgs_r[0], map2_1, map2_2, cv.INTER_LINEAR)
        axxs[1,0].imshow(res_l)
        axxs[1,0].set_title("Rectified left image")
        axxs[1,1].imshow(res_r)
        axxs[1,1].set_title("Rectified right image")
        plt.show()


    if os.path.exists(output_path):
        res = input("Output file already exists, rewrite? (y/n)")
        while res != 'n' and res != 'y':
            res = input("Output file already exists, rewrite? (y/n)")
        if res == 'y':
            np.savez_compressed(output_path, m_L=m_L, m_R=m_R, d_L=d_L,d_R=d_R, R_L=R_L, R_R=R_R, P_L=P_L, P_R=P_R)
            print(f"Saved calibration data to {output_path}")
        else:
            print("Exiting...")
    else:
        np.savez_compressed(output_path, m_L=m_L, m_R=m_R, d_L=d_L, d_R=d_R, R_L=R_L, R_R=R_R, P_L=P_L, P_R=P_R)
        print(f"Saved calibration data to {output_path}")

    return {"m_L":m_L,"m_r":m_R,"d_L":d_L,"d_R":d_R,"R_L":R_L,"R_R":R_R,"P_L":P_L,"P_R":P_R}


if __name__ == "__main__":
    parser = custom_parser.CustomParser()
    parser.add_argument("input_path", type=str, help="Path to the directory with /L_calib and /R_calib folders\n"
                                                     "(folders with left and right imgs of sterepair respectively)")

    parser.add_argument("--output", type=str, help="Path to output calibration data file")

    parser.add_argument("--grid", type=str, help="Resolution of the grid in corners in vxh form\n"
                                                 "example: 6x7 equals to 6 vetical corners and 7 horizontal corners\n"
                                                 "(default is 9x14)", default="9x14")
    parser.add_argument("--verbose", type=bool, help="If True, displays the calibration result on first stereopair",
                        default=False)
    args = parser.parse_args()

    calibrate(args)

