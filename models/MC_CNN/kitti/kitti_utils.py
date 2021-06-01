import cupy as cp
import numpy as np
import cv2
import os


from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from adcensus.cupy_adcensus import *
from skimage.color import rgb2yuv


class DisparityDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        self.all_imgs = os.listdir(os.path.join(main_dir,"disp"))

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_name = self.all_imgs[idx]
        img_loc = os.path.join(self.main_dir,"disp", img_name)
        img_l_loc = os.path.join(self.main_dir,"image_left",img_name)
        image = load_disparity_map(img_loc,img_l_loc,True)
        tensor_image = self.transform(image)
        return tensor_image,idx

def positive_disparities(disp):
    # getting rid of useless 1-sized dim
    disp = disp[0]
    for i,row in enumerate(disp):
        for j,col_value in enumerate(row):
            value = float(col_value)
            if value > 0.5:
                yield i,j,value


def getZeroes(img_id):
    if img_id == 0:
        return "0" * 5
    i = 0
    while img_id > 0:
        img_id = int(img_id / 10)
        i += 1
    return (6 - i) * "0"



def load_kitty_stereopair(dataset_dir, img_id, grayscale = False, mode ="numpy", std = True):
    img_id = int(img_id)
    path_l = os.path.join(dataset_dir, "image_left", str(getZeroes(img_id)) + str(img_id) + "_10.png")
    path_r = os.path.join(dataset_dir, "image_right", str(getZeroes(img_id)) + str(img_id) + "_10.png")
    return load_stereopair(path_l,path_r,grayscale,mode,std)


def load_stereopair(path_l,path_r,grayscale = False,mode = "numpy", std = True):
    img_l = cv2.cvtColor(cv2.imread(path_l),cv2.COLOR_BGR2RGB)
    img_r = cv2.cvtColor(cv2.imread(path_r),cv2.COLOR_BGR2RGB)
    if std:
        img_l = img_l - img_l.mean()
        img_r = img_r - img_r.mean()
        img_l /= img_l.std()
        img_r /= img_r.std()
    if (grayscale):
        img_l = rgb2yuv(img_l)
        img_r = rgb2yuv(img_r)
    if mode == "torch":
        img_l = torch.from_numpy(img_l)
        img_r = torch.from_numpy(img_r)
    elif mode == "cupy":
        img_l = cp.array(img_l)
        img_r = cp.array(img_r)

    return img_l,img_r

def load_disparity_map(disp_path,img_l_path,removal = True):
    dev_image = transforms.ToTensor()(Image.fromarray(np.array(Image.open(disp_path)).astype("uint16"))).float().cuda()
    dev_image /= 256
    if removal:
        remove_nonvisible(dev_image)
        remove_occluded(dev_image)
        dev_img_l = transforms.ToTensor()(Image.open(img_l_path)).cuda()
        dev_img_l = transforms.Grayscale()(dev_img_l)
        remove_white(dev_img_l, dev_image)
    image = dev_image.cpu()
    return image

def make_patch_2(src,dst_shape,dim3,dim4,scale,phi,trans,hshear,contrast,brightness):
    shear = 0
    center = (dst_shape - 1) / 2
    a2 = center - dim4 * scale[0] - trans[0]
    b2 = center - dim3 * scale[1] - trans[1]
    translate = np.array([[scale[0], 0, a2], [shear, scale[1], b2]])
    rotation = cv2.getRotationMatrix2D((center, center), phi, 1)
    warped = cv2.warpAffine(src.numpy(), translate, (dst_shape, dst_shape))
    warped = cv2.warpAffine(warped, rotation, (dst_shape, dst_shape))
    tensor = torch.from_numpy(warped).contiguous()
    tensor.mul_(contrast).add_(brightness)
    return tensor




