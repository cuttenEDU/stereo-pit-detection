import torch
import math
import numpy as np
from adcensus.cupy_adcensus_kernels import *

TB = 128


def normalize_forward(input, norm, output):
    s = input.shape
    s1 = cp.int32(s[1])
    s2 = cp.int32(s[2])
    s3 = cp.int32(s[3])
    n = cp.int32(int(norm.numel()))
    o_n = cp.int32(int(output.numel()))
    normalize_get_norm_kernel((math.ceil(n - 1 / TB + 1),), (TB,), (input.data_ptr(), norm.data_ptr(), s1, s2 * s3, n))
    normalize_forward_kernel((math.ceil(o_n - 1 / TB + 1),), (TB,),
                             (input.data_ptr(), norm.data_ptr(), output.data_ptr(), s2 * s3, s1 * s2 * s3, o_n))


def normalize_backward(grad_output, input, norm, grad_input):
    s = input.shape
    s1 = cp.int32(s[1])
    s2 = cp.int32(s[2])
    s3 = cp.int32(s[3])
    n = cp.int32(int(input.numel()))
    normalize_backward_input_kernel((math.ceil(n - 1 / TB + 1),), (TB,), (
    grad_output.data_ptr(), input.data_ptr(), norm.data_ptr(), grad_input.data_ptr(), s1, s2 * s3, n))

def grayscale(image,grayscale):
    h = image.shape[1]
    w = image.shape[2]
    # n = cp.int32(int(image.numel()))
    grayscale_kernel((math.ceil(w / 32),math.ceil(h/32)),(32,32),(image.data_ptr(),grayscale.data_ptr(),w,h))

def remove_nonvisible(disp):
    n = disp.numel()
    remove_nonvisible_kernel((math.ceil((n - 1)/TB + 1),),(TB,),(disp.data_ptr(),n,disp.shape[2]))

def remove_occluded(disp):
    n = disp.numel()
    remove_occluded_kernel((math.ceil((n - 1)/TB + 1),),(TB,),(disp.data_ptr(),n,disp.shape[2]))

def remove_white(img,disp):
    n = disp.numel()
    remove_white_kernel((math.ceil((n - 1)/TB + 1),),(TB,),(img.data_ptr(),disp.data_ptr(),n))

def margin_loss(input, output, gradInput, margin):
    n = output.numel()
    marginloss_kernel((math.ceil((n - 1) / TB + 1),), (TB,), (input.data_ptr(), output.data_ptr(), gradInput.data_ptr(), margin, n))


def sgm(x0, x1, vol, out, pi1, pi2, tau_so, alpha1, sgm_q1, sgm_q2, direction):
    left_p = x0.data_ptr()
    right_p = x1.data_ptr()
    vol_p = vol.data_ptr()
    out_p = out.data_ptr()
    pi1 = cp.float32(pi1)
    pi2 = cp.float32(pi2)
    tau_so = cp.float32(tau_so)
    alpha1 = cp.float32(alpha1)
    sgm_q1 = cp.float32(sgm_q1)
    sgm_q2 = cp.float32(sgm_q2)
    direction = cp.int32(direction)

    dim0 = cp.int32(vol.shape[1])
    dim1 = cp.int32(vol.shape[2])
    dim2 = cp.int32(vol.shape[3])
    for sgm_dir in range(4):
        size = dim1 if sgm_dir <= 1 else dim2
        grid = (math.ceil(size / 512),)
        block = (512,)
        sgm1_kernel(grid, block,
                   (left_p, right_p, vol_p, 0, out_p,
                    dim0, dim1, dim2,
                    pi1, pi2, tau_so, alpha1, sgm_q1, sgm_q2,
                    cp.int32(sgm_dir), direction))
    return out

def sgm2(x0, x1, input, output, tmp, pi1, pi2, tau_so, alpha1,
         sgm_q1, sgm_q2, direction):
    size1 = cp.int32(output.shape[1]*output.shape[3])
    size2 = cp.int32(output.shape[2]*output.shape[3])
    disp_max = cp.int32(output.shape[3])
    x0 = cp.asarray(x0)
    x1 = cp.asarray(x1)
    input = cp.asarray(input)
    output = cp.asarray(output)
    tmp = cp.asarray(tmp)
    pi1 = cp.float32(pi1)
    pi2 = cp.float32(pi2)
    tau_so = cp.float32(tau_so)
    alpha1 = cp.float32(alpha1)
    sgm_q1 = cp.float32(sgm_q1)
    sgm_q2 = cp.float32(sgm_q2)
    direction = cp.int32(direction)
    ishape1 = cp.int32(input.shape[1])
    ishape2 = cp.int32(input.shape[2])
    ishape3 = cp.int32(input.shape[3])


    for i in range(input.shape[2]):
        i = cp.int32(i)
        sgm2_kernel((int((size1 - 1) / disp_max) + 1,), (disp_max,),
                    (x0,x1,input,output,tmp,
                    pi1,pi2,tau_so,alpha1,sgm_q1,sgm_q2,direction,
                    ishape1,ishape2,ishape3,i,cp.int32(0)))
    for i in range(input.shape[2]):
        i = cp.int32(i)
        sgm2_kernel((int((size1 - 1) / disp_max) + 1,), (disp_max,),
                    (x0,x1,input,output,tmp,
                    pi1,pi2,tau_so,alpha1,sgm_q1,sgm_q2,direction,
                    ishape1,ishape2,ishape3,i,cp.int32(1)))
    for i in range(input.shape[1]):
        i = cp.int32(i)
        sgm2_kernel((int((size2 - 1) / disp_max) + 1,), (disp_max,),
                    (x0,x1,input,output,tmp,
                    pi1,pi2,tau_so,alpha1,sgm_q1,sgm_q2,direction,
                    ishape1,ishape2,ishape3,i,cp.int32(2)))
    for i in range(input.shape[1]):
        i = cp.int32(i)
        sgm2_kernel((int((size2 - 1) / disp_max) + 1,), (disp_max,),
                    (x0,x1,input,output,tmp,
                    pi1,pi2,tau_so,alpha1,sgm_q1,sgm_q2,direction,
                    ishape1,ishape2,ishape3,i,cp.int32(3)))

    return output

def stereo_join(input_L,input_R,output_L,output_R):
    size23 = output_L.shape[2]*output_L.shape[3]
    input_L = cp.asarray(input_L)
    input_R = cp.asarray(input_R)
    output_L = cp.asarray(output_L)
    output_R = cp.asarray(output_R)
    sj_kernel((math.ceil((size23-1)/TB + 1),),(TB,),(input_L,input_R,output_L,output_R,input_L.shape[1],output_L.shape[1],output_L.shape[3],size23))

def outlier_detection(d0,d1,outliers,disp_max):
    d_num = cp.int32(d0.numel())
    disp_max = cp.int32(disp_max)
    d0_s2 = cp.int32(d0.shape[2])
    d0 = cp.asarray(d0)
    d1 = cp.asarray(d1)
    outliers = cp.asarray(outliers)

    grid = (int((d_num-1)/TB + 1),)
    block = (TB, )
    args = (d0,d1,outliers,d_num,d0_s2,disp_max)

    outlier_detection_kernel(grid,block,args)

def interpolate_occlusion(d0,outliers):
    n = cp.int32(d0.numel())
    d0 = cp.asarray(d0)
    outliers = cp.asarray(outliers)
    out = cp.zeros_like(d0)


    o_s2 = cp.int32(d0.shape[2])


    grid = (math.ceil((n - 1)/TB + 1),)
    block = (TB,)
    args = (d0,outliers,out,n,o_s2)

    interpolate_occlusion_kernel(grid,block,args)

    return torch.as_tensor(out)

def interpolate_mismatch(d0,outliers):
    n = cp.int32(d0.numel())

    d0 = cp.asarray(d0)
    outliers = cp.asarray(outliers)
    out = cp.zeros_like(d0)
    o_s2 = cp.int32(d0.shape[1])
    o_s3 = cp.int32(d0.shape[2])

    grid = (math.ceil((n - 1)/TB + 1),)
    block = (TB,)
    args = (d0,outliers,out,n,o_s2,o_s3)

    interpolate_mismatch_kernel(grid,block,args)

    return torch.as_tensor(out)


def subpixel_enchancement(d0,vol,disp_max):
    n = cp.int32(d0.numel())
    o_s12 = cp.int32(d0.shape[1]*d0.shape[2])
    d0 = cp.asarray(d0)
    vol = cp.asarray(vol)
    out = cp.zeros_like(d0)

    grid = (int((n - 1)/TB + 1),)
    block = (TB,)
    args = (d0,vol,out,n,o_s12,disp_max)

    subpixel_enchancement_kernel(grid,block,args)

    return torch.as_tensor(out)

def median_filter(d0,ks):
    assert ks % 2 == 1 and ks <= 11
    ks = cp.int32(ks)
    n = cp.int32(d0.numel())
    o_s1 = cp.int32(d0.shape[1])
    o_s2 = cp.int32(d0.shape[2])
    d0 = cp.asarray(d0)
    out = cp.zeros_like(d0)

    grid = (int((n - 1)/TB + 1),)
    block = (TB,)
    args = (d0,out,n,o_s1,o_s2,ks/2)

    median_filter_kernel(grid,block,args)

    return torch.as_tensor(out)

def bilateral_filter(d0,blur_sigma,blur_t):
    def gaussian(sigma):
        kr = math.ceil(sigma * 3)
        ks = kr * 2 + 1
        k = cp.zeros((ks, ks),dtype=np.float32)
        for i in range(ks):
            for j in range(ks):
                y = i - kr
                x = j - kr
                k[i,j] = math.exp(-(x * x + y * y) / (2 * sigma * sigma))

        return k




    blur_t = cp.float32(blur_t)
    n = cp.int32(d0.numel())

    o_s1 = cp.int32(d0.shape[1])
    o_s2 = cp.int32(d0.shape[2])
    d0 = cp.asarray(d0)
    out = cp.zeros_like(d0)
    kernel = gaussian(blur_sigma)
    ks = cp.int32(kernel.shape[0])


    assert kernel.shape[0] % 2 == 1

    grid = (int((n - 1)/TB + 1),)
    block = (TB,)
    args = (d0,kernel,out,n,cp.int32(ks/2),o_s1,o_s2,blur_t)

    bilateral_filter_kernel(grid,block,args)

    return torch.as_tensor(out)