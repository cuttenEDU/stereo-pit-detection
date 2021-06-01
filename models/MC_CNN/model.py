from modules.Normalize import *
from modules.StereoJoin import *
import adcensus.cupy_adcensus as adcensus
import time
from matplotlib import pyplot as plt
import os


default_mc_cnn_parameters = {
        "disp_max" : 230,
        "pi1":2.3,
        "pi2":18.3,
        "tau_so":0.08,
        "alpha":1.25,
        "q1":2,
        "q2":3
}

def create_model(input_planes=3, middle=64, count=4, kernel=(3, 3)):
    model = nn.Sequential()
    assert count > 1
    model.add_module("conv-0", nn.Conv2d(input_planes, middle, kernel))
    for i in range(count-1):
        model.add_module(f"relu-{i}", nn.ReLU(True))
        model.add_module(f"conv-{i + 1}", nn.Conv2d(middle, middle, kernel))
    model.add_module("norm", Normalize())
    model.add_module("join", StereoJoin())
    return model


def save_net(model: nn.Sequential, path):
    torch.save(model.state_dict(), path)


def load_net(path, input_planes=3, middle=64, count=4, kernel=(3, 3)):
    state_dict = torch.load(path)
    model = create_model(input_planes, middle, count, kernel)
    model.load_state_dict(state_dict, strict=False)
    return model


def switch_to_prod(model: nn.Sequential):
    for i, n in enumerate(model.modules()):
        if isinstance(n, nn.Conv2d):
            n.padding = (1, 1)
    return model


def get_window_size(model: nn.Sequential):
    ws = 1
    for i, n in enumerate(model.modules()):
        if isinstance(n, nn.Conv2d):
            ws += n.kernel_size[0] - 1
    return ws



def fix_border(model,vol,direction):
    n = int((get_window_size(model) - 1) / 2)
    for i in range(n):
        vol[:,:,direction*i].copy_(vol[:,:,direction * (n-1)])

def predict(model,img_l,img_r,parameters = default_mc_cnn_parameters,save_stages_path = None):
    verbose = not save_stages_path is None

    disp_max = parameters["disp_max"]

    f, axxs = plt.subplots(6, 1, figsize=(6, 20))
    plt.subplots_adjust(hspace=0.7, wspace=0)

    if verbose:
        axxs[0].imshow(img_r.cpu().permute(1,2,0))

    assert img_l.shape == img_r.shape

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    model.norm.register_forward_hook(get_activation("norm"))

    x = torch.Tensor(2,img_l.shape[0],img_l.shape[1],img_l.shape[2]).cuda()
    x[0] = img_l
    x[1] = img_r
    with torch.no_grad():
        result = model(x)

    height, width = x.shape[2:]

    vols = torch.full((2, parameters["disp_max"], height, width), float("0"), device="cuda")
    output = activation["norm"]
    adcensus.stereo_join(output[0].unsqueeze(0), output[1].unsqueeze(0), vols[0].unsqueeze(0), vols[1].unsqueeze(0))
    fix_border(model, vols[0], -1)
    fix_border(model, vols[1], 1)
    dirs = [1, -1]
    disp = [0, 0]

    if verbose:
        vol = vols[1]
        i, values = torch.min(vol, 0)
        axxs[1].set_title("Disparity map straight from network")
        axxs[1].imshow(values.cpu(), cmap="gray")

    pi1, pi2, tau_so, alpha1, sgm_q1, sgm_q2 = 0, 0, 0, 0, 0, 0

    for i, d in enumerate(dirs):
        vol = vols[0 if d == -1 else 1].clone().unsqueeze(0)
        vol = torch.movedim(vol, 1, 3).contiguous()
        out = torch.zeros((1, vol.shape[1], vol.shape[2], vol.shape[3])).cuda()
        tmp = torch.zeros((vol.shape[2], vol.shape[3])).cuda()
        pi1 = parameters["pi1"]
        pi2 = parameters["pi2"]
        tau_so = parameters["tau_so"]
        alpha1 = parameters["alpha"]
        sgm_q1 = parameters["q1"]
        sgm_q2 = parameters["q2"]



        adcensus.sgm2(x[0], x[1], vol, out, tmp, pi1, pi2, tau_so, alpha1, sgm_q1, sgm_q2, d)
        vol.copy_(out)
        vol = torch.movedim(vol, 3, 1).contiguous()

        values, indicies = torch.min(vol, 1)
        disp[0 if d == 1 else 1] = indicies.float()

    if verbose:
        axxs[2].set_title("Left disparity map after SGM")
        axxs[2].imshow(disp[0].permute(1, 2, 0).cpu(), cmap="gray")
        axxs[3].set_title("Right disparity map after SGM")
        axxs[3].imshow(disp[1].permute(1, 2, 0).cpu(), cmap="gray")

    new_disp = [disp[0].clone(), disp[1].clone()]

    outliers = torch.zeros_like(new_disp[1]).cuda()
    adcensus.outlier_detection(new_disp[1], new_disp[0], outliers, parameters["disp_max"])
    new_disp[1] = adcensus.interpolate_occlusion(new_disp[1], outliers)
    new_disp[1] = adcensus.interpolate_mismatch(new_disp[1], outliers)
    new_disp[1] = adcensus.subpixel_enchancement(new_disp[1], vol, parameters["disp_max"])
    if verbose:
        axxs[4].set_title("Disparity map")
        axxs[4].imshow(new_disp[1].permute(1, 2, 0).cpu().type(torch.uint8), cmap="gray")
        axxs[5].set_title("Outliers")
        axxs[5].imshow(outliers.permute(1, 2, 0).cpu().type(torch.uint8), cmap="gray")
        f.suptitle(f"pi1: {pi1}\n"
                   f"pi2: {pi2}\n "
                   f"tau_so: {tau_so}\n "
                   f"a:{alpha1}\n "
                   f"sgm_q1: {sgm_q1}\n "
                   f"sgm_q2: {sgm_q2}\n "
                   f"DISP_MAX: {disp_max}\n")

        plt.show()

    if verbose:
        if not os.path.exists(save_stages_path):
            os.mkdir(save_stages_path)
        f.savefig(os.path.join(save_stages_path,str(time.time()) + ".png"))

    return new_disp[1]
