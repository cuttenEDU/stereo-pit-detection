import torch.optim as optim
from model import *
from modules.MarginLoss import *
from tqdm import *
from random import uniform
from datetime import datetime
from kitti.kitti_utils import *
from torch.utils.tensorboard import SummaryWriter
from utils import custom_parser

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(0)

torch.backends.cudnn.deterministic = True
torch.manual_seed(42)


def save_network(net,epoch,loss,path):
    savename = f"model-{datetime.now().strftime('%d.%m.%Y-%H.%M.%S')}-epoch#{epoch}-loss_{round(loss,6)}.pt"
    print(f"Saved model on epoch{epoch} as {savename}")
    save_net(net,os.path.join(path,savename))

def print_epoch_start(epoch):
    print(f"======================== STARTED EPOCH #{epoch}========================\n")

def print_epoch_finish(epoch):
    print(f"======================== FINISHED EPOCH #{epoch}========================\n")


def train(dataset_dir,save_dir,params):
    try:
        update_each = params["update_each"]

        width,height = list(map(int,params["crop"].split("x")))

        network = create_model(input_planes=3)
        ws = get_window_size(network)
        network = network.cuda()

        optimizer = optim.SGD(network.parameters(), params["lr"], 0.9)

        x_batch_shape = (params["bs"] * 2, ws, ws, 3)

        criterion = MarginLoss(0.2)

        running_loss = 0.0
        batches = 0

        writer = SummaryWriter(save_dir)

        for epoch in range(params["num_epochs"]):
            print_epoch_start(epoch)

            dataset = DisparityDataSet(dataset_dir, transforms.Compose(
                [transforms.Lambda(lambda x: transforms.functional.crop(x, 0, 0, height, width))]
            ))
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

            epoch_bar = tqdm(total=len(dataset), unit=" disparity maps processed", ncols=120)
            batches_bar = tqdm(total=float('inf'), unit=" batches processed")

            batches_amount = 0
            b = 0

            epoch_loss = 0

            x_batch_tr = torch.zeros(x_batch_shape,dtype=torch.float32)

            for img_disp,img_id in dataloader:
                img_disp = img_disp[0]
                disp_gen = positive_disparities(img_disp)
                img_l, img_r = load_kitty_stereopair(dataset_dir, img_id, False, mode="torch",std=params["std"])
                for i,j,disp_value in disp_gen:

                    if b == 0:
                        x_batch_tr = torch.zeros(x_batch_shape,dtype=torch.float32)

                    scale1 = (uniform(0.9, 1), 1)
                    phi1 = uniform(-7, 7)
                    translate1 = (0, 0)
                    shear1 = uniform(-0.1, 0.1)
                    brightness1 = uniform(-0.7,0.7)
                    contrast1 = uniform(1 / 1.3, 1.3)

                    scale2 = scale1
                    phi2 = phi1
                    translate2 = translate1
                    shear2 = shear1
                    brightness2 = brightness1 + uniform(-0.3, 0.3)
                    contrast2 = contrast1

                    d_pos = uniform(-0.5, 0.5)
                    d_neg = uniform(4, 10)
                    if uniform(0, 1) < 0.5:
                        d_neg = -d_neg


                    l_patch = make_patch_2(img_l, ws, i, j, scale1, phi1, translate1, shear1,contrast1,brightness1)

                    x_batch_tr[b * 4 + 0] = l_patch
                    x_batch_tr[b * 4 + 1] = make_patch_2(img_r, ws, i, j - disp_value + d_pos, scale2, phi2, translate2,
                                                         shear2,contrast2,brightness2)
                    x_batch_tr[b * 4 + 2] = l_patch
                    x_batch_tr[b * 4 + 3] = make_patch_2(img_r, ws, i, j - disp_value + d_neg, scale2, phi2, translate2,
                                                         shear2,contrast2,brightness2)

                    b += 1


                    if b == params["bs"]/2:
                        x_batch_tr = x_batch_tr.cuda()
                        x_batch_tr = x_batch_tr.movedim(3, 1)

                        optimizer.zero_grad()

                        net_output = network(x_batch_tr)

                        loss = criterion(net_output.flatten())

                        running_loss += loss.item()
                        epoch_loss += loss.item()

                        if batches_amount % update_each == 0 and batches_amount > 0:
                            batches_bar.update(update_each)
                            batches_bar.refresh()
                            batches_bar.set_postfix_str(
                                "current loss: " + str(round(loss.item(), 6)) + " | running loss: " + str(
                                    round(running_loss / update_each, 6)) + "| average epoch loss: " + str(round(epoch_loss/batches_amount, 6)))
                            writer.add_scalar(f"Running loss on epoch {epoch}", running_loss / update_each, batches_amount)
                            writer.add_scalar(f"Average loss on epoch {epoch}", epoch_loss / batches_amount, batches_amount)
                            writer.flush()
                            running_loss = 0

                        loss.backward()
                        optimizer.step()
                        batches_amount += 1
                        b = 0

                        del x_batch_tr
                epoch_bar.update(1)

            epoch_bar.close()
            batches_bar.close()
            save_network(network, epoch,epoch_loss,save_dir)
            print_epoch_finish(epoch)
    except Exception as e:
        print(e)
        if epoch > 0 or batches > 10000:
            save_network(network,-1)
    writer.close()
if __name__ == "__main__":

    train_params = {
        "lr":0.002,
        "bs":128,
        "num_epochs":12,
        "crop": "1242x350",
        "update_each":250,
        "std":True,
    }

    parser = custom_parser.CustomParser(description="Training script for MC_CNN model. Tailored for training on KITTI 2015 dataset")
    parser.add_argument("train_dataset", type=str, help="Path to train dataset")
    parser.add_argument("save_path", type=str, help="Model weights will be saved in this directory")
    parser.add_argument("--amount", type=int, help="Amount of disps to evaluate", default=20)
    parser.add_argument("--no_std", action="store_true", help="Disables standartization of the images")

    parser.add_argument("--lr",type=float,help=f"Learning rate for training (default is {train_params['lr']}",
                        default=train_params['lr'])
    parser.add_argument("--bs",type=int,help=f"Size of one learning mini-batch (default is {train_params['bs']}",
                        default=train_params['bs'])
    parser.add_argument("--num_epochs",type=int,help=f"Amount of epochs to train (default is {train_params['num_epochs']}",
                        default=train_params['num_epochs'])
    parser.add_argument("--crop",type=str,help=f"Crop size of images in wxh format (default is {train_params['crop']}",
                        default=train_params['crop'])
    parser.add_argument("--update_each",type=int,help=f"Updating rate of training info (default is {train_params['update_each']}",
                        default=train_params['update_each'])


    args = parser.parse_args()

    if not os.path.isdir(args.save_path):
        os.mkdir(args.save_path)

    train_params["lr"] = args.lr
    train_params["bs"] = args.bs
    train_params["num_epochs"] = args.num_epochs
    train_params["crop"] = args.crop
    train_params["update_each"] = args.update_each
    train_params["std"] = not args.no_std

    train(args.train_dataset,args.save_path,train_params)




