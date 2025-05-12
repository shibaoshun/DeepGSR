import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from swin_bm3d import *
from dataset import prepare_data, Dataset
from utilsDnCNN import *
from head_HM import init_logger

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"# 按照PCI_BUS_ID顺序从0开始排列GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = "3"# 设置当前使用的GPU设备仅为0号设备  设备名称为'/gpu:0'

parser = argparse.ArgumentParser(description="DnCNN")# 创建ArgumentParser()对象
parser.add_argument("--tr_dir", type=str, default="D:\code\jxy\denoise\data\grey\BSD400", help='path of log files')
parser.add_argument("--vl_dir", type=str, default="D:\code\jxy\denoise\data\grey\Set12", help='path of log files')
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')#指定训练模式
parser.add_argument("--noiseL", type=float, default=15, help='noise level; ignored when mode=B')#指定训练噪声水平
parser.add_argument("--gpu", type=int, default=0, help='run prepare_data or not')# 调用add_argument()方法添加参数，preprocess指示是否运行数据预处理
parser.add_argument("--preprocess", type=bool, default=True, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=1, help="Training batch size")
parser.add_argument("--epochs", type=int, default=350, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=[100,200,300,400], help="When to decay learning rate; should be less than epochs")

parser.add_argument("--lr", type=float, default=5e-5, help="Initial learning rate")

opt = parser.parse_args()# 使用parse_args()解析添加的参数
opt.val_noiseL= opt.noiseL
opt.model_dir = os.path.join("result", "noise"+str(opt.noiseL), "ckp/")
opt.outf = os.path.join("result", "noise"+str(opt.noiseL), "logs")
opt.resume_dir = os.path.join("result", "noise"+str(opt.noiseL), "ckp/epoch389.pth")
opt.resume_epoch=389
opt.resume= False

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)
    return num_params

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def check_image_size(x,window_size):
    _, _, h, w = x.size()
    mod_pad_h = (window_size - h % window_size) % window_size
    mod_pad_w = (window_size - w % window_size) % window_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x

def main():
    mkdir(opt.model_dir)
    mkdir(opt.outf)

    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True, noiseL=opt.noiseL)  # 创建一个训练数据集对象dataset_train，其中train=True表示这是用于训练的数据集
    dataset_val = Dataset(train=False, noiseL=opt.noiseL)  # 创建一个验证数据集对象dataset_val，其中train=False表示这是用于验证的数据集。
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))# 打印训练数据集中的样本数量
    # Build model 改

    x = check_image_size(torch.rand(1, 1, 180, 180), 30)  #patchsize*windowsize
    _, _, h, w = x.size()
    net_tr = BM3DTrans(img_size=(h, w))

    x = check_image_size(torch.rand(1, 1, 256, 256), 30)
    _, _, h1, w1 = x.size()
    net_vl = BM3DTrans(img_size=(h1, w1))


    criterion = torch.nn.MSELoss(reduction='sum')
    # Move to GPU
    model = net_tr.cuda()
    modelvl = net_vl.cuda()
    criterion.cuda()
    optimizer = optim.Adam(model.parameters(), betas=(0.5, 0.999), lr=opt.lr)    # training
    logger = init_logger(opt)
    num = print_network(model)
    logger.info("\tTotal number of parameters: {}".format(num))

    training_params = {}
    best_psnr = 0
    best_psnr_epoch = 0
    start = 0
    if opt.resume == True:
        ckp = torch.load(opt.resume_dir)
        # print(ckp['model'])
        model.load_state_dict(ckp['model'], False)
        optimizer.load_state_dict(ckp['optimizer'])
        # self.scheduler.load_state_dict(ckp['scheduler'])
        training_params = ckp['training_params']
        print("self.training_params", training_params)
        best_psnr = training_params['best_psnr']
        best_psnr_epoch = training_params['best_psnr_epoch']
        start = opt.resume_epoch

    step = 0#step 是一个计数器，用于跟踪训练过程中的步数

    for epoch in range(start, opt.epochs):
        current_lr = opt.lr
        epoch1=epoch+1
        if epoch1 < opt.milestone[0] or epoch1 == opt.milestone[0]:
            current_lr = opt.lr
        elif opt.milestone[0] < epoch1 < opt.milestone[1] or epoch1 == opt.milestone[1]:
            current_lr = opt.lr / 2
        elif opt.milestone[1] < epoch1 < opt.milestone[2] or epoch1 == opt.milestone[2]:
            current_lr = opt.lr / 4
        elif opt.milestone[2] < epoch1 < opt.milestone[3] or epoch1 == opt.milestone[3]:
            current_lr = opt.lr / 8
        else:
            current_lr = opt.lr / 16
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        # train
        for i, data in enumerate(loader_train, 0):
            # training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            img_train = data
            if opt.mode == 'S':
                noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=opt.noiseL/255.)
            imgn_train = img_train + noise
            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())
            noise = Variable(noise.cuda())

            out_train = model(imgn_train)

            loss = criterion(out_train, img_train)
            loss.backward()#反向传播
            optimizer.step()#更新参数

            # results 改 加保存位置
            model.eval()
            out_train = torch.clamp(out_train, 0., 1.)
            psnr_train = batch_PSNR(out_train, img_train, 1.)#计算psnr
            ssim_train = batch_SSIM(out_train, img_train)
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f SSIM_train: %.4f" %
                (epoch+1, i+1, len(loader_train), loss.item(), psnr_train,ssim_train))

            step += 1
        ## the end of each epoch
        model_filename = opt.model_dir + 'epoch%d.pth' % (epoch + 1)
        state = {'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 # 'scheduler': self.scheduler.state_dict(),
                 'training_params': training_params
                 }
        torch.save(state, model_filename)

        ckp = torch.load(os.path.join(model_filename))
        modelvl.load_state_dict(ckp['model'], False)  ##


        print('-' * 100)
        modelvl.eval()
        # validate
        psnr_val = 0
        ssim_val = 0
        for k in range(len(dataset_val)):
            img_val = torch.unsqueeze(dataset_val[k], 0)
            noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.val_noiseL/255.)
            imgn_val = img_val + noise
            img_val, imgn_val = Variable(img_val.cuda(), volatile=True), Variable(imgn_val.cuda(), volatile=True)
            with torch.no_grad():
                out_val = torch.clamp(modelvl(imgn_val), 0., 1.)

            psnr_iter = batch_PSNR(out_val, img_val, 1.)
            ssim_iter = batch_SSIM(out_val, img_val)
            print("[epoch %d][%d/%d] PSNR_val: %.4f SSIM_train: %.4f" % (epoch + 1, k+1, len(dataset_val),psnr_iter, ssim_iter))
            psnr_val += psnr_iter
            ssim_val += ssim_iter

        psnr_val /= len(dataset_val)
        ssim_val /= len(dataset_val)
        print("\n[epoch %d] PSNR_val: %.4f SSIM_train: %.4f" % (epoch+1, psnr_val, ssim_val))


        if psnr_val > best_psnr:
            best_psnr = psnr_val
            training_params['best_psnr'] = best_psnr
            training_params['best_psnr_epoch'] = epoch + 1
            best_psnr_epoch = epoch + 1
            model_filename_best = opt.model_dir + 'best.pth'
            statebest = {'model': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     # 'scheduler': self.scheduler.state_dict(),
                     'training_params': training_params
                     }
            torch.save(statebest, model_filename_best)


        logger.info(
            "\tval: current_epoch:{}  psnr_val:{:.4f}  ssim_val:{:.4f}  lr:{}  best_psnr:{:.4f}  best_psnr_epoch:{}"
                .format(epoch + 1, psnr_val, ssim_val, current_lr, best_psnr, best_psnr_epoch))

        print("best_psnr:{:.4f} best_psnr_epoch:{}".format(best_psnr, best_psnr_epoch))
        print('-' * 100)

    print('Reach the maximal epoch! Finish training')

if __name__ == "__main__":
    # opt.preprocess=True  #true的时候开始处理图像
    if opt.preprocess:
        if opt.mode == 'S':
            prepare_data(tr_path=opt.tr_dir, vl_path=opt.vl_dir,noiseL=opt.noiseL)### 准备数据  将数据分块 增强
        if opt.mode == 'B':
            prepare_data(data_path='data', patch_size=50, stride=10, aug_times=2)
    main()
