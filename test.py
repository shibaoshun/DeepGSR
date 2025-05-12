import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from swin_bm3d import *
from utilsDnCNN import *
from head_HM import init_logger_test
import time
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"### 设备号 默认为0

parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--test_dir", type=str, default="D:\code\jxy\denoise\data\grey\BSD68", help='path of log files')
parser.add_argument("--test_noiseL", type=float, default=15, help='noise level used on test set')
parser.add_argument("--test_model", type=str, default="best.pth", help='noise level used on test set')
parser.add_argument("--test_save_img", type=bool, default=False, help='run prepare_data or not')# 调用add_argument()方法添加参数，preprocess指示是否运行数据预处理

opt = parser.parse_args()
opt.model_dir = os.path.join("result", "noise"+str(opt.test_noiseL), "ckp/")
opt.logdir = os.path.join("result", "noise"+str(opt.test_noiseL), "logs")
opt.img_dir = './result/' + str(opt.test_noiseL) + '/img/'

def save_image(idx, dir, datalist):
    for i in range(len(datalist)):
        file_dir = dir[i] + str(idx)+'.png'
        plt.imsave(file_dir, datalist[i].data.cpu().numpy().squeeze(), cmap="gray")
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        print("---  There exsits folder " + path + " !  ---")

def check_image_size(x,window_size):
    _, _, h, w = x.size()
    mod_pad_h = (window_size - h % window_size) % window_size
    mod_pad_w = (window_size - w % window_size) % window_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x

def normalize(data):
    return data/255.### 输入的 data 归一化到 [0, 1] 的范围内

def main():
    logger = init_logger_test(opt)

    # Build model
    print('Loading model ...\n')
    x=check_image_size(torch.rand(1, 1, 481, 321), 30)
    _, _, h, w = x.size()
    net = BM3DTrans(img_size=(h, w))

    model=net.cuda()
    ckp = torch.load(os.path.join(opt.model_dir, opt.test_model))
    model.load_state_dict(ckp['model'], False)
    logger.info("test_model: %s " % opt.test_model)
    model.eval()
    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join(opt.test_dir, '*.png'))
    files_source.sort()
    out_dir = opt.img_dir + '/pgsr/image/'
    mkdir(out_dir)
    input_dir90 = opt.img_dir + '/input/image/'
    mkdir(input_dir90)
    gt_dir90 = opt.img_dir + '/gt/image/'
    mkdir(gt_dir90)
    # process data
    psnr_test = 0
    ssim_test = 0
    k=0
    t_start = time.time()
    i = 0
    for f in files_source:
        i += 1            # image
        Img = cv2.imread(f) #使用 cv2.imread 函数从文件 f 中读取图像
        Img = normalize(np.float32(Img[:,:,0]))
        # Img =  np.transpose(Img, (2,0, 1))
        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1)
        ISource = torch.Tensor(Img)
        # noise 生成与图像大小相同的噪声张量，并添加到原始图像上以创建带噪图像
        noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL/255.)
        # noisy image
        INoisy = ISource + noise
        ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())

        with torch.no_grad(): # this can save much memory
            Out = torch.clamp(model(INoisy), 0., 1.)

        if opt.test_save_img == True:
            X_90 = [Out, ISource, INoisy]
            dir_90 = [out_dir, gt_dir90, input_dir90]
            save_image(i, dir_90, X_90)

        psnr = batch_PSNR(Out, ISource, 1.)# the last parameter is the range of intensity of the image  计算一个batch的psnr
        ssim = batch_SSIM(Out, ISource)
        psnr_test += psnr
        ssim_test += ssim
        k+=1
        print("%s PSNR %.4f SSIM: %.4f" % (f, psnr,ssim))
        logger.info("image:{}  psnr:{:.4f}  ssim:{:.4f} "
                    .format(k, psnr, ssim))

    psnr_test /= len(files_source)
    ssim_test /= len(files_source)
    print("\nPSNR, SSIM on test data %.4f,%.4f" %( psnr_test,ssim_test))
    logger.info("\t avg_psnr:{:.4f}  avg_ssim:{:.4f}  "
                    .format( psnr_test, ssim_test))
    t_end = time.time()
    print('Test consumes time= %2.4f' % (t_end - t_start))
    print(100 * '*')
    logger.info("\t Test consumes time= %2.4f" % (t_end - t_start))

if __name__ == "__main__":
    main()
