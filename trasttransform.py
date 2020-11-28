import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
import cv2
import glob
import time
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from DerainDataset import *
from utils import *
from torch.optim.lr_scheduler import MultiStepLR
from SSIM import SSIM
from DFN import RRFBN

from PIL import Image
from pylab import *


import matplotlib
matplotlib.use('Agg')  #ssh x
import matplotlib.pyplot as plt

fp=open("Rain_12600_0.txt",'w')
parser = argparse.ArgumentParser(description="DFN_train")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=[30,50,80], help="When to decay learning rate")
parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
parser.add_argument("--save_path1", type=str, default="logs/h/", help='path to save models and log files')
parser.add_argument("--save_freq",type=int,default=1,help='save intermediate model')
parser.add_argument("--data_path1",type=str, default="../data/RainTrainH/",help='path to training data')
parser.add_argument("--use_gpu1", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id1", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter1", type=int, default=6, help='number of recursive stages')
parser.add_argument("--number_blocks", type=int, default=2, help='number of feedback blocks')
parser.add_argument("--use_cl", type=bool, default=True, help='use cl or not')
parser.add_argument("--logdir", type=str, default="logs/h/", help='path to model and log files')
parser.add_argument("--data_path", type=str, default="../data/Rain100H1/rainy", help='path to training data')
parser.add_argument("--save_path", type=str, default="results/100H/", help='path to save results')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=7, help='number of recursive stages')
opt = parser.parse_args()

if opt.use_gpu1:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id1


#注销裁剪
"""
def cutpic(im_path):
    pil_im = Image.open(f'{im_path}')
    width,height=pil_im.size
    if width%2==1:
        width=width-1
    if height%2==1:
        height-=1
    box = (0,0,width,height)
    region = pil_im.crop(box)
    region.save(f'{im_path}')
"""

def main():

    xs=[]
    y1=[]
    ys=[]
    PSNR=[]
    PSNRs=[]

    if not os.path.isdir(opt.save_path1):
        os.makedirs(opt.save_path1)
        # Load dataset
    print('Loading dataset ...\n')
    if (opt.data_path1.find('Light') != -1 or opt.data_path1.find('Heavy') != -1):
        dataset_train = newDataset(data_path=opt.data_path1)
    else:
        dataset_train = Dataset(data_path=opt.data_path1)

    loader_train = DataLoader(dataset=dataset_train, num_workers=128, batch_size=opt.batch_size, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))

    # Build model

    model = RRFBN(iterations=opt.recurrent_iter1,blocks=opt.number_blocks)
    #model = torch.nn.DataParallel(model,device_ids=[0,1])

    print_network(model)

    # loss function
    #criterion = nn.MSELoss()
    criterion = SSIM()
    #criterion=nn.L1Loss()

    # Move to GPU
    if opt.use_gpu1:
        model = model.cuda()
        criterion.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.2)  # learning rates

    # record training
    writer = SummaryWriter(opt.save_path1)

    # load the lastest model
    initial_epoch = findLastCheckpoint(save_dir=opt.save_path1)
    
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        model.load_state_dict(torch.load(os.path.join(opt.save_path1, 'net_epoch%d.pth' % initial_epoch)))



    """
    for img_name in os.listdir(opt.data_path):
            if is_image(img_name):
                img_path = os.path.join(opt.data_path, img_name)
                cutpic(img_path)

    """

    # start training
    step = 0
    for epoch in range(initial_epoch, opt.epochs):
        scheduler.step(epoch)
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])

        ## epoch training start
        for i, (input_train, target_train) in enumerate(loader_train, 0):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            input_train, target_train = Variable(input_train), Variable(target_train)

            if opt.use_gpu1:
                input_train, target_train = input_train.cuda(), target_train.cuda()

            out_train, out_list = model(input_train)
            loss = 0
            if opt.use_cl:
                pixel_metric = sum([criterion(target_train, out_list[idx])*(idx+1) for idx in range(len(out_list))])/sum(range((len(out_list)+1)))
            else:
                pixel_metric = criterion(target_train, out_train)
            loss =loss-pixel_metric
            
            loss.backward()
            optimizer.step()

            # training curve
            model.eval()
            out_train, _ = model(input_train)
            out_train = torch.clamp(out_train, 0., 1.)
            psnr_train = batch_PSNR(out_train, target_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f, pixel_metric: %.4f, PSNR: %.4f" %
                  (epoch+1, i+1, len(loader_train), loss.item(), pixel_metric.item(), psnr_train))
            fp.write("[epoch %d][%d/%d] loss: %.4f, pixel_metric: %.4f, PSNR: %.4f" %
                  (epoch+1, i+1, len(loader_train), loss.item(), pixel_metric.item(), psnr_train))
            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1
            y1.append(loss.item())
            PSNR.append(psnr_train)

        avgy=sum(y1)/len(y1)
        avgPSNR=sum(PSNR)/len(PSNR)
        ys.append(avgy)
        PSNRs.append(avgPSNR)
        y1=[]
        PSNR=[]
        xs.append(epoch+1)


        ## epoch training end

        # log the images
        model.eval()
        time_test = 0
        count = 0
        for img_name in os.listdir(opt.data_path):
            if is_image(img_name):
                img_path = os.path.join(opt.data_path, img_name)


                # input image
                y = cv2.imread(img_path)
                b, g, r = cv2.split(y)
                y = cv2.merge([r, g, b])
                #y = cv2.resize(y, (int(500), int(500)), interpolation=cv2.INTER_CUBIC)
    
                y = normalize(np.float32(y))
                y = np.expand_dims(y.transpose(2, 0, 1), 0)
                y = Variable(torch.Tensor(y))
    
                if opt.use_GPU:
                    y = y.cuda()
    
                with torch.no_grad(): #
                    if opt.use_GPU:
                        torch.cuda.synchronize()
                    start_time = time.time()
    
                    _,outlist = model(y)
                    for i in range(len(outlist)):
                        outlist[i] = torch.clamp(outlist[i], 0., 1.)
    
                    if opt.use_GPU:
                        torch.cuda.synchronize()
                    end_time = time.time()
                    dur_time = end_time - start_time
                    time_test += dur_time
    
                    print(img_name, ': ', dur_time)
    
                if opt.use_GPU:
                    jj=0
                    for out in outlist:
                        rain_steak=y-out
                        save_out = np.uint8(255 * rain_steak.data.cpu().numpy().squeeze())   #back to cpu
                        save_out = save_out.transpose(1, 2, 0)
                        b, g, r = cv2.split(save_out)
                        save_out = cv2.merge([r, g, b])
                        cv2.imwrite(os.path.join(opt.save_path,str(jj)+"_"+img_name),save_out)
                        jj=jj+1
    
                    jj=0
                    for out in outlist:
                        save_out = np.uint8(255 *out.data.cpu().numpy().squeeze())
                        save_out = save_out.transpose(1, 2, 0)
                        b, g, r = cv2.split(save_out)
                        save_out = cv2.merge([r, g, b])
                        cv2.imwrite(os.path.join(opt.save_path, str(jj) + "1_" + img_name), save_out)
                        jj=jj + 1
    
    
    
                count += 1
        print('Avg. time:', time_test/count)
        out_train, _ = model(input_train)
        out_train = torch.clamp(out_train, 0., 1.)
        im_target = utils.make_grid(target_train.data, nrow=8, normalize=True, scale_each=True)
        im_input = utils.make_grid(input_train.data, nrow=8, normalize=True, scale_each=True)
        im_derain = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        writer.add_image('clean image', im_target, epoch+1)
        writer.add_image('rainy image', im_input, epoch+1)
        writer.add_image('deraining image', im_derain, epoch+1)

        # save model
        torch.save(model.state_dict(), os.path.join(opt.save_path1, 'net_latest.pth'))
        if epoch % opt.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(opt.save_path1, 'net_epoch%d.pth' % (epoch+1)))



    
    fp.close()
    
    plt.figure()
    plt.subplot(121)
    plt.plot(xs,ys,'b')
    plt.subplot(122)
    plt.plot(xs,PSNRs,'r')
    plt.savefig("pic")

if __name__ == "__main__":
    if opt.preprocess:
        if opt.data_path1.find('RainTrainH') != -1:
            print(opt.data_path1.find('RainTrainH'))
            prepare_data_RainTrainH(data_path1=opt.data_path1, patch_size=100, stride=80)
        elif opt.data_path1.find('RainTrainL') != -1:
            prepare_data_RainTrainL(data_path1=opt.data_path1, patch_size=100, stride=80)
        elif opt.data_path1.find('Rain12600') != -1:
            prepare_data_Rain12600(data_path1=opt.data_path1, patch_size=100, stride=100)
        elif opt.data_path1.find('real') != -1:
            prepare_data_real_test(data_path1=opt.data_path1, patch_size=100, stride=80)
        else:
            print('unkown datasets: please define prepare data function in DerainDataset.py')


    main()
