from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import time
import math
from models import *
from PIL import Image

parser = argparse.ArgumentParser(description='PSMNet')
# KITTI stands for KITTI version 
parser.add_argument('--KITTI', default='2015', 
                    help='KITTI version')
# datapath stands for the place where you want to test, wiat, what?   
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/testing/',
                    help='select model')
# loadmodel stands for the trained model, which is the disparity map
parser.add_argument('--loadmodel', default='./trained/pretrained_model_KITTI2015.tar',
                    help='loading model')
# this argument is not used 
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
# The maximum disparity we allowed 
parser.add_argument('--maxdisp', default=192,
                    help='maxium disparity')
# Do we enable cuda training or not
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
# set random seed
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
# where are you go to save the path for your prediction
parser.add_argument('--save_path', type=str, default='finetune_1000', metavar='S',
                    help='path to save the predict')
# Save the figure 
parser.add_argument('--save_figure', action='store_true', help='if true, save the numpy file, not the png file')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

#this part is simply about setting the seed and cuda the seed. Not important. 
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

#this part is interesting because I really want to see the loader of the data. Notice that dataset here is 2015 version,
#only need to check the version of 2015. 
if args.KITTI == '2015':
    from dataloader import KITTI_submission_loader as DA
else:
    from dataloader import KITTI_submission_loader2012 as DA  

# you get two lists, one is the list for left image and one is the list for right image
test_left_img, test_right_img = DA.dataloader(args.datapath)

#loaded in the data that was specified. 
if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

#If you have multiple gpus you can modify here. 
model = nn.DataParallel(model, device_ids=[0])
model.cuda()

#directly load the model with our trained weights
if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

# Given a pair of left images and right images, this function would output the result
def test(imgL,imgR):
    model.eval()

    if args.cuda:
        imgL = imgL.cuda()
        imgR = imgR.cuda()     

    with torch.no_grad():
        output = model(imgL,imgR)
    output = torch.squeeze(output).data.cpu().numpy()
    return output

# The main function for the script
def main():
    #print(os.path.abspath("."))
    #transfer the data to be suited for the model
    normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    infer_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(**normal_mean_var)])    

    for inx in range(len(test_left_img)):

        imgL_o = Image.open(test_left_img[inx]).convert('RGB')
        imgR_o = Image.open(test_right_img[inx]).convert('RGB')

        imgL = infer_transform(imgL_o)
        imgR = infer_transform(imgR_o)         

        # pad to width and hight to 16 times
        if imgL.shape[1] % 16 != 0:
            times = imgL.shape[1]//16       
            top_pad = (times+1)*16 -imgL.shape[1]
        else:
            top_pad = 0

        if imgL.shape[2] % 16 != 0:
            times = imgL.shape[2]//16                       
            right_pad = (times+1)*16-imgL.shape[2]
        else:
            right_pad = 0    

        imgL = F.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
        imgR = F.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)

        start_time = time.time()
        pred_disp = test(imgL,imgR)
        print('time = %.2f' %(time.time() - start_time))

        if top_pad !=0 or right_pad != 0:
            img = pred_disp[top_pad:,:-right_pad]
        else:
            img = pred_disp

        #img = (img*256).astype('uint16')
        #img = Image.fromarray(img)
        #img.save(test_left_img[inx].split('/')[-1])
        
        if args.save_figure:
            skimage.io.imsave(args.save_path+'/'+test_left_img[inx].split('/')[-1],(img*256).astype('uint16'))
        else:
            if not os.path.isdir(args.save_path):
                os.makedirs(args.save_path)
            np.save(args.save_path+'/'+test_left_img[inx].split('/')[-1][:-4], img)
            #file_path = "/home/zhengjiwang/data/KITTI/object/training/disparity/{0:06d}.npy".format(inx)
            #disparity_gt = np.load(file_path)
            #gt_mask = disparity_gt >= -300
            #print(np.abs(disparity_gt[gt_mask] - img[gt_mask]).mean())


if __name__ == '__main__':
    main()






