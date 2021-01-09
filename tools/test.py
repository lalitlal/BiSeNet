
import sys
import os
sys.path.insert(0, '.')
import argparse
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
from os.path import basename

import lib.transform_cv2 as T
from lib.models import model_factory
from configs import cfg_factory
import time
from labels import *

torch.set_grad_enabled(False)
np.random.seed(123)


# args
parse = argparse.ArgumentParser()
parse.add_argument('--model', dest='model', type=str, default='bisenetv2',)
parse.add_argument('--weight-path', type=str, default='./res/model_final.pth',)
parse.add_argument('--img-path', dest='img_path', type=str, default='./example.png',)
args = parse.parse_args()
cfg = cfg_factory[args.model]


palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
cityspallete = [
    128, 64, 128,
    244, 35, 232,
    70, 70, 70,
    102, 102, 156,
    190, 153, 153,
    153, 153, 153,
    250, 170, 30,
    220, 220, 0,
    107, 142, 35,
    152, 251, 152,
    0, 130, 180,
    220, 20, 60,
    255, 0, 0,
    0, 0, 142,
    0, 0, 70,
    0, 60, 100,
    0, 80, 100,
    0, 0, 230,
    119, 11, 32,
]

# define model
net = model_factory[cfg.model_type](19)
net.load_state_dict(torch.load(args.weight_path, map_location='cpu'))
net.eval()
net.cuda()

# prepare data
to_tensor = T.ToTensor(
    mean=(0.3257, 0.3690, 0.3223), # city, rgb
    std=(0.2112, 0.2148, 0.2115),
)

# directory structure:
# rootdir = '../datasets/leftImg8bit/test'
# respath = '.testres/'
d = os.path.dirname(os.getcwd())
rootdir = d + '/' + 'BiSeNet/datasets/cityscapes/leftImg8bit/test'
respath = d + '/BiSeNet/testsubmission/'
# go through each image and output it into results:

start_time = time.time()
num_pics = 0
end = 0
total_time = 0
num_images = 0
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        im = cv2.imread(os.path.join(subdir, file))[:, :, ::-1]
        im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()
        start_time = time.time()
        # inference
        out = net(im)[0]
        total_time += time.time() - start_time
        out = out.argmax(dim=1).squeeze().detach().cpu().numpy()
#         pred = Image.fromarray(out.astype('uint8'))
#         pred.putpalette(cityspallete)
        
        # for the submission!
        h,w = np.shape(out)
        finalout = np.empty((h,w))
        for i in range(h):
            for j in range(w):
                finalout[i,j] = trainId2label[out[i,j]].id
        subFile = Image.fromarray(finalout.astype('uint8'))
        dirname = respath + '/' + basename(subdir)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        outpath = dirname + '/' + basename(file)
        subFile.save(outpath)
        num_images += 1

print(num_images, total_time, num_images/total_time)


