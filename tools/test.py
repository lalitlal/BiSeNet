
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
rootdir = '../datasets/leftImg8bit/test'
respath = '.testres/'
# go through each image and output it into results:
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        im = cv2.imread(os.path.join(subdir, file))[:, :, ::-1]
        im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()

        # inference
        out = net(im)[0].argmax(dim=1).squeeze().detach().cpu().numpy()
        pred = palette[out]

        dirname = respath + basename(subdir)
        print(dirname)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        outpath = dirname + '/' + basename(file)
        cv2.imwrite(outpath, pred)

