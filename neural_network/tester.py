

import torch
from torch import optim
from torch import nn
from collections import OrderedDict
from utils import *
import os
from model import net_G, net_D
# from lr_sh import  MultiStepLR

# added
import datetime
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import params
import visdom


def tester(args):
    print('Evaluation Mode...')

    image_saved_path = params.output_dir + '/' + args.model_name + '/' + args.logs + '/test_outputs'
    if not os.path.exists(image_saved_path):
        os.makedirs(image_saved_path)

    if args.use_visdom:
        vis = visdom.Visdom()

    save_file_path = params.output_dir + '/' + args.model_name
    pretrained_file_path_G = save_file_path + '/' + args.logs + '/models/G.pth'
    pretrained_file_path_D = save_file_path + '/' + args.logs + '/models/D.pth'

    print(pretrained_file_path_G)

    D = net_D(args)
    G = net_G(args)

    if not torch.cuda.is_available():
        G.load_state_dict(torch.load(pretrained_file_path_G, map_location={'cuda:0': 'cpu'}))
        D.load_state_dict(torch.load(pretrained_file_path_D, map_location={'cuda:0': 'cpu'}))
    else:
        G.load_state_dict(torch.load(pretrained_file_path_G))
        D.load_state_dict(torch.load(pretrained_file_path_D, map_location={'cuda:0': 'cpu'}))

    print('visualizing model')


    G.to(params.device)
    D.to(params.device)
    G.eval()
    D.eval()


    N = 8

    for i in range(N):

        z = generateZ(args, 1)


        fake = G(z)
        samples = fake.unsqueeze(dim=0).detach().cpu().numpy()
        y_prob = D(fake)
        y_real = torch.ones_like(y_prob)
        if not args.use_visdom:
            SavePloat_Voxels(samples, image_saved_path, 'tester_' + str(i))  # norm_
        else:
            plotVoxelVisdom(samples[0, :], vis, "tester_" + str(i))
