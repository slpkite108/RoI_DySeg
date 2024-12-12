import os
import torch
import argparse
from src import utils
from train import train
from inference import inference
from generation import generation
import setproctitle
from datetime import datetime

def run(configs, run_train, run_inference, run_generation, useDeviceNum):
    
    setproctitle.setproctitle(configs.run.checkpoint)
    
    if useDeviceNum == -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = configs.run.device_num
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(useDeviceNum)
    
    if run_train:
        print('start train')
        train(configs)

    if  run_inference:
        print('start inference')
        inference(configs)

    if  run_generation:
        print('start generation')
        generation(configs)

    return



if __name__ == "__main__":
    torch.cuda.init
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default='/home/work/jayeon/git/RoI_DySeg/config/make/preset/Default_SlimUNETR_lab[1]_[128]_lr[0.002].yml', help="path of the config yml file")
    parser.add_argument("--train", action='store_false', help='run train mode',default=True)
    parser.add_argument("--inference", action='store_false', help='run inference mode',default=True)
    parser.add_argument("--generation", action='store_false', help='run generation mode',default=True)
    parser.add_argument("--use_spec_device", type=int, help='do not manage device number',default=-1)
    opt = parser.parse_args()
    
    #config_path = '/home/work/jayeon/git/RoI_DySeg/config/make/preset/Default_SlimUNETR_lab[1]_[128]_lr[0.002].yml'
    configs = utils.Prepare(opt.config_path)
    
    run(configs, opt.train, opt.inference, opt.generation, opt.use_spec_device)
    