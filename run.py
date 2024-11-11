import os
from src import utils
from train import train
from inference import inference
from generation import generation

from datetime import datetime

def run(configs, run_train, run_inference, run_generation):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = configs.run.device_num
    
    if run_train:
        train(configs)

    if  run_inference:
        inference(configs)

    if  run_generation:
        generation(configs)
    
    
    return

if __name__ == "__main__":
    config_path = 'config/Default_SlimUNETR_Amos22_128.yml'
    configs = utils.Prepare(config_path)
    
    run_train = True
    run_inference = True
    run_generation = True
    run(configs, run_train, run_inference, run_generation)
    