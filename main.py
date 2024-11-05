import json
import utils

from datetime import datetime
from train import train_model
from run import run

# from validate import validate_model  # 검증 함수가 구현되어 있다고 가정

def main(config_path='config/config.json'):
    
    configs = utils.Prepare(config_path)
    
    accelerator = utils.getAccelerator(configs.work_dir,configs.checkpoint)
    
    logger = utils.getLogger(accelerator, configs)
    
    #interval 0.1 
    start_time = datetime.now()
    run(accelerator, logger, configs)
    end_time = datetime.now()

    
    
    exit(0)
    
if __name__ == '__main__':
    main()
