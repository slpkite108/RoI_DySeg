import logging
import os
from accelerate.logging import get_logger

def getLogger(accelerator, configs):
    logging_dir = accelerator.config.project_dir
    logger = get_logger('main', log_level="DEBUG")
    
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    logger.logger.addHandler(sh)
    
    fh = logging.FileHandler(os.path.join(logging_dir,'info.log'), encoding='utf-8', delay=True)
    fh.setLevel(logging.INFO)
    logger.logger.addHandler(fh)
    
    return logger
