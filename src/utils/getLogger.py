import logging
import os
from accelerate.logging import get_logger

def getLogger(acceletor, mode='train'):
    logging_dir = acceletor.project_dir
    logger = get_logger(mode, log_level="INFO")
    
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    logger.logger.addHandler(sh)
    
    fh = logging.FileHandler(os.path.join(logging_dir,'info.log'), encoding='utf-8', delay=True)
    fh.setLevel(logging.INFO)
    logger.logger.addHandler(fh)
    
    return logger