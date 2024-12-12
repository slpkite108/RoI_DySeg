import logging
import os
from accelerate.logging import get_logger

def getLogger(accelerator, mode='train'):
    """
    Configure and return a logger with StreamHandler and FileHandler.
    Ensures no duplicate logging and prevents FileHandler output from being shown in the stream.
    """
    logging_dir = accelerator.project_dir
    logger = get_logger(mode, log_level="INFO")
    logger.logger.propagate = False  # Prevent double logging to the root logger

    # Formatter for consistent log output
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    # Update or add StreamHandler
    stream_handler = next((h for h in logger.logger.handlers if isinstance(h, logging.StreamHandler)), None)
    if stream_handler:
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(formatter)  # Ensure proper formatting
    else:
        sh = logging.StreamHandler()
        sh.setLevel(logging.DEBUG)
        sh.setFormatter(formatter)
        logger.logger.addHandler(sh)

    # Update or add FileHandler
    file_handler = next(
        (h for h in logger.logger.handlers if isinstance(h, logging.FileHandler) 
         and h.baseFilename.endswith('info.log')), None)
    if file_handler:
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)  # Ensure proper formatting
    else:
        fh = logging.FileHandler(os.path.join(logging_dir, 'info.log'), encoding='utf-8', delay=True)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.logger.addHandler(fh)
    
    return logger
