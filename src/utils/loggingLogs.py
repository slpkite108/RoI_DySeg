import logging

def loggingLogs(logger:logging.Logger, logs:dict):
    for key, value in logs.items():
        logger.info(f"{key} : {value}")
    return