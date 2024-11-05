import utils
import train
import inference
import generation

def run(accelerator, logger, configs):
    logs = {}
        framework_obj = utils.SetUp(configs, accelerator, logger)
        
        if configs.train:
            framework_obj.train()
        
        if configs.inference:
            framework_obj.inference()
            
        if configs.generation:
            framework_obj.generation()
        
    
    logs.update(
        {
            "total runtime":end_time-start_time,
            "nvml max GPU Memory Usage":profiler.get_max_memory_usage(),
        }
    )
    
    utils.loggingLogs(logger, logs)
    
    return