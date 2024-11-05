import utils

class SetUp:
    def __init__(self, configs, accelerator, logger):
        self.configs = configs
        self.accelerator = accelerator
        self.logger = logger
        
        self.setup()
        pass
    
    def setup(self):
        try:
            self.module = utils.getModule(self.configs)
        except Exception as e:
            self.logger.error(f"Cannot Loaded Module : {self.module}")
            raise e
        
        try:
            self.train_obj, self.inference_obj, self.generation_obj = utils.setupObj(self.accelerator, self.configs)
            self.train_obj = None #optimizer, scheduler, metrics, transforms, loss, parameters, accelerator
            self.inference_obj = None # metrics, transforms,accelerator
            self.generation_obj = None # accelerator
            pass
        except Exception as e:
            self.logger.error(f'Cannot made Objs')
            pass
        pass
    
    def train(self):
        with utils.Profiler(self.configs.device_num, self.configs.current_pid, interval=self.configs.interval) as profiler:
            pass

    def inference(self):
        with utils.Profiler(self.configs.device_num, self.configs.current_pid, interval=self.configs.interval) as profiler:
            pass

    def generation(self):
        with utils.Profiler(self.configs.device_num, self.configs.current_pid, interval=self.configs.interval) as profiler:
            pass