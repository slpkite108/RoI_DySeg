from easyDict import EasyDict
import yaml

def call(opt=None):
    try:
        if opt == None:
            config = EasyDict(
                yaml.load(open("config.yml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
            )
        else:
            config = None
            raise Exception("there is no opt option. need to add")
    except Exception as e:
        print(e)
        raise e
    
    return config