import yaml
from easydict import EasyDict as edict

def Prepare(config_path):
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)  # YAML 파일을 딕셔너리로 로드
    config = edict(config_dict)  # 딕셔너리를 EasyDict로 변환
    return config