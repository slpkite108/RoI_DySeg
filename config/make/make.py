import os
import yaml
from easydict import EasyDict
import pprint
from collections import OrderedDict

# 리스트만 flow style로 출력하도록 설정하는 Custom Dumper
class FlowStyleListDumper(yaml.Dumper):
    def represent_sequence(self, tag, sequence, flow_style=None):
        # 리스트만 flow style로 지정하고 나머지는 block style
        if isinstance(sequence, list):
            return super().represent_sequence(tag, sequence, flow_style=True)
        return super().represent_sequence(tag, sequence, flow_style=False)

def recursive_override(cfg, base_path):
    for key, value in list(cfg.items()):
        if isinstance(value, dict) and 'yml_path' in value:
            path_value = value['yml_path']
            target_yaml_path = os.path.join(base_path, key, path_value)
            
            # 해당 위치의 yml 파일을 로드하여 기존 value를 덮어씌웁니다.
            if os.path.exists(target_yaml_path):
                with open(target_yaml_path, 'r') as target_file:
                    override_value = yaml.load(target_file, Loader=yaml.Loader)
                    override_value = EasyDict(override_value)
                    cfg[key].update(override_value)
                    
                    # 'yml_path' 키를 제거합니다.
                    del cfg[key]['yml_path']
                    
                    # 중첩된 구조도 재귀적으로 처리합니다.
                    recursive_override(cfg[key], base_path)
            else:
                print(f"Warning: {target_yaml_path} 파일이 존재하지 않습니다.")

def load_yaml_with_override(yml_path):
    # yml 파일을 로드합니다.
    with open(yml_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # easydict로 변환합니다.
    config = EasyDict(config)
    cwd = os.path.dirname(os.path.abspath(yml_path))
    
    # key: value가 dict이고, 해당 dict에 'path' 키가 있는 경우 중첩적으로 처리합니다.
    recursive_override(config, cwd)
    return config

def recursive_dict(cfg):
    if isinstance(cfg, EasyDict):
        cfg = {k: recursive_dict(v) for k, v in cfg.items()}
    elif isinstance(cfg, dict):
        cfg = {k: recursive_dict(v) for k, v in cfg.items()}
    return cfg

# 사용 예시
yml_path = os.path.join(os.path.dirname(__file__), 'config.yml')  # 메인 yml 파일 경로
config = load_yaml_with_override(yml_path)

file_name = f'{config.run.checkpoint}_{config.run.model.name}_lab{config.run.loader.target_labels}_[{config.run.loader.dataset.scale}]_lr[{config.train.optimizer.lr}]'
config.run.checkpoint = file_name
# 예쁘게 출력하기
pprint.pprint(config)

# 결과를 preset 폴더 아래에 yml로 저장하기
preset_folder = os.path.join(os.path.dirname(__file__), 'preset')
os.makedirs(preset_folder, exist_ok=True)
output_path = os.path.join(preset_folder, f'{file_name}.yml')

# dict로 변환하여 저장
config_dict = recursive_dict(config)
with open(output_path, 'w') as output_file:
    yaml.dump(config_dict, output_file, Dumper=FlowStyleListDumper, allow_unicode=True)

print(f"결과가 {output_path}에 저장되었습니다.")
