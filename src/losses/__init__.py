import os
import importlib

# 현재 디렉터리(즉, losses 폴더)의 파일들을 확인합니다.
module_files = [f for f in os.listdir(os.path.dirname(__file__)) if f.endswith(".py") and f != "__init__.py"]
__all__ = []

# 각 모듈을 동적으로 import하고, 모듈의 주요 클래스나 함수를 __all__ 리스트에 추가합니다.
for module_file in module_files:
    module_name = module_file[:-3]  # .py 확장자 제거
    module = importlib.import_module(f".{module_name}", package=__name__)
    
    # 모듈 내 주요 객체를 __all__ 리스트에 등록합니다.
    if hasattr(module, module_name):  # 모듈 이름과 동일한 클래스가 있는지 확인
        globals()[module_name] = getattr(module, module_name)
        __all__.append(module_name)
