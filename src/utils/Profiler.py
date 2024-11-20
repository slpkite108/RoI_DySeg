import pynvml
import torch
import threading
import time
from datetime import datetime

from .Statistics import Statistics

def get_gpu_handle(device_num):
    return pynvml.nvmlDeviceGetHandleByIndex(device_num)

def get_process_gpu_memory_info(handle, pid):
    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    for process in processes:
        if process.pid == pid:
            # NVML 메모리 사용량
            nvml_memory = process.usedGpuMemory / 1024**2  # MB 단위
            # PyTorch 메모리 사용량
            torch_memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB 단위
            torch_memory_reserved = torch.cuda.memory_reserved() / 1024**2  # MB 단위
            return {
                'nvml_memory': nvml_memory,
                'torch_memory_allocated': torch_memory_allocated,
                'torch_memory_reserved': torch_memory_reserved
            }
    return {
        'nvml_memory': 0,
        'torch_memory_allocated': 0,
        'torch_memory_reserved': 0
    }  # 해당 PID가 없는 경우 0 반환

class Profiler:
    _shared_thread = None
    _shared_stop_event = threading.Event()
    _shared_interval = 1
    _handles = []

    def __init__(self, device_num, pid, interval=None):
        if not pynvml.nvmlInit():
            pynvml.nvmlInit()
            
        self.device_num = int(device_num)
        self.pid = pid
        self.max_memory_usage = {'nvml_memory': 0, 'torch_memory_allocated': 0, 'torch_memory_reserved': 0}
        self.statistics = {
            'nvml_memory': Statistics(),
            'torch_memory_allocated': Statistics(),
            'torch_memory_reserved': Statistics()
        }
        self.handle = get_gpu_handle(self.device_num)
        Profiler._handles.append((self.handle, self))
        
        # 새로운 interval 값이 주어지면 공유 interval 업데이트
        if interval is not None:
            Profiler._shared_interval = interval

        # 공유 스레드가 없다면 생성
        if Profiler._shared_thread is None:
            Profiler._shared_thread = threading.Thread(target=self._profile_memory)
            Profiler._shared_thread.start()

    @classmethod
    def _profile_memory(cls):
        while not cls._shared_stop_event.is_set():
            for handle, profiler in cls._handles:
                memory_info = get_process_gpu_memory_info(handle, profiler.pid)
                
                # 각 메모리 사용량을 통계에 추가
                for key in memory_info:
                    profiler.statistics[key].add_number(memory_info[key])
                    if memory_info[key] > profiler.max_memory_usage[key]:
                        profiler.max_memory_usage[key] = memory_info[key]
                
            time.sleep(cls._shared_interval)

    def start(self):
        pass  # 이미 공유 스레드가 실행 중이므로 따로 할 일 없음

    def stop(self):
        Profiler._handles.remove((self.handle, self))
        if not Profiler._handles:
            Profiler._shared_stop_event.set()
            Profiler._shared_thread.join()
            Profiler._shared_thread = None
            Profiler._shared_stop_event.clear()

    def get_max_memory_usage(self):
        return self.max_memory_usage
    
    def get_statistics(self):
        return {
            'nvml_memory': {
                'sum': self.statistics['nvml_memory'].get_sum(),
                'mean': self.statistics['nvml_memory'].get_mean(),
                'median': self.statistics['nvml_memory'].get_median(),
                'min': self.statistics['nvml_memory'].get_min(),
                'max': self.statistics['nvml_memory'].get_max()
            },
            'torch_memory_allocated': {
                'sum': self.statistics['torch_memory_allocated'].get_sum(),
                'mean': self.statistics['torch_memory_allocated'].get_mean(),
                'median': self.statistics['torch_memory_allocated'].get_median(),
                'min': self.statistics['torch_memory_allocated'].get_min(),
                'max': self.statistics['torch_memory_allocated'].get_max()
            },
            'torch_memory_reserved': {
                'sum': self.statistics['torch_memory_reserved'].get_sum(),
                'mean': self.statistics['torch_memory_reserved'].get_mean(),
                'median': self.statistics['torch_memory_reserved'].get_median(),
                'min': self.statistics['torch_memory_reserved'].get_min(),
                'max': self.statistics['torch_memory_reserved'].get_max()
            }
        }

    def get_runtime(self):
        return self.runtime
    
    def shutdown(self):
        pynvml.nvmlShutdown()

    def __enter__(self):
        self.start()
        self.start_time = datetime.now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self.end_time = datetime.now()
        self.runtime = (self.end_time - self.start_time)
        if not Profiler._handles:  # 모든 프로파일러가 종료되었을 때만 NVML 종료
            self.shutdown()
