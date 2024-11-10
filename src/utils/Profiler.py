import pynvml
import threading
import time
from datetime import datetime

from .Statistics import Statistics

# NVML 초기화
pynvml.nvmlInit()

def get_gpu_handle(device_num):
    return pynvml.nvmlDeviceGetHandleByIndex(device_num)

def get_process_gpu_memory_info(handle, pid):
    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    for process in processes:
        if process.pid == pid:
            return process.usedGpuMemory / 1024**2  # 사용된 메모리 (MB 단위)
    return 0  # 해당 PID가 없는 경우 0 반환

class Profiler:
    _shared_thread = None
    _shared_stop_event = threading.Event()
    _shared_interval = 1
    _handles = []

    def __init__(self, device_num, pid, interval=None):
        self.device_num = int(device_num)
        self.pid = pid
        self.max_memory_usage = 0
        self.statistics = Statistics()  # Statistics 객체 추가
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
                current_memory_usage = get_process_gpu_memory_info(handle, profiler.pid)
                profiler.statistics.add_number(current_memory_usage)  # 통계에 현재 메모리 사용량 추가
                if current_memory_usage > profiler.max_memory_usage:
                    profiler.max_memory_usage = current_memory_usage
            time.sleep(cls._shared_interval)

    def start(self):
        # 이미 공유 스레드가 실행 중이므로 따로 할 일 없음
        pass

    def stop(self):
        # 공유 스레드를 종료하려면 모든 객체가 종료해야 함
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
            'sum': self.statistics.get_sum(),
            'mean': self.statistics.get_mean(),
            'median': self.statistics.get_median(),
            'min': self.statistics.get_min(),
            'max': self.statistics.get_max()
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

# 사용 예시
# with Profiler(framework_obj.device_num, framework_obj.current_pid, interval=0.1) as train_profiler:
#     # ... Training code ...
#     stats = train_profiler.get_statistics()
#     print("Memory Statistics:", stats)
