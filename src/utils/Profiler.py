import pynvml
import threading
import time

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
    def __init__(self, device_num, pid, interval=1):
        self.device_num = device_num
        self.pid = pid
        self.interval = interval
        self.max_memory_usage = 0
        self._stop_event = threading.Event()
        self.thread = threading.Thread(target=self._profile_memory)
        self.handle = get_gpu_handle(device_num)

    def _profile_memory(self):
        while not self._stop_event.is_set():
            current_memory_usage = get_process_gpu_memory_info(self.handle, self.pid)
            if current_memory_usage > self.max_memory_usage:
                self.max_memory_usage = current_memory_usage
            time.sleep(self.interval)

    def start(self):
        self.thread.start()

    def stop(self):
        self._stop_event.set()
        self.thread.join()

    def get_max_memory_usage(self):
        return self.max_memory_usage

    def shutdown(self):
        pynvml.nvmlShutdown()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self.shutdown()

