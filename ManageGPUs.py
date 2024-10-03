import pynvml
import numpy as np


class ManageGPUs:
    def __init__(self):
        pass

    @staticmethod
    def get_gpu_utilization():
        usages = {}
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            usages[i] = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        pynvml.nvmlShutdown()
        best_gpu_nums = dict(sorted(usages.items(), key=lambda item: item[1]))
        best_gpu_nums = (list(best_gpu_nums.keys()))
        return best_gpu_nums




