import os, psutil
import torch


class MemoryUsage:
    def __init__(self, logger=None):
        self.max_mem_GPU = 0
        self.max_mem_GPU_name = ""
        self.max_mem_CPU = 0
        self.max_mem_CPU_name = ""
        self.process = psutil.Process(os.getpid())
        self.LOG = logger
    def fmt_mem(self, value_bytes):
        return "%.02fMiB"%(value_bytes/(1024*1024),)
    def check_memory(self, name="", verbose=True):
        used_mem_GPU = torch.cuda.memory_allocated()
        if used_mem_GPU>self.max_mem_GPU:
            self.max_mem_GPU = used_mem_GPU
            self.max_mem_GPU_name = name
        used_mem_CPU = self.process.memory_info()[0]
        if used_mem_CPU>self.max_mem_CPU:
            self.max_mem_CPU = used_mem_CPU
            self.max_mem_CPU_name = name
        if verbose and (self.LOG is not None): self.LOG.info("used memory '%s': CPU %s, GPU %s", name, self.fmt_mem(used_mem_CPU), self.fmt_mem(used_mem_GPU))
    def print_max_memory(self):
        if self.LOG is not None: self.LOG.info("Max used memory: CPU %s at %s, GPU %s at '%s'",
            self.fmt_mem(self.max_mem_CPU), self.max_mem_CPU_name, self.fmt_mem(self.max_mem_GPU), self.max_mem_GPU_name)