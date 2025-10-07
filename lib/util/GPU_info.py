import psutil, subprocess, re
#, logging

#https://thispointer.com/python-get-list-of-all-running-processes-and-sort-by-highest-memory-usage/
def getInfoForPid(pid):
    for proc in psutil.process_iter():
        try:
            if proc.pid==pid:
                return proc.as_dict(['pid','name','username','memory_info'])
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    #logging.warning("Failed to get process info for PID %d.", pid)
    return {'pid':pid,'name':"unknown",'username':"unknown",'memory_info':"0"}


#filter_gpu = re.compile(r"\|\s+(\d+).+\|\s+([0-9A-F]{8}\:[0-9A-F]{2}\:[0-9A-F]{2}\.[0-9A-F])")
filter_gpu = re.compile(r"\|\s+(?P<gpu>\d+)\s+(?P<name>.+)\s+(?P<persm>\w+)\s+\|\s+(?P<busid>[0-9A-F]{8}\:[0-9A-F]{2}\:[0-9A-F]{2}\.[0-9A-F])\s+(?P<disp>\w+)\s+\|")
#filter_mem = re.compile(r".+\|\s+(\d+)MiB\s+/\s+(\d+)MiB\s+\|")#')
# | 31%   54C    P2    61W / 250W |   1421MiB / 11177MiB |     28%      Default |
filter_mem = re.compile(r"\|\s+((?P<fan>\d+)%?|N/A)\s+(?P<temp>\d+)(C|F)\s+(?P<perf>.+)\s+(?P<curpwr>\d+)W\s+/\s+(?P<maxpwr>\d+)W\s+\|\s+(?P<curmem>\d+)MiB\s+/\s+(?P<maxmem>\d+)MiB\s+\|\s+(?P<util>\d+)%\s+(?P<compm>.+)\s+\|")#')
#filter_process = re.compile(r'\|\s+(\d+)\s+(\d{1,6})\s+.+\s+([0-9a-zA-Z_/.-]+)\s+(\d+)MiB\s+\|')
filter_process = re.compile(r'\|\s+(?P<gpu>\d+)(\s+.+)?\s+(?P<pid>\d{1,6})\s+.+\s+(?P<name>[0-9a-zA-Z_/.-]+)\s+(?P<mem>\d+)MiB\s+\|')
def getGPUInfo(active_mem_threshold=0.05, verbose=False):
    info_process = subprocess.Popen('nvidia-smi', stdout=subprocess.PIPE)
    #https://stackoverflow.com/questions/18421757/live-output-from-subprocess-command
    info = {}
    last_gpu = -1
    for line in  iter(info_process.stdout.readline, b''):
        g = filter_gpu.search(str(line))
        m = filter_mem.search(str(line))
        p = filter_process.search(str(line))
        if g is not None:
            if verbose: print('{} ({})'.format(g.group("gpu"), g.group("busid")))
            last_gpu = int(g.group("gpu"))
            info[last_gpu]={'id':last_gpu, 'bus':g.group("busid"), 'processes':[]}
        elif m is not None and last_gpu!=-1:
            if verbose: print('{}: {:.03f}% Memory'.format(last_gpu, 100.0*float(m.group("curmem"))/float(m.group("maxmem"))))
            info[last_gpu].update({'mem_used':int(m.group("curmem")), 'mem_max':int(m.group("maxmem")), 'util':float(m.group("util"))/100.0, 'fan':float(-1 if m.group("fan") is None else m.group("fan"))/100.0, 'temp':float(m.group("temp")), 'pwr_used':int(m.group("curpwr")),'pwr_max':int(m.group("maxpwr")) })
            last_gpu = -1
        elif p is not None:
            pid = int(p.group("pid"))
            pInfo = getInfoForPid(pid)
            gpu_id = int(p.group("gpu"))
            if verbose: print('{} from {} ({}, {:.2f}MiB) on GPU {} ({}MiB)'.format(p.group("name"), pInfo['username'], pid, pInfo['memory_info'].vms/(1024*1024), p.group(1), p.group("mem")))
            info[gpu_id]['processes'].append({'pid':pid, 'usr':pInfo['username'], 'name': p.group("name"), 'cpu_mem':int(pInfo['memory_info'].vms/(1024*1024)), 'gpu_mem':int(p.group("mem"))})
            #print(r.group(0))
        #else:
            #print(str(line))
        #info += line
    info_process.communicate()
    for gpu_id, gpu_info in info.items():
        if len(gpu_info['processes'])==0 or (gpu_info['mem_used']/gpu_info['mem_max'])<active_mem_threshold:
            gpu_info['available']=True
        else:
            gpu_info['available']=False
    return info

def getAvailableGPU(active_mem_threshold=0.05, sorting="MEMORY"):
    available = []
    gpu_info = getGPUInfo(active_mem_threshold)
    for gpu_id, info in gpu_info.items():
        if info['available']:
            available.append(gpu_id)
    # sort by number of processes, then memory usage
    if sorting=="MEMORY":
        available.sort(key=lambda gpu_id: gpu_info[gpu_id]['mem_used']/gpu_info[gpu_id]['mem_max'])
    elif sorting=="PROCESSES_MEMORY":
        available.sort(key=lambda gpu_id: len(gpu_info[gpu_id]['processes']) + gpu_info[gpu_id]['mem_used']/gpu_info[gpu_id]['mem_max'])
    return available


def get_available_GPU_id(active_mem_threshold=0.05, default=None, sorting="MEMORY"):
    available = getAvailableGPU(active_mem_threshold, sorting=sorting)
    if available:
        return available[0]
    elif default is not None:
        print("No GPU availabe, using default", default)
        return int(default)
    else:
        raise RuntimeError("No GPU availabe")