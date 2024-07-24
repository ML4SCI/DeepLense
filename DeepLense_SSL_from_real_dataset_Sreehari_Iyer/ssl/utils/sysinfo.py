import psutil
import GPUtil
import os
import platform
from datetime import datetime

def get_processor_name():
    if platform.system() == "Windows":
        return platform.uname().processor
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo | grep 'model name' | uniq"
        return os.popen(command).read().strip().split(": ")[1]
    return "Unknown Processor"
    
def get_system_info():
    # CPU information
    cpu_info = {
        "physical_cores": psutil.cpu_count(logical=False),
        "total_cores": psutil.cpu_count(logical=True),
        "max_frequency": psutil.cpu_freq().max,
        "min_frequency": psutil.cpu_freq().min,
    }

    # Memory information
    svmem = psutil.virtual_memory()
    memory_info = {
        "total_memory": svmem.total,
        "available_memory": svmem.available,
    }

    # GPU information
    gpus = GPUtil.getGPUs()
    gpu_info = []
    for gpu in gpus:
        gpu_info.append({
            "id": gpu.id,
            "name": gpu.name,
            "load": gpu.load * 100,
            "free_memory": gpu.memoryFree,
            "used_memory": gpu.memoryUsed,
            "total_memory": gpu.memoryTotal,
        })

    # System information
    system_info = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "hostname": platform.node(),
        "ip_address": psutil.net_if_addrs()['lo'][0].address,
        "processor": get_processor_name(),
        "boot_time": datetime.fromtimestamp(psutil.boot_time()).strftime("%Y-%m-%d %H:%M:%S")
    }

    # Combine all the information into a single dictionary
    info = {
        "cpu_info": cpu_info,
        "memory_info": memory_info,
        "gpu_info": gpu_info,
        "system_info": system_info
    }

    return info