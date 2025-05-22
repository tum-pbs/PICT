# test domain memory usage and leaks
import os, gc
import torch
import PISOtorch
#import PISOtorch_simulation
#import PISOtorch_diff
from lib.util.logging import get_logger
_LOG = get_logger("MemoryTest")


assert torch.cuda.is_available()
cuda_device = torch.device("cuda")
cpu_device = torch.device("cpu")

DTYPE = torch.float64

def make_domain():

    from vortex_street_sample import make8BlockChannelFlowSetup

    res_scale = 8
    use_3D = True
    closed_bounds = True
    closed_z = False
    domain, prep_fn, layout = make8BlockChannelFlowSetup(x=16*res_scale, y=3*res_scale, z=3*res_scale if use_3D else None, x_in=1*res_scale, y_in=1*res_scale, x_pos=2*res_scale,
                                                         in_vel=4.0 if closed_bounds else 1, in_var=0.4, closed_y=closed_bounds, closed_z=closed_z, viscosity=1e-3, scale=1/res_scale, dtype=DTYPE)
    
    
    return domain

def fmt_mem(value_bytes):
    return "%.02fMiB"%(value_bytes/(1024*1024),)

def test_creation(domain_setup_fn, iterations):
    _LOG.info("Domain creation test for %04d iterations.", iterations)
    
    initial_mem = torch.cuda.memory_allocated()
    last_mem = initial_mem
    
    _LOG.info("Torch allocated memory initially: %s", fmt_mem(initial_mem))
    
    for it in range(iterations):
        domain = domain_setup_fn()
        curr_mem = torch.cuda.memory_allocated()
        _LOG.info("Torch allocated memory after creation %04d: %s", it, fmt_mem(curr_mem))
        
        #domain.Detach()
        del domain
        curr_mem = torch.cuda.memory_allocated()
        _LOG.info("Torch allocated memory after deletion %04d: %s", it, fmt_mem(curr_mem))
        
        gc.collect()
        curr_mem = torch.cuda.memory_allocated()
        _LOG.info("Torch allocated memory after gc %04d: %s", it, fmt_mem(curr_mem))
    
    curr_mem = torch.cuda.memory_allocated()
    _LOG.info("Torch allocated memory after %04d iterations: %s. Increase by %s", iterations, fmt_mem(curr_mem), fmt_mem(curr_mem-initial_mem))

def test_clone(domain_setup_fn, iterations):
    _LOG.info("Domain cloning test for %04d iterations.", iterations)
    
    initial_domain = domain_setup_fn()
    initial_mem = torch.cuda.memory_allocated()
    last_mem = initial_mem
    
    _LOG.info("Torch allocated memory initially: %s", fmt_mem(initial_mem))
    
    for it in range(iterations):
        domain = initial_domain.Clone()
        curr_mem = torch.cuda.memory_allocated()
        _LOG.info("Torch allocated memory after cloning %04d: %s", it, fmt_mem(curr_mem))
        
        #domain.Detach()
        del domain
        curr_mem = torch.cuda.memory_allocated()
        _LOG.info("Torch allocated memory after deletion %04d: %s", it, fmt_mem(curr_mem))
        
        gc.collect()
        curr_mem = torch.cuda.memory_allocated()
        _LOG.info("Torch allocated memory after gc %04d: %s", it, fmt_mem(curr_mem))
    
    curr_mem = torch.cuda.memory_allocated()
    _LOG.info("Torch allocated memory after %04d iterations: %s. Increase by %s", iterations, fmt_mem(curr_mem), fmt_mem(curr_mem-initial_mem))

def test_copy(domain_setup_fn, iterations):
    _LOG.info("Domain copy test for %04d iterations.", iterations)
    
    initial_domain = domain_setup_fn()
    initial_mem = torch.cuda.memory_allocated()
    last_mem = initial_mem
    
    _LOG.info("Torch allocated memory initially: %s", fmt_mem(initial_mem))
    
    for it in range(iterations):
        domain = initial_domain.Copy()
        #domain.PrepareSolve()
        curr_mem = torch.cuda.memory_allocated()
        _LOG.info("Torch allocated memory after copying %04d: %s", it, fmt_mem(curr_mem))
        
        #domain.Detach()
        del domain
        curr_mem = torch.cuda.memory_allocated()
        _LOG.info("Torch allocated memory after deletion %04d: %s", it, fmt_mem(curr_mem))
        
        gc.collect()
        curr_mem = torch.cuda.memory_allocated()
        _LOG.info("Torch allocated memory after gc %04d: %s", it, fmt_mem(curr_mem))
    
    curr_mem = torch.cuda.memory_allocated()
    _LOG.info("Torch allocated memory after %04d iterations: %s. Increase by %s", iterations, fmt_mem(curr_mem), fmt_mem(curr_mem-initial_mem))


def get_tests():
    #import tests.test_setups as test_setups
    setups = {"3D8BlockChannelFlow": make_domain}
    tests = {
        "Creation": (test_creation, setups),
        #"Clone": (test_clone, setups),
        #"Copy": (test_copy, setups),
    }
    return tests

def get_test_params(domain_setup_fn):
    #PISOtorch_sim.set_backend(diff=False)
    return {"domain_setup_fn":domain_setup_fn, "iterations": 10}
