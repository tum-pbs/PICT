# tests for optimization through PISO simulations
# test criterion: falling loss over n steps

from lib.util.logging import setup_run, get_logger, close_logging
from lib.util.profiling import SAMPLE

#import os, signal, itertools
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
#import domain_structs
import PISOtorch
#import PISOtorch_sim
import PISOtorch_simulation
#import numpy as np

assert torch.cuda.is_available()
cuda_device = torch.device("cuda")
cpu_device = torch.device("cpu")

import lib.data.shapes as shapes

LOG = get_logger("test_optim")

DTYPE = torch.float64


def MSE(a, b):
    return torch.mean((a - b)**2)
def SSE(a, b):
    return torch.sum((a - b)**2)

#_loss_fn = SSE

def test_optim_sim(domain_fn, time_step, it, opt_it, corr=2, substeps=1, static=False, 
                   opt_vel=False, opt_dens=False, loss_vel=False, loss_dens=False):
    
    target_domain, is_non_ortho, prep_fn = domain_fn()
    sim = PISOtorch_simulation.Simulation(target_domain, time_step=time_step, corrector_steps=corr, substeps=substeps,
        pressure_time_step_normalized=True, differentiable=False, log_images=False,
        non_orthogonal=is_non_ortho, advect_non_ortho_steps=2 if is_non_ortho else 1, pressure_non_ortho_steps=4 if is_non_ortho else 1,
        prep_fn = prep_fn)
    sim.run(iterations=it, static=static)

    domain, is_non_ortho, prep_fn = domain_fn()
    sim.differentiable = True
    sim.prep_fn = prep_fn

    if opt_vel:
        for block in domain.getBlocks():
            block.velocity.zero_()
    if opt_dens:
        for block in domain.getBlocks():
            block.passiveScalar.zero_()
    
    
    var_grids = []
    for block in domain.getBlocks():
        if opt_vel:
            var_grids.append(block.velocity)
        if opt_dens:
            var_grids.append(block.passiveScalar)
    for grid in var_grids: grid.requires_grad_(True)
    optimizer = torch.optim.SGD(var_grids, lr=0.1, momentum=0.0)

    def loss_fn(domain):
        loss = 0
        for block_idx in range(domain.getNumBlocks()):
            if loss_vel:
                loss = loss + SSE(domain.getBlock(block_idx).velocity, target_domain.getBlock(block_idx).velocity)
            if loss_dens:
                loss = loss + SSE(domain.getBlock(block_idx).passiveScalar, target_domain.getBlock(block_idx).passiveScalar)
        return loss

    last_loss = float("inf")
    loss_falling = True
    for optim_setp in range(opt_it):
        optimizer.zero_grad()
        adv_domain = domain.Copy("optDomain%04d"%optim_setp)
        adv_domain.PrepareSolve()
        sim.domain = adv_domain
        sim.run(iterations=it, static=static) #, name="step{:04d}".format(optim_setp))
        loss = loss_fn(adv_domain)

        LOG.info("step %d, loss: %s", optim_setp, loss)
        if loss.isnan().any():
            raise RuntimeError("loss is NaN.")
        elif(loss>=last_loss and not loss<1e-5):
            loss_falling = False
        last_loss = loss

        loss.backward()
        optimizer.step()
    
    if not loss_falling:
        raise RuntimeError("Optimization test failed: loss not falling or converged.")


def test_optim_density_static(domain_fn, time_step, it, opt_it):
    test_optim_sim(domain_fn, time_step, it, opt_it,
                   corr=1, substeps=1, static=True, opt_dens=True, loss_dens=True)

def test_optim_velocity_static(domain_fn, time_step, it, opt_it):
    test_optim_sim(domain_fn, time_step, it, opt_it,
                   corr=1, substeps=1, static=True, opt_vel=True, loss_dens=True)

def test_optim_velocity_loss_density(domain_fn, time_step, it, opt_it):
    test_optim_sim(domain_fn, time_step, it, opt_it,
                   corr=2, substeps=1, static=False, opt_vel=True, loss_dens=True)

def test_optim_velocity_loss_velocity(domain_fn, time_step, it, opt_it):
    test_optim_sim(domain_fn, time_step, it, opt_it,
                   corr=2, substeps=1, static=False, opt_vel=True, loss_vel=True)

def test_optim_density(domain_fn, time_step, it, opt_it):
    test_optim_sim(domain_fn, time_step, it, opt_it,
                   corr=2, substeps=1, static=False, opt_dens=True, loss_dens=True)

def test_optim_vel_div_free(domain_fn, time_step, it, opt_it):
    # optimize a velocity s.t. it matches a target after going through pressure projection
    
    
    #time_step = torch.ones([1], dtype=DTYPE, device=cpu_device)*time_step
    target_domain, is_non_ortho, prep_fn = domain_fn()
    
    sim = PISOtorch_simulation.Simulation(target_domain, time_step=time_step,
        pressure_time_step_normalized=True, differentiable=False, log_images=False,
        non_orthogonal=is_non_ortho, advect_non_ortho_steps=2 if is_non_ortho else 1, pressure_non_ortho_steps=4 if is_non_ortho else 1,
        prep_fn = prep_fn)
    #target_dir = os.path.join(run_dir, "target")
    #os.makedirs(target_dir)
    #save_domain_images(target_domain, target_dir, 0, layout=None, norm_p=True, max_mag=1, mode3D="", vel_exr=False)
    sim.make_divergence_free(iterations=it)
    #save_domain_images(target_domain, target_dir, 1, layout=None, norm_p=True, max_mag=1, mode3D="", vel_exr=False)
    
    domain, is_non_ortho, prep_fn = domain_fn()
    sim.differentiable = True
    sim.prep_fn = prep_fn
    
    var_velocities = [block.velocity for block in domain.getBlocks()]
    for vel in var_velocities:
        vel.zero_()
        vel.requires_grad_(True)
    optimizer = torch.optim.SGD(var_velocities, lr=0.1, momentum=0.0)
    #init_dir = os.path.join(run_dir, "initial")
    #os.makedirs(init_dir)
    #divfree_dir = os.path.join(run_dir, "div-free")
    #os.makedirs(divfree_dir)

    last_loss = float("inf")
    loss_falling = True
    for optim_setp in range(opt_it):
        optimizer.zero_grad()
        adv_domain = domain.Copy("optDomain%04d"%optim_setp)
        adv_domain.PrepareSolve()
        
        sim.domain = adv_domain
        sim.make_divergence_free(iterations=it)
        #save_domain_images(domain, init_dir, optim_setp, layout=None, norm_p=True, max_mag=1, mode3D="", vel_exr=False)
        #save_domain_images(adv_domain, divfree_dir, optim_setp, layout=None, norm_p=True, max_mag=1, mode3D="", vel_exr=False)
        loss = 0
        for blockIdx in range(domain.getNumBlocks()):
            loss = loss + SSE(adv_domain.getBlock(blockIdx).velocity, target_domain.getBlock(blockIdx).velocity)
        #loss = torch.sum(torch.as_tensor([SSE(ab.velocity, tb.velocity) for ab, tb in zip(adv_domain.getBlocks(), target_domain.getBlocks())]))
        #loss = MSE(adv_domain.scalarResult, torch.flatten(target_domain.getBlock(0).passiveScalar))
        LOG.info("loss: %s", loss)
        if(loss>=last_loss and not loss<1e-5):
            loss_falling = False
        last_loss = loss
        loss.backward()
        #LOG.info("grad: %s", domain.getBlock(0).velocity.grad.data)
        optimizer.step()
        #LOG.info("vel: %s", domain.getBlock(0).velocity)
    if not loss_falling:
        raise RuntimeError("Optimization test failed: loss not falling or converged.")



def get_tests():
    import tests.test_setups as test_setups
    dens_static_setups = dict(test_setups.setups_simple_combined)
    del_keys = [k for k in dens_static_setups if k.endswith("_closed")]
    for k in del_keys:
        del dens_static_setups[k] #TODO
    tests = {
        "DensityStatic": (test_optim_density_static, dens_static_setups),
        "VelocityStatic": (test_optim_velocity_static, test_setups.setups_simple_combined),
        "VelDivFree": (test_optim_vel_div_free, test_setups.setups_simple_combined),
        "VelocityFromDensity": (test_optim_velocity_loss_density, test_setups.setups_simple_combined),
        "VelocityFromVelocity": (test_optim_velocity_loss_velocity, test_setups.setups_simple_combined),
        "DensityFromDensity": (test_optim_density, test_setups.setups_simple_combined),
    }
    return tests

def get_test_params(domain_setup_fn):
    #PISOtorch_sim.set_backend(diff=True)
    return {"domain_fn":domain_setup_fn, "time_step": 0.5, "it": 5, "opt_it": 10}
