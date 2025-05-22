
from lib.util.logging import setup_run, get_logger, close_logging
from lib.util.profiling import SAMPLE

import os#, signal, itertools
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
#import numpy as np
import PISOtorch
#import PISOtorch_sim
import PISOtorch_simulation

assert torch.cuda.is_available()
cuda_device = torch.device("cuda")
cpu_device = torch.device("cpu")

import lib.data.shapes as shapes
from lib.util.output import *

LOG = get_logger("test_pisosim")

DTYPE = torch.float64


def has_transform(domain):
    return any(block.hasTransform() for block in domain.getBlocks())

def get_mass_tolerance(domain):
    return 0.02 if not has_transform(domain) else 0.04

def get_cell_size(block):
    if block.hasTransform():
        dims = block.getSpatialDims()
        p = [0,dims+1] + list(range(1, dims+1))
        return torch.permute(block.transform[...,-1:], p)
    else:
        return 1

def get_velocity_divergence(domain):
    return torch.max(torch.abs(PISOtorch.ComputeVelocityDivergence(domain)))

def get_scalar_result_mass(domain):
    return torch.sum(domain.scalarResult)
def get_scalar_mass(domain):
    mass = 0
    for block in domain.getBlocks():
        cell_size = get_cell_size(block)
        mass = mass + torch.sum(block.passiveScalar*cell_size)
    return mass

def get_velocity_result_energy(domain):
    return torch.sum(torch.abs(domain.velocityResult))
def get_velocity_energy(domain):
    e = 0
    for block in domain.getBlocks():
        e = e + torch.sum(torch.abs(block.velocity))
    return e

def get_velocity_result_magnitude(domain:PISOtorch.Domain):
    vel = torch.reshape(domain.velocityResult, (domain.getSpatialDims(), domain.getTotalSize()))
    # vel_max = torch.max(torch.abs(vel)).cpu().numpy().tolist()
    # if vel_max>0:
    return torch.sum(torch.linalg.vector_norm(vel, dim=0))
    # return 0
def get_velocity_magnitude(domain:PISOtorch.Domain):
    mag = 0
    for block in domain.getBlocks():
        vel = torch.reshape(block.velocity, (domain.getSpatialDims(), block.getStrides().w))
        if block.hasTransform():
            cell_size = torch.reshape(block.transform[...,-1:], (1, block.getStrides().w))
            mag = mag + torch.sum(torch.linalg.vector_norm(vel, dim=0)*cell_size)
        else:
            mag = mag + torch.sum(torch.linalg.vector_norm(vel, dim=0))
    return mag
    
def get_velocity_result_mass(domain):
    return torch.sum(domain.velocityResult)
def get_velocity_mass(domain):
    mass = 0
    for block in domain.getBlocks():
        cell_size = get_cell_size(block)
        mass = mass + torch.sum(block.velocity*cell_size)
    return torch.abs(mass)

def check_magnitude(mag, last_mag, tol, closed, it):
    
    mag_lower = last_mag*(1-tol)
    mag_upper = last_mag*(1+tol)
    if((mag<mag_lower or mag_upper<mag) and not (closed or it==0)):
        return false
    return true


def test_static_advection(domain_setup_fn, time_step, iterations, out_dir=None, differentiable=False):
    if out_dir is not None:
        os.makedirs(out_dir)
    #time_step = torch.ones([1], dtype=DTYPE, device=cpu_device)*time_step
    domain, is_non_ortho, prep_fn = domain_setup_fn()
    
    div = get_velocity_divergence(domain)
    last_m = get_scalar_mass(domain)
    m_tol = get_mass_tolerance(domain)
    m_ok = True
    domain_closed = domain.hasPrescribedBoundary()
    has_viscosity = domain.viscosity!=0
    LOG.info("initial: divergence %.02e, mass %.02e", div.cpu().numpy(), last_m.cpu().numpy())
    
    sim = PISOtorch_simulation.Simulation(domain, time_step=time_step,
        non_orthogonal=is_non_ortho, advect_non_ortho_steps=2 if is_non_ortho else 1, pressure_non_ortho_steps=4 if is_non_ortho else 1,
        prep_fn = prep_fn,
        differentiable=differentiable, log_images=False)
    
    if out_dir: save_velocity_image(domain, out_dir, "v", 0, max_mag=1, layout="H")
    if out_dir: save_scalar_image(domain, out_dir, "d", 0, layout="H")
    for it in range(iterations):
        sim.advect_static(iterations=1)
        m = get_scalar_mass(domain)
        LOG.info("step %d: mass %.02e", it, m.cpu().numpy())
        if not (domain_closed and has_viscosity) and (m<last_m*(1-m_tol) or last_m*(1+m_tol)<m):
            m_ok = False
        last_m = m
        if out_dir: save_scalar_image(domain, out_dir, "d", it+1, layout="H")
    if not m_ok:
        raise RuntimeError("Static advection test failed: scalar not conserved with tolerance %f."%m_tol)

def test_velocity_advection(domain_setup_fn, time_step, iterations, out_dir=None, differentiable=False):
    if out_dir is not None:
        os.makedirs(out_dir)
    #time_step = torch.ones([1], dtype=DTYPE, device=cpu_device)*time_step
    domain, is_non_ortho, prep_fn = domain_setup_fn()
    
    div = get_velocity_divergence(domain)
    last_vm = get_velocity_mass(domain)
    mag = get_velocity_magnitude(domain)
    vm_tol = get_mass_tolerance(domain)
    vm_ok = True
    domain_closed = domain.hasPrescribedBoundary()
    LOG.info("initial: divergence %.02e, vel mass %.02e (mag %.02e)", div.cpu().numpy(), last_vm.cpu().numpy(), mag.cpu().numpy())
    
    # corrector_steps=0: no pressure correction, only advect velocity
    sim = PISOtorch_simulation.Simulation(domain, time_step=time_step, corrector_steps=0, advect_passive_scalar=False,
        non_orthogonal=is_non_ortho, advect_non_ortho_steps=2 if is_non_ortho else 1, pressure_non_ortho_steps=4 if is_non_ortho else 1,
        prep_fn = prep_fn,
        differentiable=differentiable, log_images=False)
    
    if out_dir: save_velocity_image(domain, out_dir, "v", 0, max_mag=1, layout="H")
    for it in range(iterations):
        sim._PISO_split_step(iterations=1)
        div = get_velocity_divergence(domain)
        vm = get_velocity_mass(domain)
        mag = get_velocity_magnitude(domain)
        LOG.info("step %d: divergence %.02e, vel mass %.02e (mag %.02e)", it, div.cpu().numpy(), vm.cpu().numpy(), mag.cpu().numpy())
        if not domain_closed and (vm<last_vm*(1-vm_tol) or last_vm*(1+vm_tol)<vm):
            vm_ok = False
        last_vm = vm
        if out_dir: save_velocity_image(domain, out_dir, "v", it+1, max_mag=1, layout="H")
    if not vm_ok:
        raise RuntimeError("Velocity advection test failed: velocity not conserved with tolerance %f."%vm_tol)

def test_veloctiy_div_free(domain_setup_fn, time_step, iterations, out_dir=None, differentiable=False):
    if out_dir is not None:
        os.makedirs(out_dir)
    #time_step = torch.ones([1], dtype=DTYPE, device=cpu_device)*time_step
    domain, is_non_ortho, prep_fn = domain_setup_fn()
    
    last_div = get_velocity_divergence(domain)
    div_tol = 2e-5
    div_falling = True
    m = get_scalar_mass(domain)
    last_vm = get_velocity_mass(domain)
    mag = get_velocity_magnitude(domain)
    vm_tol = get_mass_tolerance(domain)
    vm_ok = True
    domain_closed = domain.hasPrescribedBoundary()
    LOG.info("initial: divergence %.02e, mass %.02e, vel mass %.02e (mag %.02e)", last_div.cpu().numpy(), m.cpu().numpy(), last_vm.cpu().numpy(), mag.cpu().numpy())
    
    sim = PISOtorch_simulation.Simulation(domain, time_step=time_step,
        non_orthogonal=is_non_ortho, advect_non_ortho_steps=2 if is_non_ortho else 1, pressure_non_ortho_steps=4 if is_non_ortho else 1,
        prep_fn = prep_fn,
        differentiable=differentiable, log_images=False)
    
    #LOG.info("divergence: \n%s", PISOtorch_sim.PISOtorch.ComputeVelocityDivergence(domain).cpu())
    if out_dir: save_velocity_image(domain, out_dir, "v", 0, max_mag=1, layout="H")
    for it in range(iterations):
        sim.make_divergence_free(iterations=1) # ignores time-step
        div = get_velocity_divergence(domain)
        m = get_scalar_mass(domain)
        vm = get_velocity_mass(domain)
        mag = get_velocity_magnitude(domain)
        LOG.info("step %d: divergence %.02e, mass %.02e, vel mass %.02e (mag %.02e)", it, div.cpu().numpy(), m.cpu().numpy(), vm.cpu().numpy(), mag.cpu().numpy())
        if(div>=last_div): # and not div<div_tol):
            div_falling = False
        if div<div_tol:
            div_falling = True
        last_div = div
        if not domain_closed and (vm<last_vm*(1-vm_tol) or last_vm*(1+vm_tol)<vm):
            vm_ok = False
        last_vm = vm
        if out_dir: save_velocity_image(domain, out_dir, "v", it+1, max_mag=1, layout="H")
    if not div_falling:
        raise RuntimeError("Velocity divergence test failed: divergence not falling or converged.")
    if not vm_ok:
        raise RuntimeError("Velocity divergence test failed: velocity not conserved with tolerance %f."%vm_tol)

def test_PISO_sim(domain_setup_fn, time_step, iterations, out_dir=None, differentiable=False):
    if out_dir is not None:
        os.makedirs(out_dir)
    #time_step = torch.ones([1], dtype=DTYPE, device=cpu_device)*time_step
    domain, is_non_ortho, prep_fn = domain_setup_fn()
    
    last_div = get_velocity_divergence(domain)
    div_tol = 2e-5
    div_falling = True
    m = get_scalar_mass(domain)
    last_vm = get_velocity_mass(domain)
    mag = get_velocity_magnitude(domain)
    vm_tol = get_mass_tolerance(domain)
    vm_ok = True
    domain_closed = domain.hasPrescribedBoundary()
    LOG.info("initial: divergence %.02e, mass %.02e, vel mass %.02e (mag %.02e)", last_div.cpu().numpy(), m.cpu().numpy(), last_vm.cpu().numpy(), mag.cpu().numpy())
    
    sim = PISOtorch_simulation.Simulation(domain, time_step=time_step,
        non_orthogonal=is_non_ortho, advect_non_ortho_steps=2 if is_non_ortho else 1, pressure_non_ortho_steps=4 if is_non_ortho else 1,
        prep_fn = prep_fn,
        #pressure_return_best_result=True,
        differentiable=differentiable, log_images=False)
    
    #LOG.info("divergence: \n%s", PISOtorch_sim.PISOtorch.ComputeVelocityDivergence(domain).cpu())
    if out_dir: save_velocity_image(domain, out_dir, "v", 0, max_mag=1, layout="H")
    for it in range(iterations):
        sim._PISO_split_step(iterations=1)
        div = get_velocity_divergence(domain)
        m = get_scalar_mass(domain)
        vm = get_velocity_mass(domain)
        mag = get_velocity_magnitude(domain)
        LOG.info("step %d: divergence %.02e, mass %.02e, vel mass %.02e (mag %.02e)", it, div.cpu().numpy(), m.cpu().numpy(), vm.cpu().numpy(), mag.cpu().numpy())
        if(div>=last_div): # and not div<div_tol):
            div_falling = False
        if div<div_tol:
            div_falling = True
        last_div = div
        if not domain_closed and (vm<last_vm*(1-vm_tol) or last_vm*(1+vm_tol)<vm):
            vm_ok = False
        last_vm = vm
        if out_dir: save_velocity_image(domain, out_dir, "v", it+1, max_mag=1, layout="H")
    if not div_falling:
        raise RuntimeError("Full PISO step test failed: divergence not falling or converged.")
    if not vm_ok:
        raise RuntimeError("Full PISO step test failed: velocity not conserved with tolerance %f."%vm_tol)

def get_tests():
    import tests.test_setups as test_setups
    setups = test_setups.setups_simple_combined
    #setups = test_setups.setups_nonortho
    tests = {
        #"AdvectStatic": (test_static_advection, setups),
        #"AdvectVelocity": (test_velocity_advection, setups),
        #"VelocityDivFree": (test_veloctiy_div_free, setups),
        "PISOsim": (test_PISO_sim, setups),
    }
    return tests

def get_test_params(domain_setup_fn):
    #PISOtorch_sim.set_backend(diff=False)
    return {"domain_setup_fn":domain_setup_fn, "time_step": 0.01, "iterations": 5, "differentiable": False}

def get_test_params_diff(domain_setup_fn):
    #PISOtorch_sim.set_backend(diff=True)
    return {"domain_setup_fn":domain_setup_fn, "time_step": 0.01, "iterations": 5, "differentiable": True}
