import os
from lib.util.logging import setup_run, get_logger, close_logging

import numpy as np


if __name__=="__main__":
    from lib.util.GPU_info import get_available_GPU_id
    cudaID = None #str, to set a fixed GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = cudaID or str(get_available_GPU_id(active_mem_threshold=0.8, default=None))


import torch
import PISOtorch # domain data structures and core PISO functions. check /extensions/PISOtorch.cpp to see what is available
import PISOtorch_simulation # uses core PISO functions to make full simulation steps
import lib.data.shapes as shapes

# PISOtorch is only implemented for GPU

assert torch.cuda.is_available()
cuda_device = torch.device("cuda")
cpu_device = torch.device("cpu")

# Decaying turbulence in a periodic box [0, 2pi]^2 (or [0, 2pi]^3 in 3D), starting from a random velocity field.

def make_filtered_velocity_field(res:int, domain_size:float, max_velocity:float, peak_wavenumber:float, dims:int=2, seed:int=None, dtype=torch.float32):
    # Random divergence-free velocity field with most energy around 'peak_wavenumber', like JAX-CFD's filtered_velocity_field:
    # white noise is filtered with a log-normal spectral density, the divergent part is removed, and it is scaled to 'max_velocity'.
    # dims: 2 or 3 spatial dimensions, the field has res cells per axis
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    noise = torch.randn([dims] + [res]*dims, generator=generator, dtype=torch.float64) # C[D]HW
    k_axis = 2*np.pi*torch.fft.fftfreq(res, d=domain_size/res, dtype=torch.float64) # angular wavenumbers
    k = torch.meshgrid(*[k_axis]*dims, indexing="ij")[::-1] # k_x, k_y[, k_z]
    k_mag = torch.sqrt(sum(k_i**2 for k_i in k))
    k_mag[(0,)*dims] = 1 # avoid division by 0, the mean is removed below
    # log-normal spectral density with mode peak_wavenumber and log-variance 0.25, divided by k**(dims-1)
    log_k = torch.log(k_mag)
    spectral_filter = torch.exp(-(np.log(peak_wavenumber) + 0.25 - log_k)**2/(2*0.25) - log_k) / k_mag**(dims-1)
    spectral_filter[(0,)*dims] = 0
    fft_dims = tuple(range(1, dims+1))
    vel_hat = torch.fft.fftn(noise, dim=fft_dims) * spectral_filter
    # projection to divergence-free
    k_dot_vel = sum(k_i*vel_hat_i for k_i, vel_hat_i in zip(k, vel_hat)) / k_mag**2
    vel_hat = torch.stack([vel_hat_i - k_i*k_dot_vel for k_i, vel_hat_i in zip(k, vel_hat)])
    velocity = torch.fft.ifftn(vel_hat, dim=fft_dims).real
    velocity = velocity * (max_velocity/torch.max(torch.linalg.vector_norm(velocity, dim=0)))
    return velocity.reshape([1, dims] + [res]*dims).to(dtype) # NC[D]HW

def makePeriodicTurbulenceDomain(res:int, domain_size:float, viscosity:float, velocity:torch.Tensor, dtype=torch.float32):
    # single block with uniform grid, all boundaries are periodic
    # res: number of cells per axis
    # domain_size: physical size of the square (cubic in 3D) domain
    # velocity: initial velocity, NC[D]HW with spatial shape (res, res[, res]), its shape sets the dimensionality
    dims = velocity.dim() - 2

    grid = shapes.make_wall_refined_ortho_grid(res, res, corner_upper=(domain_size, domain_size), wall_refinement=[], dtype=dtype)
    if dims==3:
        grid = shapes.extrude_grid_z(grid, res, start_z=0, end_z=domain_size)

    viscosity = torch.ones([1], dtype=dtype, device=cpu_device)*viscosity
    domain = PISOtorch.Domain(dims, viscosity, passiveScalarChannels=0, name="DomainTurbulence", device=cuda_device, dtype=dtype)

    # boundaries are periodic by default
    domain.CreateBlock(velocity=velocity.to(cuda_device).contiguous(), vertexCoordinates=grid.to(cuda_device).contiguous(), name="BlockTurbulence")

    domain.PrepareSolve()

    return domain

RUN_ID = 0

def decaying_turbulence_sample(log_dir, name, iterations=100, time_step=0.1, dims=2, res=256, viscosity=1e-3, max_velocity=4.2, peak_wavenumber=4, seed=0,
        dp=False, STOP_FN=None):
    # dims: 2 or 3 spatial dimensions
    # res: number of cells per axis
    # max_velocity: maximum velocity magnitude of the initial velocity field
    # peak_wavenumber: wavenumber with the most energy in the initial velocity field
    # seed: random seed for the initial velocity field
    global RUN_ID
    if STOP_FN():
        return
    LOG = get_logger("DecayingTurbulence")
    dtype = torch.float64 if dp else torch.float32

    domain_size = 2*np.pi

    LOG.info("%dD decaying turbulence #%d '%s' with resolution %d, viscosity %.02e, initial max velocity %.02e at peak wavenumber %.02e (seed %s), %d iterations with time step %.02e",
             dims, RUN_ID, name, res, viscosity, max_velocity, peak_wavenumber, seed, iterations, time_step)
    log_dir = os.path.join(log_dir, "%04d_decaying_turbulence_%s"%(RUN_ID, name))

    velocity = make_filtered_velocity_field(res, domain_size, max_velocity, peak_wavenumber, dims=dims, seed=seed, dtype=dtype)
    domain = makePeriodicTurbulenceDomain(res, domain_size, viscosity, velocity, dtype=dtype)

    sim = PISOtorch_simulation.Simulation(domain=domain, prep_fn=None,
            substeps="ADAPTIVE", adaptive_CFL=0.5, time_step=time_step, corrector_steps=2, pressure_tol=1e-6,
            advect_non_ortho_steps=1, pressure_non_ortho_steps=1, pressure_return_best_result=True,
            velocity_corrector="FD", non_orthogonal=False,
            log_interval=1, norm_vel=True,
            log_dir=log_dir, save_domain_name="domain",
            stop_fn=STOP_FN)

    # make the initial velocity divergence-free w.r.t. the discretization of the solver
    sim.make_divergence_free()

    LOG.info("Initial max velocity magnitude: %.03e", domain.getMaxVelocityMagnitude(True, False).item())

    sim.run(iterations)

    RUN_ID += 1


if __name__=="__main__":
    # create a new directory <time-step>_<name> in ./test_runs to use as base output directory
    run_dir = setup_run("./test_runs",
        name="decaying-turbulence-sample"
    )
    LOG = get_logger("Main")
    stop_handler = PISOtorch_simulation.StopHandler(LOG)
    stop_handler.register_signal()

    decaying_turbulence_sample(run_dir, name="r256_visc1e-3", res=256, viscosity=1e-3, iterations=200, time_step=0.1, dp=False, STOP_FN=stop_handler)

    # higher resolution
    #decaying_turbulence_sample(run_dir, name="r512_visc1e-3", res=512, viscosity=1e-3, iterations=200, time_step=0.1, dp=False, STOP_FN=stop_handler)

    # 3D with 128^3 cells. The viscosity is higher than in 2D to resolve the smallest eddies of the energy cascade,
    # the kinetic energy decays to ca. 3% of its initial value in 10 and to 1% in 20 time units
    #decaying_turbulence_sample(run_dir, name="3D_r128_visc3e-3", dims=3, res=128, viscosity=3e-3, iterations=200, time_step=0.1, dp=False, STOP_FN=stop_handler)

    stop_handler.unregister_signal()
    close_logging()
