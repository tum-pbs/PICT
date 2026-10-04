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

# 2D decaying turbulence in a periodic box [0, 2pi]^2, starting from a random velocity field.

def make_filtered_velocity_field(res:int, domain_size:float, max_velocity:float, peak_wavenumber:float, seed:int=None, dtype=torch.float32):
    # Random divergence-free velocity field with most energy around 'peak_wavenumber', like JAX-CFD's filtered_velocity_field:
    # white noise is filtered with a log-normal spectral density, the divergent part is removed, and it is scaled to 'max_velocity'.
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    noise = torch.randn([2, res, res], generator=generator, dtype=torch.float64) # CHW
    k_axis = 2*np.pi*torch.fft.fftfreq(res, d=domain_size/res, dtype=torch.float64) # angular wavenumbers
    k_y, k_x = torch.meshgrid(k_axis, k_axis, indexing="ij")
    k_mag = torch.sqrt(k_x**2 + k_y**2)
    k_mag[0,0] = 1 # avoid division by 0, the mean is removed below
    # log-normal spectral density with mode peak_wavenumber and log-variance 0.25, divided by k**(dims-1)
    log_k = torch.log(k_mag)
    spectral_filter = torch.exp(-(np.log(peak_wavenumber) + 0.25 - log_k)**2/(2*0.25) - log_k) / k_mag
    spectral_filter[0,0] = 0
    vel_hat = torch.fft.fft2(noise) * spectral_filter
    # projection to divergence-free
    k_dot_vel = (k_x*vel_hat[0] + k_y*vel_hat[1]) / k_mag**2
    vel_hat = torch.stack([vel_hat[0] - k_x*k_dot_vel, vel_hat[1] - k_y*k_dot_vel])
    velocity = torch.fft.ifft2(vel_hat).real
    velocity = velocity * (max_velocity/torch.max(torch.linalg.vector_norm(velocity, dim=0)))
    return velocity.reshape([1, 2, res, res]).to(dtype) # NCHW

def makePeriodicTurbulenceDomain(res:int, domain_size:float, viscosity:float, velocity:torch.Tensor, dtype=torch.float32):
    # single block with uniform grid, all boundaries are periodic
    # res: number of cells per axis
    # domain_size: physical size of the square domain
    # velocity: initial velocity, NCHW with spatial shape (res, res)
    dims = 2

    grid = shapes.make_wall_refined_ortho_grid(res, res, corner_upper=(domain_size, domain_size), wall_refinement=[], dtype=dtype)

    viscosity = torch.ones([1], dtype=dtype, device=cpu_device)*viscosity
    domain = PISOtorch.Domain(dims, viscosity, passiveScalarChannels=0, name="DomainTurbulence", device=cuda_device, dtype=dtype)

    # boundaries are periodic by default
    domain.CreateBlock(velocity=velocity.to(cuda_device).contiguous(), vertexCoordinates=grid.to(cuda_device).contiguous(), name="BlockTurbulence")

    domain.PrepareSolve()

    return domain

RUN_ID = 0

def decaying_turbulence_sample(log_dir, name, iterations=100, time_step=0.1, res=256, viscosity=1e-3, max_velocity=4.2, peak_wavenumber=4, seed=0,
        dp=False, STOP_FN=None):
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

    LOG.info("2D decaying turbulence #%d '%s' with resolution %d, viscosity %.02e, initial max velocity %.02e at peak wavenumber %.02e (seed %s), %d iterations with time step %.02e",
             RUN_ID, name, res, viscosity, max_velocity, peak_wavenumber, seed, iterations, time_step)
    log_dir = os.path.join(log_dir, "%04d_decaying_turbulence_%s"%(RUN_ID, name))

    velocity = make_filtered_velocity_field(res, domain_size, max_velocity, peak_wavenumber, seed=seed, dtype=dtype)
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

    stop_handler.unregister_signal()
    close_logging()
