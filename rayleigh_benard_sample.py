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
from lib.util.output import plot_grids

# PISOtorch is only implemented for GPU

assert torch.cuda.is_available()
cuda_device = torch.device("cuda")
cpu_device = torch.device("cpu")

# 2D Rayleigh-Bénard convection: a fluid layer between a hot lower and a cold upper wall, periodic in x.
# The temperature is a passive scalar with its own diffusivity, it drives the flow through a buoyancy force.
# The setup is non-dimensionalized with the layer height and the free-fall velocity, so the buoyancy force is just the temperature.
# Setup adapted from the code of "FluidGym" (Becktepe et al., ICML 2026), where it is the uncontrolled flow of the RBC environments.
# See buoyancy_sample.py for a simpler buoyancy setup.

def get_resampling_settings(grids, pixel_size):
    # The non-uniform grid is resampled to a uniform grid in physical space for output.
    # The output shape matches the aspect ratio of the domain s.t. the pixels are square.
    # Cells that are larger than the pixels leave holes in the resampled data, which are filled from neighboring pixels.
    # fill_max_steps is just large enough to close the holes of the largest cells.
    coords = torch.cat([grid.reshape(2, -1) for grid in grids], dim=-1)
    domain_size = (torch.max(coords, dim=-1)[0] - torch.min(coords, dim=-1)[0]).tolist()
    resample_shape = [int(np.ceil(size/pixel_size)) for size in domain_size] # xyz
    max_cell_size = max(torch.max(torch.abs(grid[:,:,1:,1:] - grid[:,:,:-1,:-1])).item() for grid in grids)
    fill_max_steps = int(np.ceil(0.5*max_cell_size/pixel_size))
    return resample_shape, fill_max_steps

def makeRayleighBenardSetup(res_x:int, res_y:int, length:float, height:float, viscosity:float, diffusivity:float, T_hot:float=1.0, T_cold:float=0.0,
        wall_refinement_base:float=1.02, T_noise:float=0.1, vel_noise:float=0.05, seed:int=None, dtype=torch.float32):
    # single block, periodic in x, walls with fixed temperature at the bottom (hot) and the top (cold)
    # res_x, res_y: number of cells
    # length, height: physical size of the domain, the walls are at y=-height/2 and y=height/2
    # viscosity: viscosity of the flow
    # diffusivity: diffusivity of the temperature
    # wall_refinement_base: the cells grow by this factor from the walls towards the center
    # T_noise: standard deviation of the random perturbation of the initial temperature, relative to T_hot-T_cold
    # vel_noise: standard deviation of the random initial velocity
    # seed: random seed for the initial perturbation
    dims = 2

    grid = shapes.make_wall_refined_ortho_grid(res_x, res_y, corner_lower=(0, -height/2), corner_upper=(length, height/2),
        wall_refinement=["-y", "+y"], base=wall_refinement_base, dtype=dtype)

    viscosity = torch.ones([1], dtype=dtype, device=cpu_device)*viscosity
    domain = PISOtorch.Domain(dims, viscosity, passiveScalarChannels=1, name="DomainRayleighBenard", device=cuda_device, dtype=dtype)
    # the passive scalar (temperature) has its own diffusivity, which is set as a global scalar viscosity
    domain.setScalarViscosity(torch.ones([1], dtype=dtype, device=cuda_device)*diffusivity)

    # initial state: linear temperature profile (pure conduction) and fluid at rest, both with random perturbations
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    y = shapes.coords_to_center_coords(grid)[:,1:2] # N1HW
    temperature = T_hot + (T_cold - T_hot)*(y/height + 0.5)
    temperature = temperature + T_noise*(T_hot - T_cold)*torch.randn(temperature.shape, generator=generator, dtype=dtype)
    temperature = torch.clamp(temperature, min(T_hot, T_cold), max(T_hot, T_cold))
    velocity = vel_noise*torch.randn([1, dims, res_y, res_x], generator=generator, dtype=dtype)

    block = domain.CreateBlock(velocity=velocity.to(cuda_device).contiguous(), passiveScalar=temperature.to(cuda_device).contiguous(),
        vertexCoordinates=grid.to(cuda_device).contiguous(), name="BlockRayleighBenard")

    # boundaries are periodic by default, close the walls and set their temperature
    block.CloseBoundary("-y")
    block.CloseBoundary("+y")
    block.getBoundary("-y").setPassiveScalar(torch.tensor([[T_hot]], dtype=dtype, device=cuda_device))
    block.getBoundary("+y").setPassiveScalar(torch.tensor([[T_cold]], dtype=dtype, device=cuda_device))

    # buoyancy: the temperature is used as upward force, set as velocity source before the velocity of every step is computed
    vel_src_x_pad = torch.zeros_like(block.passiveScalar)
    def buoyancy_fn(domain, **kwargs):
        block = domain.getBlock(0)
        block.setVelocitySource(torch.cat([vel_src_x_pad, block.passiveScalar], dim=1))
        domain.UpdateDomainData()

    domain.PrepareSolve()

    return domain, {"PRE_VELOCITY_SETUP": buoyancy_fn}, grid

def make_nusselt_fn(domain, Ra:float, Pr:float):
    # Nusselt number, the heat transport relative to pure conduction: Nu = 1 + sqrt(Ra*Pr)*<u_y*T>, with the volume average <.>
    cell_volume = domain.getBlock(0).transform[0,...,-1] # determinant of the cell transformations, HW
    def nusselt_fn(domain):
        block = domain.getBlock(0)
        vel_y_T = block.velocity[0,1]*block.passiveScalar[0,0]
        return (1 + np.sqrt(Ra*Pr)*torch.sum(vel_y_T*cell_volume)/torch.sum(cell_volume)).item()
    return nusselt_fn

RUN_ID = 0

def rayleigh_benard_sample(log_dir, name, iterations=100, time_step=1.0, res_x=96, Ra=8e5, Pr=0.7, aspect_ratio=np.pi, adaptive_CFL=0.5, seed=0,
        dp=False, STOP_FN=None):
    # res_x: number of cells along x. The cells are about twice as fine in y, which is refined towards the walls.
    # Ra: Rayleigh number
    # Pr: Prandtl number
    # aspect_ratio: length/height of the domain
    # adaptive_CFL: CFL number of the adaptive time stepping. FluidGym uses 0.5 for Ra>1e5.
    # seed: random seed for the perturbation of the initial temperature and velocity
    global RUN_ID
    if STOP_FN():
        return
    LOG = get_logger("RayleighBenard")
    dtype = torch.float64 if dp else torch.float32

    # free-fall units: the layer height and the free-fall velocity are 1
    height = 1.0
    length = aspect_ratio*height
    viscosity = np.sqrt(Pr/Ra)
    diffusivity = 1/np.sqrt(Ra*Pr)
    res_y = round(2*res_x/aspect_ratio)

    LOG.info("2D Rayleigh-Benard convection #%d '%s' with resolution %dx%d, Ra %.02e, Pr %.02e (viscosity %.02e, diffusivity %.02e), %d iterations with time step %.02e",
             RUN_ID, name, res_x, res_y, Ra, Pr, viscosity, diffusivity, iterations, time_step)
    log_dir = os.path.join(log_dir, "%04d_rayleigh_benard_%s"%(RUN_ID, name))

    domain, prep_fn, grid = makeRayleighBenardSetup(res_x, res_y, length, height, viscosity, diffusivity, seed=seed, dtype=dtype)
    plot_grids([grid], path=log_dir, linewidth=0.5)

    # resample with 2.5 pixels per cell in x
    resample_shape, resample_fill_steps = get_resampling_settings([grid], pixel_size=length/(2.5*res_x))
    LOG.info("output resampling shape %s with up to %d fill steps", resample_shape, resample_fill_steps)

    nusselt_fn = make_nusselt_fn(domain, Ra, Pr)
    def log_fn(domain, it, **kwargs):
        LOG.info("Nusselt number at t=%.02f: %.04f", it*time_step, nusselt_fn(domain))

    sim = PISOtorch_simulation.Simulation(domain=domain, prep_fn=prep_fn,
            substeps="ADAPTIVE", adaptive_CFL=adaptive_CFL, time_step=time_step, corrector_steps=2, pressure_tol=1e-5,
            advect_non_ortho_steps=1, pressure_non_ortho_steps=1, pressure_return_best_result=True,
            velocity_corrector="FD", non_orthogonal=False,
            log_interval=1, norm_vel=True, log_fn=log_fn,
            output_resampling_shape=resample_shape, output_resampling_fill_max_steps=resample_fill_steps,
            log_dir=log_dir, save_domain_name="domain",
            stop_fn=STOP_FN)

    # make the random initial velocity divergence-free
    sim.make_divergence_free()

    sim.run(iterations)

    RUN_ID += 1


if __name__=="__main__":
    # create a new directory <time-step>_<name> in ./test_runs to use as base output directory
    run_dir = setup_run("./test_runs",
        name="rayleigh-benard-sample"
    )
    LOG = get_logger("Main")
    stop_handler = PISOtorch_simulation.StopHandler(LOG)
    stop_handler.register_signal()

    # at Ra=8e5 the plumes oscillate periodically, the Nusselt number varies between ca. 4 and 12
    rayleigh_benard_sample(run_dir, name="Ra8e5_r96", res_x=96, Ra=8e5, Pr=0.7, adaptive_CFL=0.5, iterations=300, time_step=1.0, dp=False, STOP_FN=stop_handler)

    # lower Rayleigh numbers, as in the easy and medium RBC environments of FluidGym. The convection becomes steady after ca. 50 and 200 time units.
    #rayleigh_benard_sample(run_dir, name="Ra8e4_r96", res_x=96, Ra=8e4, Pr=0.7, iterations=300, time_step=1.0, dp=False, STOP_FN=stop_handler)
    #rayleigh_benard_sample(run_dir, name="Ra4e5_r96", res_x=96, Ra=4e5, Pr=0.7, adaptive_CFL=0.5, iterations=300, time_step=1.0, dp=False, STOP_FN=stop_handler)

    stop_handler.unregister_signal()
    close_logging()
