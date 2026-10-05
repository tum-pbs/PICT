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
import lib.data.TCF_tools as TCF_tools
import lib.util.domain_io as domain_io
from lib.util.output import plot_grids

# PISOtorch is only implemented for GPU

assert torch.cuda.is_available()
cuda_device = torch.device("cuda")
cpu_device = torch.device("cpu")

# 3D turbulent channel flow between two walls (y), periodic in stream-wise (x) and span-wise (z) direction.
# A forcing that balances the mean wall shear stress drives the flow, so the mass flow rate stays constant.
# The flow starts from a turbulent mean profile with curl noise and needs some time to become fully turbulent.
# Setup adapted from the code of "FluidGym" (Becktepe et al., ICML 2026), where it is the uncontrolled flow of the channel flow environments.
# See channel_flow_sim.py for the channel flow at Re_tau=550 of the PICT paper, including turbulence statistics.

def make_y_weights(N=1, ny_half=48):
    # vertex positions in [0,1], refined towards both walls with a growth factor of 1.2**(N/2) between neighboring cells.
    # the result has 2*(ny_half//N) cells.
    ny = 2*(ny_half//N)

    r = 1.2**(N/2) # cell size growth factor
    h0 = 0.5*(1-r)/(1-r**(ny/2)) # cell size at the walls
    h = 0 # current distance from the wall
    y = [0]*ny
    for i in range((ny-2)//2):
        h += h0*(r**i)
        y[i] = h
        y[ny-i-2] = 1-h
    y[ny//2-1] = 0.5
    y[ny-1] = 1.0

    return [0] + y

def make_channel_grid(res_x:int, res_y:int, res_z:int, length:float, depth:float, delta:float=1, refinement_strength:int=1, dtype=torch.float32):
    # res_x, res_z: number of cells in stream- and span-wise direction
    # res_y: number of cells between the walls, must be even
    # length, depth: size in stream- and span-wise direction
    # delta: channel half-height, the walls are at y=-delta and y=delta
    # refinement_strength: stronger refinement towards the walls, see make_y_weights
    assert (res_x%4)==0
    assert (res_y%2)==0

    y_weights = make_y_weights(N=refinement_strength, ny_half=(res_y//2)*refinement_strength)
    corners = [(-length/2, -delta), (length/2, -delta), (-length/2, delta), (length/2, delta)]
    grid = shapes.generate_grid_vertices_2D([len(y_weights), res_x+1], corners, None, x_weights=y_weights, dtype=dtype)
    grid = shapes.extrude_grid_z(grid, res_z, start_z=-depth/2, end_z=depth/2)

    return grid

def make_reichardt_profile(domain, u_wall:float):
    # mean velocity profile of turbulent channel flow (Reichardt's law of the wall), at the cell centers of block 0
    k = 0.41
    def reichardt_profile(y_wall):
        y11 = y_wall/11.0
        return (1/k)*torch.log(1 + k*y_wall) + 7.8*(1 - torch.exp(-y11) - y11*torch.exp(-y_wall/3))

    pos_y = domain.getBlock(0).getCellCoordinates()[0,1,0,:,0] #NCDHW -> H
    wall_distance = (1 - torch.abs(pos_y)) * u_wall / domain.viscosity.to(cuda_device) # in wall units, assumes delta=1

    return reichardt_profile(wall_distance) * u_wall

def makeChannelFlowSetup(res_x:int, res_y:int, res_z:int, length:float, depth:float, Re_wall:float, refinement_strength:int=1, init_with_noise:bool=True,
        dtype=torch.float32):
    # single block, closed at -y and +y (walls), periodic in x and z
    # Re_wall: target friction Reynolds number Re_tau. The center-line velocity is 1 and the half-height delta is 1.
    # init_with_noise: perturb the initial mean profile with curl noise, needs the compiled SimplexNoiseVariations: python extensions/noise/setup_noise.py install
    dims = 3
    delta = 1
    Re_center = TCF_tools.Re_wall_to_cl(Re_wall)
    u_wall = Re_wall/Re_center # friction velocity
    viscosity = delta/Re_center

    grid = make_channel_grid(res_x, res_y, res_z, length, depth, delta=delta, refinement_strength=refinement_strength, dtype=dtype)

    viscosity = torch.ones([1], dtype=dtype, device=cpu_device)*viscosity
    domain = PISOtorch.Domain(dims, viscosity, passiveScalarChannels=0, name="DomainChannel", device=cuda_device, dtype=dtype)

    # create block from the mesh on the domain (missing settings, fields and transformation metrics are created automatically)
    block = domain.CreateBlock(vertexCoordinates=grid.to(cuda_device).contiguous(), name="BlockChannel")
    # closing -y also closes +y as they are connected via the default periodic boundary
    block.CloseBoundary("-y")

    # initial velocity: turbulent mean profile in x
    vel_init = make_reichardt_profile(domain, u_wall)
    vel_init = vel_init.view(1,1,1,res_y,1).expand(-1,-1,res_z,-1,res_x)
    vel_init = torch.cat([vel_init] + [torch.zeros_like(vel_init)]*(dims-1), dim=1)
    if init_with_noise:
        import SimplexNoiseVariations
        curl_noise = SimplexNoiseVariations.GenerateSimplexNoiseVariation([res_x,res_y,res_z], cuda_device, [2/res_x,2/res_y,2/res_z], [0]*3,
            SimplexNoiseVariations.NoiseVariation.CURL)
        curl_noise *= 0.5*vel_init/torch.max(torch.linalg.vector_norm(curl_noise, dim=1))
        vel_init = vel_init + curl_noise
    block.setVelocity(vel_init.to(dtype).contiguous())

    domain.PrepareSolve()

    return domain, grid

def make_dynamic_forcing_fn(domain, delta:float=1):
    # A stream-wise velocity source that balances the mean shear stress at the walls, set before every step.
    # Returns the callback and a function to compute the mean wall shear stress.
    dims = domain.getSpatialDims()
    pos_y = torch.mean(domain.getBlock(0).getCellCoordinates()[0,1], dim=(0,2))
    wall_distance = (delta + pos_y[0].item(), delta - pos_y[-1].item()) # distance of the first cell centers from the walls

    def get_wall_shear_stress(domain):
        block = domain.getBlock(0)
        mean_vel_u = torch.mean(block.velocity[0,0], dim=(0,2)) # mean over x and z
        viscosity = domain.viscosity.to(domain.getDevice())
        return (viscosity*mean_vel_u[0]/wall_distance[0] + viscosity*mean_vel_u[-1]/wall_distance[1])*0.5

    def dynamic_forcing_fn(domain, **kwargs):
        G = torch.cat([get_wall_shear_stress(domain).reshape([1,1]), torch.zeros([1,dims-1], dtype=domain.getDtype(), device=domain.getDevice())], dim=1) # NC
        domain.getBlock(0).setVelocitySource(G)

    return dynamic_forcing_fn, get_wall_shear_stress

RUN_ID = 0

def turbulent_channel_flow_sample(log_dir, name, iterations=100, time_step=1.0, res_x=64, res_y=64, res_z=64, length=np.pi, depth=np.pi/2, Re_wall=180,
        adaptive_CFL=0.1, init_with_noise=True, load_domain_path=None, dp=False, STOP_FN=None):
    # res_x, res_y, res_z: number of cells in stream-wise, wall-normal and span-wise direction
    # length, depth: stream- and span-wise size of the channel, the channel height is 2
    # Re_wall: target friction Reynolds number Re_tau
    # adaptive_CFL: CFL number of the adaptive time stepping. Higher values can result in a laminar flow.
    # load_domain_path: continue from a saved domain (e.g. '<run_dir>/domain' of a previous run) instead of the initial velocity profile.
    #   Use the same Re_wall. The flow from the initial profile becomes turbulent after ca. 8 ETT in the small channel at Re_tau=180,
    #   so it can be useful to warm up the simulation once.
    global RUN_ID
    if STOP_FN():
        return
    LOG = get_logger("ChannelFlow")
    dtype = torch.float64 if dp else torch.float32

    Re_center = TCF_tools.Re_wall_to_cl(Re_wall)
    u_wall = Re_wall/Re_center
    viscosity = 1/Re_center
    ETT = TCF_tools.t_to_ETT(iterations*time_step, u_wall) # simulated time in eddy turnover times
    log_dir = os.path.join(log_dir, "%04d_channel_flow_%s"%(RUN_ID, name))

    if load_domain_path:
        LOG.info("3D turbulent channel flow #%d '%s' loaded from '%s', Re_tau %.01f, %d iterations with time step %.02e (%.01f ETT)",
                 RUN_ID, name, load_domain_path, Re_wall, iterations, time_step, ETT)
        domain = domain_io.load_domain(load_domain_path, dtype=dtype, device=cuda_device)
        domain.PrepareSolve()
        grid = domain.getBlock(0).vertexCoordinates
        res_z, res_y, res_x = [size - 1 for size in grid.shape[-3:]]
        length = (grid[0,0,0,0,-1] - grid[0,0,0,0,0]).item()
        depth = (grid[0,2,-1,0,0] - grid[0,2,0,0,0]).item()
    else:
        LOG.info("3D turbulent channel flow #%d '%s' with resolution %dx%dx%d, size %.02fx2x%.02f, Re_tau %.01f (Re_center %.01f, viscosity %.02e), %d iterations with time step %.02e (%.01f ETT)",
                 RUN_ID, name, res_x, res_y, res_z, length, depth, Re_wall, Re_center, viscosity, iterations, time_step, ETT)
        domain, grid = makeChannelFlowSetup(res_x, res_y, res_z, length, depth, Re_wall, init_with_noise=init_with_noise, dtype=dtype)
    plot_grids([grid[:,:2,0].cpu()], path=log_dir, linewidth=0.5)

    # the forcing depends on the velocity and is updated before every step
    forcing_fn, get_wall_shear_stress = make_dynamic_forcing_fn(domain)
    prep_fn = {"PRE": forcing_fn}

    # resample to cubic cells with half the stream-wise cell size
    pixel_size = length/(2*res_x)
    resample_shape = [int(np.round(size/pixel_size)) for size in [length, 2, depth]]
    max_cell_size_y = torch.max(grid[0,1,0,1:,0] - grid[0,1,0,:-1,0]).item()
    resample_fill_steps = int(np.ceil(0.5*max(max_cell_size_y, length/res_x, depth/res_z)/pixel_size))
    LOG.info("output resampling shape %s with up to %d fill steps", resample_shape, resample_fill_steps)

    def log_fn(domain, it, **kwargs):
        # the friction Reynolds number from the current wall shear stress shows if the flow is turbulent
        LOG.info("Re_tau at %.02f ETT: %.01f", TCF_tools.t_to_ETT(it*time_step, u_wall), np.sqrt(get_wall_shear_stress(domain).item())/viscosity)

    sim = PISOtorch_simulation.Simulation(domain=domain, prep_fn=prep_fn,
            substeps="ADAPTIVE", adaptive_CFL=adaptive_CFL, time_step=time_step, corrector_steps=2,
            advection_use_BiCG=True, advection_tol=1e-6, pressure_tol=1e-6,
            advect_non_ortho_steps=1, pressure_non_ortho_steps=1, pressure_return_best_result=True,
            velocity_corrector="FD", non_orthogonal=True,
            log_interval=1, norm_vel=True, log_fn=log_fn,
            output_resampling_shape=resample_shape, output_resampling_fill_max_steps=resample_fill_steps,
            log_dir=log_dir, save_domain_name="domain",
            stop_fn=STOP_FN)

    sim.preconditionBiCG = False
    sim.BiCG_precondition_fallback = True

    if not load_domain_path:
        # make the initial velocity divergence-free
        PISOtorch.CopyVelocityResultFromBlocks(domain)
        sim.make_divergence_free()

    sim.run(iterations)

    RUN_ID += 1


if __name__=="__main__":
    # create a new directory <time-step>_<name> in ./test_runs to use as base output directory
    run_dir = setup_run("./test_runs",
        name="turbulent-channel-flow-sample"
    )
    LOG = get_logger("Main")
    stop_handler = PISOtorch_simulation.StopHandler(LOG)
    stop_handler.register_signal()

    # small channel (pi x 2 x pi/2) at Re_tau=180, 400 iterations are ca. 17 ETT
    turbulent_channel_flow_sample(run_dir, name="Re180_small", res_x=64, res_y=64, res_z=64, length=np.pi, depth=np.pi/2, Re_wall=180,
        iterations=400, time_step=1.0, dp=False, STOP_FN=stop_handler)

    # continue a previous simulation, e.g. after a warm-up
    #turbulent_channel_flow_sample(run_dir, name="Re180_small_continued", Re_wall=180, load_domain_path="./test_runs/<run_dir>/0000_channel_flow_Re180_small/domain",
    #    iterations=200, time_step=1.0, dp=False, STOP_FN=stop_handler)

    # large channel (2pi x 2 x pi) at Re_tau=180, as in the large channel flow environments of FluidGym
    #turbulent_channel_flow_sample(run_dir, name="Re180_large", res_x=128, res_y=64, res_z=128, length=2*np.pi, depth=np.pi, Re_wall=180,
    #    iterations=400, time_step=1.0, dp=False, STOP_FN=stop_handler)

    stop_handler.unregister_signal()
    close_logging()
