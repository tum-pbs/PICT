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

# 2D Karman vortex street behind a rectangular obstacle in a closed channel, with the grid refined towards the obstacle.
# Setup adapted from the code of "INC: An Indirect Neural Corrector for Auto-Regressive Hybrid PDE Solvers" (Wei et al., NeurIPS 2025).
# See vortex_street_sample.py for the basic 8-block setup.

def make_weights_ratio(res, ratio, refinement):
    # Weights (normalized vertex positions) for 'res' cells where the largest cell is about 'ratio' times the smallest.
    # In contrast to shapes.make_weights_exp the size ratio does not change with the resolution.
    # refinement: "START" or "END" for geometric growth away from that side, "BOTH" for a logistic (sigmoid) distribution.
    if ratio==1:
        return shapes.make_weights_linear(res)
    num_points = 1000 # resolution of the underlying continuous distribution. Changing it changes the grid.
    if refinement=="BOTH":
        k = 2*np.log(4*ratio)/num_points
        weights = 1/(1 + np.exp(-k*(np.arange(num_points+1) - num_points/2)))
        weights = (weights - weights.min())/(weights.max() - weights.min())
    else:
        sizes = ratio**(np.arange(num_points)/(num_points-1))
        if refinement=="END":
            sizes = sizes[::-1]
        weights = np.concatenate([[0], np.cumsum(sizes)/np.sum(sizes)])
    return np.interp(np.linspace(0, 1, res+1), np.linspace(0, 1, num_points+1), weights).tolist()

def make_refined_ortho_grid(res_x, res_y, corner_lower, corner_upper, wall_refinement=[], ratio=(1,1), dtype=torch.float32):
    # like shapes.make_wall_refined_ortho_grid, but with make_weights_ratio for the cell sizes
    def get_weights(res, lower, upper, ratio):
        if lower in wall_refinement:
            return make_weights_ratio(res, ratio, "BOTH" if upper in wall_refinement else "START")
        if upper in wall_refinement:
            return make_weights_ratio(res, ratio, "END")
        return None
    corners = [tuple(corner_lower), (corner_upper[0], corner_lower[1]), (corner_lower[0], corner_upper[1]), tuple(corner_upper)]
    # generate_grid_vertices_2D: x_weights distribute the vertices along y, y_weights along x
    return shapes.generate_grid_vertices_2D([res_y+1, res_x+1], corners, None,
        x_weights=get_weights(res_y, "-y", "+y", ratio[1]), y_weights=get_weights(res_x, "-x", "+x", ratio[0]), dtype=dtype)

def make_wake_perturbation(res_x, res_y, half_width, dtype=torch.float32):
    # Upward velocity in a horizontal band of 2*half_width cells through the center of the wake block,
    # following the first part of a sine wave. It breaks the symmetry of the flow to trigger the vortex shedding.
    num_samples = max(30, 2*half_width)
    amplitude = max(5, 10*half_width/30)
    profile = amplitude*torch.sin(torch.linspace(0, 2*np.pi, num_samples, dtype=dtype))[:2*half_width]
    velocity = torch.zeros([1, 2, res_y, res_x], dtype=dtype)
    velocity[0, 1, res_y//2-half_width:res_y//2+half_width, :] = profile.reshape([-1, 1])
    return velocity

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

def make8BlockKarmanSetup(x:int, y:int, x_in:int, y_in:int, x_pos:int, in_vel:float, in_var:float=0.5, viscosity=0.0, scale:float=None,
        refinement_ratio:float=3, obstacle_res_factor:float=1.5, wake_perturbation=True, dtype=torch.float32):
    # 8 blocks arranged and connected as a "ring", s.t. the missing center block creates an obstacle via closed bounds.
    # The upper and lower bounds of the channel are closed.
    # x: total length (stream-wise) of the channel in cells
    # y: total height (normal to stream) of the channel in cells
    # x_in: the x-size of the obstacle in cells
    # y_in: the y-size of the obstacle in cells
    # x_pos: distance in cells of the obstacle from the inflow (the obstacle is centered in y)
    # in_vel: the magnitude of the inflow velocity
    # in_var: the variance of the gaussian inflow profile
    # viscosity: viscosity of the flow
    # scale: None for untransformed grid. otherwise the cell size is multiplied by this, used for normalization with different resolutions
    # refinement_ratio: ratio of largest to smallest cell in each block, the cells are refined towards the obstacle
    # obstacle_res_factor: the blocks next to the obstacle use this factor more cells while the physical size of the setup stays the same
    # wake_perturbation: perturb the initial velocity behind the obstacle to trigger the vortex shedding
    dims = 2
    if obstacle_res_factor<1:
        raise ValueError("obstacle_res_factor must be at least 1.")
    if obstacle_res_factor>1:
        x = x + int((obstacle_res_factor-1)*x_in)
        y = y + int((obstacle_res_factor-1)*y_in)
        x_in = int(obstacle_res_factor*x_in)
        y_in = int(obstacle_res_factor*y_in)
    assert x>=(x_in-2)
    assert y>=(y_in-2)
    x_sizes = [x_pos, x_in, x - (x_pos+x_in)]
    y_sizes = [(y-y_in)//2, y_in, (y-y_in) - (y-y_in)//2]

    # make the grids
    obs_half_x = x_sizes[1]/(2*obstacle_res_factor)
    obs_half_y = y_sizes[1]/(2*obstacle_res_factor)
    x_poss = [-x_sizes[0]-obs_half_x, -obs_half_x, obs_half_x, obs_half_x+x_sizes[2]]
    y_poss = [-y_sizes[0]-obs_half_y, -obs_half_y, obs_half_y, obs_half_y+y_sizes[2]]
    if scale is not None:
        x_poss = [_*scale for _ in x_poss]
        y_poss = [_*scale for _ in y_poss]

    def make_grid(x_idx, y_idx, wall_refinement):
        return make_refined_ortho_grid(x_sizes[x_idx], y_sizes[y_idx], corner_lower=(x_poss[x_idx],y_poss[y_idx]), corner_upper=(x_poss[x_idx+1],y_poss[y_idx+1]),
            wall_refinement=wall_refinement, ratio=(refinement_ratio, refinement_ratio), dtype=dtype)
    # each block is refined towards the obstacle
    Gbl = make_grid(0, 0, ["+x","+y"])
    Gbm = make_grid(1, 0, ["-x","+x","+y"])
    Gbr = make_grid(2, 0, ["-x","+y"])

    Gml = make_grid(0, 1, ["+x","-y","+y"])
    #Gmm = obstacle
    Gmr = make_grid(2, 1, ["-x","-y","+y"])

    Gtl = make_grid(0, 2, ["+x","-y"])
    Gtm = make_grid(1, 2, ["-x","+x","-y"])
    Gtr = make_grid(2, 2, ["-x","-y"])

    grids = [Gtl, Gtm, Gtr, Gml, Gmr, Gbl, Gbm, Gbr]

    viscosity = torch.ones([1], dtype=dtype, device=cpu_device)*viscosity
    domain = PISOtorch.Domain(dims, viscosity, passiveScalarChannels=1, name="Domain8BlockKarman", device=cuda_device, dtype=dtype)

    # make the blocks
    Btl = domain.CreateBlock(vertexCoordinates=Gtl.to(cuda_device), name="BlockTopLeft")
    Btm = domain.CreateBlock(vertexCoordinates=Gtm.to(cuda_device), name="BlockTopMiddle")
    Btr = domain.CreateBlock(vertexCoordinates=Gtr.to(cuda_device), name="BlockTopRight")
    Bml = domain.CreateBlock(vertexCoordinates=Gml.to(cuda_device), name="BlockMiddleLeft")
    Bmr = domain.CreateBlock(vertexCoordinates=Gmr.to(cuda_device), name="BlockMiddleRight")
    Bbl = domain.CreateBlock(vertexCoordinates=Gbl.to(cuda_device), name="BlockBotLeft")
    Bbm = domain.CreateBlock(vertexCoordinates=Gbm.to(cuda_device), name="BlockBotMiddle")
    Bbr = domain.CreateBlock(vertexCoordinates=Gbr.to(cuda_device), name="BlockBotRight")

    layout = [[5,6,7],[3,-1,4],[0,1,2]] # used by output formatting

    if wake_perturbation:
        Bmr.setVelocity(make_wake_perturbation(x_sizes[2], y_sizes[1], (y_sizes[1]+1)//int(obstacle_res_factor*4), dtype=dtype).to(cuda_device).contiguous())

    # set boundaries
    # close obstacle
    Btm.CloseBoundary("-y")
    Bml.CloseBoundary("+x")
    Bmr.CloseBoundary("-x")
    Bbm.CloseBoundary("+y")

    # connect 8-block ring
    # the outer boundaries of the channel are closed as the default periodic boundaries are replaced by the connections
    Btl.ConnectBlock("+x", Btm, "-x", "-y")
    Btm.ConnectBlock("+x", Btr, "-x", "-y")

    Btr.ConnectBlock("-y", Bmr, "+y", "-x")
    Bmr.ConnectBlock("-y", Bbr, "+y", "-x")

    Bbl.ConnectBlock("+x", Bbm, "-x", "-y")
    Bbm.ConnectBlock("+x", Bbr, "-x", "-y")

    Btl.ConnectBlock("-y", Bml, "+y", "-x")
    Bml.ConnectBlock("-y", Bbl, "+y", "-x")

    # specify the inflow
    # gaussian profile over the cells of the channel height
    inflow = shapes.get_grid_normal_dist([y], [0], [in_var], dtype=dtype)
    inflow = torch.reshape(inflow, [1,1,y,1])*in_vel
    mean_in = torch.mean(inflow).item()
    max_in = torch.max(inflow).item()
    char_vel = torch.tensor([[max_in, 0]], device=cuda_device, dtype=dtype) # NC
    inflow = torch.cat([inflow, torch.zeros_like(inflow)], axis=1)
    inflow_blocks = [_.to(cuda_device).contiguous() for _ in torch.split(inflow, y_sizes, dim=-2)]

    Btl.getBoundary("-x").setVelocity(inflow_blocks[2])
    Bml.getBoundary("-x").setVelocity(inflow_blocks[1])
    Bbl.getBoundary("-x").setVelocity(inflow_blocks[0])

    # passive scalar inflow in front of the obstacle
    inflow_scalar_mid = shapes.get_grid_normal_dist([y_in], [0], [in_var], dtype=dtype)
    Bml.getBoundary("-x").setPassiveScalar(torch.reshape(inflow_scalar_mid, [1,1,y_in,1]).to(cuda_device).contiguous())

    # also set up outflow boundaries at the end of the channel.
    # these will need special treatment during the simulation
    out_vel = torch.tensor([mean_in, 0], device=cuda_device, dtype=dtype).reshape([1,dims,1,1])

    Btr.getBoundary("+x").setVelocity((torch.ones_like(inflow_blocks[2]) * out_vel).contiguous())
    Bmr.getBoundary("+x").setVelocity((torch.ones_like(inflow_blocks[1]) * out_vel).contiguous())
    Bbr.getBoundary("+x").setVelocity((torch.ones_like(inflow_blocks[0]) * out_vel).contiguous())

    # create varying passive scalar for outflow
    Btr.getBoundary("+x").CreatePassiveScalar(False)
    Bmr.getBoundary("+x").CreatePassiveScalar(False)
    Bbr.getBoundary("+x").CreatePassiveScalar(False)

    # make the boundaries consistent/divergence-free
    out_bounds = [Btr.getBoundary("+x"), Bmr.getBoundary("+x"), Bbr.getBoundary("+x")]
    PISOtorch_simulation.balance_boundary_fluxes(domain, out_bounds)

    # callback function to update the outflow during the simulation
    out_bound_indices = [domain.getBlocks().index(Btr), domain.getBlocks().index(Bmr), domain.getBlocks().index(Bbr)]
    def prep_fn(domain, time_step, **kwargs):
        out_bounds = [domain.getBlock(idx).getBoundary("+x") for idx in out_bound_indices]
        PISOtorch_simulation.update_advective_boundaries(domain, out_bounds, char_vel, time_step.cuda())

    domain.PrepareSolve()

    return domain, {"PRE": prep_fn}, layout, grids

RUN_ID = 0

def karman_vortex_street_sample(log_dir, name, iterations=100, time_step=0.1, res_scale=16, Re=600, obstacle_height=2.0, wake_perturbation=True,
        dp=False, STOP_FN=None):
    # res_scale: resolution of the outer blocks in cells per unit length. The blocks next to the obstacle use 1.5x that resolution.
    #   res_scale=16/downsample_factor of the INC code.
    # Re: Reynolds number based on the obstacle height and the inflow velocity scale
    # obstacle_height: physical height of the obstacle, the channel is 16x8 and the obstacle is 1.5 wide.
    global RUN_ID
    if STOP_FN():
        return
    LOG = get_logger("KarmanVortexStreet")
    dtype = torch.float64 if dp else torch.float32

    in_vel = 1.0
    viscosity = obstacle_height*in_vel/Re

    LOG.info("2D Karman vortex street #%d '%s' with obstacle height %.02e, resolution scale %d, Re %.02e (viscosity %.02e), %d iterations with time step %.02e",
             RUN_ID, name, obstacle_height, res_scale, Re, viscosity, iterations, time_step)
    log_dir = os.path.join(log_dir, "%04d_karman_vortex_street_%s"%(RUN_ID, name))

    domain, prep_fn, layout, grids = make8BlockKarmanSetup(x=16*res_scale, y=8*res_scale, x_in=int(1.5*res_scale), y_in=int(obstacle_height*res_scale), x_pos=3*res_scale,
        in_vel=in_vel, in_var=0.5, viscosity=viscosity, scale=1/res_scale, refinement_ratio=3, obstacle_res_factor=1.5, wake_perturbation=wake_perturbation, dtype=dtype)
    plot_grids(grids, color=["r","g","b","y","k","b","g","r"], path=log_dir, linewidth=0.5)

    # resample with twice the resolution of the outer blocks to resolve the refined cells around the obstacle
    resample_shape, resample_fill_steps = get_resampling_settings(grids, pixel_size=1/(2*res_scale))
    LOG.info("output resampling shape %s with up to %d fill steps", resample_shape, resample_fill_steps)

    max_vel = domain.getMaxVelocity(True, True).cpu().numpy()
    LOG.info("Domain max vel: %s", max_vel)

    sim = PISOtorch_simulation.Simulation(domain=domain, block_layout=layout, prep_fn=None,
            substeps="ADAPTIVE", time_step=time_step, corrector_steps=2, pressure_tol=1e-6,
            advect_non_ortho_steps=1, pressure_non_ortho_steps=1, pressure_return_best_result=True,
            velocity_corrector="FD", non_orthogonal=False,
            log_interval=1, norm_vel=True,
            output_resampling_shape=resample_shape, output_resampling_fill_max_steps=resample_fill_steps,
            log_dir=log_dir, save_domain_name="domain",
            stop_fn=STOP_FN)

    # initialize the domain with velocity that matches in- and outflow
    sim.make_divergence_free()

    # set the outflow update only after the initial projection, which uses a time step of 1
    sim.prep_fn = prep_fn
    sim.run(iterations)

    RUN_ID += 1


if __name__=="__main__":
    # create a new directory <time-step>_<name> in ./test_runs to use as base output directory
    run_dir = setup_run("./test_runs",
        name="karman-vortex-street-sample"
    )
    LOG = get_logger("Main")
    stop_handler = PISOtorch_simulation.StopHandler(LOG)
    stop_handler.register_signal()

    karman_vortex_street_sample(run_dir, name="Re600_r16", res_scale=16, Re=600, iterations=800, time_step=0.1, dp=False, STOP_FN=stop_handler)

    # lower resolution, faster but with stronger numerical oscillations at the obstacle corners
    #karman_vortex_street_sample(run_dir, name="Re600_r8", res_scale=8, Re=600, iterations=800, time_step=0.1, dp=False, STOP_FN=stop_handler)

    stop_handler.unregister_signal()
    close_logging()
