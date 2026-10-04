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

# 2D backward-facing step (BFS) with refined grid and outflow buffer blocks.
# Setup adapted from the code of "INC: An Indirect Neural Corrector for Auto-Regressive Hybrid PDE Solvers" (Wei et al., NeurIPS 2025).

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

def growth_factor_from_sizes(res, x, x0, eps=1e-5):
    from scipy.optimize import least_squares
    # res: number of cells
    # x: total size
    # x0: base cell size
    # finds exponent b s.t. x = sum[i=0, res-1](x0 * b**i) = x0*(1-b**res)/(1-b)
    if not (res>0 and x>x0 and x0>0 and eps>0): raise ValueError("invalid input")
    if np.isclose(res, x/x0):
        return 1

    def func(b):
        return x0*(1-np.power(b,res))/(1-b) - x

    if res<x/x0:
        b0, lb, ub = 1+eps, 1+eps, 2
    else:
        b0, lb, ub = 1-eps, eps, 1-eps

    return least_squares(func, [b0], bounds=([lb], [ub])).x[0]

def make_BFS_grids(h:float, s:float, l:float, L:float, buffer_length:float, res_l:int, res_L:int, res_y:int, res_buffer:int, ratio=(20,2), dtype=torch.float32):
    """
    The step corner is at x=l, the upper channel wall at y=H=h+s, the lower wall at y=0.
    h: height of the inflow channel
    s: step height
    l: length of the inflow channel before the step
    L: length of the channel after the step
    buffer_length: length of the outflow buffer after the channel
    res_*: number of cells along the respective length. res_y is the cell count of the full height H.
    ratio: largest/smallest cell size. x: refined towards the step, y: refined towards the walls of the upper and lower blocks.

    x-axis goes left to right
    y-axis goes bottom to top
    """
    H = h + s
    res_h = int(res_y*h/H) # cells above the step
    xs = [0, l, l+L, l+L+buffer_length]
    ys = [0, s, H]

    # channel after the step, split at the height of the step
    Gtm = make_refined_ortho_grid(res_L, res_h, (xs[1],ys[1]), (xs[2],ys[2]), wall_refinement=["-x","-y","+y"], ratio=ratio, dtype=dtype)
    Gbm = make_refined_ortho_grid(res_L, res_y-res_h, (xs[1],ys[0]), (xs[2],ys[1]), wall_refinement=["-x","-y","+y"], ratio=ratio, dtype=dtype)

    # inflow channel and outflow buffer grow exponentially from the cell size of the connected block for a smooth transition
    weights_y_top = make_weights_ratio(res_h, ratio[1], "BOTH")
    weights_y_bot = make_weights_ratio(res_y-res_h, ratio[1], "BOTH")
    cell_x_first = (Gtm[0,0,0,1] - Gtm[0,0,0,0]).item()
    cell_x_last = (Gtm[0,0,0,-1] - Gtm[0,0,0,-2]).item()
    weights_x_in = shapes.make_weights_exp(res_l, base=growth_factor_from_sizes(res_l, l, cell_x_first), refinement="END")
    weights_x_out = shapes.make_weights_exp(res_buffer, base=growth_factor_from_sizes(res_buffer, buffer_length, cell_x_last), refinement="START")

    def make_grid(cells_x, cells_y, x0, x1, y0, y1, weights_x, weights_y):
        return shapes.generate_grid_vertices_2D((cells_y+1, cells_x+1), corner_vertices=[(x0,y0), (x1,y0), (x0,y1), (x1,y1)],
            x_weights=weights_y, y_weights=weights_x, dtype=dtype)
    Gtl = make_grid(res_l, res_h, xs[0], xs[1], ys[1], ys[2], weights_x_in, weights_y_top)
    Gtr = make_grid(res_buffer, res_h, xs[2], xs[3], ys[1], ys[2], weights_x_out, weights_y_top)
    Gbr = make_grid(res_buffer, res_y-res_h, xs[2], xs[3], ys[0], ys[1], weights_x_out, weights_y_bot)

    return [Gtl, Gtm, Gbm, Gtr, Gbr]

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

def make5BlockBackwardFacingStepSetup(grids, h:float, s:float, in_vel:float=1, in_var:float=0.4, viscosity:float=0.0, buffer_viscosity_ratio:float=1,
        dtype=torch.float32):
    # grids: the 2D coordinate grids to make the blocks from, as generated by make_BFS_grids
    # in_vel: mean velocity of the parabolic inflow profile
    # in_var: the variance of the gaussian inflow profile of the passive scalar
    # viscosity: viscosity of the flow
    # buffer_viscosity_ratio: increase the viscosity in the outflow buffer up to this factor to damp vortices before they reach the outflow. 1 to disable.
    dims = 2
    assert len(grids)==5
    H = h + s
    Gtl, Gtm, Gbm, Gtr, Gbr = grids
    res_h = Gtl.size(-2)-1

    viscosity = torch.ones([1], dtype=dtype, device=cpu_device)*viscosity
    domain = PISOtorch.Domain(dims, viscosity, passiveScalarChannels=1, name="Domain5BlockBFS", device=cuda_device, dtype=dtype)

    # make the blocks
    Btl = domain.CreateBlock(vertexCoordinates=Gtl.to(cuda_device), name="BlockInflow")
    Btm = domain.CreateBlock(vertexCoordinates=Gtm.to(cuda_device), name="BlockTop")
    Bbm = domain.CreateBlock(vertexCoordinates=Gbm.to(cuda_device), name="BlockBottom")
    Btr = domain.CreateBlock(vertexCoordinates=Gtr.to(cuda_device), name="BlockBufferTop")
    Bbr = domain.CreateBlock(vertexCoordinates=Gbr.to(cuda_device), name="BlockBufferBottom")

    blocks = [Btl, Btm, Bbm, Btr, Bbr]
    layout = [[-1,2,4],[0,1,3]] # used by output formatting

    # set boundaries
    # all outer boundaries are walls, in- and outflow are set below
    for block in blocks:
        block.CloseAllBoundaries()

    # connect the blocks
    # parameters are: block_from, side of block_from to connect, block_to, side of block_to to connect, axes
    Btl.ConnectBlock("+x", Btm, "-x", "-y")
    Btm.ConnectBlock("-y", Bbm, "+y", "-x")
    Btm.ConnectBlock("+x", Btr, "-x", "-y")
    Bbm.ConnectBlock("+x", Bbr, "-x", "-y")
    Btr.ConnectBlock("-y", Bbr, "+y", "-x")

    if buffer_viscosity_ratio!=1:
        # per-cell viscosity in the outflow buffer: exponential increase over the first half, then constant
        for block in [Btr, Bbr]:
            size = block.getSizes()
            ramp = torch.linspace(0, 1, size.x//2, dtype=dtype)
            ramp = torch.cat([ramp, torch.ones([size.x - size.x//2], dtype=dtype)])
            block_viscosity = viscosity * torch.pow(buffer_viscosity_ratio, ramp)
            block.setViscosity(block_viscosity.reshape([1,1,1,size.x]).expand(-1,-1,size.y,-1).to(cuda_device).contiguous())

    # specify the inflow
    # parabolic (Poiseuille) profile with mean velocity in_vel, sampled at the cell centers of the inflow face
    y_in = shapes.coords_to_center_coords(Gtl)[0,1,:,0] - s
    inflow = 6*in_vel*(y_in/h)*(1 - y_in/h)
    inflow = torch.stack([inflow, torch.zeros_like(inflow)]).reshape([1,dims,res_h,1])
    Btl.getBoundary("-x").setVelocity(inflow.to(cuda_device).contiguous())

    # passive scalar inflow
    inflow_scalar = shapes.get_grid_normal_dist([res_h], [0], [in_var], dtype=dtype)
    Btl.getBoundary("-x").setPassiveScalar(inflow_scalar.reshape([1,1,res_h,1]).to(cuda_device).contiguous())

    # also set up outflow boundaries at the end of the buffer.
    # these will need special treatment during the simulation
    # start with a parabolic profile over the full height, the mean velocity follows from mass conservation
    out_bounds = []
    for block, grid in [(Btr, Gtr), (Bbr, Gbr)]:
        y_out = shapes.coords_to_center_coords(grid)[0,1,:,-1]
        outflow = 6*in_vel*(h/H)*(y_out/H)*(1 - y_out/H)
        outflow = torch.stack([outflow, torch.zeros_like(outflow)]).reshape([1,dims,-1,1])
        bound = block.getBoundary("+x")
        bound.setVelocity(outflow.to(cuda_device).contiguous())
        bound.CreatePassiveScalar(False) # varying passive scalar for the outflow
        out_bounds.append(bound)

    # make the boundaries consistent/divergence-free
    PISOtorch_simulation.balance_boundary_fluxes(domain, out_bounds)

    # the outflow is advected with the mean velocity of the channel after the step
    in_flux = torch.sum(inflow[0,0,:,0] * (Gtl[0,1,1:,0] - Gtl[0,1,:-1,0])).item()
    char_vel = torch.tensor([[in_flux/H, 0]], device=cuda_device, dtype=dtype) # NC

    # callback function to update the outflow during the simulation
    out_bound_indices = [domain.getBlocks().index(Btr), domain.getBlocks().index(Bbr)]
    def prep_fn(domain, time_step, **kwargs):
        out_bounds = [domain.getBlock(idx).getBoundary("+x") for idx in out_bound_indices]
        PISOtorch_simulation.update_advective_boundaries(domain, out_bounds, char_vel, time_step.cuda())

    domain.PrepareSolve()

    return domain, {"PRE": prep_fn}, layout

RUN_ID = 0

def backward_facing_step_sample(log_dir, name, iterations=100, time_step=0.1, res_scale=8, Re=1400, step_height=1.0, buffer_viscosity_ratio=1,
        dp=False, STOP_FN=None):
    # res_scale: resolution of the channel after the step in cells per inflow height h.
    #   The cross-stream resolution is 4x higher. res_scale=16/downsample_factor of the INC code.
    # Re: Reynolds number based on the mean inflow velocity and the hydraulic diameter of the inflow channel (2h)
    # step_height: step height relative to the inflow height h
    global RUN_ID
    if STOP_FN():
        return
    LOG = get_logger("BFS")
    dtype = torch.float64 if dp else torch.float32

    h = 1.0
    s = step_height*h
    H = h + s
    in_vel = 1.0
    viscosity = in_vel*2*h/Re

    LOG.info("2D BFS #%d '%s' with step height %.02e, resolution scale %d, Re %.02e (viscosity %.02e), %d iterations with time step %.02e",
             RUN_ID, name, s, res_scale, Re, viscosity, iterations, time_step)
    log_dir = os.path.join(log_dir, "%04d_BFS_%s"%(RUN_ID, name))

    grids = make_BFS_grids(h=h, s=s, l=5*h, L=32*h, buffer_length=3*h,
        res_l=4*res_scale, res_L=32*res_scale, res_y=int(4*res_scale*H), res_buffer=2*res_scale, ratio=(20,2), dtype=dtype)
    plot_grids(grids, color=["r","g","b","y","k"], path=log_dir, linewidth=0.5)

    # resample with the average cell size of the cross-stream direction
    resample_shape, resample_fill_steps = get_resampling_settings(grids, pixel_size=H/int(4*res_scale*H))
    LOG.info("output resampling shape %s with up to %d fill steps", resample_shape, resample_fill_steps)

    domain, prep_fn, layout = make5BlockBackwardFacingStepSetup(grids, h=h, s=s, in_vel=in_vel, in_var=0.4, viscosity=viscosity,
        buffer_viscosity_ratio=buffer_viscosity_ratio, dtype=dtype)

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
    sim.make_divergence_free(max_iter=sim.linear_solve_max_iterations)

    # set the outflow update only after the initial projection, which uses a time step of 1
    sim.prep_fn = prep_fn
    sim.run(iterations)

    RUN_ID += 1


if __name__=="__main__":
    # create a new directory <time-step>_<name> in ./test_runs to use as base output directory
    run_dir = setup_run("./test_runs",
        name="backward-facing-step-sample"
    )
    LOG = get_logger("Main")
    stop_handler = PISOtorch_simulation.StopHandler(LOG)
    stop_handler.register_signal()

    backward_facing_step_sample(run_dir, name="Re1400_r8", res_scale=8, Re=1400, iterations=1000, time_step=0.1, dp=False, STOP_FN=stop_handler)

    # higher resolution
    #backward_facing_step_sample(run_dir, name="Re1400_r16", res_scale=16, Re=1400, iterations=1000, time_step=0.1, dp=False, STOP_FN=stop_handler)

    # with damping in the outflow buffer
    #backward_facing_step_sample(run_dir, name="Re1400_r8_buffer-visc10", res_scale=8, Re=1400, buffer_viscosity_ratio=10, iterations=1000, time_step=0.1, dp=False, STOP_FN=stop_handler)

    stop_handler.unregister_signal()
    close_logging()
