import os, signal
from lib.util.logging import setup_run, get_logger, close_logging
from lib.util.profiling import SAMPLE

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

def growth_factor_from_sizes(res, x, x0, eps=1e-5, print_info_fn=None):
    from scipy.optimize import least_squares
    # res: number of cells
    # x: total size
    # x0: base cell size
    # finds exponent b s.t. x = sum[i=0, res-1](x0 * b**i) = x0*(1-b**res)/(1-b)
    if not (res>0 and x>x0 and x0>0 and eps>0): raise ValueError("invalid input")
    
    res = np.asarray([res])
    x = np.asarray([x])
    x0 = np.asarray([x0])
    
    if print_info_fn: print_info_fn("least squares: res=%.03e, x==%.03e, x0==%.03e, x/x0=%.03e.", res, x, x0, x/x0)
    
    if np.isclose(res, x/x0):
        return 1
    
    def func(b):
        return x0*(1-np.power(b,res))/(1-b) - x
    
    if res<x/x0:
        b0 = 1+eps
        lb = b0
        ub = 2
    else:
        b0 = 1-eps
        lb = 0+eps
        ub = b0
    
    b0 = np.asarray([b0])
    lb = np.asarray([lb])
    ub = np.asarray([ub])
    
    if print_info_fn: print_info_fn("least squares: b0=%.03e, lb==%.03e, ub==%.03e.", b0, lb, ub)
    
    b = least_squares(func, b0, bounds=(lb, ub))
    
    if print_info_fn: print_info_fn("least squares: %s.", b)
    
    return b.x

def make_refined_grid(
        top_y_res:int, top_size:float, top_grow_factor:float,
        bot_y_res:int, bot_size:float, bot_grow_factor:float,
        in_x_res:int, in_size:float, in_grow_factor:float,
        out_x_res:int, out_size:float, out_grow_factor:float,
        dtype=torch.float32):
    """
    the obstacle is centered on (0,0)
    obs_x_res, obs_y_res: number of cells along the obstacle boundary
    obs_x_size, obs_y_size: physical size of the obstacle
    top_bot_size: physical space above and below the obstacle
    in_size: physical space before the obstacle
    out_size: physical space after the obstacle
    *_grow_factor: how fast the grid size grows in that direction (exponential base)
    
    x-axis goes left to right
    y-axis goes bottom to top
    """
    
    # block corner coordinates
    block_corners_x = [-in_size, 0, out_size]
    block_corners_y = [-bot_size, 0, top_size]
    
    
    # y-resolution and -weights for top and bottom grids
    top_weights = shapes.make_weights_exp(top_y_res, base=top_grow_factor, refinement="BOTH")
    bot_weights = shapes.make_weights_exp(bot_y_res, base=bot_grow_factor, refinement="BOTH")
    
    # x-resolution and -weights for outflow grids
    out_weights = shapes.make_weights_exp(out_x_res, base=out_grow_factor, refinement="START")
    
    ### Make the block grids
    # out
    Gbr = shapes.generate_grid_vertices_2D((bot_y_res+1, out_x_res+1),
        corner_vertices=[(block_corners_x[1], block_corners_y[0]) ,(block_corners_x[2], block_corners_y[0]),
                         (block_corners_x[1], block_corners_y[1]), (block_corners_x[2], block_corners_y[1])],
        x_weights = bot_weights, y_weights = out_weights, dtype=dtype)
    
    Gtr = shapes.generate_grid_vertices_2D((top_y_res+1, out_x_res+1),
        corner_vertices=[(block_corners_x[1], block_corners_y[1]) ,(block_corners_x[2], block_corners_y[1]),
                         (block_corners_x[1], block_corners_y[2]), (block_corners_x[2], block_corners_y[2])],
        x_weights = top_weights, y_weights = out_weights, dtype=dtype)
    

    # in
    if in_grow_factor=="AUTO":
        min_x_size = Gtr[0,0,0,1] - Gtr[0,0,0,0]
        in_grow_factor = growth_factor_from_sizes(in_x_res, block_corners_x[1] - block_corners_x[0], min_x_size, print_info_fn=LOG.info)
        #block_corners_x[0] = - min_x_size * (1 - in_grow_factor**(in_x_res)) / (1 - in_grow_factor)
        #LOG.info("overwriting inflow size to %.03e to match step cell size %.03e (%.03e, %.03e)", -block_corners_x[0], min_x_size, Gtr[0,0,0,1], Gtr[0,0,0,0])
        LOG.info("overwriting inflow growth factor to %.03e to match step cell size %.03e (%.03e, %.03e)", in_grow_factor, min_x_size, Gtr[0,0,0,1], Gtr[0,0,0,0])
    
    # x-resolution and -weights for inflow grids
    in_weights = shapes.make_weights_exp(in_x_res, base=in_grow_factor, refinement="END")
    
    Gtl = shapes.generate_grid_vertices_2D((top_y_res+1, in_x_res+1),
        corner_vertices=[(block_corners_x[0], block_corners_y[1]) ,(block_corners_x[1], block_corners_y[1]),
                         (block_corners_x[0], block_corners_y[2]), (block_corners_x[1], block_corners_y[2])],
        x_weights = top_weights, y_weights = in_weights, dtype=dtype)
    
    grids = [Gtl, Gtr, Gbr]
    
    return grids

def make3BlockBackwardFacingStepSetupRefined(grids, z:int=0, z_size:float=1, in_vel:float=1, in_var:float=0.4, closed_y=False, closed_z=False,
        viscosity=0.0, scale:float=None, dtype=torch.float32) -> PISOtorch.Domain:
    # 8 blocks arranged and connected as a "ring", s.t. the missing center block creates an obstacle via closed bounds
    # grids: the 2D coordinate grids to make the blocks from, as generated by make_refined_grid
    # in_vel: the magnitude of the inflow velocity
    # in_var: the variance of the gaussian inflow profile when using closed bonunds
    # closed_y: whether the upper and lower bounds of the channel should be closed (periodic otherwise)
    # closed_z: whether the z-bounds of the channel should be closed (periodic otherwise)
    # viscosity: viscosity of the flow
    # scale: None for untransformed grid. otherwise the cell size is multiplied by this, used for normalization with different resolutions
    dims = 2 if z is None else 3
    assert len(grids)==3
    
    init_block_vel = True
    
    #x_sizes = [grids[0].size(-1)-1, grids[2].size(-1)-1]
    #x = np.sum(x_sizes)
    y_sizes = [grids[2].size(-2)-1, grids[0].size(-2)-1]
    #y = np.sum(y_sizes)

    x_size = grids[1][0,0,0,-1] - grids[0][0,0,0,0]
    y_size = grids[1][0,1,-1,0] - grids[2][0,1,0,0]
    resample_res = 32
    resample_shape = [int(x_size*resample_res), int(y_size*resample_res)] # xyz
    
    #vel = [in_vel] + [0]*(dims-1)
    
    if dims==3:
        grids = [shapes.extrude_grid_z(grid, z, end_z=z_size, weights_z=None, exp_base=1.05) for grid in grids]
    
    Gtl, Gtr, Gbr = grids
    
    
    viscosity = torch.ones([1], dtype=dtype, device=cpu_device)*viscosity
    domain = PISOtorch.Domain(dims, viscosity, passiveScalarChannels=0, name="Domain8BlockChannelFlow", device=cuda_device, dtype=dtype)
    
    # make the blocks
    Btl = domain.CreateBlock(vertexCoordinates=Gtl.to(cuda_device), name="BlockTopLeft")
    Btr = domain.CreateBlock(vertexCoordinates=Gtr.to(cuda_device), name="BlockTopRight")
    Bbr = domain.CreateBlock(vertexCoordinates=Gbr.to(cuda_device), name="BlockBotRight")

    blocks = [Btl, Btr, Bbr]
    layout = [[-1,2],[0,1]] # used by output formatting

    # set boundaries
    # close obstacle
    Btl.CloseBoundary("-y")
    Bbr.CloseBoundary("-x")
    if not closed_y:
        # connect to be periodic
        axes = ["-x"] if dims==2 else ["-z", "-x"]
        Btr.ConnectBlock("+y", Bbr, "-y", *axes)
    
    if dims==3 and closed_z:
        for block in blocks:
            block.CloseBoundary("-z") # also closes +z from default periodic
            #block.CloseBoundary("+z")
    
    # connect 8-block ring
    axes_x = ["-y"] if dims==2 else ["-y", "-z"]
    axes_y = ["-x"] if dims==2 else ["-z", "-x"]
    #Btl.ConnectBlock("+x", Btm, "-x", *axes_x)
    #Btm.ConnectBlock("+x", Btr, "-x", *axes_x)
    Btl.ConnectBlock("+x", Btr, "-x", *axes_x)

    Btr.ConnectBlock("-y", Bbr, "+y", *axes_y)

    # on block connection specification:
    # parameters are: block_from, side of block_from to connect, block_to, side of block_to to connect, axes
    # the "axes"-parameter is a bit tricky: it is based on the value o f"side of block_from to connect"
    # and goes over the remaining axes by increasing index, wrapping back to 0 (x) if necessary.
    # It specifies the other axis to connect to. usually this will be simply increasing to keep the setup consistent.
    # The sign indicates whether this connection axis should the inverted (+) or not (-)

    # PISOtorch.ConnectBlocks sets a valid 2-way connection
    
    # specify the inflow
    in_shape = [y_sizes[-1]] if dims==2 else [z,y_sizes[-1]]
    bdims = dims-1
    # when the boundaries normal to the inflow are closed we use a gauss profile
    inflow = shapes.get_grid_normal_dist(in_shape, [0]*bdims,[in_var]*bdims, dtype=dtype)
    inflow = (torch.reshape(inflow, [1,1]+in_shape+[1])*in_vel)
    #else:
        # othewise the inflow is constant
    #    inflow = torch.ones([1,1]+in_shape+[1], dtype=dtype, device=cpu_device)*in_vel
    in_size = Gtl[0,1,-1,0] - Gtl[0,1,0,0]
    in_flux = inflow * torch.abs(Gtl[0:1,1:2,1:,0:1] - Gtl[0:1,1:2,:-1,0:1]) # 11y1
    mean_in = torch.sum(in_flux) / in_size
    #max_in = torch.max(inflow)
    
    inflow = torch.cat([inflow]+[torch.zeros_like(inflow)]*bdims, axis=1)
    inflow = inflow.to(cuda_device).contiguous()
    
    Btl.getBoundary("-x").setVelocity(inflow.to(cuda_device).contiguous())
    if init_block_vel:
        Btl.setVelocity((torch.ones_like(Btl.velocity) * inflow).contiguous())
    
    if domain.hasPassiveScalar():
        inflow_scalar = shapes.get_grid_normal_dist(in_shape, [0]*bdims,[in_var]*bdims, dtype=dtype)
        inflow_scalar = (torch.reshape(inflow_scalar, [1,1]+in_shape+[1]))
        Btl.getBoundary("-x").setPassiveScalar(inflow_scalar.to(cuda_device))
    
    # also set up outflow boundaries at the end of the channel.
    # these will need special treatment during the simulation
    out_size = Gtr[0,1,-1,0] - Gbr[0,1,0,0]
    #char_vel = torch.tensor([[max_in]+[0]*(dims-1)], device=cuda_device, dtype=dtype) # NC
    mean_out = mean_in*in_size/out_size
    char_vel = torch.tensor([[mean_out]+[0]*(dims-1)], device=cuda_device, dtype=dtype) # NC
    out_vel = torch.reshape(torch.FloatTensor([mean_out.cpu(),0] if dims==2 else [mean_out.cpu(),0,0]), [1,dims]+[1]*dims).cuda()
    
    Btr.getBoundary("+x").setVelocity(torch.ones((1,2,y_sizes[1],1), device=cuda_device, dtype=dtype) * out_vel)
    Bbr.getBoundary("+x").setVelocity(torch.ones((1,2,y_sizes[0],1), device=cuda_device, dtype=dtype) * out_vel)
    if init_block_vel:
        Btr.setVelocity((torch.ones_like(Btr.velocity) * out_vel).contiguous())
        Bbr.setVelocity((torch.ones_like(Bbr.velocity) * out_vel).contiguous())

    if domain.hasPassiveScalar():
        # create varying passive scalar for outflow
        Btr.getBoundary("+x").CreatePassiveScalar(domain.getPassiveScalarChannels(),False)
        Bbr.getBoundary("+x").CreatePassiveScalar(domain.getPassiveScalarChannels(),False)
    
    # update once to make the boundaries already consistent/divergence-free
    out_bounds = [Btr.getBoundary("+x"), Bbr.getBoundary("+x")]
    PISOtorch_simulation.balance_boundary_fluxes(domain, out_bounds)
        
    # we use the prep_fn to pass the update of the outflow boundaries
    out_bound_indices = [(1,"+x"), (2,"+x")]
    prep_fn = lambda domain, time_step, **kwargs: PISOtorch_simulation.update_advective_boundaries(domain, [domain.getBlock(idx).getBoundary(bound) for idx, bound in out_bound_indices], char_vel, time_step.cuda())

    domain.PrepareSolve()

    return domain, {"PRE": prep_fn}, layout, resample_shape

RUN_ID = 0

def run_BFS(log_dir, name, iterations=100, time_step=1, res_scale=8, viscosity=5e-2, closed_bounds=False, use_3D=False, dp=False, STOP_FN=None):
    global RUN_ID
    if STOP_FN():
        return
    LOG = get_logger("VortexStreet")
    substeps = -1 #-1 for adaptive, -2 for adaptive based in initial conditions (including boundaries)
    dtype = torch.float64 if dp else torch.float32

    y_res_scale = 64
    top_bot_grow_factor_global = 2
    top_bot_grow_factor = top_bot_grow_factor_global**(1/(y_res_scale//2 - 1))#1.05
    
    out_scale = 36 #8
    out_grow_factor_global = 20 if res_scale<30 else 7 #7
    out_grow_factor = out_grow_factor_global**(1/(out_scale*res_scale - 1))
    
    in_scale = 5
    in_grow_factor_global = out_grow_factor_global
    in_grow_factor = "AUTO" #in_grow_factor_global**(1/(in_scale*res_scale - 1)) #out_grow_factor #"AUTO"

    LOG.info("%dD BFS #%d '%s' with %s bounds, resolution scale %d, %d iterations with time step %.02e, viscosity %.02e",
             3 if use_3D else 2, RUN_ID, name, "closed" if closed_bounds else "open",
             res_scale, iterations, time_step, viscosity)
    
    LOG.info("grid refinement (exp base): top/bot=%s, in=%s, out=%s (global %s)", top_bot_grow_factor, in_grow_factor, out_grow_factor, out_grow_factor_global)
    
    vel = 1.0
    
    if substeps==-2:
        #overwrite the default time-step estimation since the velocity around the obstacle can be larger than the inflow
        ts = 5e-2 * 8 * (scale if scale is not None else 1) * 1/vel
        substeps = max(1, int(time_step/ts)) #* 8

    #time_step = ts
    LOG.info("setting time step to %.02e, substeps to %d", time_step, substeps)
    
    grids = make_refined_grid(
        top_y_res=1*y_res_scale, top_size=1, top_grow_factor=top_bot_grow_factor,
        bot_y_res=1*y_res_scale, bot_size=1, bot_grow_factor=top_bot_grow_factor,
        in_x_res=2*res_scale, in_size=5, in_grow_factor=in_grow_factor,
        out_x_res=out_scale*res_scale, out_size=out_scale, out_grow_factor=out_grow_factor,
        dtype=dtype)
    plot_grids(grids, color=["r","g","b"], path=log_dir)
    
    domain, prep_fn, layout, resample_shape = make3BlockBackwardFacingStepSetupRefined(grids, z=3*res_scale if use_3D else None, z_size=3,
                                in_vel=4.0 if closed_bounds else vel, in_var=0.4, closed_y=closed_bounds, closed_z=closed_bounds, viscosity=viscosity, dtype=dtype)
    
    max_vel = domain.getMaxVelocity(True, True).cpu().numpy()
    LOG.info("Domain max vel: %s", max_vel)
    log_dir=os.path.join(log_dir, "%04d_vortex_street_%s"%(RUN_ID, name))
    
    sim = PISOtorch_simulation.Simulation(domain=domain, block_layout=layout, prep_fn=None,
            substeps="ADAPTIVE" if substeps==-1 else substeps, time_step=time_step, corrector_steps=2,
            advection_tol=1e-7, pressure_tol=4e-7,
            advect_non_ortho_steps=1, pressure_non_ortho_steps=1, pressure_return_best_result=True,
            velocity_corrector="FD", non_orthogonal=False,
            log_interval=1, norm_vel=True,
            output_resampling_shape=resample_shape, output_resampling_fill_max_steps=32,
            log_dir=log_dir, save_domain_name="domain",
            stop_fn=STOP_FN)
    
    sim.preconditionBiCG = False
    sim.BiCG_precondition_fallback = True
    
    sim.make_divergence_free()
    
    sim.prep_fn = prep_fn
    sim.run(iterations)
    
    RUN_ID += 1


if __name__=="__main__":
    # create a new directory <time-step>_learning_sample in ./test_runs to use as base output directory
    run_dir = setup_run("./test_runs",
        name="BFS-hao-v4e-4_in-auto_refined.05_it200-ts.25-adaptive_CG-maxIT5k_ortho_vel-init_char-vel-mean-out" #
    )
    LOG = get_logger("Main")
    stop_handler = PISOtorch_simulation.StopHandler(LOG)
    stop_handler.register_signal()


    #vortex_street_sample(run_dir, name="turbulent_open_r32", res_scale=32, iterations=2, time_step=0.1, viscosity=5e-3, closed_bounds=False, use_3D=False, dp=False, STOP_FN=stop_handler)
    
    run_BFS(run_dir, name="closed_r16-y64_out-g20", res_scale=16, iterations=200, time_step=0.25, viscosity=4e-4, closed_bounds=True, use_3D=False, dp=False, STOP_FN=stop_handler)
    #run_BFS(run_dir, name="closed_r32-y64_out-g7", res_scale=32, iterations=4, time_step=0.01, viscosity=4e-4, closed_bounds=True, use_3D=False, dp=False, STOP_FN=stop_handler)


    #vortex_street_sample(run_dir, name="turbulent_open_r32_3D", res_scale=32, iterations=10, time_step=0.1, viscosity=4e-4, closed_bounds=False, use_3D=True, dp=True, STOP_FN=stop_handler)
    
    close_logging()
