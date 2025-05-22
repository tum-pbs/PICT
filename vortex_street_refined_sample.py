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

def make_refined_grid(obs_x_res:int, obs_y_res:int, obs_x_size:float, obs_y_size:float,
        top_bot_size:float, top_bot_grow_factor:float,
        in_size:float, in_grow_factor:float,
        out_size:float, out_grow_factor:float,
        dtype=torch.float32):
    """
    the obstacle is centered on (0,0)
    obs_x_res, obs_y_res: number of cells along the obstacle boundary
    obs_x_size, obs_y_size: physical size of the obstacle
    top_bot_size: physical space above and below the obstacle
    in_size: physical space before the obstacle
    out_size_size: physical space after the obstacle
    *_grow_factor: how fast the grid size grows in that direction (exponential base)
    
    x-axis goes left to right
    y-axis goes bottom to top
    """
    
    # block corner coordinates
    block_corners_x = [-(in_size + obs_x_size/2), -obs_x_size/2, obs_x_size/2, obs_x_size/2 + out_size]
    block_corners_y = [-(top_bot_size + obs_y_size/2), -obs_y_size/2, obs_y_size/2, obs_y_size/2 + top_bot_size]
    
    # cell sizes along the obstacle, to match other grids to
    obs_cell_size_x = obs_x_size / obs_x_res
    obs_cell_size_y = obs_y_size / obs_y_res
    
    def grow_res_to_size(base_size, target_size, exp_base):
        res = 0
        size = 0
        while size<target_size:
            size += base_size * (exp_base**res)
            res += 1
        return res
    
    # y-resolution and -weights for top and bottom grids
    top_bot_y_res = grow_res_to_size(obs_cell_size_y, top_bot_size, top_bot_grow_factor)
    top_weights = shapes.make_weights_exp(top_bot_y_res, base=top_bot_grow_factor, refinement="START")
    bot_weights = shapes.make_weights_exp(top_bot_y_res, base=top_bot_grow_factor, refinement="END")
    
    # x-resolution and -weights for inflow grids
    in_x_res = grow_res_to_size(obs_cell_size_x, in_size, in_grow_factor)
    in_weights = shapes.make_weights_exp(in_x_res, base=in_grow_factor, refinement="END")
    
    # x-resolution and -weights for outflow grids
    out_x_res = grow_res_to_size(obs_cell_size_x, out_size, out_grow_factor)
    out_weights = shapes.make_weights_exp(out_x_res, base=out_grow_factor, refinement="START")
    
    ### Make the block grids
    # bottom row
    Gbl = shapes.generate_grid_vertices_2D((top_bot_y_res+1, in_x_res+1),
        corner_vertices=[(block_corners_x[0], block_corners_y[0]) ,(block_corners_x[1], block_corners_y[0]),
                         (block_corners_x[0], block_corners_y[1]), (block_corners_x[1], block_corners_y[1])],
        x_weights = bot_weights, y_weights = in_weights, dtype=dtype)
    Gbm = shapes.generate_grid_vertices_2D((top_bot_y_res+1, obs_x_res+1),
        corner_vertices=[(block_corners_x[1], block_corners_y[0]) ,(block_corners_x[2], block_corners_y[0]),
                         (block_corners_x[1], block_corners_y[1]), (block_corners_x[2], block_corners_y[1])],
        x_weights = bot_weights, y_weights = None, dtype=dtype)
    Gbr = shapes.generate_grid_vertices_2D((top_bot_y_res+1, out_x_res+1),
        corner_vertices=[(block_corners_x[2], block_corners_y[0]) ,(block_corners_x[3], block_corners_y[0]),
                         (block_corners_x[2], block_corners_y[1]), (block_corners_x[3], block_corners_y[1])],
        x_weights = bot_weights, y_weights = out_weights, dtype=dtype)
    # mid row
    Gml = shapes.generate_grid_vertices_2D((obs_y_res+1, in_x_res+1),
        corner_vertices=[(block_corners_x[0], block_corners_y[1]) ,(block_corners_x[1], block_corners_y[1]),
                         (block_corners_x[0], block_corners_y[2]), (block_corners_x[1], block_corners_y[2])],
        x_weights = None, y_weights = in_weights, dtype=dtype)
    Gmr = shapes.generate_grid_vertices_2D((obs_y_res+1, out_x_res+1),
        corner_vertices=[(block_corners_x[2], block_corners_y[1]) ,(block_corners_x[3], block_corners_y[1]),
                         (block_corners_x[2], block_corners_y[2]), (block_corners_x[3], block_corners_y[2])],
        x_weights = None, y_weights = out_weights, dtype=dtype)
    # top row
    Gtl = shapes.generate_grid_vertices_2D((top_bot_y_res+1, in_x_res+1),
        corner_vertices=[(block_corners_x[0], block_corners_y[2]) ,(block_corners_x[1], block_corners_y[2]),
                         (block_corners_x[0], block_corners_y[3]), (block_corners_x[1], block_corners_y[3])],
        x_weights = top_weights, y_weights = in_weights, dtype=dtype)
    Gtm = shapes.generate_grid_vertices_2D((top_bot_y_res+1, obs_x_res+1),
        corner_vertices=[(block_corners_x[1], block_corners_y[2]) ,(block_corners_x[2], block_corners_y[2]),
                         (block_corners_x[1], block_corners_y[3]), (block_corners_x[2], block_corners_y[3])],
        x_weights = top_weights, y_weights = None, dtype=dtype)
    Gtr = shapes.generate_grid_vertices_2D((top_bot_y_res+1, out_x_res+1),
        corner_vertices=[(block_corners_x[2], block_corners_y[2]) ,(block_corners_x[3], block_corners_y[2]),
                         (block_corners_x[2], block_corners_y[3]), (block_corners_x[3], block_corners_y[3])],
        x_weights = top_weights, y_weights = out_weights, dtype=dtype)
    
    grids = [Gtl, Gtm, Gtr, Gml, Gmr, Gbl, Gbm, Gbr]
    
    return grids

def make8BlockChannelFlowSetupRefined(grids, z:int=0, z_size:float=1, in_vel:float=1, in_var:float=0.4, closed_y=False, closed_z=False,
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
    assert len(grids)==8
    
    x_sizes = [grids[0].size(-1)-1, grids[1].size(-1)-1, grids[2].size(-1)-1]
    x = np.sum(x_sizes)
    x_in = x_sizes[1]
    y_sizes = [grids[5].size(-2)-1, grids[3].size(-2)-1, grids[0].size(-2)-1]
    y = np.sum(y_sizes)
    y_in = y_sizes[1]
    
    #vel = [in_vel] + [0]*(dims-1)
    
    if dims==3:
        grids = [shapes.extrude_grid_z(grid, z, end_z=z_size, weights_z=None, exp_base=1.05) for grid in grids]
    
    Gtl, Gtm, Gtr, Gml, Gmr, Gbl, Gbm, Gbr = grids
    
    
    viscosity = torch.ones([1], dtype=dtype, device=cpu_device)*viscosity
    domain = PISOtorch.Domain(dims, viscosity, passiveScalarChannels=0, name="Domain8BlockChannelFlow", device=cuda_device, dtype=dtype)
    
    # make the blocks
    Btl = domain.CreateBlock(vertexCoordinates=Gtl.to(cuda_device), name="BlockTopLeft")
    Btm = domain.CreateBlock(vertexCoordinates=Gtm.to(cuda_device), name="BlockTopMiddle")
    Btr = domain.CreateBlock(vertexCoordinates=Gtr.to(cuda_device), name="BlockTopRight")
    Bml = domain.CreateBlock(vertexCoordinates=Gml.to(cuda_device), name="BlockMiddleLeft")
    Bmr = domain.CreateBlock(vertexCoordinates=Gmr.to(cuda_device), name="BlockMiddleRight")
    Bbl = domain.CreateBlock(vertexCoordinates=Gbl.to(cuda_device), name="BlockBotLeft")
    Bbm = domain.CreateBlock(vertexCoordinates=Gbm.to(cuda_device), name="BlockBotMiddle")
    Bbr = domain.CreateBlock(vertexCoordinates=Gbr.to(cuda_device), name="BlockBotRight")

    blocks = [Btl, Btm, Btr, Bml, Bmr, Bbl, Bbm, Bbr]
    layout = [[5,6,7],[3,-1,4],[0,1,2]] # used by output formatting

    # set boundaries
    # close obstacle
    Btm.CloseBoundary("-y")
    Bml.CloseBoundary("+x")
    Bmr.CloseBoundary("-x")
    Bbm.CloseBoundary("+y")
    if not closed_y:
        # connect to be periodic
        axes = ["-x"] if dims==2 else ["-z", "-x"]
        Btl.ConnectBlock("+y", Bbl, "-y", *axes)
        Btm.ConnectBlock("+y", Bbm, "-y", *axes)
        Btr.ConnectBlock("+y", Bbr, "-y", *axes)
    
    if dims==3 and closed_z:
        for block in blocks:
            block.CloseBoundary("-z") # also closes +z from default periodic
            #block.CloseBoundary("+z")
    
    # connect 8-block ring
    axes_x = ["-y"] if dims==2 else ["-y", "-z"]
    axes_y = ["-x"] if dims==2 else ["-z", "-x"]
    Btl.ConnectBlock("+x", Btm, "-x", *axes_x)
    Btm.ConnectBlock("+x", Btr, "-x", *axes_x)

    Btr.ConnectBlock("-y", Bmr, "+y", *axes_y)
    Bmr.ConnectBlock("-y", Bbr, "+y", *axes_y)
    
    Bbl.ConnectBlock("+x", Bbm, "-x", *axes_x)
    Bbm.ConnectBlock("+x", Bbr, "-x", *axes_x)

    Btl.ConnectBlock("-y", Bml, "+y", *axes_y)
    Bml.ConnectBlock("-y", Bbl, "+y", *axes_y)

    # on block connection specification:
    # parameters are: block_from, side of block_from to connect, block_to, side of block_to to connect, axes
    # the "axes"-parameter is a bit tricky: it is based on the value o f"side of block_from to connect"
    # and goes over the remaining axes by increasing index, wrapping back to 0 (x) if necessary.
    # It specifies the other axis to connect to. usually this will be simply increasing to keep the setup consistent.
    # The sign indicates whether this connection axis should the inverted (+) or not (-)

    # PISOtorch.ConnectBlocks sets a valid 2-way connection
    
    # specify the inflow
    in_shape = [y] if dims==2 else [z,y]
    bdims = dims-1
    if closed_y:
        # when the boundaries normal to the inflow are closed we use a gauss profile
        inflow = shapes.get_grid_normal_dist(in_shape, [0]*bdims,[in_var]*bdims, dtype=dtype)
        inflow = (torch.reshape(inflow, [1,1]+in_shape+[1])*in_vel)
    else:
        # othewise the inflow is constant
        inflow = torch.ones([1,1]+in_shape+[1], dtype=dtype, device=cpu_device)*in_vel
    mean_in = torch.mean(inflow)
    max_in = torch.max(inflow)
    char_vel = torch.tensor([[max_in]+[0]*(dims-1)], device=cuda_device, dtype=dtype) # NC
    inflow = torch.cat([inflow]+[torch.zeros_like(inflow)]*bdims, axis=1)
    inflow_blocks = [_.cuda() for _ in torch.split(inflow, y_sizes, dim=-2)]
    
    Btl.getBoundary("-x").setVelocity(inflow_blocks[2])
    Bml.getBoundary("-x").setVelocity(inflow_blocks[1])
    Bbl.getBoundary("-x").setVelocity(inflow_blocks[0])
    
    if domain.hasPassiveScalar():
        in_shape = [y_in] if dims==2 else [z,y_in]
        inflow_scalar_mid = shapes.get_grid_normal_dist(in_shape, [0]*bdims,[in_var]*bdims, dtype=dtype)
        inflow_scalar_mid = (torch.reshape(inflow_scalar_mid, [1,1]+in_shape+[1]))
        Bml.getBoundary("-x").setPassiveScalar(inflow_scalar_mid.to(cuda_device))
    
    # also set up outflow boundaries at the end of the channel.
    # these will need special treatment during the simulation
    out_vel = torch.reshape(torch.FloatTensor([mean_in.cpu(),0] if dims==2 else [mean_in.cpu(),0,0]), [1,dims]+[1]*dims).cuda()
    
    Btr.getBoundary("+x").setVelocity(torch.ones_like(inflow_blocks[2]) * out_vel)
    Bmr.getBoundary("+x").setVelocity(torch.ones_like(inflow_blocks[1]) * out_vel)
    Bbr.getBoundary("+x").setVelocity(torch.ones_like(inflow_blocks[0]) * out_vel)

    if domain.hasPassiveScalar():
        # create varying passive scalar for outflow
        Btr.getBoundary("+x").CreatePassiveScalar(domain.getPassiveScalarChannels(),False)
        Bmr.getBoundary("+x").CreatePassiveScalar(domain.getPassiveScalarChannels(),False)
        Bbr.getBoundary("+x").CreatePassiveScalar(domain.getPassiveScalarChannels(),False)
        
    # we use the prep_fn to pass the update of the outflow boundaries
    
    out_bounds = [Btr.getBoundary("+x"), Bmr.getBoundary("+x"), Bbr.getBoundary("+x")]
    # make the boundaries consistent/divergence-free
    PISOtorch_simulation.balance_boundary_fluxes(domain, out_bounds)
    # callback function to update the outflow during the simulation
    prep_fn = lambda domain, time_step, **kwargs: PISOtorch_simulation.update_advective_boundaries(domain, out_bounds, char_vel, time_step.cuda())

    domain.PrepareSolve()

    return domain, {"PRE": prep_fn}, layout

RUN_ID = 0

def vortex_street_sample(log_dir, name, iterations=100, time_step=1, res_scale=8, viscosity=5e-2, closed_bounds=False, use_3D=False, dp=False, STOP_FN=None):
    global RUN_ID
    if STOP_FN():
        return
    LOG = get_logger("VortexStreet")
    substeps = -1 #-1 for adaptive, -2 for adaptive based in initial conditions (including boundaries)
    scale = 1/res_scale # normalizing with resolution s.t. different res_scale have the same physical size
    dtype = torch.float64 if dp else torch.float32

    grow_factor = 1.20

    LOG.info("%dD vortex street #%d '%s' with %s bounds, resolution scale %d (=> scale %.02e), grid refinement %.02e, %d iterations with time step %.02e, viscosity %.02e",
             3 if use_3D else 2, RUN_ID, name, "closed" if closed_bounds else "open",
             res_scale, (scale if scale is not None else 1), grow_factor, iterations, time_step, viscosity)
    
    
    vel = 1.0
    
    if substeps==-2:
        #overwrite the default time-step estimation since the velocity around the obstacle can be larger than the inflow
        ts = 5e-2 * 8 * (scale if scale is not None else 1) * 1/vel
        substeps = max(1, int(time_step/ts)) #* 8

    #time_step = ts
    LOG.info("setting time step to %.02e, substeps to %d", time_step, substeps)
    
    grids = make_refined_grid(obs_x_res=res_scale, obs_y_res=res_scale, obs_x_size=1, obs_y_size=1, top_bot_size=2, top_bot_grow_factor=grow_factor,
                                in_size=2, in_grow_factor=grow_factor, out_size=8, out_grow_factor=grow_factor, dtype=dtype)
    plot_grids(grids, color=["r","g","b","y","k","b","g","r"], path=log_dir)
    
    domain, prep_fn, layout = make8BlockChannelFlowSetupRefined(grids, z=3*res_scale if use_3D else None, z_size=3,
                                                         in_vel=4.0 if closed_bounds else vel, in_var=0.4, closed_y=closed_bounds, closed_z=closed_bounds, viscosity=viscosity, scale=scale, dtype=dtype)
    
    max_vel = domain.getMaxVelocity(True, True).cpu().numpy()
    LOG.info("Domain max vel: %s", max_vel)
    log_dir=os.path.join(log_dir, "%04d_vortex_street_%s"%(RUN_ID, name))
    
    sim = PISOtorch_simulation.Simulation(domain=domain, block_layout=layout, prep_fn=prep_fn,
            substeps="ADAPTIVE" if substeps==-1 else substeps, time_step=time_step, corrector_steps=2, pressure_tol=1e-8,
            advect_non_ortho_steps=1, pressure_non_ortho_steps=1, pressure_return_best_result=True,
            velocity_corrector="FD", non_orthogonal=False,
            log_interval=1, norm_vel=True,
            log_dir=log_dir, save_domain_name="domain",
            stop_fn=STOP_FN)
    
    sim.make_divergence_free()
    sim.run(iterations)
    
    RUN_ID += 1


if __name__=="__main__":
    # create a new directory <time-step>_learning_sample in ./test_runs to use as base output directory
    run_dir = setup_run("./test_runs",
        name="vortex_street_refined.20_sample_ts-0.1-adaptive_ortho"
    )
    LOG = get_logger("Main")
    stop_handler = PISOtorch_simulation.StopHandler(LOG)
    stop_handler.register_signal()


    #vortex_street_sample(run_dir, name="turbulent_open_r32", res_scale=32, iterations=2, time_step=0.1, viscosity=5e-3, closed_bounds=False, use_3D=False, dp=False, STOP_FN=stop_handler)
    
    vortex_street_sample(run_dir, name="turbulent_closed_r32", res_scale=32, iterations=10, time_step=0.1, viscosity=5e-3, closed_bounds=True, use_3D=False, dp=False, STOP_FN=stop_handler)


    #vortex_street_sample(run_dir, name="turbulent_open_r32_3D", res_scale=32, iterations=10, time_step=0.1, viscosity=4e-4, closed_bounds=False, use_3D=True, dp=True, STOP_FN=stop_handler)
