import os, signal
from lib.util.logging import setup_run, get_logger, close_logging
from lib.util.profiling import SAMPLE

import numpy as np


if __name__=="__main__":
    from lib.util.GPU_info import get_available_GPU_id
    cuda_id = None#"7" #
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id or str(get_available_GPU_id(active_mem_threshold=0.8, default=None))


import torch
import PISOtorch # domain data structures and core PISO functions. check /extensions/PISOtorch.cpp to see what is available
import PISOtorch_simulation # uses core PISO functions to make full simulation steps
import lib.data.shapes as shapes
from lib.util.output import plot_grids
from lib.util.memory_usage import MemoryUsage # after torch

# PISOtorch is only implemented for GPU

assert torch.cuda.is_available()
cuda_device = torch.device("cuda")
cpu_device = torch.device("cpu")

def make8BlockChannelFlowSetup(x:int, y:int, z:int, x_in:int, y_in:int, x_pos:int,
        in_vel:float, in_var:float=0.4, closed_y=False, closed_z=False, viscosity=0.0,
        scalar_channels:int=1, scale:float=None, dtype=torch.float32) -> PISOtorch.Domain:
    # 8 blocks arranged and connected as a "ring", s.t. the missing center block creates an obstacle via closed bounds
    # x: total length (stream-wise) of the channel
    # y: total hight (normal to stream) of the channel
    # z (optional, None for 2D): total depth of the channel
    # x_in: the x-size of the obstacle
    # y_in: the y-size of the obstacle
    # x_pos: distance in cells of the obstacle from the inflow (the obstacle is centered in y)
    # in_vel: the magnitude of the inflow velocity
    # in_var: the variance of the gaussian inflow profile when using closed bonunds
    # closed_y: whether the upper and lower bounds of the channel should be closed (periodic otherwise)
    # closed_z: whether the z-bounds of the channel should be closed (periodic otherwise)
    # viscosity: viscosity of the flow
    # scale: None for untransformed grid. otherwise the cell size is multiplied by this, used for normalization with different resolutions
    dims = 2 if z is None else 3
    assert x>=(x_in-2)
    assert y>=(y_in-2)
    x_sizes = [x_pos, x_in, x - (x_pos+x_in)]
    y_sizes = [(y-y_in)//2, y_in, (y-y_in) - (y-y_in)//2]

    has_transformation = scale is not None
    
    vel = [in_vel] + [0]*(dims-1)
    
    # make the grids
    x_poss = [-x_sizes[0]-(x_sizes[1]/2), -x_sizes[1]/2, x_sizes[1]/2, x_sizes[1]/2+x_sizes[2]]
    y_poss = [-y_sizes[0]-(y_sizes[1]/2), -y_sizes[1]/2, y_sizes[1]/2, y_sizes[1]/2+y_sizes[2]]
    if scale is not None:
        x_poss = [_*scale for _ in x_poss]
        y_poss = [_*scale for _ in y_poss]
    # make_wall_refined_ortho_grid(res_x, res_y, corner_lower=(0,0), corner_upper=(1,1), wall_refinement=[], base=1.05, dtype=torch.float32)
    Gbl = shapes.make_wall_refined_ortho_grid(x_sizes[0], y_sizes[0], dtype=dtype,
        corner_lower=(x_poss[0],y_poss[0]), corner_upper=(x_poss[1],y_poss[1]))
    Gbm = shapes.make_wall_refined_ortho_grid(x_sizes[1], y_sizes[0], dtype=dtype,
        corner_lower=(x_poss[1],y_poss[0]), corner_upper=(x_poss[2],y_poss[1]))
    Gbr = shapes.make_wall_refined_ortho_grid(x_sizes[2], y_sizes[0], dtype=dtype,
        corner_lower=(x_poss[2],y_poss[0]), corner_upper=(x_poss[3],y_poss[1]))
    
    Gml = shapes.make_wall_refined_ortho_grid(x_sizes[0], y_sizes[1], dtype=dtype,
        corner_lower=(x_poss[0],y_poss[1]), corner_upper=(x_poss[1],y_poss[2]))
    #Gmm = obstacle
    Gmr = shapes.make_wall_refined_ortho_grid(x_sizes[2], y_sizes[1], dtype=dtype,
        corner_lower=(x_poss[2],y_poss[1]), corner_upper=(x_poss[3],y_poss[2]))
        
    Gtl = shapes.make_wall_refined_ortho_grid(x_sizes[0], y_sizes[2], dtype=dtype,
        corner_lower=(x_poss[0],y_poss[2]), corner_upper=(x_poss[1],y_poss[3]))
    Gtm = shapes.make_wall_refined_ortho_grid(x_sizes[1], y_sizes[2], dtype=dtype,
        corner_lower=(x_poss[1],y_poss[2]), corner_upper=(x_poss[2],y_poss[3]))
    Gtr = shapes.make_wall_refined_ortho_grid(x_sizes[2], y_sizes[2], dtype=dtype,
        corner_lower=(x_poss[2],y_poss[2]), corner_upper=(x_poss[3],y_poss[3]))
    
    grids = [Gtl, Gtm, Gtr, Gml, Gmr, Gbl, Gbm, Gbr]

    if dims==3:
        grids = [shapes.extrude_grid_z(grid, z, end_z=z*scale if scale is not None else z, weights_z=None, exp_base=1.05) for grid in grids]
        Gtl, Gtm, Gtr, Gml, Gmr, Gbl, Gbm, Gbr = grids
    
    
    viscosity = torch.ones([1], dtype=dtype, device=cpu_device)*viscosity
    domain = PISOtorch.Domain(dims, viscosity, name="Domain8BlockChannelFlow", device=cuda_device, dtype=dtype, passiveScalarChannels=scalar_channels)
    
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
    
    if domain.hasPassiveScalar():
        # set free slip condition for passive scalar
        scalar_boundary_type = PISOtorch.BoundaryConditionType.NEUMANN # can also be a list when using multiple scalar channels
        Btm.getBoundary("-y").setPassiveScalarType(scalar_boundary_type)
        Bml.getBoundary("+x").setPassiveScalarType(scalar_boundary_type)
        Bmr.getBoundary("-x").setPassiveScalarType(scalar_boundary_type)
        Bbm.getBoundary("+y").setPassiveScalarType(scalar_boundary_type)
        if closed_y:
            Btl.getBoundary("+y").setPassiveScalarType(scalar_boundary_type)
            Btm.getBoundary("+y").setPassiveScalarType(scalar_boundary_type)
            Btr.getBoundary("+y").setPassiveScalarType(scalar_boundary_type)
            Bbl.getBoundary("-y").setPassiveScalarType(scalar_boundary_type)
            Bbm.getBoundary("-y").setPassiveScalarType(scalar_boundary_type)
            Bbr.getBoundary("-y").setPassiveScalarType(scalar_boundary_type)

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
        if domain.getPassiveScalarChannels()>1:
            inflow_scalar_mid = inflow_scalar_mid.expand(*([-1,domain.getPassiveScalarChannels()] + [-1]*dims))
        Bml.getBoundary("-x").setPassiveScalar(inflow_scalar_mid.to(cuda_device))
    
    # also set up outflow boundaries at the end of the channel.
    # these will need special treatment during the simulation
    out_vel = torch.reshape(torch.FloatTensor([mean_in.cpu(),0] if dims==2 else [mean_in.cpu(),0,0]), [1,dims]+[1]*dims).cuda()
    
    Btr.getBoundary("+x").setVelocity(torch.ones_like(inflow_blocks[2]) * out_vel)
    Bmr.getBoundary("+x").setVelocity(torch.ones_like(inflow_blocks[1]) * out_vel)
    Bbr.getBoundary("+x").setVelocity(torch.ones_like(inflow_blocks[0]) * out_vel)
    
    if domain.hasPassiveScalar():
        # create varying passive scalar for outflow
        Btr.getBoundary("+x").CreatePassiveScalar(False)
        Bmr.getBoundary("+x").CreatePassiveScalar(False)
        Bbr.getBoundary("+x").CreatePassiveScalar(False)
        
    # we use the prep_fn to pass the update of the outflow boundaries
    
    # make the boundaries consistent/divergence-free
    out_bounds = [Btr.getBoundary("+x"), Bmr.getBoundary("+x"), Bbr.getBoundary("+x")]
    PISOtorch_simulation.balance_boundary_fluxes(domain, out_bounds)
    
    # callback function to update the outflow during the simulation
    out_bound_indices = [domain.getBlocks().index(Btr), domain.getBlocks().index(Bmr), domain.getBlocks().index(Bbr)]
    def prep_fn(domain, time_step, **kwargs):
        out_bounds = [domain.getBlock(idx).getBoundary("+x") for idx in out_bound_indices]
        PISOtorch_simulation.update_advective_boundaries(domain, out_bounds, char_vel, time_step.cuda())

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

    LOG.info("%dD vortex street #%d '%s' with %s bounds, resolution scale %d (=> scale %.02e), %d iterations with time step %.02e, viscosity %.02e",
             3 if use_3D else 2, RUN_ID, name, "closed" if closed_bounds else "open",
             res_scale, (scale if scale is not None else 1), iterations, time_step, viscosity)
    
    
    vel = 1.0
    
    if substeps==-2:
        #overwrite the default time-step estimation since the velocity around the obstacle can be larger than the inflow
        ts = 5e-2 * 8 * (scale if scale is not None else 1) * 1/vel
        substeps = max(1, int(time_step/ts)) #* 8

    #time_step = ts
    LOG.info("setting time step to %.02e, substeps to %d", time_step, substeps)

    domain, prep_fn, layout = make8BlockChannelFlowSetup(x=16*res_scale, y=3*res_scale, z=3*res_scale if use_3D else None, x_in=1*res_scale, y_in=1*res_scale, x_pos=2*res_scale,
                                                         in_vel=4.0 if closed_bounds else vel, in_var=0.4, closed_y=closed_bounds, closed_z=closed_bounds, viscosity=viscosity, scale=scale, dtype=dtype)
    
    max_vel = domain.getMaxVelocity(True, True).cpu().numpy()
    LOG.info("Domain max vel: %s", max_vel)
    log_dir=os.path.join(log_dir, "%04d_vortex_street_%s"%(RUN_ID, name))
    
    
    mem_usage = MemoryUsage(logger=LOG)
    def mem_usage_log_fn(total_step, **kwargs):
        mem_usage.check_memory("step %04d"%(total_step,))
    
    sim = PISOtorch_simulation.Simulation(domain=domain, block_layout=layout, prep_fn=prep_fn,
            substeps="ADAPTIVE", time_step=time_step, corrector_steps=2, pressure_tol=1e-5,
            advect_non_ortho_steps=1, pressure_non_ortho_steps=1, pressure_return_best_result=True,
            velocity_corrector="FD", non_orthogonal=False,
            log_interval=1, norm_vel=True, log_fn=mem_usage_log_fn,
            log_dir=log_dir, save_domain_name="domain",
            stop_fn=STOP_FN)
    
    if not use_3D:
        plot_grids(domain.getVertexCoordinates(), color=["r","g","b","y","k","b","g","r"], path=log_dir)
    
    # initialize the domain with velocity that matches in- and outflow
    sim.make_divergence_free()
    
    sim.run(iterations)
    
    mem_usage.print_max_memory()
    
    RUN_ID += 1


if __name__=="__main__":
    # create a new directory <time-step>_learning_sample in ./test_runs to use as base output directory
    run_dir = setup_run("./test_runs",
        name="vortex_street_sample_ts-0.1-adaptive_ortho_2D_closed"
    )
    LOG = get_logger("Main")
    stop_handler = PISOtorch_simulation.StopHandler(LOG)
    stop_handler.register_signal()


    #vortex_street_sample(run_dir, name="turbulent_open_r8", res_scale=8, iterations=100, time_step=0.5, viscosity=5e-3, closed_bounds=False, use_3D=False, dp=False, STOP_FN=stop_handler)
    
    #vortex_street_sample(run_dir, name="turbulent_closed_r8", res_scale=8, iterations=100, time_step=0.5, viscosity=5e-3, closed_bounds=True, use_3D=False, dp=False, STOP_FN=stop_handler)

    #vortex_street_sample(run_dir, name="laminar_open_r8", res_scale=8, iterations=100, time_step=0.5, viscosity=5e-2, closed_bounds=False, use_3D=False, dp=False, STOP_FN=stop_handler)

    #vortex_street_sample(run_dir, name="turbulent_open_r32", res_scale=32, iterations=500, time_step=0.1, viscosity=5e-3, closed_bounds=False, use_3D=False, dp=False, STOP_FN=stop_handler)
    
    vortex_street_sample(run_dir, name="turbulent_closed_r32", res_scale=32, iterations=200, time_step=0.1, viscosity=5e-3, closed_bounds=True, use_3D=False, dp=False, STOP_FN=stop_handler)

    #vortex_street_sample(run_dir, name="laminar_open_r32", res_scale=32, iterations=100, time_step=0.5, viscosity=5e-2, closed_bounds=False, use_3D=False, dp=False, STOP_FN=stop_handler)

    #vortex_street_sample(run_dir, name="turbulent_open_r32_3D", res_scale=32, iterations=100, time_step=0.1, viscosity=5e-3, closed_bounds=False, use_3D=True, dp=False, STOP_FN=stop_handler)
    
    stop_handler.unregister_signal()
    close_logging()
