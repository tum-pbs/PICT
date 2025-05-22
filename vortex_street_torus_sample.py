import os, signal
from lib.util.logging import setup_run, get_logger, close_logging
from lib.util.profiling import SAMPLE

import numpy as np


if __name__=="__main__":
    from lib.util.GPU_info import get_available_GPU_id
    cuda_id = None# "7"
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

def make4BlockTorus_VortexStreet(x, r1=0.5, r2=8, 
        z:int=0, z_size=1, z_closed=False,
        rot_vel=0, obs_vel_z=0, vel_in=1,
        scaling=None, rotation_distortion_max_angle=None, z_sine_scale=None,
        viscosity=1e-3, dtype=torch.float32):
    # Create a torus with 4 segments (quadrants). Inflow at the outer boundary of the left segments, outflow at the right.
    # The middle is an obstacle (closed boundaries) that can rotate (using tangetial boundary velocity)
    # x: angular resolution of a segment. complete torus angular resolution is 4*x.
    #    the radial y-resolution is calculated based on the radius to result in approx. square cells
    # r1: inner radius of the torus (radius of the obstacle)
    # r2: outer radius of the torus. measured from the center, so r2>r1.
    # rot_vel: velocity at the boundary of the obstacle
    # vel_in: inflow velocity
    # rotation_distortion_max_angle: distort the inner part of the torus by rotating grid vertices around the center by up to this amount. degrees.
    # z_sine_scale: distort the z-direction using a cosine scaling.
    # scaling: int or list of int. scale factor of the mesh in x- and y-direction after the torus was created.
    
    dims = 3 if z>0 else 2
    
    viscosity = torch.tensor([viscosity], dtype=dtype, device=cpu_device)
    domain = PISOtorch.Domain(dims, viscosity, name="DomainTorusParts", dtype=dtype, device=cuda_device, passiveScalarChannels=0)
    
    # coordinates are centered on the obstacle
    grid_rotation = rotation_distortion_max_angle is not None
    angle = rotation_distortion_max_angle
    grid_z_sine = z_sine_scale is not None
    z_scale = z_sine_scale
    
    # make the torus grids
    # torus generation: 
    # - angle starts at x-axis and goes counterclockwise
    # - grid: x goes along angle, y goes inner to outer
    # - coords: y is up
    # - return: torch.tensor with shape NCHW
    torus_tr_coords = shapes.make_torus_2D(x, r1=r1, r2=r2, start_angle=90, angle=-90, dtype=dtype) # y up, x right
    torus_br_coords = shapes.make_torus_2D(x, r1=r1, r2=r2, start_angle=0, angle=-90, dtype=dtype) # y right, x down
    torus_bl_coords = shapes.make_torus_2D(x, r1=r1, r2=r2, start_angle=-90, angle=-90, dtype=dtype) # y down, x left
    torus_tl_coords = shapes.make_torus_2D(x, r1=r1, r2=r2, start_angle=-180, angle=-90, dtype=dtype) # y left, x up
    
    
    torus_coords = [torus_tr_coords, torus_br_coords, torus_bl_coords, torus_tl_coords]
    y = torus_tr_coords.size(-2)-1
    
    if grid_rotation:
        # an option to distord the grid. experiment to check the effect of bad meshes on the simulation results
        distance_scaling = shapes.make_rotation_distance_scaling_fn_sine(angle, r1*1.1, r2*0.9)
        torus_coords = [shapes.rotate_grid(grid, angle=angle, distance_scaling=distance_scaling, center=(0,0)) for grid in torus_coords]
    
    if scaling is not None:
        if not isinstance(scaling, (list, tuple)):
            scaling = [scaling]*dims
        if not len(scaling)==dims:
            raise ValueError("Invalid scaling")
        LOG.info("Torus vortex street scaling: %s", scaling)
        
        scale_factor = torch.tensor(scaling, dtype=dtype).reshape([1,dims]+[1]*dims)
        
        torus_coords = [_*scale_factor for _ in torus_coords]
    
    if dims==3:
        # extrude the grid in z for 3D
        torus_coords = [shapes.extrude_grid_z(grid, z, end_z=z_size, weights_z=None, exp_base=1.05) for grid in torus_coords]
        if grid_z_sine:
            if z_closed:
                raise RuntimeError("z distortion requires periodic z boundaries")
            x_max = torch.max(torch.tensor([torch.max(grid[:,0]) for grid in torus_coords]))
            x_min = torch.min(torch.tensor([torch.min(grid[:,0]) for grid in torus_coords]))
            y_max = torch.max(torch.tensor([torch.max(grid[:,1]) for grid in torus_coords]))
            y_min = torch.min(torch.tensor([torch.min(grid[:,1]) for grid in torus_coords]))
            x_size = x_max - x_min
            y_size = y_max - y_min
            x_norm = (2*np.pi)/(x_size)
            y_norm = (2*np.pi)/(y_size)
            
            for grid in torus_coords:
                x_pos, y_pos, z_pos = torch.split(grid, 1, dim=1)
                z_offset_x = (torch.cos((x_pos - x_min)*x_norm)+1)*0.5*z_scale
                z_offset_y = (torch.cos((y_pos - y_min)*y_norm)+1)*0.5*z_scale
                
                grid[:,-1:] += z_offset_x - z_offset_y
    
    torus_coords = [_.to(cuda_device).contiguous() for _ in torus_coords]
    
    # make torus segment blocks
    for t_idx, coords in enumerate(torus_coords):
        block = domain.CreateBlock(vertexCoordinates=coords, name="Torus%d"%t_idx) #make_block_with_transform(transform, name="Torus%d"%t_idx, dtype=dtype)
        block.CloseAllBoundaries()
    
    torus_blocks = domain.getBlocks()
    torus_tr, torus_br, torus_bl, torus_tl = torus_blocks
    
    
    # connect segments
    axes = ["-y", "-z"] if dims==3 else ["-y"]
    for block_idx in range(len(torus_blocks)):
        torus_blocks[block_idx].ConnectBlock("+x", torus_blocks[(block_idx+1)%(len(torus_blocks))], "-x", *axes)
    
    if rot_vel!=0 or (dims==3 and obs_vel_z!=0):
        # rotating obstacle, done by prescribing a tangential velocity on the inner boundary
        x_total = x*len(torus_coords)
        angle = -360
        deg_step = angle/x_total
        start_angle = -90 + deg_step * 0.5
        rad_step = np.deg2rad(deg_step)
        start_rad = np.deg2rad(start_angle)

        def get_tangential_vel(i):
            vel = [-np.sin(start_rad + rad_step*i)*rot_vel, np.cos(start_rad + rad_step*i)*rot_vel]
            if dims==3:
                vel.append(obs_vel_z)
            return vel
        
        bound_vel = [get_tangential_vel(i) for i in range(x_total)]
        bound_vel = np.asarray(bound_vel) # WC
        bound_vel = np.moveaxis(bound_vel, -1, 0) # CW
        bound_vel = np.reshape(bound_vel, (1,dims,1,x_total))
        bound_vel = torch.tensor(bound_vel, device=cuda_device, dtype=dtype)
        if dims==3:
            bound_vel = bound_vel.reshape((1,dims,1,1,x_total)).repeat(1,1,z,1,1)
        bound_vel_parts = torch.split(bound_vel, x, dim=-1)

        for block, bound_vel_part in zip(torus_blocks, bound_vel_parts):
            block.getBoundary("-y").setVelocity(bound_vel_part.to(dtype).to(cuda_device).contiguous())
    
    if dims==3 and not z_closed:
        for block in torus_blocks:
            block.MakePeriodic("z")
    
    # initialize velocity
    if True:
        for block in torus_blocks:
            block.velocity[:,0,...] = vel_in
    
    # make inflow
    vel_in = torch.tensor([[vel_in] + [0]*(dims-1)], dtype=dtype, device=cuda_device) #NC
    torus_tl.getBoundary("+y").setVelocity(vel_in)
    torus_bl.getBoundary("+y").setVelocity(vel_in)

    # make outflow
    out_bounds = [torus_tr.getBoundary("+y"), torus_br.getBoundary("+y")]
    for out_bound in out_bounds:
        out_bound.setVelocity(vel_in)
        out_bound.makeVelocityVarying() # new tensor
        if domain.hasPassiveScalar():
            out_bound.CreatePassiveScalar(domain.getPassiveScalarChannels(), False)
    
    # make the boundaries consistent/divergence-free
    PISOtorch_simulation.balance_boundary_fluxes(domain, out_bounds)

    # callback function to update the outflow during the simulation
    prep_fn = lambda domain, time_step, **kwargs: PISOtorch_simulation.update_advective_boundaries(domain, out_bounds, vel_in, time_step.cuda())
    
    domain.PrepareSolve()
    
    resample_res_scale = 8
    resample_shape = [resample_res_scale*y,resample_res_scale*y]
    if dims==3:
        resample_shape.append(z)
    
    return domain, {"PRE": prep_fn}, resample_shape



RUN_ID = 0

def vortex_street_sample(log_dir, name, iterations=100, time_step=1, dp=False, STOP_FN=None, **domain_args):
    global RUN_ID
    if STOP_FN():
        return
    LOG = get_logger("VortexStreet")
    dtype = torch.float64 if dp else torch.float32
    
    x = domain_args["x"]
    res_z = domain_args.get("z", 0)

    LOG.info("%dD vortex street #%d '%s', base resolution %d, %d iterations with time step %.02e",
             3 if res_z>0 else 2, RUN_ID, name,
             x, iterations, time_step)
    

    domain, prep_fn, resample_shape = make4BlockTorus_VortexStreet(dtype=dtype, **domain_args)
    
    max_vel = domain.getMaxVelocity(True, True).cpu().numpy()
    LOG.info("Domain max vel: %s", max_vel)
    log_dir=os.path.join(log_dir, "%04d_vortex_street_%s"%(RUN_ID, name))
    
    grids = domain.getVertexCoordinates()
    if domain.getSpatialDims()==3:
        grids = [grid[:,:2,0,:,:] for grid in grids]
    #grid_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    grid_colors = ["tab:blue", "tab:blue", "tab:red", "tab:red"]
    plot_grids(grids, color=grid_colors[:len(grids)], path=log_dir, type="pdf", linewidth=1.0)
    
    mem_usage = MemoryUsage(logger=LOG)
    def mem_usage_log_fn(total_step, **kwargs):
        mem_usage.check_memory("step %04d"%(total_step,))
    
    if iterations>0:
        sim = PISOtorch_simulation.Simulation(domain=domain, output_resampling_shape=resample_shape, prep_fn=prep_fn,
                substeps="ADAPTIVE", time_step=time_step, corrector_steps=2, advection_tol=1e-6, pressure_tol=1e-6, # "ADAPTIVE"
                advect_non_ortho_steps=2, pressure_non_ortho_steps=4, pressure_return_best_result=True,
                velocity_corrector="FD", non_orthogonal=True,
                log_interval=1, norm_vel=True, log_fn=mem_usage_log_fn,
                log_dir=log_dir, save_domain_name="domain",
                output_resampling_fill_max_steps=16,
                stop_fn=STOP_FN)
                
        sim.preconditionBiCG = False
        sim.BiCG_precondition_fallback = True
        
        # run a pressure projection step on the domain to create a velocity field that matches the boundaries
        sim.make_divergence_free()
        
        sim.run(iterations)
    
    mem_usage.print_max_memory()
    
    RUN_ID += 1


if __name__=="__main__":
    # the topology is actually circular or cylindical, but the name 'torus' stuck...
    # create a new directory <time-step>_<name> in ./test_runs to use as base output directory
    run_dir = setup_run("./test_runs",
        name="vortex-street-torus-sample_rot2_fill-16_plot-grid"
    )
    LOG = get_logger("Main")
    stop_handler = PISOtorch_simulation.StopHandler(LOG)
    stop_handler.register_signal()


    # 2D example
    vortex_street_sample(run_dir, name="2D_r32_rot-2_it200", iterations=200, time_step=0.25, dp=False, STOP_FN=stop_handler,
         x=32, rot_vel=-2)
    
    # 3D example
    #vortex_street_sample(run_dir, name="3D_r32-z32-zClosed_rot2_it200_visc1e-3", iterations=200, time_step=0.25, dp=False, STOP_FN=stop_handler,
    #   x=32, z=32, rot_vel=2, viscosity=1e-3, z_closed=True)

    stop_handler.unregister_signal()
    close_logging()