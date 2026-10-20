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
import lib.util.domain_io as domain_io
from lib.util.output import plot_grids

# PISOtorch is only implemented for GPU

assert torch.cuda.is_available()
cuda_device = torch.device("cuda")
cpu_device = torch.device("cpu")

# Flow around a cylinder in a channel, the 2D benchmark setup of Schäfer & Turek (1996) scaled to a cylinder diameter of 1.
# The cylinder sits slightly below the center of the channel, which triggers the vortex shedding without a perturbation.
# For 3D the grid is extruded in span-wise direction with periodic boundaries.
# Setup adapted from the code of "FluidGym" (Becktepe et al., ICML 2026), where it is the uncontrolled flow of the cylinder environments.
# See vortex_street_torus_sample.py for a cylinder in an open domain.

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

def make_cylinder_channel_grids(res:int, cylinder_radius:float=0.5, cylinder_offset_y:float=0.05, ring_thickness:float=0.5, quad_thickness:float=1.0,
        channel_length:float=22.0, wake_refinement_base:float=0.95, dtype=torch.float32):
    """
    4 grids form a ring around the cylinder (O-grid), a 5th grid covers the wake. The cylinder is centered on (0,0).
    res: number of cells along a quarter of the cylinder surface, the resolution of the other parts follows from it.
    cylinder_offset_y: the cylinder is closer to the lower wall by 2x this distance.
    ring_thickness: thickness of the circular inner part of the ring grids.
    quad_thickness: distance from the circular part to the inflow and to the wake grid.
        The channel height is 2*(cylinder_radius + ring_thickness + quad_thickness + cylinder_offset_y).
    channel_length: total length of the channel, starting at the inflow.
    wake_refinement_base: the cells of the wake grid shrink by this factor from the walls towards the center line.

    x-axis goes left to right
    y-axis goes bottom to top
    """
    r_ring = cylinder_radius + ring_thickness # outer radius of the circular part
    quad_thickness_y = quad_thickness + cylinder_offset_y
    x_quad = r_ring + quad_thickness # inflow at -x_quad, wake grid from x_quad
    x_out = channel_length - x_quad # outflow
    y_top = r_ring + quad_thickness_y + cylinder_offset_y # upper wall
    y_bot = -(r_ring + quad_thickness_y - cylinder_offset_y) # lower wall

    # circular part, a quarter ring per grid
    # torus generation: angles in degree start at the x-axis and go counterclockwise, grid x goes along the angle, grid y from inner to outer radius
    ring_top = shapes.make_torus_2D(res, r1=cylinder_radius, r2=r_ring, start_angle=135, angle=-90, dtype=dtype) # y up, x right
    ring_right = shapes.make_torus_2D(res, r1=cylinder_radius, r2=r_ring, start_angle=45, angle=-90, dtype=dtype) # y right, x down
    ring_right = torch.flip(torch.movedim(ring_right, -1, -2), (-2,)) # y up, x right
    ring_bot = shapes.make_torus_2D(res, r1=cylinder_radius, r2=r_ring, start_angle=-45, angle=-90, dtype=dtype) # y down, x left
    ring_bot = torch.flip(ring_bot, (-2, -1)) # y up, x right
    ring_left = shapes.make_torus_2D(res, r1=cylinder_radius, r2=r_ring, start_angle=-135, angle=-90, dtype=dtype) # y left, x up
    ring_left = torch.flip(torch.movedim(ring_left, -1, -2), (-1,)) # y up, x right
    res_ring = ring_top.size(-2) - 1 # cells in radial direction

    # outer part from the circle to the walls, the inflow and the wake grid
    # corner order: (-x,-y), (+x,-y), (-x,+y), (+x,+y). the edges towards the circle follow the outer border of the circular part, the others are straight.
    c = np.sin(np.deg2rad(45))*r_ring # corners on the circle
    res_quad = int(np.ceil(quad_thickness_y/ring_thickness*res_ring)) # vertices in radial direction
    def make_border(t):
        return torch.movedim(t, 0, 1).cpu().clone().numpy().tolist() # CW -> list of (x,y)
    # border order: -x, +x, -y, +y
    quad_top = shapes.generate_grid_vertices_2D([res_quad, res+1], [(-c, c), (c, c), (-x_quad, y_top), (x_quad, y_top)],
        [None, None, make_border(ring_top[0,:,-1,:]), None], dtype=dtype)
    quad_bot = shapes.generate_grid_vertices_2D([res_quad, res+1], [(-x_quad, y_bot), (x_quad, y_bot), (-c, -c), (c, -c)],
        [None, None, None, make_border(ring_bot[0,:,0,:])], dtype=dtype)
    quad_left = shapes.generate_grid_vertices_2D([res+1, res_quad], [(-x_quad, y_bot), (-c, -c), (-x_quad, y_top), (-c, c)],
        [None, make_border(ring_left[0,:,:,0]), None, None], dtype=dtype)
    # the right part is refined like the wake grid s.t. the cells match at the connection
    wake_weights = shapes.make_weights_exp(res=res, base=wake_refinement_base, refinement="BOTH")
    quad_right = shapes.generate_grid_vertices_2D([res+1, res_quad], [(c, -c), (x_quad, y_bot), (c, c), (x_quad, y_top)],
        [make_border(ring_right[0,:,:,-1]), None, None, None], x_weights=wake_weights, dtype=dtype)

    # merge circular and outer parts, the shared vertices are only kept once
    grid_left = torch.cat([quad_left[...,:-1], ring_left], dim=-1)
    grid_top = torch.cat([ring_top[...,:-1,:], quad_top], dim=-2)
    grid_right = torch.cat([ring_right[...,:-1], quad_right], dim=-1)
    grid_bot = torch.cat([quad_bot[...,:-1,:], ring_bot], dim=-2)

    # wake, cells are about as wide as the radial cells of the outer part
    res_wake = int(res_quad/quad_thickness_y*(x_out - x_quad))
    grid_wake = shapes.make_wall_refined_ortho_grid(res_wake, res, corner_lower=(x_quad, y_bot), corner_upper=(x_out, y_top),
        wall_refinement=["-y", "+y"], base=wake_refinement_base, dtype=dtype)

    return [grid_left, grid_top, grid_right, grid_bot, grid_wake]

def make5BlockCylinderChannelSetup(grids, in_vel:float=1.0, viscosity:float=0.0, res_z:int=0, depth:float=4.0, dtype=torch.float32):
    # grids: the 2D coordinate grids to make the blocks from, as generated by make_cylinder_channel_grids
    # in_vel: mean velocity of the parabolic inflow profile
    # viscosity: viscosity of the flow
    # res_z: number of cells in span-wise direction for a 3D setup, 0 for 2D.
    #   The 2D grids are extruded from z=-depth/2 to z=depth/2 and the span-wise boundaries are periodic.
    dims = 3 if res_z>0 else 2
    assert len(grids)==5
    y_bot = grids[0][0,1,0,0].item()
    H = grids[0][0,1,-1,0].item() - y_bot
    inflow_y = grids[0][0,1,:,0]
    outflow_y = grids[4][0,1,:,-1]
    if dims==3:
        grids = [shapes.extrude_grid_z(grid, res_z, start_z=-depth/2, end_z=depth/2) for grid in grids]
    Gl, Gt, Gr, Gb, Gw = grids

    viscosity = torch.ones([1], dtype=dtype, device=cpu_device)*viscosity
    domain = PISOtorch.Domain(dims, viscosity, passiveScalarChannels=0, name="Domain5BlockCylinderChannel", device=cuda_device, dtype=dtype)

    # make the blocks
    Bl = domain.CreateBlock(vertexCoordinates=Gl.to(cuda_device), name="BlockCylinderLeft")
    Bt = domain.CreateBlock(vertexCoordinates=Gt.to(cuda_device), name="BlockCylinderTop")
    Br = domain.CreateBlock(vertexCoordinates=Gr.to(cuda_device), name="BlockCylinderRight")
    Bb = domain.CreateBlock(vertexCoordinates=Gb.to(cuda_device), name="BlockCylinderBottom")
    Bw = domain.CreateBlock(vertexCoordinates=Gw.to(cuda_device), name="BlockWake")

    layout = [[-1,3,-1,-1],[0,-1,2,4],[-1,1,-1,-1]] # used by output formatting

    # set boundaries
    # cylinder
    Bl.CloseBoundary("+x")
    Bt.CloseBoundary("-y")
    Br.CloseBoundary("-x")
    Bb.CloseBoundary("+y")
    # channel walls
    Bt.CloseBoundary("+y")
    Bb.CloseBoundary("-y")
    Bw.CloseBoundary("-y")
    Bw.CloseBoundary("+y")
    if dims==3:
        for block in [Bl, Bt, Br, Bb, Bw]:
            block.MakePeriodic("z")

    # connect the ring around the cylinder and the wake
    # parameters are: block_from, side of block_from to connect, block_to, side of block_to to connect, axes
    # for 3D the axes are given for both remaining dimensions, in the order (y,z) for x-faces and (z,x) for y-faces
    if dims==2:
        Bl.ConnectBlock("+y", Bt, "-x", "+y")
        Bl.ConnectBlock("-y", Bb, "-x", "-y")
        Br.ConnectBlock("+y", Bt, "+x", "-y")
        Br.ConnectBlock("-y", Bb, "+x", "+y")
        Br.ConnectBlock("+x", Bw, "-x", "-y")
    else:
        Bl.ConnectBlock("+y", Bt, "-x", "-z", "+y")
        Bl.ConnectBlock("-y", Bb, "-x", "-z", "-y")
        Br.ConnectBlock("+y", Bt, "+x", "-z", "-y")
        Br.ConnectBlock("-y", Bb, "+x", "-z", "+y")
        Br.ConnectBlock("+x", Bw, "-x", "-y", "-z")

    def parabolic_profile(y):
        # parabolic (Poiseuille) profile over the full channel height with mean velocity in_vel, sampled at the face centers of a boundary.
        # y: vertex y-coordinates of the boundary (the cell centers are shifted towards the cylinder in the curved grid of the left block)
        y = 0.5*(y[1:] + y[:-1]) - y_bot
        u = 6*in_vel*(y/H)*(1 - y/H)
        if dims==2:
            u = u.reshape([1,1,-1,1])
        else:
            u = u.reshape([1,1,1,-1,1]).repeat(1,1,res_z,1,1)
        return torch.cat([u] + [torch.zeros_like(u)]*(dims-1), dim=1).to(cuda_device).contiguous()

    # specify the inflow
    Bl.CloseBoundary("-x", parabolic_profile(inflow_y))

    # also set up the outflow boundary at the end of the channel.
    # this will need special treatment during the simulation
    Bw.CloseBoundary("+x", parabolic_profile(outflow_y))
    out_bound = Bw.getBoundary("+x")
    out_bound.makeVelocityVarying()

    # make the boundaries consistent/divergence-free
    PISOtorch_simulation.balance_boundary_fluxes(domain, [out_bound])

    # callback function to update the outflow during the simulation
    char_vel = torch.tensor([[in_vel] + [0]*(dims-1)], device=cuda_device, dtype=dtype) # NC
    out_block_idx = domain.getBlocks().index(Bw)
    def prep_fn(domain, time_step, **kwargs):
        out_bounds = [domain.getBlock(out_block_idx).getBoundary("+x")]
        PISOtorch_simulation.update_advective_boundaries(domain, out_bounds, char_vel, time_step.cuda())

    domain.PrepareSolve()

    return domain, {"PRE": prep_fn}, layout

def set_velocity_from_2D(domain, domain_2d, noise:float=0.0, seed:int=None):
    # Initialize a 3D domain with the velocity of a 2D domain of the same setup, copied to every span-wise position.
    # The varying boundary velocities (outflow) are copied as well, an outflow that does not match the flow breaks the span-wise symmetry.
    # noise: standard deviation of random velocity added to break the span-wise symmetry
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    for block, block_2d in zip(domain.getBlocks(), domain_2d.getBlocks()):
        velocity = torch.zeros_like(block.velocity) # NCDHW
        velocity[:,:2] = block_2d.velocity.unsqueeze(2)
        velocity += noise*torch.randn(velocity.shape, generator=generator, dtype=velocity.dtype).to(velocity.device)
        block.setVelocity(velocity.contiguous())
        for side in range(4): # x- and y-faces
            bound, bound_2d = block.getBoundary(side), block_2d.getBoundary(side)
            if isinstance(bound, PISOtorch.FixedBoundary) and not bound.isVelocityStatic:
                bound_velocity = torch.zeros_like(bound.velocity)
                bound_velocity[:,:2] = bound_2d.velocity.unsqueeze(2)
                bound.setVelocity(bound_velocity.contiguous())
    domain.UpdateDomainData()

RUN_ID = 0

def cylinder_channel_sample(log_dir, name, iterations=100, time_step=0.1, res=32, Re=100, res_z=0, load_2d_domain_path=None, dp=False, STOP_FN=None):
    # res: number of cells along a quarter of the cylinder surface. FluidGym uses 24 for Re=100 and 32 for Re 250 and 500.
    # Re: Reynolds number based on the cylinder diameter and the mean inflow velocity
    # res_z: number of span-wise cells for a 3D simulation with a span of 4 (FluidGym's 3D cylinder environments use res), 0 for 2D
    # load_2d_domain_path: 3D only. Start from the developed flow of a 2D simulation of the same setup, e.g. '<run_dir>/domain'
    #   of a previous 2D run. It is copied to every span-wise position, with random noise to break the span-wise symmetry.
    global RUN_ID
    if STOP_FN():
        return
    LOG = get_logger("CylinderChannel")
    dtype = torch.float64 if dp else torch.float32

    diameter = 1.0
    depth = 4.0 # span of the 3D setup
    in_vel = 1.0
    viscosity = in_vel*diameter/Re

    LOG.info("%dD cylinder channel #%d '%s' with resolution %d, Re %.02e (viscosity %.02e), %d iterations with time step %.02e",
             3 if res_z>0 else 2, RUN_ID, name, res, Re, viscosity, iterations, time_step)
    log_dir = os.path.join(log_dir, "%04d_cylinder_channel_%s"%(RUN_ID, name))

    grids = make_cylinder_channel_grids(res, cylinder_radius=diameter/2, cylinder_offset_y=0.05, ring_thickness=diameter/2, quad_thickness=diameter,
        channel_length=22.0, wake_refinement_base=0.95, dtype=dtype)
    plot_grids(grids, color=["r","b","y","g","k"], path=log_dir, linewidth=0.5)

    # resample with 4*res pixels over the channel height, 2*res in 3D
    channel_height = (grids[0][0,1,-1,0] - grids[0][0,1,0,0]).item()
    pixel_size = channel_height/((4 if res_z==0 else 2)*res)
    resample_shape, resample_fill_steps = get_resampling_settings(grids, pixel_size=pixel_size)
    if res_z>0:
        resample_shape.append(int(np.ceil(depth/pixel_size)))
    LOG.info("%d cells, output resampling shape %s with up to %d fill steps", sum((g.size(-1)-1)*(g.size(-2)-1) for g in grids)*max(res_z,1),
        resample_shape, resample_fill_steps)

    domain, prep_fn, layout = make5BlockCylinderChannelSetup(grids, in_vel=in_vel, viscosity=viscosity, res_z=res_z, depth=depth, dtype=dtype)
    if load_2d_domain_path:
        LOG.info("velocity initialized from the 2D domain '%s'", load_2d_domain_path)
        domain_2d = domain_io.load_domain(load_2d_domain_path, dtype=dtype, device=cuda_device)
        set_velocity_from_2D(domain, domain_2d, noise=0.025, seed=0) # noise as in the initialization of FluidGym's cylinder environments
        del domain_2d

    max_vel = domain.getMaxVelocity(True, True).cpu().numpy()
    LOG.info("Domain max vel: %s", max_vel)

    sim = PISOtorch_simulation.Simulation(domain=domain, block_layout=layout, prep_fn=None,
            substeps="ADAPTIVE", time_step=time_step, corrector_steps=2, pressure_tol=1e-5,
            advect_non_ortho_steps=1, pressure_non_ortho_steps=1, pressure_return_best_result=True,
            velocity_corrector="FD", non_orthogonal=True,
            log_interval=1, norm_vel=True,
            output_resampling_shape=resample_shape, output_resampling_fill_max_steps=resample_fill_steps,
            log_dir=log_dir, save_domain_name="domain",
            stop_fn=STOP_FN)

    sim.solver_double_fallback = True
    sim.preconditionBiCG = False
    sim.BiCG_precondition_fallback = True

    # initialize the domain with velocity that matches in- and outflow
    sim.make_divergence_free()

    # set the outflow update only after the initial projection, which uses a time step of 1
    sim.prep_fn = prep_fn
    sim.run(iterations)

    RUN_ID += 1


if __name__=="__main__":
    # create a new directory <time-step>_<name> in ./test_runs to use as base output directory
    run_dir = setup_run("./test_runs",
        name="cylinder-channel-sample"
    )
    LOG = get_logger("Main")
    stop_handler = PISOtorch_simulation.StopHandler(LOG)
    stop_handler.register_signal()

    # the vortex shedding starts after ca. 20 and is fully developed after ca. 45 time units
    cylinder_channel_sample(run_dir, name="Re100_r32", res=32, Re=100, iterations=1000, time_step=0.1, dp=False, STOP_FN=stop_handler)

    # higher Reynolds numbers, as in the medium and hard cylinder environments of FluidGym
    #cylinder_channel_sample(run_dir, name="Re250_r32", res=32, Re=250, iterations=1000, time_step=0.1, dp=False, STOP_FN=stop_handler)
    #cylinder_channel_sample(run_dir, name="Re500_r32", res=32, Re=500, iterations=1000, time_step=0.1, dp=False, STOP_FN=stop_handler)

    # 3D with 32 cells over a periodic span of 4 (0.75M cells), as in the medium 3D cylinder environment of FluidGym,
    # starting from the developed 2D flow of the Re=250 run above (uncomment it). At Re=100 the wake stays two-dimensional.
    #cylinder_channel_sample(run_dir, name="Re250_r32_3D", res=32, Re=250, res_z=32,
    #    load_2d_domain_path=os.path.join(run_dir, "0001_cylinder_channel_Re250_r32", "domain"), iterations=400, time_step=0.2, dp=False, STOP_FN=stop_handler)

    stop_handler.unregister_signal()
    close_logging()
