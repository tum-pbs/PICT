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

# Flow around a NACA 0012 airfoil at an angle of attack in a channel, the flow separates from the upper side and sheds vortices.
# The grid consists of 6 blocks: in front of the airfoil, around it (front, top, bottom) and two blocks for the wake.
# For 3D the grid is extruded in span-wise direction with periodic boundaries.
# Setup adapted from the code of "FluidGym" (Becktepe et al., ICML 2026), where it is the uncontrolled flow of the airfoil environments.
# See airfoil_sample.py for an airfoil on a C-grid in an open domain.

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

# read airfoil geometry from file and rotate it around the leading edge
def read_airfoil(path, attack_angle:float=0, dtype=torch.float32):
    # attack_angle: angle of attack in degrees, positive values turn the trailing edge down
    coords = []
    with open(path, "r") as file:
        name = file.readline().strip()
        for line in file:
            x,y = line.split()
            coords.append((float(x), float(y)))

    coords = torch.tensor(coords, device=cpu_device, dtype=dtype) #WC
    if attack_angle!=0:
        angle = torch.tensor(np.deg2rad(-attack_angle), dtype=dtype)
        rotation = torch.tensor([[torch.cos(angle), -torch.sin(angle)], [torch.sin(angle), torch.cos(angle)]], dtype=dtype)
        coords = torch.matmul(coords, rotation.T)
    coords = torch.movedim(coords, 1,0) #CW
    coords = torch.reshape(coords, (1,2,1,-1)) #NCHW

    return name, coords

def distance_to_point(o, d, p):
    # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    # o, d: origin and direction of line, NCHW
    # p: point, NCHW
    p1 = o
    p2 = o + d

    d1 = p2[:,0] - p1[:,0]
    d2 = p2[:,1] - p1[:,1]

    a = torch.abs(d1*(p1[:,1] - p[:,1]) - (p1[:,0] - p[:,0])*d2)
    b = torch.sqrt(d1*d1 + d2*d2)

    distance = (a/b).reshape((1,1,o.size(-2),o.size(-1)))

    return distance

def ray_rectangle_intersection(normals, half_height, width_left, attack_angle):
    # Distributes the points of the airfoil's front part on the upper, left and lower side of a rectangle around it (the outer boundary of the front grid).
    # The points are assigned by the angle of their surface normal, the points with the normals closest to the rectangle's corners become the corners.
    # normals: surface normals of the airfoil points, NCHW
    # returns: points on the rectangle (NCHW), indices of the upper and lower corner
    device = normals.device
    normal_angles = torch.atan2(normals[:,1], normals[:,0])

    # angles in [-90, 90] with 0 pointing left
    normal_angles = 180 - normal_angles*180.0/np.pi
    normal_angles = torch.where(normal_angles<180, normal_angles, -360 + normal_angles)
    normal_angles -= attack_angle

    corner_angle = np.rad2deg(np.arctan2(half_height, width_left))

    upper_idxs = (normal_angles>0).squeeze()
    angles_upper = normal_angles[0,0,upper_idxs]
    angles_lower = normal_angles[0,0,~upper_idxs]

    closest_idx_top = torch.argmin(torch.abs(angles_upper - corner_angle), dim=-1)
    closest_idx_bot = torch.argmin(torch.abs(angles_lower + corner_angle), dim=-1)

    # upper side and upper part of the left side
    n_fill_x_top = angles_upper.shape[-1] - closest_idx_top - 1
    x_upper = torch.cat([
        torch.linspace(0, -width_left, int(angles_upper.shape[-1] - n_fill_x_top + 1), device=device)[1:-1],
        torch.tensor([-width_left]*(n_fill_x_top + 1), device=device)], dim=0)
    x_fill_y_top = angles_upper.shape[-1] - n_fill_x_top
    y_upper = torch.cat([
        torch.tensor([half_height]*x_fill_y_top, device=device),
        torch.linspace(half_height, 0, int(angles_upper.shape[-1] - x_fill_y_top + 2), device=device)[1:-1]], dim=0)

    # lower part of the left side and lower side
    n_fill_x_bot = closest_idx_bot
    x_lower = torch.cat([
        torch.tensor([-width_left]*(n_fill_x_bot + 1), device=device),
        torch.linspace(-width_left, 0, int(angles_lower.shape[-1] - closest_idx_bot + 1), device=device)[1:-1]], dim=0)
    y_lower = torch.cat([
        torch.linspace(0, -half_height, int(n_fill_x_bot + 1), device=device)[:-1],
        torch.tensor([-half_height]*(angles_lower.shape[-1] - n_fill_x_bot), device=device)], dim=0)

    intersection = torch.cat([torch.stack([x_upper, y_upper], dim=0), torch.stack([x_lower, y_lower], dim=0)], dim=-1).unsqueeze(0).unsqueeze(2)

    return intersection, closest_idx_top, x_upper.shape[-1] + closest_idx_bot

def make_airfoil_channel_grids(airfoil_path:str, attack_angle:float=10, channel_height:float=1.4, inflow_x:float=-1.5, outflow_x:float=4.5,
        front_x:float=-0.5, res_normal:int=96, normal_base:float=0.97, tail_grow_mul:float=1.01, dtype=torch.float32):
    """
    6 grids: left (between inflow and front grid), around the airfoil (front, top, bottom), and two for the wake (upper, lower).
    The leading edge of the airfoil is at (0,0), the channel walls at y=-channel_height/2 and y=channel_height/2.
    attack_angle: angle of attack in degrees.
    inflow_x, outflow_x: x-position of in- and outflow.
    front_x: x-position of the boundary between left and front grid.
    res_normal: number of vertices in airfoil-normal direction, the cells are refined towards the airfoil.
    normal_base: cell size factor between neighboring cells in airfoil-normal direction.
    tail_grow_mul: cell size factor between neighboring cells in x-direction in the wake grids.

    x-axis goes left to right
    y-axis goes bottom to top
    """
    half_height = channel_height/2
    front_width = -front_x
    normal_weights = shapes.make_weights_exp(res=res_normal - 1, base=normal_base, refinement="START")
    normal_weights_reversed = shapes.make_weights_exp(res=res_normal - 1, base=normal_base, refinement="END")

    # airfoil coordinates start at the trailing edge and go counterclockwise
    _, airfoil_coords = read_airfoil(airfoil_path, attack_angle=attack_angle, dtype=dtype) #NCHW
    airfoil_res = airfoil_coords.size(-1)
    airfoil_len_x = torch.max(airfoil_coords[:,0])
    airfoil_end = airfoil_coords[:,:,:,:1]

    point_start_top = torch.tensor([0, half_height], dtype=dtype).reshape((1,2,1,1))
    point_end_top = torch.tensor([airfoil_len_x, half_height], dtype=dtype).reshape((1,2,1,1))
    point_start_bot = torch.tensor([0, -half_height], dtype=dtype).reshape((1,2,1,1))
    point_end_bot = torch.tensor([airfoil_len_x, -half_height], dtype=dtype).reshape((1,2,1,1))

    # surface normals of the airfoil points
    airfoil_end_spacing = torch.linalg.vector_norm(airfoil_coords[0,:,0,1] - airfoil_coords[0,:,0,0])
    airfoil_end_extend = airfoil_end + torch.tensor([airfoil_end_spacing, 0], dtype=dtype).reshape((1,2,1,1))
    airfoil_coords_extended = torch.cat([airfoil_end_extend, airfoil_coords, airfoil_end_extend], dim=-1)
    airfoil_spacing = airfoil_coords_extended[0,:,0,2:] - airfoil_coords_extended[0,:,0,:-2] # CW
    airfoil_normals = torch.flip(airfoil_spacing, dims=(0,)) * torch.tensor([1, -1], dtype=dtype).reshape((2,1))
    airfoil_normals /= torch.linalg.vector_norm(airfoil_normals, dim=0)
    airfoil_normals = torch.reshape(airfoil_normals, (1,2,1,-1))

    # the wake grids start with the smallest spacing of the airfoil points and grow exponentially
    airfoil_spacing = airfoil_coords_extended[0,:,0,1:] - airfoil_coords_extended[0,:,0,:-1] # CW
    airfoil_spacing = torch.linalg.vector_norm(airfoil_spacing, dim=0) # W
    min_size = torch.min(airfoil_spacing).numpy().tolist()
    tail_sizes = [min_size]
    tail_dist = min_size
    while tail_dist<half_height:
        size = tail_sizes[-1]*tail_grow_mul
        tail_sizes.append(size)
        tail_dist = tail_dist + size
    tail_weights = [0] + (np.cumsum(tail_sizes) / tail_dist).tolist()
    tail_res_x = len(tail_weights)

    # split the airfoil surface into the parts of the top, front and bottom grids.
    # each airfoil point gets a point on the outer boundary: on the upper and lower wall, and on a rectangle in front of the airfoil
    distances_top = distance_to_point(airfoil_coords[...,:airfoil_res//2], airfoil_normals[...,:airfoil_res//2], point_start_top)
    min_d_top_idx = torch.min(distances_top[0,0,0,:], dim=-1)[1]
    distances_bot = distance_to_point(airfoil_coords[...,airfoil_res//2:], airfoil_normals[...,airfoil_res//2:], point_start_bot)
    min_d_bot_idx = torch.min(distances_bot[0,0,0,:], dim=-1)[1] + airfoil_res//2

    # upper outer boundary, right to left
    steps_top = min_d_top_idx
    step_top = (point_start_top - point_end_top) / steps_top
    coords_top = torch.cat([point_end_top + step_top*i for i in range(steps_top+1)], dim=-1)

    # lower outer boundary, left to right
    steps_bot = (airfoil_res-1) - min_d_bot_idx
    step_bot = (point_end_bot - point_start_bot) / steps_bot
    coords_bot = torch.cat([point_start_bot + step_bot*i for i in range(steps_bot+1)], dim=-1)

    # front outer boundary
    coords_front, upper_corner_idx, lower_corner_idx = ray_rectangle_intersection(airfoil_normals[...,min_d_top_idx+1:min_d_bot_idx],
        half_height=half_height, width_left=front_width, attack_angle=attack_angle)
    coords_outer = torch.cat([coords_top, coords_front, coords_bot], dim=-1)

    # move the corners of the front grid further around the leading edge
    upper_corner_idx += 7
    lower_corner_idx += 7

    top_slice = slice(0, coords_bot.shape[-1] + upper_corner_idx + 3)
    front_slice = slice(coords_bot.shape[-1] + upper_corner_idx + 2, coords_bot.shape[-1] + lower_corner_idx + 3)
    bot_slice = slice(coords_bot.shape[-1] + lower_corner_idx + 2, None)

    # top part: flip from right-to-left to left-to-right
    airfoil_coords_top = torch.flip(airfoil_coords[...,top_slice], dims=(-1,))
    airfoil_res_top = coords_outer[...,top_slice].shape[-1]
    # front part: from right-to-left (along x) to top-to-bottom (along y)
    airfoil_coords_front = torch.flip(torch.movedim(airfoil_coords[...,front_slice], -1, -2), dims=(-2,))
    airfoil_res_front = coords_outer[...,front_slice].shape[-1]
    # bottom part, already left-to-right
    airfoil_coords_bot = airfoil_coords[...,bot_slice]
    airfoil_res_bot = coords_outer[...,bot_slice].shape[-1]

    start_top = airfoil_coords_top[0,:,0,0].tolist()
    start_bot = airfoil_coords_bot[0,:,0,0].tolist()
    end_top = airfoil_coords_top[0,:,0,-1].tolist()
    end_bot = airfoil_coords_bot[0,:,0,-1].tolist()

    def make_border(t):
        return torch.movedim(t, 0, 1).cpu().clone().numpy().tolist() # CW -> list of (x,y)

    # corner order: (-x,-y), (+x,-y), (-x,+y), (+x,+y); border order: -x, +x, -y, +y.
    # only the borders on the airfoil are given, the others are straight lines between the corners.
    grid_left = shapes.generate_grid_vertices_2D([airfoil_res_front, int(0.75*res_normal)],
        [(inflow_x, -half_height), (front_x, -half_height), (inflow_x, half_height), (front_x, half_height)], None, dtype=dtype)
    grid_top = shapes.generate_grid_vertices_2D([res_normal, airfoil_res_top],
        [tuple(start_top), tuple(end_top), (front_x, half_height), (end_top[0], half_height)],
        [None, None, make_border(airfoil_coords_top[0,:,0,:]), None], x_weights=normal_weights_reversed, dtype=dtype)
    grid_front = shapes.generate_grid_vertices_2D([airfoil_res_front, res_normal],
        [(front_x, -half_height), tuple(start_bot), (front_x, half_height), tuple(start_top)],
        [None, make_border(airfoil_coords_front[0,:,:,0]), None, None], y_weights=normal_weights, dtype=dtype)
    grid_bot = shapes.generate_grid_vertices_2D([res_normal, airfoil_res_bot],
        [(front_x, -half_height), (end_bot[0], -half_height), tuple(start_bot), tuple(end_bot)],
        [None, None, None, make_border(airfoil_coords_bot[0,:,0,:])], x_weights=normal_weights, dtype=dtype)
    grid_tail_upper = shapes.generate_grid_vertices_2D([res_normal, tail_res_x],
        [tuple(end_top), (outflow_x, end_top[1]), (end_top[0], half_height), (outflow_x, half_height)], None,
        x_weights=normal_weights_reversed, y_weights=tail_weights, dtype=dtype)
    grid_tail_lower = shapes.generate_grid_vertices_2D([res_normal, tail_res_x],
        [(end_bot[0], -half_height), (outflow_x, -half_height), tuple(end_bot), (outflow_x, end_bot[1])], None,
        x_weights=normal_weights, y_weights=tail_weights, dtype=dtype)

    return [grid_left, grid_front, grid_top, grid_bot, grid_tail_upper, grid_tail_lower]

def make6BlockAirfoilChannelSetup(grids, in_vel:float=1.0, viscosity:float=0.0, res_z:int=0, depth:float=1.4, dtype=torch.float32):
    # grids: the 2D coordinate grids to make the blocks from, as generated by make_airfoil_channel_grids
    # in_vel: mean velocity of the parabolic inflow profile
    # viscosity: viscosity of the flow
    # res_z: number of cells in span-wise direction for a 3D setup, 0 for 2D.
    #   The 2D grids are extruded from z=-depth/2 to z=depth/2 and the span-wise boundaries are periodic.
    dims = 3 if res_z>0 else 2
    assert len(grids)==6
    y_bot = grids[0][0,1,0,0].item()
    H = grids[0][0,1,-1,0].item() - y_bot
    inflow_y = grids[0][0,1,:,0]
    if dims==3:
        grids = [shapes.extrude_grid_z(grid, res_z, start_z=-depth/2, end_z=depth/2) for grid in grids]
    Gl, Gf, Gt, Gb, Gtu, Gtl = grids

    viscosity = torch.ones([1], dtype=dtype, device=cpu_device)*viscosity
    domain = PISOtorch.Domain(dims, viscosity, passiveScalarChannels=0, name="Domain6BlockAirfoilChannel", device=cuda_device, dtype=dtype)

    # make the blocks
    Bl = domain.CreateBlock(vertexCoordinates=Gl.to(cuda_device), name="BlockLeft")
    Bf = domain.CreateBlock(vertexCoordinates=Gf.to(cuda_device), name="BlockAirfoilFront")
    Bt = domain.CreateBlock(vertexCoordinates=Gt.to(cuda_device), name="BlockAirfoilTop")
    Bb = domain.CreateBlock(vertexCoordinates=Gb.to(cuda_device), name="BlockAirfoilBottom")
    Btu = domain.CreateBlock(vertexCoordinates=Gtu.to(cuda_device), name="BlockTailUpper")
    Btl = domain.CreateBlock(vertexCoordinates=Gtl.to(cuda_device), name="BlockTailLower")

    layout = [[-1,3,5],[0,1,-1],[-1,2,4]] # used by output formatting

    # set boundaries
    # airfoil
    Bf.CloseBoundary("+x")
    Bt.CloseBoundary("-y")
    Bb.CloseBoundary("+y")
    # channel walls
    Bl.CloseBoundary("-y")
    Bl.CloseBoundary("+y")
    Bt.CloseBoundary("+y")
    Btu.CloseBoundary("+y")
    Btl.CloseBoundary("-y")
    if dims==3:
        for block in [Bl, Bf, Bt, Bb, Btu, Btl]:
            block.MakePeriodic("z")

    # connect the blocks
    # parameters are: block_from, side of block_from to connect, block_to, side of block_to to connect, axes
    # for 3D the axes are given for both remaining dimensions, in the order (y,z) for x-faces and (z,x) for y-faces
    if dims==2:
        Bl.ConnectBlock("+x", Bf, "-x", "-y")
        Bf.ConnectBlock("+y", Bt, "-x", "+y")
        Bf.ConnectBlock("-y", Bb, "-x", "-y")
        Bt.ConnectBlock("+x", Btu, "-x", "-y")
        Bb.ConnectBlock("+x", Btl, "-x", "-y")
        Btu.ConnectBlock("-y", Btl, "+y", "-x")
    else:
        Bl.ConnectBlock("+x", Bf, "-x", "-y", "-z")
        Bf.ConnectBlock("+y", Bt, "-x", "-z", "+y")
        Bf.ConnectBlock("-y", Bb, "-x", "-z", "-y")
        Bt.ConnectBlock("+x", Btu, "-x", "-y", "-z")
        Bb.ConnectBlock("+x", Btl, "-x", "-y", "-z")
        Btu.ConnectBlock("-y", Btl, "+y", "-z", "-x")

    # specify the inflow
    # parabolic (Poiseuille) profile with mean velocity in_vel, sampled at the face centers
    y = 0.5*(inflow_y[1:] + inflow_y[:-1]) - y_bot
    inflow = 6*in_vel*(y/H)*(1 - y/H)
    if dims==2:
        inflow = inflow.reshape([1,1,-1,1])
    else:
        inflow = inflow.reshape([1,1,1,-1,1]).repeat(1,1,res_z,1,1)
    inflow = torch.cat([inflow] + [torch.zeros_like(inflow)]*(dims-1), dim=1)
    Bl.CloseBoundary("-x", inflow.to(cuda_device).contiguous())

    # also set up outflow boundaries at the end of the channel.
    # these will need special treatment during the simulation
    out_vel = torch.tensor([[in_vel] + [0]*(dims-1)], device=cuda_device, dtype=dtype) # NC
    out_bounds = []
    for block in [Btu, Btl]:
        block.CloseBoundary("+x", out_vel)
        block.getBoundary("+x").makeVelocityVarying()
        out_bounds.append(block.getBoundary("+x"))

    # make the boundaries consistent/divergence-free
    PISOtorch_simulation.balance_boundary_fluxes(domain, out_bounds)

    # callback function to update the outflow during the simulation
    char_vel = out_vel.clone()
    out_block_indices = [domain.getBlocks().index(Btu), domain.getBlocks().index(Btl)]
    def prep_fn(domain, time_step, **kwargs):
        out_bounds = [domain.getBlock(idx).getBoundary("+x") for idx in out_block_indices]
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

def airfoil_channel_sample(log_dir, name, iterations=100, time_step=0.25, Re=3000, attack_angle=10, in_vel=0.3, res_z=0, load_2d_domain_path=None,
        dp=False, STOP_FN=None):
    # Re: Reynolds number based on the chord length and the mean inflow velocity
    # attack_angle: angle of attack in degrees, from 0 to 20
    # in_vel: mean inflow velocity
    # res_z: number of span-wise cells for a 3D simulation with a span of 1.4 (FluidGym's 3D airfoil environments use 96), 0 for 2D
    # load_2d_domain_path: 3D only. Like FluidGym, start from the developed flow of a 2D simulation of the same setup, e.g. '<run_dir>/domain'
    #   of a previous 2D run. It is copied to every span-wise position, with random noise to break the span-wise symmetry.
    global RUN_ID
    if STOP_FN():
        return
    LOG = get_logger("AirfoilChannel")
    dtype = torch.float64 if dp else torch.float32

    chord = 1.0
    depth = 1.4 # span of the 3D setup
    viscosity = in_vel*chord/Re
    airfoil_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_airfoils", "naca0012_sharp.dat")

    LOG.info("%dD airfoil channel #%d '%s' with angle of attack %.01f, Re %.02e (viscosity %.02e), %d iterations with time step %.02e",
             3 if res_z>0 else 2, RUN_ID, name, attack_angle, Re, viscosity, iterations, time_step)
    log_dir = os.path.join(log_dir, "%04d_airfoil_channel_%s"%(RUN_ID, name))

    grids = make_airfoil_channel_grids(airfoil_path, attack_angle=attack_angle, channel_height=1.4, inflow_x=-1.5, outflow_x=4.5, front_x=-0.5,
        res_normal=96, normal_base=0.97, tail_grow_mul=1.01, dtype=dtype)
    plot_grids(grids, color=["k","r","b","g","y","y"], path=log_dir, linewidth=0.5)

    # resample with 100 pixels per chord length, 50 in 3D
    resample_shape, resample_fill_steps = get_resampling_settings(grids, pixel_size=chord/(100 if res_z==0 else 50))
    if res_z>0:
        resample_shape.append(int(np.ceil(depth*50/chord)))
    LOG.info("%d cells, output resampling shape %s with up to %d fill steps", sum((g.size(-1)-1)*(g.size(-2)-1) for g in grids)*max(res_z,1),
        resample_shape, resample_fill_steps)

    domain, prep_fn, layout = make6BlockAirfoilChannelSetup(grids, in_vel=in_vel, viscosity=viscosity, res_z=res_z, depth=depth, dtype=dtype)
    if load_2d_domain_path:
        LOG.info("velocity initialized from the 2D domain '%s'", load_2d_domain_path)
        domain_2d = domain_io.load_domain(load_2d_domain_path, dtype=dtype, device=cuda_device)
        set_velocity_from_2D(domain, domain_2d, noise=0.01, seed=0) # noise as in the initialization of FluidGym's environments
        del domain_2d

    # FluidGym uses a pressure tolerance of 1e-7 (1e-8 in 3D), which is at the precision limit of the single precision CG solver for
    # this grid in 2D: many pressure solves then run to the iteration limit, making the simulation ca. 5x slower.
    # In 3D 1e-6 is not enough: the solver errors grow to an instability at the outflow, where the two wake blocks meet.
    sim = PISOtorch_simulation.Simulation(domain=domain, block_layout=layout, prep_fn=None,
            substeps="ADAPTIVE", time_step=time_step, corrector_steps=2, advection_tol=1e-6, pressure_tol=1e-6 if res_z==0 else 1e-7,
            advect_non_ortho_steps=2, pressure_non_ortho_steps=4, pressure_return_best_result=True,
            velocity_corrector="FD", non_orthogonal=True,
            log_interval=1, norm_vel=True,
            output_resampling_shape=resample_shape, output_resampling_fill_max_steps=resample_fill_steps,
            log_dir=log_dir, save_domain_name="domain",
            stop_fn=STOP_FN)

    sim.solver_double_fallback = True
    sim.preconditionBiCG = False
    sim.BiCG_precondition_fallback = True

    # initialize the domain with velocity that matches in- and outflow
    sim.make_divergence_free(max_iter=sim.linear_solve_max_iterations)

    # set the outflow update only after the initial projection, which uses a time step of 1
    sim.prep_fn = prep_fn
    sim.run(iterations)

    RUN_ID += 1


if __name__=="__main__":
    # create a new directory <time-step>_<name> in ./test_runs to use as base output directory
    run_dir = setup_run("./test_runs",
        name="airfoil-channel-sample"
    )
    LOG = get_logger("Main")
    stop_handler = PISOtorch_simulation.StopHandler(LOG)
    stop_handler.register_signal()

    # 240 iterations are 60 time units, the inflow passes the airfoil ca. 18 times
    airfoil_channel_sample(run_dir, name="Re3000_aoa10", Re=3000, attack_angle=10, iterations=240, time_step=0.25, dp=False, STOP_FN=stop_handler)

    # lower and higher Reynolds numbers, as in the easy and hard airfoil environments of FluidGym
    #airfoil_channel_sample(run_dir, name="Re1000_aoa10", Re=1000, attack_angle=10, iterations=240, time_step=0.25, dp=False, STOP_FN=stop_handler)
    #airfoil_channel_sample(run_dir, name="Re5000_aoa10", Re=5000, attack_angle=10, iterations=240, time_step=0.25, dp=False, STOP_FN=stop_handler)

    # 3D with 96 cells over a periodic span of 1.4 (4.5M cells), as in the 3D airfoil environments of FluidGym,
    # starting from the developed 2D flow of the first run above
    #airfoil_channel_sample(run_dir, name="Re3000_aoa10_3D", Re=3000, attack_angle=10, res_z=96,
    #    load_2d_domain_path=os.path.join(run_dir, "0000_airfoil_channel_Re3000_aoa10", "domain"), iterations=120, time_step=0.25, dp=False, STOP_FN=stop_handler)

    stop_handler.unregister_signal()
    close_logging()
