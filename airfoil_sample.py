import os, signal
from lib.util.logging import setup_run, get_logger, close_logging
from lib.util.profiling import SAMPLE

import numpy as np
from matplotlib import pyplot as plt 
import matplotlib.cm as cm 


if __name__=="__main__":
    # choose which GPU to use
    cudaID = None #"7"
    if cudaID is None:
        from lib.util.GPU_info import getAvailableGPU
        gpu_available = getAvailableGPU(active_mem_threshold=0.8) #active_mem_threshold=0.05
        #print('available GPU:', gpu_available)
        if gpu_available:
            cudaID = str(gpu_available[0])
        else:
            cudaID = "0"
            print('No GPU available, using', cudaID)
            #exit(4)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cudaID)


import torch
import PISOtorch # domain data structures and core PISO functions. check /extensions/PISOtorch.cpp to see what is available
#import PISOtorch_sim # uses core PISO functions to make full simulation steps
import PISOtorch_simulation
import lib.data.shapes as shapes
from lib.util.output import save_scalar_image, save_domain_images, plot_grids
from lib.data.resample import sample_coords_from_uniform_grid

# PISOtorch is only implemented for GPU

assert torch.cuda.is_available()
cuda_device = torch.device("cuda:0")
cpu_device = torch.device("cpu")

# read airfoil geometry from file
def read_airfoil(path, dtype=torch.float32):
    coords = []
    with open(path, "r") as file:
        name = file.readline().strip()
        for line in file:
            x,y = line.split()
            x = float(x)
            y = float(y)
            coords.append((x,y))
    
    coords = torch.tensor(coords, device=cpu_device, dtype=dtype) #WC
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

def ray_circle_intersection(o, d, r):
    # https://stackoverflow.com/questions/1073336/circle-line-segment-collision-detection-algorithm
    # assumes circle centered on origin, origin of ray in circle
    # o, d: origin and direction of line, NCHW
    # r: radius of circle, centered on (0,0)
    # f = o

    #r /= 2

    a = d[:,0] * d[:,0] + d[:,1] * d[:,1] # dot(d, d)
    b = 2*(o[:,0] * d[:,0] + o[:,1] * d[:,1]) # 2*dot(f, d)
    c = (o[:,0] * o[:,0] + o[:,1] * o[:,1]) - r*r # dot(f, f) - r*r

    discriminant = b*b - 4*a*c
    
    discriminant = torch.sqrt(discriminant)

    t2 = (-b + discriminant)/(2*a) #NHW

    t2 = t2.reshape((1,1, o.size(-2), o.size(-1)))

    LOG.info("t2: %s", t2)

    intersection = o + d*t2

    return intersection

def makeAirfoil_v2(airfoil_path:str, airfoil_res_div:int=1, vel_in:float=1, viscosity=1e-4, dtype=torch.float32):
    # C-grid, sharp trailing edge
    # assumes airfoil centered on x-axis with origin at tip
    # coordinates start at trailing edge and go counter-clockwise
    
    #plot_lw = 0.2
    grid_r = 2
    normal_base_mul = 0.5
    normal_grow_mul = 1.03
    tail_length = 4
    tail_grow_mul = 1.01
    grid_res = "AUTO"
    
    #flip_tuY = False #DON'T USE, it leads to inconsistent coordiante orientation, causing issues
    
    vel_init = [vel_in,0]
    bound_vel = [vel_in, 0]
    
    
    ### MAKE AIRFOIL GRID ###
    
    name, airfoil_coords = read_airfoil(airfoil_path) #NCHW
    
    
    airfoil_res = airfoil_coords.size(-1)
    airfoil_len_x = torch.max(airfoil_coords[:,0])
    
    airfoil_end = airfoil_coords[:,:,:,:1]
    
    point_start_top = torch.tensor([0,grid_r], device=cpu_device, dtype=dtype).reshape((1,2,1,1))
    point_end_top = torch.tensor([airfoil_len_x,grid_r], device=cpu_device, dtype=dtype).reshape((1,2,1,1))
    point_start_bot = torch.tensor([0,-grid_r], device=cpu_device, dtype=dtype).reshape((1,2,1,1))
    point_end_bot = torch.tensor([airfoil_len_x,-grid_r], device=cpu_device, dtype=dtype).reshape((1,2,1,1))
    
    if airfoil_res_div>1:
        div = int(airfoil_res_div)
        airfoil_coords = torch.nn.AvgPool2d([1,div])(airfoil_coords)
        airfoil_coords = torch.cat([airfoil_end, airfoil_coords, airfoil_end], dim=-1)
        airfoil_res = airfoil_coords.size(-1)
    
    LOG.info("Airfoil '%s' has length %s and %d points.", name, airfoil_len_x, airfoil_res)
    
    airfoil_end_spacing = torch.linalg.vector_norm(airfoil_coords[0,:,0,1] - airfoil_coords[0,:,0,0])
    airfoil_end_extend = airfoil_end + torch.tensor([airfoil_end_spacing, 0], device=cpu_device, dtype=dtype).reshape((1,2,1,1))
    airfoil_coords_extended = torch.cat([ airfoil_end_extend, airfoil_coords, airfoil_end_extend], dim=-1)
    
    #LOG.info("Airfoil end spacing %s", airfoil_end_spacing)
    
    # extrude normal
    airfoil_spacing = (airfoil_coords_extended[0,:,0,2:] - airfoil_coords_extended[0,:,0,:-2]) # CW
    airfoil_normals = torch.flip(airfoil_spacing, dims=(0,)) * torch.tensor([1, -1], device=cpu_device, dtype=dtype).reshape((2,1))
    airfoil_normals /= torch.linalg.vector_norm(airfoil_normals, dim=0)
    airfoil_normals = torch.reshape(airfoil_normals, (1,2,1,-1))
    
    airfoil_spacing = (airfoil_coords_extended[0,:,0,1:] - airfoil_coords_extended[0,:,0,:-1]) # CW
    airfoil_spacing = torch.linalg.vector_norm(airfoil_spacing, dim=0) # W
    min_size = torch.min(airfoil_spacing).numpy().tolist()
    LOG.info("airfoil grid minimum spacing %.03e", min_size)
    min_size = min_size * normal_base_mul
    
    #airfoil_spacing = torch.reshape(airfoil_spacing, (1,1,1,-1))
    if True:
        normal_sizes = [min_size]
        normal_dist = min_size
        while normal_dist<grid_r:
            size = normal_sizes[-1]*normal_grow_mul
            normal_sizes.append(size)
            normal_dist = normal_dist + size
        #LOG.info("Normal sizes %s", normal_sizes)
        normal_weights = [0] + (np.cumsum(normal_sizes) / normal_dist).tolist()
        normal_weights_inverse = [0] + (np.cumsum(normal_sizes[::-1]) / normal_dist).tolist()
        normal_res = len(normal_weights)
        
        #LOG.info("Normal weights, inverse\n%s\n%s", normal_weights, normal_weights_inverse)
        
        tail_sizes = [min_size]
        tail_dist = min_size
        while tail_dist<grid_r:
            size = tail_sizes[-1]*tail_grow_mul
            tail_sizes.append(size)
            tail_dist = tail_dist + size
        tail_weights = [0] + (np.cumsum(tail_sizes) / tail_dist).tolist()
        tail_res_x = len(tail_weights)
    else:
        normal_weights = None
        normal_weights_inverse = None
        normal_res = int(torch.ceil(grid_r / torch.max(airfoil_spacing)).numpy().tolist())
        tail_weights = None
        tail_res_x = int(np.round(normal_res * tail_length/grid_r))
        
    
    #normal_res = int(torch.ceil(grid_r / torch.max(airfoil_spacing)).numpy().tolist())
    LOG.info("Airfoil grid res %s, normal %s, tail %s", airfoil_res, normal_res, tail_res_x)
    
    
    distances_top = distance_to_point(airfoil_coords[...,:airfoil_res//2], airfoil_normals[...,:airfoil_res//2], point_start_top)
    min_d_top, min_d_top_idx = torch.min(distances_top[0,0,0,:], dim=-1)
    distances_bot = distance_to_point(airfoil_coords[...,airfoil_res//2:], airfoil_normals[...,airfoil_res//2:], point_start_bot)
    min_d_bot, min_d_bot_idx = torch.min(distances_bot[0,0,0,:], dim=-1)
    min_d_bot_idx += airfoil_res//2
    
    #LOG.info("Corner intersect: top %s (%s), bot %s (%s)", min_d_top_idx, min_d_top, min_d_bot_idx, min_d_bot)

    # upper outer boundary
    dist_top = point_start_top - point_end_top
    steps_top = min_d_top_idx
    step_top = dist_top / steps_top
    coords_top = torch.cat([point_end_top + step_top * i for i in range(steps_top+1)], dim=-1)

    # lower outer boundary
    dist_bot = point_end_bot - point_start_bot
    steps_bot = (airfoil_res-1)-min_d_bot_idx
    step_bot = dist_bot / steps_bot
    coords_bot = torch.cat([point_start_bot + step_bot * i for i in range(steps_bot+1)], dim=-1)

    # front outer boundary

    coords_front = ray_circle_intersection(airfoil_coords[...,min_d_top_idx+1:min_d_bot_idx], airfoil_normals[...,min_d_top_idx+1:min_d_bot_idx], grid_r)

    coords_outer = torch.cat([coords_top, coords_front, coords_bot], dim=-1)

    #LOG.info("Outer coords: %s, %s, %s: %s", coords_top.shape, coords_front.shape, coords_bot.shape, coords_outer.shape)

    airfoil_end_x , airfoil_end_y = airfoil_end[0,:,0,0].numpy().tolist()
    point_end_top_x, point_end_top_y = point_end_top[0,:,0,0].numpy().tolist()
    point_end_bot_x, point_end_bot_y = point_end_bot[0,:,0,0].numpy().tolist()
    airfoil_grid_corners = [(airfoil_end_x , airfoil_end_y), (airfoil_end_x , airfoil_end_y), (point_end_top_x, point_end_top_y), (point_end_bot_x, point_end_bot_y)]
    airfoil_coords_flat = torch.movedim(airfoil_coords.reshape(2,-1), 0,1).numpy().tolist()
    coords_outer_flat = torch.movedim(coords_outer.reshape(2,-1), 0,1).numpy().tolist()
    airfoil_grid = shapes.generate_grid_vertices_2D([normal_res, airfoil_res], airfoil_grid_corners, [None,None, airfoil_coords_flat, coords_outer_flat],
        x_weights=normal_weights, dtype=dtype)
    
    

    #tail_res_x = int(np.round(normal_res * tail_length/grid_r))
    corners_tail_upper = [(point_end_top_x, point_end_top_y),(point_end_top_x+tail_length, point_end_top_y), (airfoil_end_x , airfoil_end_y),(airfoil_end_x+tail_length, airfoil_end_y)]
    corners_tail_lower = [(airfoil_end_x , airfoil_end_y),(airfoil_end_x+tail_length, airfoil_end_y), (point_end_bot_x, point_end_bot_y),(point_end_bot_x+tail_length, point_end_bot_y)]
    
    coords_tail_upper = shapes.generate_grid_vertices_2D([normal_res, tail_res_x], corners_tail_upper, None,
        x_weights=normal_weights_inverse, y_weights=tail_weights, dtype=dtype)
    coords_tail_lower = shapes.generate_grid_vertices_2D([normal_res, tail_res_x], corners_tail_lower, None,
        x_weights=normal_weights, y_weights=tail_weights, dtype=dtype)
    
    
    airfoil_grid = airfoil_grid.to(cuda_device)
    coords_tail_upper = coords_tail_upper.to(cuda_device)
    coords_tail_lower = coords_tail_lower.to(cuda_device)
    
    
    ### MAKE DOMAIN AND BLOCKS ###
    
    viscosity = torch.tensor([viscosity], dtype=dtype, device=cpu_device)
    domain = PISOtorch.Domain(2, viscosity, name="DomainNACA0012", dtype=dtype, device=cuda_device, passiveScalarChannels=0)
    
    block_airfoil = domain.CreateBlock(vertexCoordinates=airfoil_grid, name="Airfoil")
    
    block_tail_upper = domain.CreateBlock(vertexCoordinates=coords_tail_upper, name="TailUpper")
    block_tail_lower = domain.CreateBlock(vertexCoordinates=coords_tail_lower, name="TailLower")

    block_airfoil.ConnectBlock("-x", block_tail_upper, "-x", "+y")
    block_airfoil.ConnectBlock("+x", block_tail_lower, "-x", "-y")
    block_tail_upper.ConnectBlock("+y", block_tail_lower, "-y", "-x")
    
    # inflow
    in_vel = torch.tensor([bound_vel], device=cuda_device, dtype=dtype)
    block_airfoil.CloseBoundary("+y", in_vel)
    
    
    block_tail_upper.CloseBoundary("-y", in_vel)
    block_tail_lower.CloseBoundary("+y", in_vel)
    
    # outflow
    block_tail_upper.CloseBoundary("+x", in_vel)
    block_tail_upper.getBoundary("+x").makeVelocityVarying()
    block_tail_lower.CloseBoundary("+x", in_vel)
    block_tail_lower.getBoundary("+x").makeVelocityVarying()
    
    # init passive scalar outflow, if set on block
    if domain.hasPassiveScalar():
        block_tail_upper.getBoundary("+x").setPassiveScalar(block_tail_upper.passiveScalar[...,-1:].clone().contiguous())
        block_tail_lower.getBoundary("+x").setPassiveScalar(block_tail_lower.passiveScalar[...,-1:].clone().contiguous())
    
    #LOG.info("boundary transforms:\nUpper %s\nLower %s", block_tail_upper.getBoundary("+x").transform, block_tail_lower.getBoundary("+x").transform)

    out_bounds = [block_tail_upper.getBoundary("+x"), block_tail_lower.getBoundary("+x")]
    # make the boundaries consistent/divergence-free
    PISOtorch_simulation.balance_boundary_fluxes(domain, out_bounds)
    
    # callback function to update the outflow during the simulation
    char_vel = in_vel.clone()
    def update_outflow(domain, time_step, **kwargs):
        out_bounds = [domain.getBlock(1).getBoundary("+x"), domain.getBlock(2).getBoundary("+x")]
        PISOtorch_simulation.update_advective_boundaries(domain, out_bounds, char_vel, time_step.cuda())
    prep_fn = update_outflow

    
    for block in domain.getBlocks():
        block.velocity[:,0,:,:] = vel_in

    domain.PrepareSolve()
    
    resample_res = [  #x,y
        normal_res-1 + steps_top + tail_res_x-1,
        (normal_res-1)*2]
    
    resample_res_scale = 4
    resample_res = [_*resample_res_scale for _ in resample_res]

    return domain, resample_res, {"PRE": prep_fn}

def airfoil_sample(run_dir, name="airfoil", iterations=100, time_step=1, dp=False, STOP_FN=None, **domain_args):
    
    dtype = torch.float64 if dp else torch.float32
    log_dir = os.path.join(run_dir, name)
    
    domain, out_shape, prep_fn = makeAirfoil_v2(dtype=dtype, **domain_args)
    
    
    from lib.util.domain_io import save_domain
    #domain_path = os.path.join(log_dir, "domain_init")
    #save_domain(domain, domain_path)
    
    
    grids = domain.getVertexCoordinates()
    if domain.getSpatialDims()==3:
        grids = [grid[:,:2,0,:,:] for grid in grids]
    grid_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    plot_grids(grids, color=grid_colors[:len(grids)], path=log_dir, type="pdf", linewidth=0.5)
    
    from lib.util.memory_usage import MemoryUsage
    
    mem_usage = MemoryUsage(logger=LOG)
    def mem_usage_log_fn(total_step, **kwargs):
        mem_usage.check_memory("step %04d"%(total_step,))
    
    if iterations>0:
        LOG.info("runSim, out shape = %s", out_shape)
        sim = PISOtorch_simulation.Simulation(domain=domain, prep_fn=prep_fn,
                substeps="ADAPTIVE", time_step=time_step, corrector_steps=2, advection_tol=1e-6, pressure_tol=1e-7, #"ADAPTIVE"
                advect_non_ortho_steps=2, pressure_non_ortho_steps=4, pressure_return_best_result=True,
                velocity_corrector="FD", non_orthogonal=True,
                log_interval=1, norm_vel=True, log_fn=mem_usage_log_fn,
                log_dir=log_dir, output_resampling_shape=out_shape, save_domain_name="domain",
                output_resampling_fill_max_steps=16,
                stop_fn=stop_handler)
        
        sim.preconditionBiCG = False
        sim.BiCG_precondition_fallback = True
        
        sim.make_divergence_free()
        
        sim.run(iterations)
    
    mem_usage.print_max_memory()

if __name__=="__main__":
    run_dir = setup_run("./test_runs",
        name="Airfoil-div1-expW-n1.03-t1.01_full_v1e-4_it200-ada-ts.1"
    )
    LOG = get_logger("Main")
    stop_handler = PISOtorch_simulation.StopHandler(LOG)
    stop_handler.register_signal()
    
    LOG.info("Airfoil test")
    
    airfoil_path = os.path.join("sample_airfoils", "naca0012_sharp.dat")
    
    airfoil_sample(run_dir, name="naca0012_sharp", iterations=200, time_step=0.1, dp=False, STOP_FN=stop_handler,
        airfoil_res_div=1, airfoil_path=airfoil_path)
    
    stop_handler.unregister_signal()
    close_logging()
