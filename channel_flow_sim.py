import os, glob, csv
# use GPU with lowest memory used, if available
from lib.util.GPU_info import get_available_GPU_id
cudaID = None #'7'
cudaID = cudaID or str(get_available_GPU_id(active_mem_threshold=0.3, default=None))
os.environ["CUDA_VISIBLE_DEVICES"] = cudaID

import numpy as np
import torch
assert torch.cuda.is_available()
cuda_device = torch.device("cuda")
cpu_device = torch.device("cpu")

import PISOtorch # core CUDA module
import PISOtorch_simulation # helper for PISO loop
import lib.data.shapes as shapes # mesh generation
from lib.util.output import plot_grids, save_block_data_image, ttonp, ntonp
from lib.util.logging import setup_run, get_logger, close_logging # logging and output
import lib.util.domain_io as domain_io

from lib.util.profiling import SAMPLE
from lib.data.torroja import TorrojaProfile, TorrojaBalances # loaders for reference statistics
import lib.data.TCF_tools as TCF_tools # functionality for recording and plotting TCF statistics
from lib.data.TCF_tools import VelocityStats

#from matplotlib import pyplot as plt
#from matplotlib.ticker import MultipleLocator, AutoMinorLocator


def parse_adaptive_substeps(log_path):
    import re
    re_sim_start = re.compile(r"Starting sim with (\d+) iterations")
    re_substeps = re.compile(r"used (\d+) substeps")
    sim_steps = 0
    current_step = None
    
    sim_substeps = []
    sim_frames = []
    
    log_path = os.path.join(log_path, "logfile.log")
    with open(log_path, "r") as log_file:
        for line in log_file:
            if not sim_steps:
                result_sim_start = re_sim_start.search(line)
                if result_sim_start is not None:
                    sim_steps = int(result_sim_start.group(1))
                    re_step = re.compile(r"It: (\d+)/" + str(sim_steps) + r", substeps:adaptive")
            else:
                if current_step is None:
                    result_step = re_step.search(line)
                    if result_step is not None:
                        current_step = int(result_step.group(1))
                else:
                    result_substeps = re_substeps.search(line)
                    if result_substeps is not None:
                        substeps = int(result_substeps.group(1))
                        sim_substeps.append(substeps)
                        sim_frames.append(current_step)
                        current_step = None
            
    return sim_frames, sim_substeps





def load_domain_from_run(run_id:str, base_path:str="./test_runs", sub_dir=None, dtype=torch.float32, with_scalar:bool=True, frame:int=None):
    load_path = os.path.join(base_path, "%s_*"%(run_id,))
    if sub_dir is not None:
        load_path = os.path.join(load_path, sub_dir)
    if frame is None:
        load_path = os.path.join(load_path, "domain.json")
    else:
        load_path = os.path.join(load_path, "domain_frame_%06d.json"%(frame,))
    load_path = glob.glob(load_path)
    
    if len(load_path)==0: raise IOError("No domain found for run id %s, frame %s"%(run_id, frame))
    if len(load_path)!=1: raise IOError("No unique domain found for run id %s, frame %s"%(run_id, frame))
    
    load_path = load_path[0][:-5] # without file type extension ".json"
    domain = domain_io.load_domain(load_path, dtype=dtype, device=cuda_device, with_scalar=with_scalar)
    
    LOG.info("loaded domain R%s", run_id)
    
    return domain

def make_y_weights_Re550(log, N=1, ny_half=48):
    #N=1
    ny = 2*(ny_half//N)
    #if not (ny>0 and (ny%2)==0):
    #    raise ValueError("ny must be positive and even")

    r = 1.2**(N/2) #exponential growth rate?
    h0 = 0.5*(1-r)/(1-r**(ny/2)) #cell y-size at boundary
    h = 0 # current distance from boundary
    y = [0]*ny
    for i in range((ny-2)//2):
        h += h0*(r**i)
        y[i] = h
        y[ny-i-2] = 1-h
    y[ny//2-1] = 0.5
    y[ny-1] = 1.0

    y = [0] + y

    #log.info("Re 550 y coords: %d\n%s", len(y), y)

    return y

def make_grid_Re550(run_dir, log, x=128, y_half=48, yN=1, dims=3, dtype=torch.float32, global_scale=None, _use_cos_y=False):
    assert (x%4)==0 #in [128,192,256]
    assert dims in [2,3]

    delta = 1 # =y_size/2
    x_size = 2*np.pi*delta
    y_size = delta * 2
    z_size = np.pi*delta
    
    if global_scale is not None:
        y_weights = shapes.make_weights_exp_global(y_half*2, global_scale, "BOTH", log_fn=LOG.info)
    elif _use_cos_y:
        y_weights = shapes.make_weights_cos(y_half*2, "BOTH")
    else:
        y_weights = make_y_weights_Re550(log, ny_half=y_half*yN, N=yN)
    y_sizes = np.asarray(y_weights)
    y_sizes = y_sizes[1:] - y_sizes[:-1]
    log.info("Re550 grid y min=%.03e, max=%.03e, ratio=%.03e", np.min(y_sizes), np.max(y_sizes), np.max(y_sizes)/np.min(y_sizes))


    corners = [(-x_size/2, -delta), (x_size/2, -delta), (-x_size/2, delta), (x_size/2, delta)]

    y = len(y_weights) - 1
    z = x//2

    grid = shapes.generate_grid_vertices_2D([y+1,x+1],corners, None, x_weights=y_weights, dtype=dtype)
    plot_grids([grid], path=run_dir)
    if dims==3:
        grid = shapes.extrude_grid_z(grid, z, start_z=-z_size/2, end_z=z_size/2)
    grid = grid.cuda().contiguous()

    return grid

def get_viscosity_wall_distance(block, domain):
    # assumes channel flow centered on 0 with delta=1
    pos_y = block.getCellCoordinates()[:,1:2]
    wall_distance = (1 - torch.abs(pos_y)) * u_wall / domain.viscosity.to(cuda_device)
    
    return wall_distance # NCDHW with C=1

def get_van_driest_sqr(block, domain):
    wall_distance = get_viscosity_wall_distance(block, domain)
    van_driest_scale = 1 - torch.exp(- wall_distance * (1.0/25.0))
    return van_driest_scale*van_driest_scale

def make_reichardt_profile(domain, u_wall=1):

    k = 0.41
    k_inv = 1/k
    def reichardt_profile(y_wall):
        y11 = y_wall/11.0
        return k_inv * torch.log(1+k*y_wall) + 7.8*(1 - torch.exp(-y11) - y11*torch.exp(-y_wall/3))

    pos_y = domain.getBlock(0).getCellCoordinates()[0,1,0,:,0] #NCDHW -> H
    wall_distance = (1 - torch.abs(pos_y)) * u_wall / domain.viscosity.to(cuda_device)

    u = reichardt_profile(wall_distance)

    return u * u_wall

def coarsen_grid(vertex_coords, transforms, factors, data_tensors):
    # factors: DHW
    assert (isinstance(factors, int) and factors>1) or (isinstance(factors, list) and len(factors)==3 and all(isinstance(factor, int) and factor>0 for factor in factors)), "factors must be int or list of 3 ints"
    if isinstance(factors, int):
        factors = [factors]*3
    
    assert isinstance(data_tensors, list) and len(data_tensors)>0 and all(isinstance(t, torch.Tensor) for t in data_tensors), "data_tensors must be a list of tensors"
    
    assert isinstance(vertex_coords, torch.Tensor) and vertex_coords.dim()==5 and vertex_coords.size(0)==1 and vertex_coords.size(1)==3, "vertex_coords must be a 3D NCDHW tensor"
    sizes = [vertex_coords.size(i+2)-1 for i in range(3)] #zyx
    assert all((size%factor)==0 for size, factor in zip(sizes, factors)), "all spatial sizes must be divisible by their factor"
    assert all(transforms.size(i+1)==sizes[i] for i in range(3)), "transforms sizes do not match"
    assert all(all(t.size(i+2)==sizes[i] for i in range(3)) for t in data_tensors), "data tensor sizes do not match"
    
    
    pooler = torch.nn.AvgPool3d(kernel_size=factors, stride=factors, divisor_override=1) # sum pooling
    
    low_vertex_coords = vertex_coords[:,:,::factors[0],::factors[1],::factors[2]]
    
    cell_sizes = torch.unsqueeze(transforms[...,-1], 1) # NDHWC -> NCDHW
    low_cell_sizes = pooler(cell_sizes)
    
    low_data_tensors = []
    for t in data_tensors:
        low_data_tensors.append(pooler(t * cell_sizes) / low_cell_sizes)
    
    return low_vertex_coords, low_data_tensors

def coarsen_Re550_domain(domain, factors, make_div_free=False):
    dims = domain.getSpatialDims()
    
    viscosity = domain.viscosity.detach().clone()
    low_domain = PISOtorch.Domain(dims, viscosity, passiveScalarChannels=0, name="ChannelDomain", device=cuda_device, dtype=dtype)
    
    block = domain.getBlock(0)
    low_vertex_coords, low_data_tensors = coarsen_grid(block.vertexCoordinates.detach(), block.transform.detach(), factors, [block.velocity.detach(), block.pressure.detach()])

    # create block from the mesh on the domain (missing settings, fields and transformation metrics are created automatically)
    low_block = low_domain.CreateBlock(velocity=low_data_tensors[0].contiguous(), pressure=low_data_tensors[1].contiguous(),
        vertexCoordinates=low_vertex_coords.contiguous(), name="ChannelBlock")
    low_block.CloseBoundary("-y")

    #block.velocity[0,0] += 0.5*max_vel

    G = block.velocitySource.detach().clone() # NC, static velocity source
    low_block.setVelocitySource(G)
    
    if make_div_free:
        low_domain.PrepareSolve()
        sim = PISOtorch_simulation.Simulation(domain=low_domain, pressure_tol=1e-8)
        sim.make_divergence_free()
    
    return low_domain

def set_dynamic_forcing(domain, prep_fn):
    pos_y = torch.mean(domain.getBlock(0).getCellCoordinates()[0,1], dim=(0,2))
    d_y = (1+pos_y[0].cpu().numpy(), 1-pos_y[-1].cpu().numpy())
    
    def pfn_set_forcing(domain, **kwargs):
        block = domain.getBlock(0)
        viscosity = domain.viscosity.to(domain.getDevice())
        
        mean_vel_u = torch.mean(block.velocity[0,0], dim=(0,2))
        tau_wall_n = viscosity * mean_vel_u[0]/d_y[0]
        tau_wall_p = viscosity * mean_vel_u[-1]/d_y[-1]
        
        forcing = (tau_wall_n + tau_wall_p)*0.5
        G = torch.tensor([[forcing]+[0]*(dims-1)], dtype=domain.getDtype(), device=domain.getDevice()) # NC, static velocity source
        block.setVelocitySource(G)
    
    PISOtorch_simulation.append_prep_fn(prep_fn, "PRE", pfn_set_forcing)

def pfn_check_matrix(domain, **kwargs):
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import norm
    
    norm_ord = None
    
    C = csr_matrix((domain.C.value.cpu(), domain.C.index.cpu(), domain.C.row.cpu()), shape=[domain.C.getRows()]*2)
    
    metric = norm(C - C.transpose(copy=True)) / norm(C)
    
    LOG.info("C symmetry metric %.03e", metric)

def check_CFL(domain, out_dir, it, **kwargs):
    if out_dir is not None:
        save_block_data_image([block.getVelocity(True) for block in domain.getBlocks()], domain, out_dir, "comVel-zMax-norm", it, normalize=True, axis3D=0, mode3D="max")
    block = domain.getBlock(0)
    domain_max_comp_vel = ntonp(domain.getMaxVelocity(True, True))
    comp_vel = ttonp(torch.abs(block.getVelocity(True))) #NCDHW
    #LOG.info("comp vel shape: %s", comp_vel.shape)
    com_vel_max_y = np.max(comp_vel, axis=(0,2,4)) #torch.max(comp_vel, dim=(0,2,4)).values #CH
    #LOG.info("Max y shape: %s", com_vel_max_y.shape)
    #com_vel_max_c, com_vel_max_c_idx = torch.max(com_vel_max_y, dim=1) #C
    com_vel_max_c = np.max(com_vel_max_y, axis=1)
    com_vel_max_c_idx = np.argmax(com_vel_max_y, axis=1)
    LOG.info("Max comp vel: %s, y-indices: %s", ntonp(com_vel_max_c), ntonp(com_vel_max_c_idx))
    LOG.info("Domain max comp vel: %s", domain_max_comp_vel)

if __name__=="__main__":

    case = "EVAL" # "SIM", "EVAL"

    if case=="SIM":
        run_dir = setup_run("./test_runs",
            name="channel-flow-data_Re550-d_r64-y32-CFL.1_ETT20-it200_Init-R-noise_SP"
            )
        LOG = get_logger("Main")
        LOG.info("GPU #%s", cudaID)
        stop_handler = PISOtorch_simulation.StopHandler(LOG)
        stop_handler.register_signal()

        dtype = torch.float32
        
        # PARAMETERS #

        Re_wall = 550
        
        # Stream-wise resolution. 64 for training data, up to 256 for high resolution simulation.
        x = 64 # 64, 128, 192, 256
        # Wall-normal resolution, also influences z-resolution. 32 for training data, up to 128 for high resolution simulation.
        y = 32 # 32, 64, 96, 128. must be divisible by 2
        # Controls how strongly the grid is refined in wall-normal direction.
        refinement_strength = 3 # 3 for x=64 training data, 1 for high resolution simulation.
        
        target_ETT = 20 # total time to simulate. I used 20 ETT for warm up before accumulating statistics.
        iterations = 200 # iterations for output, simulation steps will be further divided by adaptive time stepping.
        # every 'iteration' will have a time of target_ETT/iterations.
        adaptive_CFL = 0.1 # condition for the adaptive time stepper. Higher values might result in a laminar flow.
        save_steps = True # will save after every 'iteration'.

        init_with_noise = True # needs compiled SimplexNoiseVariations: python extensions/noise/setup_noise.py install
        
        # you can load a previous simulation based on its time stamp. This will overwrite x and y
        #load_run_id = "<time>-<stamp>", e.g. "251008-114126"
        load_run_id = None
        
        # END PARAMETERS #

        delta = 1 # =y_size/2
        x_size = 2*np.pi*delta
        y_size = delta * 2
        z_size = np.pi*delta

        
        Re_center = TCF_tools.Re_wall_to_cl(Re_wall)
        u_wall = Re_wall / Re_center

        viscosity = delta/Re_center
        forcing = "DYNAMIC"
        
        target_t = TCF_tools.ETT_to_t(target_ETT, u_wall, delta) # target_t_wall * delta / u_wall
        target_t_wall = TCF_tools.t_to_t_wall(target_t, viscosity, u_wall)
        
        time_step = target_t / iterations # internal time step
        
        LOG.info("Dynamic TCF setup: Re_wall=%.03e, Re_center=%.03e, u_wall=%.03e, visc=%.03e, forcing=%s, ETT=%.03e -> T+=%0.3e, T=%.03e (it %d -> ts=%.03e)",
                Re_wall, Re_center, u_wall, viscosity, forcing, target_ETT, target_t_wall, target_t, iterations, time_step)
        
        # the simulation can use a static Smagorinsky SGS model with this coefficient
        C_smag = 0 # this is NOT squared later
        use_van_driest = False # use van Driest scaling to reduce the influence of the SGS model near the wall
        

        load_run_frame = None #None=last frame
        load_vel_run_id = None
        
        
        down_factors = None
        #down_factors = [3,1,3]
        #down_factors = 3
        down_make_div_free = False
        
        #vel_up_factors = [2,1,2]
        vel_up_factors = [1,2,1]
        #up_make_div_free = False
        
        compare_steps = False
        compare_low_div_free = True
        compare_steps_high = []
        compare_steps_low = []

        if load_run_id:
            domain = load_domain_from_run(load_run_id, dtype=dtype, with_scalar=False, frame=load_run_frame)
            if down_factors is not None:
                domain = coarsen_Re550_domain(domain, down_factors)
                if down_make_div_free:
                    raise NotImplementedError("get div-free from buoyancy learning")
            block = domain.getBlock(0)
            block_size = block.getSizes()
            dims = domain.getSpatialDims()
            x = block_size.x
            y = block_size.y
            z = block_size.z
            grid = block.vertexCoordinates
            prep_fn = {}
            
            if compare_steps:
                LOG.info("Loading reference sim steps.")
                #compare_steps_high = [load_domain_from_run(load_run_id, dtype=dtype, with_scalar=False, frame=i+load_run_frame) for i in range(iterations)]
                # These are not intialized (domain.PrepareSolve()) to save memory.
                #compare_steps_low = [coarsen_Re550_domain(d, down_factors) for d in compare_steps_high]
                compare_steps_low = [coarsen_Re550_domain(load_domain_from_run(load_run_id, dtype=dtype, with_scalar=False, frame=i+load_run_frame), down_factors, make_div_free=compare_low_div_free)
                    for i in range(iterations+1)]
        else:

            y_half = y//2 #48
            grid_N = refinement_strength
            dims = 3

            grid = make_grid_Re550(run_dir, LOG, x=x, y_half=y_half, yN=grid_N, dtype=dtype, global_scale=None)
            y = grid.size(-2)-1
            z = grid.size(-3)-1

            vel_init = None
            
            # create domain
            viscosity = torch.tensor([viscosity], dtype=dtype)
            domain = PISOtorch.Domain(dims, viscosity, passiveScalarChannels=0, name="ChannelDomain", device=cuda_device, dtype=dtype)
            prep_fn = {}

            # create block from the mesh on the domain (missing settings, fields and transformation metrics are created automatically)
            block = domain.CreateBlock(velocity=vel_init, vertexCoordinates=grid, name="ChannelBlock")
            block.CloseBoundary("-y")

            
            if forcing!="DYNAMIC":
                G = torch.tensor([[forcing]+[0]*(dims-1)], dtype=dtype, device=cuda_device) # NC, static velocity source
                block.setVelocitySource(G)
            
            if load_vel_run_id is not None:
                LOG.info("loading velocity from %s for initialization", load_vel_run_id)
                low_domain = load_domain_from_run(load_vel_run_id, dtype=dtype, with_scalar=False, frame=load_run_frame)
                low_vel = low_domain.getBlock(0).velocity
                del low_domain
                for dim, factor in enumerate(vel_up_factors):
                    if factor>1:
                        low_vel = low_vel.repeat_interleave(factor, dim=-1-dim)
                block.setVelocity(low_vel.contiguous())
                del low_vel
            else:
                vel_init = make_reichardt_profile(domain, u_wall=u_wall)
                vel_init_max = torch.max(vel_init)
                LOG.info("Reichardt Profile max vel: %.03e", vel_init_max.cpu().numpy())
                vel_init = vel_init.view(1,1,1,y,1).expand(-1,-1,z,-1,x)
                vel_init = torch.cat([vel_init] + [torch.zeros_like(vel_init)]*(dims-1), dim=1)

                if init_with_noise:
                    import SimplexNoiseVariations
                    LOG.info("Adding curl noise to initial velocity profile.")

                    curl_noise = SimplexNoiseVariations.GenerateSimplexNoiseVariation([x,y,z], cuda_device, [2/x,2/y,2/z], [0]*3, SimplexNoiseVariations.NoiseVariation.CURL)
                    curl_mag = torch.linalg.vector_norm(curl_noise, dim=1)
                    curl_mag_max = torch.max(curl_mag)
                    curl_noise *= 0.5*vel_init/curl_mag_max
                    vel_init += curl_noise
            
                vel_init = vel_init.to(dtype).contiguous()
                block.setVelocity(vel_init)
        
        # finalize domain
        domain.PrepareSolve()
        
        if forcing=="DYNAMIC":
            set_dynamic_forcing(domain, prep_fn)

        resample_norm = y/y_size * 0.5
        if dims==2:
            output_resampling_shape = [int(x_size*resample_norm), int(y_size*resample_norm)]
        else:
            output_resampling_shape = [int(x_size*resample_norm),  int(y_size*resample_norm), int(z_size*resample_norm)]
        
        vel_stats = VelocityStats(domain, os.path.join(run_dir, "stats"), LOG, record_online_steps=True, u_wall=u_wall)
        vel_stats.plot_u_vel_max = 30
        vel_stats.plot_vel_rms_max = 4
        ref_profiles = TorrojaProfile("./data/tcf_torroja", 550)
        vel_stats.set_references(ref_profiles)
        #vel_stats.load_references()
        prep_fn["POST"] = vel_stats.record_vel_stats
        
        use_SGS = False
        if C_smag!=0:
            LOG.info("using SMAG SGS with C=%0.3e", C_smag)
            use_SGS = True
            SGS_coefficient = torch.tensor([C_smag], dtype=dtype, device=cpu_device)
            
            if use_van_driest:
                LOG.info("using van Driest wall distance scaling")
                van_driest_scale_sqr = [get_van_driest_sqr(block, domain)]
            
            def get_SGS_viscosity(domain):
                return PISOtorch.SGSviscosityIncompressibleSmagorinsky(domain, SGS_coefficient)
            
            def add_block_SGS_viscosity(domain, **kwargs):
                domain.UpdateDomainData()
                
                SGS_viscosities = get_SGS_viscosity(domain)
                base_viscosity = domain.viscosity.to(cuda_device)
                
                for idx, (block, visc) in enumerate(zip(domain.getBlocks(), SGS_viscosities)):
                    if use_van_driest:
                        visc = visc * van_driest_scale_sqr[idx]
                    visc = visc + base_viscosity
                    block.setViscosity(visc)
                
                domain.UpdateDomainData()
            
            PISOtorch_simulation.append_prep_fn(prep_fn, "PRE", add_block_SGS_viscosity)
            
            def log_SGS_viscosity(domain, out_dir, it, **kwargs):
                SGS_viscosities = [block.viscosity for block in domain.getBlocks()]#get_SGS_viscosity(domain)
                SGS_viscosities_flat = torch.cat([v.view(-1) for v in SGS_viscosities])
                LOG.info("viscosity (SGS + base %s): mean %s, min %s, max %s", domain.viscosity.cpu().numpy(),
                    torch.mean(SGS_viscosities_flat).cpu().numpy(),
                    torch.min(SGS_viscosities_flat).cpu().numpy(),
                    torch.max(SGS_viscosities_flat).cpu().numpy())
                save_block_data_image(SGS_viscosities, domain, out_dir, "vSGS-z-norm", it, normalize=True)
                if domain.hasVertexCoordinates():
                    save_block_data_image(SGS_viscosities, domain, out_dir, "vSGS-r-z-norm", it, normalize=True,
                        vertex_coord_list=domain.getVertexCoordinates(), resampling_out_shape=output_resampling_shape, fill_max_steps=20)
        
        def log_step_comparison(domain, out_dir, it, **kwargs):
            reference_domain = compare_steps_low[it]
            vel_differences = [torch.abs(block.velocity - ref_block.velocity) for block, ref_block in zip(domain.getBlocks(), reference_domain.getBlocks())]
            vel_differences_flat = torch.cat([v.view(3,-1) for v in vel_differences], dim=-1)
            LOG.info("velocity absolute difference: mean %s, min %s, max %s",
                torch.mean(vel_differences_flat, dim=-1).cpu().numpy(),
                torch.min(vel_differences_flat, dim=-1)[0].cpu().numpy(),
                torch.max(vel_differences_flat, dim=-1)[0].cpu().numpy())
            
            save_block_data_image(vel_differences, domain, out_dir, "uDiff-z-norm", it, normalize=True)
            if domain.hasVertexCoordinates():
                save_block_data_image(vel_differences, domain, out_dir, "uDiff-r-z-norm", it, normalize=True,
                    vertex_coord_list=domain.getVertexCoordinates(), resampling_out_shape=output_resampling_shape, fill_max_steps=20)
        
        # setup simulation
        sim = PISOtorch_simulation.Simulation(domain=domain, prep_fn=prep_fn,
            substeps="ADAPTIVE", time_step=time_step, corrector_steps=2,
            advection_use_BiCG=True,
            advection_tol=1e-6, pressure_tol=1e-7,
            adaptive_CFL=adaptive_CFL,
            advect_non_ortho_steps=1, pressure_non_ortho_steps=1, pressure_return_best_result=True,
            velocity_corrector="FD", non_orthogonal=True,
            log_interval=1, norm_vel=True, log_fn=None,
            #output_resampling_coords=[grid], output_resampling_shape=output_resampling_shape,
            #output_resampling_fill_max_steps=20,
            log_dir=run_dir, save_domain_name="domain",
            stop_fn=stop_handler)
        
        #sim.linear_solve_max_iterations = 10000
        sim.solver_double_fallback = False
        sim.preconditionBiCG = False
        sim.BiCG_precondition_fallback = True
        
        sim.print_adaptive_step_info = True
        
        # def pfn_initial_BiCG_preconditioned(total_step, **kwargs):
            # if total_step>0:
                # sim.preconditionBiCG = False
        # PISOtorch_simulation.append_prep_fn(prep_fn, "PRE", pfn_initial_BiCG_preconditioned)
        
        #PISOtorch_simulation.append_prep_fn(prep_fn, "POST_VELOCITY_SETUP", pfn_check_matrix)
        #PISOtorch_simulation.append_prep_fn(prep_fn, "POST_VELOCITY_SETUP", lambda **kwargs: stop_handler.stop())

        # use prev result
        PISOtorch.CopyVelocityResultFromBlocks(domain)
        
        def log_fn(domain, it, **kwargs):
            vel_stats.log_vel_stats(domain, it, with_reference=True, **kwargs)
            vel_stats.plot_wall_stats(name="TCF550", file_type="pdf")
            if save_steps:
                sim.save_domain("domain_frame_%06d"%(it))
            if use_SGS and it>0:
                log_SGS_viscosity(domain=domain, it=it, **kwargs)
            if compare_steps:
                log_step_comparison(domain=domain, it=it, **kwargs)
            check_CFL(domain, it=it, **kwargs)
        
        sim.log_fn = log_fn
        #sim.log_fn(domain, out_dir=run_dir, it=0)
        check_CFL(domain, out_dir=None, it=-1)
        
        sim.make_divergence_free()
        sim.log_fn(domain, out_dir=run_dir, it=0)
        # run simulation
        sim.run(iterations=iterations)
        
        if sim.total_step>1:
            vel_stats.save_vel_stats()
            ref_budgets = TorrojaBalances("./data/tcf_torroja", 550)
            vel_stats.plot_final_stats(reference_budgets=ref_budgets, total_time=target_t)
            
            #avg_u_wall = vel_stats.get_avg_u_wall()
            #LOG.info("Avg T+=%.03e", TCF_tools.t_to_t_wall(target_t, viscosity, avg_u_wall))
            #LOG.info("Avg u_wall %.03e -> Re_wall %.03e.", avg_u_wall, avg_u_wall / ntonp(domain.viscosity)[0])

        stop_handler.unregister_signal()

    
    elif case=="EVAL": # load sim for further investigation
        
        dtype = torch.float64
        runs = [ # can load multiple runs
            {
                "run_id": "251008-114126", # time-stamp of the run to load
                "Re_wall": 550, # target Re_wall of the run
                "Re_wall_ref": 550, # Re_wall of the reference to plot against. None to disable comparison
                "frames": 100, # number of frames (iterations) from the end of the simulation to use for statistics, can be "ALL"
                "avg_steps": "PARSE", # average number of simulation steps per frame/iteration, can use "PARSE" to parse actual steps from log
                "short_name": "Re550-r64-y32exp", # name of sub-directory for this run evaluation
                "energy_budgets": False, # bool, wether to plot energy budgets, only works when using ALL frames
            },
        ]
        
        run_dir = setup_run("./test_runs",
            name="TCF550-r64-y32exp_vel-stats_f100"
            )
        LOG = get_logger("Main")
        
        for run_idx, run in enumerate(runs):
            
            LOG.info("Stats for run '%s' (R%s)", run["short_name"], run["run_id"])
            sub_dir = os.path.join(run_dir, "%04d_R%s_%s_f%s"%(run_idx, run["run_id"], run["short_name"], run["frames"]))
            
            
            
            load_path = "./test_runs/%s_*"%(run["run_id"],)
            load_path = glob.glob(load_path)[0]
            
            LOG.info("load domain")
            domain = domain_io.load_domain(os.path.join(load_path, "domain"), dtype=dtype, device=cuda_device)
            
            
            Re_wall = run["Re_wall"]
            #assert Re_wall in [180, 550], "Unsupported Re_wall"
            Re_center = TCF_tools.Re_wall_to_cl(Re_wall)
            u_wall = Re_wall / Re_center #1
            #vel_stats.load_references()
            
            if run['frames']=="ALL":
                start = 0
            elif run["avg_steps"]=="PARSE":
                
                sim_frames, sim_substeps = parse_adaptive_substeps(os.path.join(load_path, "log"))
                if len(sim_substeps)<run["frames"]:
                    raise ValueError("The simulation does not have that many frames.")
                end_steps = sum(sim_substeps[-run["frames"]:])
                avg_end_step = end_steps//run["frames"]
                LOG.info("parsed log: %d frames, end steps: %d, avg: %d", run["frames"], end_steps, avg_end_step)
                start = -end_steps
            else:
                end_steps = run["frames"]*run["avg_steps"]
                start = -end_steps
            
            with_energy_budgets = run.get("energy_budgets", False)
            assert not with_energy_budgets or start==0, "slicing energy budgets is not supported"
            
            vel_stats = VelocityStats(domain, sub_dir, LOG, record_online_steps=True, use_energy_budgets=with_energy_budgets, u_wall=u_wall)
            
            only_old = run.get("only_old", False)

            save_stats = run.get("save_stats", start!=0)
            if save_stats:
                LOG.info("will re-save stats after loading")
            
            LOG.info("load reference stats")
            vel_stats.plot_u_vel_max = 25
            vel_stats.plot_vel_rms_max = 3.5
            
            Re_wall_ref = run.get("Re_wall_ref", None)
            if Re_wall_ref is not None:
                assert Re_wall_ref in [180,550], "Unsupported Re_wall"
                profiles_path = "./data/tcf_torroja"
                profiles = TorrojaProfile(profiles_path, Re_wall)
                vel_stats.set_references(profiles)
                #if Re_wall==180:
                    #vel_stats.load_references() # KMM Re180 references
                if "foam_ref" in run:
                    from lib.data.OpenFOAM_profile import OpenFOAMProfile
                    foam_ref = OpenFOAMProfile(run["foam_ref"])
                    vel_stats.add_additional_reference_stats(foam_ref, linestyle="--")
                
                ref_budgets = None
                if with_energy_budgets:
                    ref_budgets = TorrojaBalances(profiles_path, Re_wall)
            
            LOG.info("load test stats")
            stats_path = run.get("stats_dir", os.path.join(load_path, "stats"))
            vel_stats.load_vel_stats(stats_path, start=start, load_online=not only_old, load_energy_budgets=with_energy_budgets, device=cuda_device, dtype=dtype, _load_old=only_old)
            
            if save_stats:
                LOG.info("save stats")
                vel_stats.save_vel_stats(save_steps=False)
            
            LOG.info("plot stats")
            vel_stats.plot_final_stats(file_type="pdf", reference_budgets=ref_budgets)

    close_logging()
