import os, glob, csv
import gc
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

#import SimplexNoiseVariations
import PISOtorch # core CUDA module
import PISOtorch_simulation # helper for PISO loop
import lib.data.shapes as shapes # mesh generation
from lib.util.output import plot_grids, save_block_data_image, save_np_png #, save_domain_images
from lib.util.logging import setup_run, get_logger, close_logging # logging and output
import lib.util.domain_io as domain_io
from lib.util.profiling import SAMPLE
from lib.modules.multiblock_cnn import MultiblockConv, MultiblockReLU
from lib.util.memory_usage import MemoryUsage

from lib.data.online_statistics import WelfordOnlineParallel_Torch, CovarianceOnlineParallel_Torch, PSDOnline_Torch, MultivariateMomentsOnlineParallel_Torch, TurbulentEnergyBudgetsOnlineParallel_Torch
from lib.data.TCF_tools import VelocityStats, PISOTCFProfile
import lib.data.TCF_tools as TCF_tools

from lib.data.torroja import TorrojaProfile, TorrojaBalances, TorrojaSpectra

from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

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

def plot_velocity_stats(domain, path):
    vel = domain.getBlock(0).velocity[0] #NCDHW -> CDHW
    mean_vel_u = torch.mean(vel[0], [0,2]) #CDHW -> H
    pos_y = domain.getBlock(0).getCellCoordinates()[0,1,0,:,0] #NCDHW -> H
    
    mean_vel_u = mean_vel_u.cpu().numpy()
    pos_y = pos_y.cpu().numpy()
    
    plt.clf()
    plt.plot(pos_y, mean_vel_u)
    plt.savefig(os.path.join(path, "u_mean.svg"))
    plt.clf()
    
    mean_vel = torch.mean(vel, [1,3], keepdims=True) # C1H1
    vel_RMS = torch.sqrt(torch.mean(torch.square(vel - mean_vel), [1,3])) #CH
    vel_RMS_u = vel_RMS[0].cpu().numpy()
    vel_RMS_v = vel_RMS[1].cpu().numpy()
    vel_RMS_w = vel_RMS[2].cpu().numpy()
    
    plt.plot(pos_y, vel_RMS_u, "-r")
    plt.plot(pos_y, vel_RMS_v, "-g")
    plt.plot(pos_y, vel_RMS_w, "-b")
    plt.savefig(os.path.join(path, "vel_RMS.svg"))
    plt.clf()

def torch_squeeze_multidim(tensor, dims):
    # dims must be sorted ascending
    for dim in reversed(dims):
        tensor = torch.squeeze(tensor, dim=dim)
    return tensor


def load_domain_from_run(run_id:str, base_path:str="./test_runs", dtype=torch.float32, with_scalar:bool=True, frame:int=None, name=None):
    load_path = os.path.join(base_path, ("%s_*/"+ ("domain" if name is None else name) + ".json")%(run_id,) if frame is None else ("%s_*/" + ("domain_frame_%06d" if name is None else name) + ".json")%(run_id,frame))
    load_path = glob.glob(load_path)
    
    if len(load_path)==0: raise IOError("No domain found for run id %s, frame %s"%(run_id, frame))
    if len(load_path)!=1: raise IOError("No unique domain found for run id %s, frame %s"%(run_id, frame))
    
    load_path = load_path[0][:-5] # without file type extension ".json"
    domain = domain_io.load_domain(load_path, dtype=dtype, device=cuda_device, with_scalar=with_scalar)
    
    LOG.info("loaded domain R%s, frame %s", run_id, frame)
    
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

def make_grid_Re550(run_dir, log, x=128, y_half=48, yN=1, dims=3, dtype=torch.float32):
    assert (x%4)==0 #in [128,192,256]
    assert dims in [2,3]

    delta = 1 # =y_size/2
    x_size = 2*np.pi*delta
    y_size = delta * 2
    z_size = np.pi*delta

    y_weights = make_y_weights_Re550(log, ny_half=y_half, N=yN)
    corners = [(-x_size/2, -delta), (x_size/2, -delta), (-x_size/2, delta), (x_size/2, delta)]

    y = len(y_weights) - 1
    z = x//2

    grid = shapes.generate_grid_vertices_2D([y+1,x+1],corners, None, x_weights=y_weights, dtype=dtype)
    plot_grids([grid], path=run_dir)
    if dims==3:
        grid = shapes.extrude_grid_z(grid, z, start_z=-z_size/2, end_z=z_size/2)
    grid = grid.cuda().contiguous()

    return grid

def get_viscosity_wall_distance(block, domain, u_wall):
    # assumes channel flow centered on 0 with delta=1
    pos_y = block.getCellCoordinates()[:,1:2] # N1DHW
    wall_distance = TCF_tools.pos_to_pos_wall(1 - torch.abs(pos_y), domain.viscosity.to(cuda_device), u_wall) #(1 - torch.abs(pos_y)) * u_wall / domain.viscosity.to(cuda_device)
    
    return wall_distance # NCDHW with C=1

def get_van_driest_sqr(block, domain, u_wall):
    wall_distance = get_viscosity_wall_distance(block, domain, u_wall)
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

def interpolate_ref_statistics(ref_statistics, pos_y, stat_keys=[]):
    assert isinstance(ref_statistics, (TorrojaProfile, PISOTCFProfile))
    ref_pos_y = ref_statistics.get_full_pos_y()
    assert isinstance(pos_y, np.ndarray)
    stats = []
    for key in stat_keys:
        if (not isinstance(ref_statistics, PISOTCFProfile)) and (key in ['V+','W+']):
            stats.append(np.zeros_like(pos_y))
        else:
            stats.append(np.interp(pos_y, ref_pos_y, ref_statistics.get_full_data(key)))
    return stats

def tensor_to_numpy(tensor):
    return tensor.detach().cpu().numpy()
ttonp = tensor_to_numpy

def numerical_to_numpy(data):
    if isinstance(data, (int, float)):
        return np.asarray(data)
    if isinstance(data, torch.Tensor):
        return tensor_to_numpy(data)
    if isinstance(data, np.ndarray):
        return data
    raise TypeError
ntonp = numerical_to_numpy


# simple conv-net to learn velocity -> forcing and/or viscosity
class CorrectionNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, seed=34512468, use_multiblock_conv=True, domain=None):
        torch.manual_seed(seed)
        super(CorrectionNet, self).__init__()
        filters = [in_channels,8,64,64,32,16,8,4]
        if use_multiblock_conv:
            assert isinstance(domain, PISOtorch.Domain)
            self.conv_layers = [MultiblockConv(domain=domain, fixed_boundary_padding="ZERO",
                                    in_channels=filters[i], out_channels=filters[i+1], kernel_size=kernel_size, stride=stride,
                                    device=cuda_device, dtype=dtype)
                                for i in range(len(filters)-1)]
        else:
            self.conv_layers = [torch.nn.Conv3d(filters[i], filters[i+1], kernel_size, stride, padding="same", device=cuda_device, dtype=dtype)
                for i in range(len(filters)-1)]
        self.conv_layers = torch.nn.ModuleList(self.conv_layers) #for correct parameters() and state_dict()
        self.conv_final = torch.nn.Conv3d(filters[-1], out_channels, 1, 1, padding="same", device=cuda_device, dtype=dtype)
        LOG.info("CorrectionNet filter sizes: in %d, hidden %s, out %d", in_channels, filters[1:], out_channels)
        LOG.info("CorrectionNet parameters: total %d, trainable %d", self.get_num_parameters(), self.get_num_parameters(only_trainable=True))
    
    def get_num_parameters(self, only_trainable=False):
        params = self.parameters()
        if only_trainable:
            params = [p for p in params if p.requires_grad]
        num_params = np.sum([np.prod(p.size()) for p in params])
        return num_params
    
    def forward(self, x):
        for conv_layer in self.conv_layers:
            if isinstance(conv_layer, MultiblockConv):
                x = conv_layer([x])[0]
            else:
                x = conv_layer(x)
            x = torch.nn.functional.relu(x)
        x = self.conv_final(x)
        return x

# some loss functions
def MSE(a, b):
    return torch.mean((a - b)**2)
def SSE(a, b):
    return torch.sum((a - b)**2)
    


class TCFcorrectionTrainer:
    def __init__(self, log_path, num_steps, ref_domain, base_forcing, #base_viscosity,
            u_wall,
            time_step, substeps, adaptive_CFL=0.8, fixed_frame=None,
            use_network=False, lr_net=1e-4, network_time_stride=1,
            use_velocity=False, velocity_residual=True, velocity_loss_scale=0, velocity_constraint=None, min_velocity=None, max_velocity=None,
            velocity_divergence_loss_scale=0,
            use_forcing=False, lr_forcing=1e-4, forcing_loss_scale=0, forcing_constraint=None, min_forcing=None, max_forcing=None,
            forcing_divergence_loss_scale=0, use_forcing_offset=False,
            use_viscosity=False, lr_viscosity=1e-4, viscosity_constraint="CLAMP", min_viscosity=1e-5, max_viscosity=None,
            use_SMAG=False, lr_SMAG_C=1e-4, use_van_driest=True, learn_wall_scaling=False,
            input_velocity=True, input_transformation=False, input_wall_distance=True,
            loss_scale_velocity=1, ref_budgets=None,
            ref_profiles = None, total_statistics_loss_scale=1, frame_statistics_loss_scale=0,
            loss_scale_vel_mean=0, loss_scale_vel_mean_v=0, loss_scale_vel_mean_w=0,
            loss_scale_vel_u_var=0, loss_scale_vel_v_var=0, loss_scale_vel_w_var=0, loss_scale_vel_uv_cov=0,
            record_individual_grads=False, stop_fn=lambda: False, data_seed=None, pre_steps_seed=None):
        
        # constraint can be "CLAMP", or for network training also "TANH"
        
        self.log_path = log_path
        self.log = get_logger("Trainer")
        self.stop_fn = stop_fn
        self.pos_y = ttonp(domain.getBlock(0).getCellCoordinates()[0,1,0,:,0]) #NCDHW -> H
        self.d_y = (1+self.pos_y[0], 1-self.pos_y[-1])
        self.u_wall = u_wall
        
        self.mem_usage = MemoryUsage(logger=self.log)
        self.max_memory_GPU = 8 * (1024**3) # 8GB
        
        assert time_step>0
        self.time_step = time_step
        assert substeps=="ADAPTIVE" or substeps>0
        self.substeps = substeps
        assert adaptive_CFL>0 and adaptive_CFL<1
        self.adaptive_CFL = adaptive_CFL
        self.base_forcing = torch.tensor([[base_forcing]+[0,0]], dtype=dtype, device=cuda_device) if base_forcing!="DYNAMIC" else base_forcing # NC, static velocity source
        self.base_viscosity = domain.viscosity.detach().to(cuda_device) #torch.tensor([base_viscosity], dtype=dtype, device=cuda_device)
        base_viscosity = ttonp(self.base_viscosity)[0]
        self.num_steps = num_steps
        
        self.data_rng = np.random.default_rng(data_seed)
        self.pre_steps_rng = np.random.default_rng(pre_steps_seed)

        assert input_velocity
        self.input_velocity = input_velocity
        self.input_transformation = input_transformation
        self.input_wall_distance = None
        if input_wall_distance:
            with torch.no_grad():
                self.input_wall_distance = (1 - torch.abs(domain.getBlock(0).getCellCoordinates()[:,1:2]) ) # this is in [0,1]
                #self.input_wall_distance = TCF_tools.pos_to_pos_wall(self.input_wall_distance, domain.viscosity.to(cuda_device), self.u_wall)
        assert (use_velocity and use_network) or use_forcing or use_viscosity or use_SMAG
        self.use_network = use_network
        self.net_time_stride = 1
        
        self.use_velocity = use_velocity
        self.velocity_residual = velocity_residual
        self.velocity_loss_scale = velocity_loss_scale
        self.velocity_divergence_loss_scale = velocity_divergence_loss_scale
        
        self.use_forcing = use_forcing
        self.forcing_loss_scale = forcing_loss_scale
        self.forcing_divergence_loss_scale = forcing_divergence_loss_scale
        
        self.use_viscosity = use_viscosity
        self.use_SMAG = use_SMAG
        
        if use_network:
            self.net_input_grads = False
            assert isinstance(network_time_stride, int) and network_time_stride>0
            self.net_time_stride = network_time_stride
            in_channels = (3 if self.input_velocity else 0) + (1 if self.input_transformation else 0) + (1 if input_wall_distance else 0) #velocity, cell size
            out_channels = (3 if use_forcing else 0) + (1 if use_viscosity else 0) + (3 if use_velocity else 0)
            self.net = CorrectionNet(in_channels, out_channels, use_multiblock_conv=True, domain=ref_domain)
            opt_vars = list(self.net.parameters())
            self.optimizer = torch.optim.Adam(opt_vars, lr=lr_net)
            assert fixed_frame in list(range(num_steps)) or fixed_frame is None
            self.fixed_frame = fixed_frame
        
        elif use_SMAG:
            self.fixed_frame = fixed_frame
            self.min_SMAG_C, self.max_SMAG_C = 1e-5, 1
            start_SMAG_C = 0.01
            self.learn_SMAG_wall_scaling = learn_wall_scaling
            if self.learn_SMAG_wall_scaling:
                #block_size = ref_domain.getBlock(0).getSizes()
                if use_van_driest:
                    # initialize with wall scaling
                    self.SMAG_C = torch.mean(get_van_driest_sqr(ref_domain.getBlock(0), ref_domain, self.u_wall), dim=(0,2,4), keepdim=True) #N1DHW -> 1,1,1,y,1
                    self.SMAG_C = (self.SMAG_C * start_SMAG_C).to(cuda_device).contiguous()
                else:
                    self.SMAG_C = (torch.ones([1,1,1,block_size.y,1], dtype=ref_domain.getDtype(), device=ref_domain.getDevice())*0.01).contiguous()
                self.SMAG_C.requires_grad = True
            else:
                self.SMAG_C = torch.tensor(start_SMAG_C, dtype=ref_domain.getDtype(), device=ref_domain.getDevice(), requires_grad=True)

            self.use_van_driest = use_van_driest and not self.learn_SMAG_wall_scaling
            if self.use_van_driest:
                self.van_driest_scale_sqr = [get_van_driest_sqr(block, ref_domain, self.u_wall) for block in ref_domain.getBlocks()]
            
            #self.optimizer_SMAG = torch.optim.SGD([self.SMAG_C], lr=lr_SMAG_C, momentum=0)
            self.optimizer_SMAG = torch.optim.Adam([self.SMAG_C], lr=lr_SMAG_C)

        else:
            assert fixed_frame in list(range(num_steps))
            self.fixed_frame = fixed_frame
            if use_velocity:
                raise NotImplementedError
            if use_forcing:
                self.learned_forcings = [torch.zeros_like(block.velocity, requires_grad=True) for frame in range(num_steps) for block in ref_domain.getBlocks()]
                self.optimizer_forcing = torch.optim.SGD(self.learned_forcings, lr=lr_forcing, momentum=0)
            if use_viscosity:
                base_visc = ref_domain.viscosity.detach().to(cuda_device)
                self.learned_viscosities = [(torch.ones_like(block.pressure)*base_visc).contiguous() for frame in range(num_steps) for block in ref_domain.getBlocks()]
                for visc in self.learned_viscosities:
                    visc.requires_grad = True
                self.optimizer_viscosity = torch.optim.SGD(self.learned_viscosities, lr=lr_viscosity, momentum=0)
        
        if use_velocity:
            assert velocity_constraint in [None, "CLAMP", "TANH"] and torch.all(min_velocity<=max_velocity)
            self.velocity_constraint = velocity_constraint
            self.min_velocity, self.max_velocity = min_velocity, max_velocity
        else:
            self.velocity_constraint = None
            self.min_velocity, self.max_velocity = 0,0
        if use_forcing:
            assert forcing_constraint in [None, "CLAMP", "TANH"] and torch.all(min_forcing<=max_forcing)
            self.forcing_constraint = forcing_constraint
            self.min_forcing, self.max_forcing = min_forcing, max_forcing
            self.use_forcing_offset = use_forcing_offset
            self.forcing_offset = None
        else:
            self.forcing_constraint = None
            self.min_forcing, self.max_forcing = 0,0
        if use_viscosity:
            assert viscosity_constraint=="CLAMP" and 0<=min_viscosity and (max_viscosity is None or min_viscosity<=max_viscosity)
            self.viscosity_constraint = viscosity_constraint
            self.min_viscosity, self.max_viscosity = min_viscosity - base_viscosity, max_viscosity #min_viscosity - base_viscosity, None
        else:
            self.viscosity_constraint = None
            self.min_viscosity, self.max_viscosity = 0,0
        
        self.SMAG_Cs = []
        self.SMAG_C_grads = []
        self.velocity_stats = []
        self.velocity_grad_stats = []
        self.forcing_stats = []
        self.forcing_grad_stats = []
        self.viscosity_stats = []
        self.viscosity_grad_stats = []
        self.stats_steps = []

        
        self.use_velocity_loss = loss_scale_velocity>0
        self.loss_scale_velocity = loss_scale_velocity
        self.loss_fn = MSE
        
        self.total_statistics_loss_scale = total_statistics_loss_scale
        self.frame_statistics_loss_scale = frame_statistics_loss_scale
        self.use_statistics_loss_vel = loss_scale_vel_mean>0 or loss_scale_vel_u_var>0 or loss_scale_vel_v_var>0 or loss_scale_vel_w_var>0 
        self.use_statistics_loss_cov =  loss_scale_vel_uv_cov>0
        assert self.use_velocity_loss or self.use_statistics_loss_vel or self.use_statistics_loss_cov, "no loss specified."
        self._is_total_statistics_loss_active = (self.use_statistics_loss_vel or self.use_statistics_loss_cov) and self.total_statistics_loss_scale>0
        self._is_frame_statistics_loss_active = (self.use_statistics_loss_vel or self.use_statistics_loss_cov) and self.frame_statistics_loss_scale>0
        self._is_any_statistics_loss_active = self._is_total_statistics_loss_active or self._is_frame_statistics_loss_active
        
        if (self.use_statistics_loss_vel or self.use_statistics_loss_cov) and ref_profiles is None:
            raise ValueError("No reference profile provided for statistics loss")
        self.ref_profiles = ref_profiles#TorrojaProfile("./data/tcf_torroja", 550)
        self.ref_budgets = ref_budgets
        
        self.statistics_losses = {}
        self.record_individual_grads = False
        if self._is_any_statistics_loss_active:
            if loss_scale_vel_mean>0:
                self.statistics_losses["U+"] = {"scale": loss_scale_vel_mean,
                    "ref": torch.tensor(interpolate_ref_statistics(self.ref_profiles, self.pos_y, ["U+"])[0], device=ref_domain.getDevice(), dtype=ref_domain.getDtype()),
                    "loss_fn": lambda vel_stats, cov_stats: self.statistics_losses["U+"]["scale"] * self.loss_fn(self._vel_to_wall(vel_stats.mean[0,0]), self.statistics_losses["U+"]["ref"])}
            if loss_scale_vel_mean_v>0:
                if not isinstance(self.ref_profiles, PISOTCFProfile):
                    self.log.info("Only PISOTCFProfile supports reference V+ mean statistics. using 0 instead.")
                    #raise TypeError("Only PISOTCFProfile supports reference V+ and W+ mean statistics.")
                self.statistics_losses["V+"] = {"scale": loss_scale_vel_mean_v,
                    "ref": torch.tensor(interpolate_ref_statistics(self.ref_profiles, self.pos_y, ["V+"])[0], device=ref_domain.getDevice(), dtype=ref_domain.getDtype()),
                    "loss_fn": lambda vel_stats, cov_stats: self.statistics_losses["V+"]["scale"] * self.loss_fn(self._vel_to_wall(vel_stats.mean[0,1]), self.statistics_losses["V+"]["ref"])}
            if loss_scale_vel_mean_w>0:
                if not isinstance(self.ref_profiles, PISOTCFProfile):
                    self.log.info("Only PISOTCFProfile supports reference W+ mean statistics. using 0 instead.")
                    #raise TypeError("Only PISOTCFProfile supports reference V+ and W+ mean statistics.")
                self.statistics_losses["W+"] = {"scale": loss_scale_vel_mean_w,
                    "ref": torch.tensor(interpolate_ref_statistics(self.ref_profiles, self.pos_y, ["W+"])[0], device=ref_domain.getDevice(), dtype=ref_domain.getDtype()),
                    "loss_fn": lambda vel_stats, cov_stats: self.statistics_losses["W+"]["scale"] * self.loss_fn(self._vel_to_wall(vel_stats.mean[0,2]), self.statistics_losses["W+"]["ref"])}
            if loss_scale_vel_u_var>0:
                self.statistics_losses["u'+"] = {"scale": loss_scale_vel_u_var,
                    "ref": torch.tensor(interpolate_ref_statistics(self.ref_profiles, self.pos_y, ["u'+"])[0], device=ref_domain.getDevice(), dtype=ref_domain.getDtype()),
                    "loss_fn": lambda vel_stats, cov_stats: self.statistics_losses["u'+"]["scale"] * self.loss_fn(self._vel_to_wall(vel_stats.standard_deviation[0,0]), self.statistics_losses["u'+"]["ref"])}
            if loss_scale_vel_v_var>0:
                self.statistics_losses["v'+"] = {"scale": loss_scale_vel_v_var,
                    "ref": torch.tensor(interpolate_ref_statistics(self.ref_profiles, self.pos_y, ["v'+"])[0], device=ref_domain.getDevice(), dtype=ref_domain.getDtype()),
                    "loss_fn": lambda vel_stats, cov_stats: self.statistics_losses["v'+"]["scale"] * self.loss_fn(self._vel_to_wall(vel_stats.standard_deviation[0,1]), self.statistics_losses["v'+"]["ref"])}
            if loss_scale_vel_w_var>0:
                self.statistics_losses["w'+"] = {"scale": loss_scale_vel_w_var,
                    "ref": torch.tensor(interpolate_ref_statistics(self.ref_profiles, self.pos_y, ["w'+"])[0], device=ref_domain.getDevice(), dtype=ref_domain.getDtype()),
                    "loss_fn": lambda vel_stats, cov_stats: self.statistics_losses["w'+"]["scale"] * self.loss_fn(self._vel_to_wall(vel_stats.standard_deviation[0,2]), self.statistics_losses["w'+"]["ref"])}
            if loss_scale_vel_uv_cov>0:
                self.statistics_losses["uv'+"] = {"scale": loss_scale_vel_uv_cov,
                    "ref": torch.tensor(interpolate_ref_statistics(self.ref_profiles, self.pos_y, ["uv'+"])[0], device=ref_domain.getDevice(), dtype=ref_domain.getDtype()),
                    "loss_fn": lambda vel_stats, cov_stats: self.statistics_losses["uv'+"]["scale"] * self.loss_fn(self._vel_to_wall(-cov_stats.covariance[0,0], order=2), self.statistics_losses["uv'+"]["ref"])}
            
            self.record_individual_grads = record_individual_grads
            if len(self.statistics_losses)>0:
                for k, v in self.statistics_losses.items():
                    v["steps"] = []
                    if self._is_total_statistics_loss_active:
                        v["losses"] = []
                    if self._is_frame_statistics_loss_active:
                        v["frame_losses"] = []
                    
                    if self.use_velocity:
                        v["velocity_grads"] = []
                        v["velocity_grad_steps"] = []
                        v["velocity_frame_grads"] = []
                        v["velocity_frame_grad_steps"] = []
                    if self.use_forcing:
                        v["forcing_grads"] = []
                        v["forcing_grad_steps"] = []
                        v["forcing_frame_grads"] = []
                        v["forcing_frame_grad_steps"] = []
                    if self.use_viscosity:
                        v["viscosity_grads"] = []
                        v["viscosity_grad_steps"] = []
                        v["viscosity_frame_grads"] = []
                        v["viscosity_frame_grad_steps"] = []
            
            if self._is_total_statistics_loss_active:
                if self.use_statistics_loss_vel:
                    self.loss_vel_stats = WelfordOnlineParallel_Torch([2,4])
                if self.use_statistics_loss_cov:
                    self.loss_cov_stats = CovarianceOnlineParallel_Torch([2,4])
        
        self.record_loss_per_frame = False
        self.losses = []
        
        self.losses_velocity = []
        self.grads_loss_velocity = []
        self.steps_grad_loss_velocity = []
        self.grads_div_velocity = []
        self.steps_grad_div_velocity = []
        self.apply_velocity_div_grad = False
        
        self.losses_forcing = []
        self.grads_loss_forcing = []
        self.steps_grad_loss_forcing = []
        self.grads_div_forcing = []
        self.steps_grad_div_forcing = []
        self.apply_forcing_div_grad = False
        
        self.reset_losses()
        self.opt_step = 0
        self.exclude_advection_solve_gradients = False
        self.exclude_pressure_solve_gradients = False
        self.preconditionBiCG = False
        self.BiCG_precondition_fallback = True
        
        self.evals = 0
        self.is_training = False
        self.record_stats = False
    
    def _vel_to_wall(self, data, order=1):
        return TCF_tools.vel_to_vel_wall(data, self.u_wall, order=order)

    def set_lr_net(self, lr):
        if self.use_network:
            for g in self.optimizer.param_groups:
                g['lr'] = lr


    def load_reference(self, load_run_id, down_factors, make_div_free=True, load_down=False,
            train_start=0, train_end=None):
        if load_down:
            self.references_low = [
                    load_domain_from_run(load_run_id, dtype=dtype, with_scalar=False, frame=i, name="domain_down_frame_%06d")
                for i in range(self.num_steps+1)]
        elif down_factors is not None:
            self.references_low = [coarsen_Re550_domain(
                    load_domain_from_run(load_run_id, dtype=dtype, with_scalar=False, frame=i),
                    down_factors, make_div_free=make_div_free)
                for i in range(self.num_steps+1)]
        else:
            self.references_low = [
                    load_domain_from_run(load_run_id, dtype=dtype, with_scalar=False, frame=i)
                for i in range(self.num_steps+1)]
        
        if train_end is None:
            train_end = len(self.references_low)
        assert 0<=train_start and train_start<train_end and train_end<=len(self.references_low)
        self.train_start = train_start
        self.train_end = train_end
    
    def get_reference_clone(self, frame):
        domain = self.references_low[frame].Clone()
        #domain.Detach() # does not work on uninitialized domain
        return domain
    
    def get_resampling_params(self, domain):
        dims = 3
        grids = domain.getVertexCoordinates()
        if len(grids)!=1:
            raise ValueError("only single-block domains supported.")
        
        sizes = domain.getBlock(0).getSizes() # resolution
        # physical size
        x_size = np.abs(ttonp(grids[0][0,0,0,0,-1] - grids[0][0,0,0,0,0]))
        y_size = np.abs(ttonp(grids[0][0,1,0,-1,0] - grids[0][0,1,0,0,0]))
        z_size = np.abs(ttonp(grids[0][0,2,-1,0,0] - grids[0][0,2,0,0,0]))
        
        #self.log.info("get_resampling_params: %s, %s, %s, %s", sizes, x_size, y_size, z_size)
        
        resample_norm = sizes.y/y_size * 0.5
        if dims==2:
            output_resampling_shape = [int(x_size*resample_norm), int(y_size*resample_norm)]
        else:
            output_resampling_shape = [int(x_size*resample_norm),  int(y_size*resample_norm), int(z_size*resample_norm)]
        
        return grids, output_resampling_shape
    
    def _get_base_forcing(self, domain, broadcastable=True):
        if self.base_forcing!="DYNAMIC":
            forcing = self.base_forcing
        else:
            block = domain.getBlock(0)
            viscosity = domain.viscosity.to(domain.getDevice())
            
            mean_vel_u = torch.mean(block.velocity[0,0], dim=(0,2))
            tau_wall_n = viscosity * mean_vel_u[0]/self.d_y[0]
            tau_wall_p = viscosity * mean_vel_u[-1]/self.d_y[-1]
            
            forcing = (tau_wall_n + tau_wall_p)*0.5
            forcing = torch.tensor([[forcing]+[0,0]], dtype=domain.getDtype(), device=domain.getDevice())
        if broadcastable:
            forcing = forcing.reshape([1,3,1,1,1])
        return forcing

    def pfn_apply_base_forcing(self, domain, **kwargs):
        for block in domain.getBlocks():
            block.setVelocitySource(self._get_base_forcing(domain, broadcastable=False))
        domain.UpdateDomainData()

    def pfn_apply_correction(self, domain, total_step, **kwargs):
        with SAMPLE("apply correction"):
            if self.use_network:
                if (total_step%self.net_time_stride) != 0:
                    return
                if not domain.getNumBlocks()==1:
                    raise ValueError("only single-block domains supported.")
                block = domain.getBlock(0)
                #self.log.info("applying network correction in step %d", total_step)
                
                net_input = []
                if self.input_velocity:
                    net_input.append(block.velocity)
                if self.input_transformation:
                    net_input.append(torch.unsqueeze(block.transform[...,-1], 1)) # NDHWC -> NCDHW
                if self.input_wall_distance is not None:
                    net_input.append(self.input_wall_distance)
                net_input = torch.cat(net_input, dim=1)
                if not self.net_input_grads:
                    net_input = net_input.detach() # avoid random gradient from network initialization to past frames
                with SAMPLE("net"):
                    net_output = self.net(net_input)
                
                if self.use_forcing and self.use_viscosity and not self.use_velocity:
                    forcing, viscosity = torch.split(net_output, [3,1], dim=1)
                elif self.use_velocity:
                    velocity = net_output * 0.1
                elif self.use_forcing:
                    forcing = net_output
                elif self.use_viscosity:
                    viscosity = net_output * 0.01
                else:
                    raise RuntimeError
                
                if self.use_velocity:
                    if self.velocity_constraint == "TANH":
                        velocity = (torch.tanh(velocity) + 1)*(0.5*(self.max_velocity - self.min_velocity)) + self.min_velocity
                    if self.record_stats:
                        if self.is_training: velocity.retain_grad()
                        self.learned_velocities.append(velocity) # for stats
                    
                    if self.is_training:
                        if self.velocity_loss_scale>0:
                            self.velocity_loss = self.velocity_loss + self.loss_fn(velocity, 0)
                        if self.velocity_divergence_loss_scale>0:
                            velocity = self.add_velocity_divergence_pressure_grad(velocity)
                    
                    if self.velocity_constraint == "CLAMP":
                        velocity = torch.clamp(velocity, self.min_velocity, self.max_velocity)
                    
                    if self.velocity_residual:
                        block.setVelocity(velocity + block.velocity)
                    else:
                        block.setVelocity(velocity)
                
                if self.use_forcing:
                    if self.forcing_constraint == "TANH":
                        forcing = (torch.tanh(forcing) + 1)*(0.5*(self.max_forcing - self.min_forcing)) + self.min_forcing
                    if self.record_stats:
                        if self.is_training: forcing.retain_grad()
                        self.learned_forcings.append(forcing) # for stats
                    
                    if self.is_training:
                        if self.forcing_loss_scale>0:
                            self.forcing_loss = self.forcing_loss + self.loss_fn(forcing, 0)
                        if self.forcing_divergence_loss_scale>0:
                            forcing = self.add_forcing_divergence_pressure_grad(forcing)
                    
                    if self.forcing_constraint == "CLAMP":
                        forcing = torch.clamp(forcing, self.min_forcing, self.max_forcing)
                    block.setVelocitySource(forcing + self._get_base_forcing(domain))
                
                if self.use_viscosity:
                    if self.record_stats:
                        if self.is_training: viscosity.retain_grad()
                        self.learned_viscosities.append(viscosity)
                    if self.viscosity_constraint == "CLAMP":
                        viscosity = torch.clamp(viscosity, self.min_viscosity, self.max_viscosity) # negative viscosity breaks the simulation
                    block.setViscosity(viscosity + self.base_viscosity)
            
            elif self.use_SMAG:
                # SGSviscosityIncompressibleSmagorinsky is not differentiable. We use C=1 here and multiply the real C later for differentiability
                block_SGS_viscosity = PISOtorch.SGSviscosityIncompressibleSmagorinsky(domain, torch.ones([1], device=cpu_device, dtype=domain.getDtype()))
                base_viscosity = domain.viscosity.to(cuda_device)
                for idx, (block, visc) in enumerate(zip(domain.getBlocks(), block_SGS_viscosity)):
                    if self.use_van_driest:
                        visc = visc * self.van_driest_scale_sqr[idx]
                    visc = visc * self.SMAG_C
                    visc = visc + base_viscosity
                    block.setViscosity(visc)
                
                domain.UpdateDomainData()
            else:
                num_blocks = domain.getNumBlocks()
                for idx_block, block in enumerate(domain.getBlocks()):
                    if self.use_forcing:
                        block.setVelocitySource(self.learned_forcings[total_step*num_blocks + idx_block] + self._get_base_forcing(domain))
                    if self.use_viscosity:
                        block.setViscosity(self.learned_viscosities[total_step*num_blocks + idx_block])
                domain.UpdateDomainData()
        
    def pfn_clear_correction(self, domain, **kwargs):
        for block in domain.getBlocks():
            if self.use_forcing:
                block.setVelocitySource(self._get_base_forcing(domain, broadcastable=False))
            if self.use_viscosity or self.use_SMAG:
                block.clearViscosity()
        domain.UpdateDomainData()

    def lfn_add_loss(self, domain, it, **kwargs):
        with SAMPLE("loss"):
            loss = 0
            frame = self.frame_offset + it
            has_frame_loss = False

            #num_cells = 0
            if self.use_velocity_loss:
                with SAMPLE("vel"):
                    for block, ref_block in zip(domain.getBlocks(), self.references_low[frame].getBlocks()):
                        loss = loss + self.loss_fn(block.velocity, ref_block.velocity) * self.loss_scale_velocity
                        #num_cells += block.getStrides().w
                has_frame_loss = True
            
            if self._is_frame_statistics_loss_active:
                with SAMPLE("frame stats"):
                    block = domain.getBlock(0)
                    
                    frame_vel_stats = None
                    if self.use_statistics_loss_vel:
                        frame_vel_stats = WelfordOnlineParallel_Torch([2,4])
                        frame_vel_stats.update_from_data(block.velocity)
                    
                    frame_cov_stats = None
                    if self.use_statistics_loss_cov:
                        frame_cov_stats = CovarianceOnlineParallel_Torch([2,4])
                        frame_cov_stats.update_from_data(block.velocity[:,:1],block.velocity[:,1:2])
                    
                    for key, stats_loss in self.statistics_losses.items():
                        temp_loss = stats_loss["loss_fn"](frame_vel_stats, frame_cov_stats) * self.frame_statistics_loss_scale
                        if self.record_stats:
                            stats_loss["frame_losses"][-1].append(ntonp(temp_loss))
                            if self.record_individual_grads and self.is_training:
                                stats_loss["frame_loss_tensor"] = stats_loss["frame_loss_tensor"] + temp_loss
                        loss = loss + temp_loss
                        has_frame_loss = True
            
            if has_frame_loss:
                with SAMPLE("norm"):
                    loss = loss / domain.getNumBlocks()#num_cells #
                    
                    if self.record_loss_per_frame:
                        self.frame_losses.append(loss)
                    
                    self.loss = self.loss + loss

    def pfn_add_loss_statistics(self, domain, **kwargs):
        with SAMPLE("loss stats"):
            if self.use_statistics_loss_vel:
                self.loss_vel_stats.update_from_data(domain.getBlock(0).velocity)
            if self.use_statistics_loss_cov:
                self.loss_cov_stats.update_from_data(domain.getBlock(0).velocity[:,:1],domain.getBlock(0).velocity[:,1:2])
    
    def get_forcing_divergence_pressure_grad(self, forcing):
        with torch.no_grad():
            domain = self.get_reference_clone(0)
            domain.getBlock(0).setVelocity(forcing.clone()) #Don't overwrite saved forcing
            domain.getBlock(0).clearVelocitySource()
            domain.PrepareSolve()
            sim = PISOtorch_simulation.Simulation(domain=domain, pressure_tol=1e-7, pressure_return_best_result=True)
            sim.make_divergence_free()
            pressure_grad_flat = PISOtorch.ComputePressureGradient(domain, False, 0)
            domain.setVelocityResult(pressure_grad_flat)
            domain.UpdateDomainData()
            PISOtorch.CopyVelocityResultToBlocks(domain)
            return domain.getBlock(0).velocity
    
    def add_forcing_divergence_pressure_grad(self, forcing):
        
        class DivergenceGradientAdder(torch.autograd.Function):
                @staticmethod
                def forward(ctx, forcing):
                    ctx.save_for_backward(forcing)
                    return forcing
                
                @staticmethod
                @torch.autograd.function.once_differentiable
                def backward(ctx, forcing_grad):
                    if ctx.needs_input_grad[0]:
                        if self.apply_forcing_div_grad:
                            with SAMPLE("DivergenceGradientAdder.backwards"):
                                forcing = ctx.saved_tensors[0]
                                try:
                                    forcing_div_grad = self.get_forcing_divergence_pressure_grad(forcing) * self.forcing_divergence_loss_scale
                                except:
                                    self.log.exception("Forcing divergence pressure solve failed!")
                                    self.stop_fn.stop()
                                    np.savez_compressed(os.path.join(self.log_path, "_ERR_S.npz"), forcing=forcing)
                                    domain = self.get_reference_clone(0)
                                    save_block_data_image(forcing, domain, self.log_path, "_ERR_S-x-norm", it, normalize=True, axis3D=2)
                                    save_block_data_image(forcing, domain, self.log_path, "_ERR_S-y-norm", it, normalize=True, axis3D=1)
                                    save_block_data_image(forcing, domain, self.log_path, "_ERR_S-z-norm", it, normalize=True, axis3D=0)
                                else:
                                    if self.record_stats:
                                        self.grads_div_forcing_frames.append(ttonp(torch.mean(torch.abs(forcing_div_grad))))
                                    forcing_grad = forcing_grad + forcing_div_grad
                        return forcing_grad
                    else:
                        return None
        
        return DivergenceGradientAdder.apply(forcing)
    
    def add_velocity_divergence_pressure_grad(self, velocity):
        
        class VelocityDivergenceGradientAdder(torch.autograd.Function):
                @staticmethod
                def forward(ctx, velocity):
                    ctx.save_for_backward(velocity)
                    return velocity
                
                @staticmethod
                @torch.autograd.function.once_differentiable
                def backward(ctx, velocity_grad):
                    if ctx.needs_input_grad[0]:
                        if self.apply_velocity_div_grad:
                            with SAMPLE("VelocityDivergenceGradientAdder.backwards"):
                                velocity = ctx.saved_tensors[0]
                                try:
                                    velocity_div_grad = self.get_forcing_divergence_pressure_grad(velocity) * self.velocity_divergence_loss_scale
                                except:
                                    self.log.exception("Velocity divergence pressure solve failed!")
                                    self.stop_fn.stop()
                                    np.savez_compressed(os.path.join(self.log_path, "_ERR_v.npz"), velocity=velocity)
                                    domain = self.get_reference_clone(0)
                                    save_block_data_image(velocity, domain, self.log_path, "_ERR_v-x-norm", it, normalize=True, axis3D=2)
                                    save_block_data_image(velocity, domain, self.log_path, "_ERR_v-y-norm", it, normalize=True, axis3D=1)
                                    save_block_data_image(velocity, domain, self.log_path, "_ERR_v-z-norm", it, normalize=True, axis3D=0)
                                else:
                                    if self.record_stats:
                                        self.grads_div_velocity_frames.append(ttonp(torch.mean(torch.abs(velocity_div_grad))))
                                    velocity_grad = velocity_grad + velocity_div_grad
                        return velocity_grad
                    else:
                        return None
        
        return VelocityDivergenceGradientAdder.apply(velocity)
    
    @property
    def learned_velocity_grads(self):
        return [f.grad for f in self.learned_velocities if f.grad is not None]
    
    @property
    def learned_forcing_grads(self):
        return [f.grad for f in self.learned_forcings if f.grad is not None]
    @property
    def learned_viscosity_grads(self):
        return [f.grad for f in self.learned_viscosities if f.grad is not None]
    
    def reset_losses(self):
        self.loss = 0
        self.frame_losses = []
        self.velocity_loss = 0
        self.grads_div_velocity_frames = []
        self.forcing_loss = 0
        self.grads_div_forcing_frames = []
        if self.use_network:
            if self.use_velocity:
                self.learned_velocities = []
            if self.use_forcing:
                self.learned_forcings = []
            if self.use_viscosity:
                self.learned_viscosities = []
        if self.total_statistics_loss_scale>0:
            if self.use_statistics_loss_vel:
                self.loss_vel_stats = WelfordOnlineParallel_Torch([2,4])
            if self.use_statistics_loss_cov:
                self.loss_cov_stats = CovarianceOnlineParallel_Torch([2,4])
    
    def step_optimizer(self):
        if self.use_network:
            self.optimizer.step()
        elif self.use_SMAG:
            self.optimizer_SMAG.step()
        else:
            if self.use_forcing:
                self.optimizer_forcing.step()
            if self.use_viscosity:
                self.optimizer_viscosity.step()
    
    def zero_grad(self):
        if self.use_network:
            self.optimizer.zero_grad()
            # these are necessary to cleanly handle the per-loss forcing/viscosity gradient reporting
            if self.use_velocity:
                for velocity_grad in self.learned_velocity_grads:
                    velocity_grad.detach_()
                    velocity_grad.zero_()
            if self.use_forcing:
                for forcing_grad in self.learned_forcing_grads:
                    forcing_grad.detach_()
                    forcing_grad.zero_()
            if self.use_viscosity:
                for viscosity_grad in self.learned_viscosity_grads:
                    viscosity_grad.detach_()
                    viscosity_grad.zero_()
        elif self.use_SMAG:
            self.optimizer_SMAG.zero_grad()
        else:
            if self.use_forcing:
                self.optimizer_forcing.zero_grad()
            if self.use_viscosity:
                self.optimizer_viscosity.zero_grad()
    
    def clamp_learned_SMAG_C(self):
        if self.use_SMAG:
            with torch.no_grad():
                self.SMAG_C.copy_(torch.clamp(self.SMAG_C, self.min_SMAG_C, self.max_SMAG_C))

    def clamp_learned_viscosity(self):
        if not self.use_network and not self.use_SMAG and self.use_viscosity:
            with torch.no_grad():
                for visc in self.learned_viscosities:
                    visc.copy_(torch.clamp(visc, self.min_viscosity, self.max_viscosity))
    
    def _get_pre_steps(self, pre_steps):
        if isinstance(pre_steps, (int,)) and pre_steps>=0:
            return pre_steps
        elif isinstance(pre_steps, (list, tuple)) and len(pre_steps)<3:
            return self.pre_steps_rng.integers(*pre_steps)
        elif callable(pre_steps):
            return pre_steps(self.opt_step)
        else:
            raise ValueError("Unsupported pre-steps: %s", pre_steps)

    def optimization_step(self, step, train_sim_steps=1, pre_steps=0, log_images=True, batch_size=1):
        self.reset_losses()
        self.is_training = True
        
        #if batch_size!=1 and self.record_stats:
        #    raise NotImplementedError("batches not supported when recording loss statistics")
        base_record_stats = self.record_stats
        
        pre_steps = self._get_pre_steps(pre_steps)

        with SAMPLE("opt step"):
            min_frame = self.train_start
            max_frame = (self.train_end + 1 - train_sim_steps) if self.use_velocity_loss else self.train_end
            
            batch_loss = 0
            batch_velocity_loss = 0
            batch_forcing_loss = 0
            for batch in range(batch_size):
                # loss statistics and gradient recording does not support batches (recording logistics and zero_grad), but first sample should be fine
                self.record_stats = base_record_stats and batch==0
                
                with SAMPLE("batch sample"):
                    step_path = os.path.join(self.log_path, "step_%04d"%self.opt_step) if log_images and batch==0 else None
                    if self.fixed_frame is not None:
                        start_frame = self.fixed_frame
                    else:
                        start_frame = self.data_rng.integers(min_frame, max_frame) #np.random.randint(min_frame, max_frame)

                    self.frame_offset = start_frame
                    #self.lfn_reset_forcing_substep()
                    
                    with SAMPLE("setup domain"):
                        opt_domain = self.get_reference_clone(start_frame) # there are issues with freeing a domain that was used for differentiable mode.
                        opt_domain.PrepareSolve()
                        
                    
                    if self.record_stats and self._is_frame_statistics_loss_active:
                        for key, stats_loss in self.statistics_losses.items():
                            stats_loss["frame_losses"].append([])
                            if self.record_individual_grads:
                                stats_loss["frame_loss_tensor"] = 0
                    
                    with SAMPLE("FWD"):
                        prep_fn = {}
                        PISOtorch_simulation.append_prep_fn(prep_fn, "PRE", self.pfn_apply_correction)
                        #PISOtorch_simulation.append_prep_fn(prep_fn, "POST", self.pfn_clear_correction)
                        
                        sim = PISOtorch_simulation.Simulation(opt_domain, prep_fn=prep_fn,
                            substeps=self.substeps, time_step=self.time_step,
                            advection_tol=1e-6, pressure_tol=1e-6,
                            non_orthogonal=False, differentiable=True, 
                            log_images=log_images and batch==0, #(opt_step==0 or opt_step==(self.optimization_steps-1)), #only log initial and last step
                            #output_resampling_coords=[grid], output_resampling_shape=output_resampling_shape,
                            #output_resampling_fill_max_steps=20,
                            #log_fn = self.lfn_add_loss, #if self.use_velocity_loss else None,
                            log_interval=1 if self.record_stats else 0, norm_vel=True,
                            log_dir=step_path, stop_fn=self.stop_fn)
                        
                        sim.exclude_advection_solve_gradients = self.exclude_advection_solve_gradients
                        sim.exclude_pressure_solve_gradients = self.exclude_pressure_solve_gradients
                        sim.preconditionBiCG = self.preconditionBiCG
                        sim.BiCG_precondition_fallback = self.BiCG_precondition_fallback
                        
                        
                        if pre_steps>0:
                            with SAMPLE("PRE"), torch.no_grad():
                                self.is_training = False
                                sim.differentiable = False
                                sim.run(pre_steps, log_domain=False)
                                sim.differentiable = True
                                self.is_training = True
                        
                        sim.log_fn = self.lfn_add_loss
                        if self._is_total_statistics_loss_active:
                            PISOtorch_simulation.append_prep_fn(prep_fn, "POST", self.pfn_add_loss_statistics)
                        
                        sim.run(train_sim_steps, log_domain=False)
                    
                    self.apply_velocity_div_grad = False
                    self.apply_forcing_div_grad = False
                    #self.log.info("loss %s", ttonp(self.loss))
                    loss = 0
                    loss_norm = train_sim_steps*batch_size
                    #if self.use_velocity_loss:
                    loss = loss + self.loss / loss_norm
                    
                    if self.record_stats and self._is_any_statistics_loss_active:
                        for key, stats_loss in self.statistics_losses.items():
                            stats_loss["steps"].append(self.opt_step)
                    
                    if self._is_total_statistics_loss_active:
                        for key, stats_loss in self.statistics_losses.items():
                            temp_loss = stats_loss["loss_fn"](self.loss_vel_stats, self.loss_cov_stats) * (self.total_statistics_loss_scale / batch_size)
                            self.log.info("%s loss: %.03e", key, ttonp(temp_loss))
                            if self.record_stats:
                                stats_loss["losses"].append(ttonp(temp_loss))
                                if self.record_individual_grads and (self.use_velocity or self.use_forcing or self.use_viscosity):
                                    with SAMPLE("loss " + key + " grad"):
                                        temp_loss.backward(retain_graph=True)
                                        with torch.no_grad():
                                            if self.use_velocity:
                                                stats_loss["velocity_grads"].append(np.mean([ttonp(torch.mean(torch.abs(g))) for g in self.learned_velocity_grads]))
                                                stats_loss["velocity_grad_steps"].append(self.opt_step)
                                                self.log.info("-> mean abs velocity grad: %.03e", stats_loss["velocity_grads"][-1])
                                            if self.use_forcing:
                                                stats_loss["forcing_grads"].append(np.mean([ttonp(torch.mean(torch.abs(g))) for g in self.learned_forcing_grads]))
                                                stats_loss["forcing_grad_steps"].append(self.opt_step)
                                                self.log.info("-> mean abs forcing grad: %.03e", stats_loss["forcing_grads"][-1])
                                            if self.use_viscosity:
                                                stats_loss["viscosity_grads"].append(np.mean([ttonp(torch.mean(torch.abs(g))) for g in self.learned_viscosity_grads]))
                                                stats_loss["viscosity_grad_steps"].append(self.opt_step)
                                                self.log.info("-> mean abs viscosity grad: %.03e", stats_loss["viscosity_grads"][-1])
                                        self.zero_grad()
                            loss = loss + temp_loss
                            del temp_loss
                    
                    if self._is_frame_statistics_loss_active and self.record_stats and self.record_individual_grads:
                        for key, stats_loss in self.statistics_losses.items():
                            with SAMPLE("loss " + key + " frame grad"):
                                temp_loss = stats_loss["frame_loss_tensor"] / loss_norm
                                self.log.info("%s frame loss: %.03e", key, ttonp(temp_loss))
                                temp_loss.backward(retain_graph=True)
                            with torch.no_grad():
                                if self.use_velocity:
                                    stats_loss["velocity_frame_grads"].append(np.mean([ttonp(torch.mean(torch.abs(g))) for g in self.learned_velocity_grads]))
                                    stats_loss["velocity_frame_grad_steps"].append(self.opt_step)
                                    self.log.info("-> mean abs velocity grad: %.03e", stats_loss["velocity_frame_grads"][-1])
                                if self.use_forcing:
                                    stats_loss["forcing_frame_grads"].append(np.mean([ttonp(torch.mean(torch.abs(g))) for g in self.learned_forcing_grads]))
                                    stats_loss["forcing_frame_grad_steps"].append(self.opt_step)
                                    self.log.info("-> mean abs forcing grad: %.03e", stats_loss["forcing_frame_grads"][-1])
                                if self.use_viscosity:
                                    stats_loss["viscosity_frame_grads"].append(np.mean([ttonp(torch.mean(torch.abs(g))) for g in self.learned_viscosity_grads]))
                                    stats_loss["viscosity_frame_grad_steps"].append(self.opt_step)
                                    self.log.info("-> mean abs viscosity grad: %.03e", stats_loss["viscosity_frame_grads"][-1])
                            del stats_loss["frame_loss_tensor"]
                            del temp_loss
                            self.zero_grad()
                    
                    batch_loss += ntonp(loss)
                    
                    if self.use_velocity:
                        if self.velocity_loss_scale>0:
                            loss_f = self.velocity_loss_scale * self.velocity_loss / loss_norm
                            if self.record_stats and self.record_individual_grads:
                                loss_f.backward(retain_graph=True)
                                with torch.no_grad():
                                    self.grads_loss_velocity.append(np.mean([ttonp(torch.mean(torch.abs(g))) for g in self.learned_velocity_grads]))
                                    self.steps_grad_loss_velocity.append(self.opt_step)
                                    self.log.info("mean abs velocity regularization grad: %.03e", self.grads_loss_velocity[-1])
                                self.zero_grad()
                            batch_velocity_loss += ntonp(loss_f)
                            loss = loss + loss_f
                        
                        if self.velocity_divergence_loss_scale>0:
                            self.apply_velocity_div_grad = True
                    
                    if self.use_forcing:
                        if self.forcing_loss_scale>0:
                            loss_f = self.forcing_loss_scale * self.forcing_loss / loss_norm
                            if self.record_stats and self.record_individual_grads:
                                loss_f.backward(retain_graph=True)
                                with torch.no_grad():
                                    self.grads_loss_forcing.append(np.mean([ttonp(torch.mean(torch.abs(g))) for g in self.learned_forcing_grads]))
                                    self.steps_grad_loss_forcing.append(self.opt_step)
                                    self.log.info("mean abs forcing regularization grad: %.03e", self.grads_loss_forcing[-1])
                                self.zero_grad()
                            batch_forcing_loss += ntonp(loss_f)
                            loss = loss + loss_f
                        
                        if self.forcing_divergence_loss_scale>0:
                            self.apply_forcing_div_grad = True
                    
                    self.mem_usage.check_memory("after loss %d steps"%train_sim_steps)
                    
                    
                    with SAMPLE("BWD"):
                        loss.backward()
                    
                    self.mem_usage.check_memory("after bwd %d steps"%train_sim_steps)
                    
                    if self.record_stats:
                        with torch.no_grad():
                            if self.use_velocity:
                                velocity_means = [ttonp(torch.mean(f)) for f in self.learned_velocities]
                                velocity_abs_means = [ttonp(torch.mean(torch.abs(f))) for f in self.learned_velocities]
                                velocity_mins = [ttonp(torch.min(f)) for f in self.learned_velocities]
                                velocity_maxs = [ttonp(torch.max(f)) for f in self.learned_velocities]
                                velocity_grad_means = [ttonp(torch.mean(torch.abs(g))) for g in self.learned_velocity_grads]
                                self.velocity_stats.append((np.mean(velocity_means), np.mean(velocity_abs_means), np.min(velocity_mins), np.max(velocity_maxs)))
                                self.velocity_grad_stats.append(np.mean(velocity_grad_means))
                                self.log.info("velocities: mean %s, mean abs %s, min %s, max %s, mean abs grad %s",
                                              *(self.velocity_stats[-1]), self.velocity_grad_stats[-1])
                                if self.velocity_divergence_loss_scale>0:
                                    if len(self.grads_div_velocity_frames)!=train_sim_steps:
                                        raise RuntimeError("Expected %d velocity divergence gradients, but got %d"%(train_sim_steps, len(self.grads_div_velocity_frames)))
                                    self.grads_div_velocity.append(np.mean(self.grads_div_velocity_frames))
                                    self.steps_grad_div_velocity.append(self.opt_step)
                                    self.log.info("mean abs velocity divergence grad: %.03e", self.grads_div_velocity[-1])
                            if self.use_forcing:
                                forcing_means = [ttonp(torch.mean(f)) for f in self.learned_forcings]
                                forcing_abs_means = [ttonp(torch.mean(torch.abs(f))) for f in self.learned_forcings]
                                forcing_mins = [ttonp(torch.min(f)) for f in self.learned_forcings]
                                forcing_maxs = [ttonp(torch.max(f)) for f in self.learned_forcings]
                                forcing_grad_means = [ttonp(torch.mean(torch.abs(g))) for g in self.learned_forcing_grads]
                                self.forcing_stats.append((np.mean(forcing_means), np.mean(forcing_abs_means), np.min(forcing_mins), np.max(forcing_maxs)))
                                self.forcing_grad_stats.append(np.mean(forcing_grad_means))
                                #self.log.info("forcings: %s, abs grads: %s", forcing_means, forcing_grad_means)
                                self.log.info("forcings: mean %s, mean abs %s, min %s, max %s, mean abs grad %s",
                                              *(self.forcing_stats[-1]), self.forcing_grad_stats[-1])
                                if self.forcing_divergence_loss_scale>0:
                                    if len(self.grads_div_forcing_frames)!=train_sim_steps:
                                        raise RuntimeError("Expected %d forcing divergence gradients, but got %d"%(train_sim_steps, len(self.grads_div_forcing_frames)))
                                    self.grads_div_forcing.append(np.mean(self.grads_div_forcing_frames))
                                    self.steps_grad_div_forcing.append(self.opt_step)
                                    self.log.info("mean abs forcing divergence grad: %.03e", self.grads_div_forcing[-1])
                            if self.use_viscosity:
                                viscosity_means = [ttonp(torch.mean(f)) for f in self.learned_viscosities]
                                viscosity_grad_means = [ttonp(torch.mean(torch.abs(g))) for g in self.learned_viscosity_grads]
                                self.viscosity_stats.append(np.mean(viscosity_means))
                                self.viscosity_grad_stats.append(np.mean(viscosity_grad_means))
                                self.log.info("viscosities: %s, abs grads: %s", viscosity_means, viscosity_grad_means)
                            if self.use_SMAG:
                                self.SMAG_Cs.append(ttonp(torch.mean(self.SMAG_C)))
                                self.SMAG_C_grads.append(ttonp(torch.mean(torch.abs(self.SMAG_C.grad))))
                                self.log.info("SMAG C: %s, abs grad: %s", self.SMAG_Cs[-1], self.SMAG_C_grads[-1])
                    
                    del sim
                    self.apply_velocity_div_grad = False
                    self.apply_forcing_div_grad = False
                    del loss
                    self.reset_losses()
                    opt_domain.Detach()
                    del opt_domain
            # END batch
            self.record_stats = base_record_stats
            
            self.log.info("step %d, frame %d: normalized loss %.08e, velocity loss %.08e, forcing loss %.08e", self.opt_step, self.frame_offset, batch_loss, batch_velocity_loss, batch_forcing_loss)
            
            if self.record_stats:
                self.losses.append(batch_loss)
                if self.use_velocity and self.velocity_loss_scale>0:
                    self.losses_velocity.append(batch_velocity_loss)
                if self.use_forcing and self.forcing_loss_scale>0:
                    self.losses_forcing.append(batch_forcing_loss)
                self.stats_steps.append(self.opt_step)
            
            with SAMPLE("update"):
                self.step_optimizer()
                
                self.zero_grad()

            if torch.cuda.memory_allocated() > self.max_memory_GPU:
                self.log.info("Running garbage collection.")
                gc.collect()
            # self.mem_usage.check_memory("after step and zero_grad (gc.collect) %d steps"%train_sim_steps)
            self.mem_usage.check_memory("after step and zero_grad %d steps"%train_sim_steps)

            if self.use_SMAG:
                self.clamp_learned_SMAG_C()
            elif not self.use_network:
                self.clamp_learned_viscosity()

            self.opt_step += 1
            self.is_training = False
            
            #if self.stop_fn():
            #    break

    
    def run_optimization(self, optimization_steps, train_sim_steps, pre_steps=0, batch_size=1, stats_interval=1, grads_interval=-1, eval_interval=0, **eval_args):
        self.log.info("Running optimization %d steps with %d simulation steps and batch size %d", optimization_steps, train_sim_steps, batch_size)
        for opt_step in range(optimization_steps):
            
            self.record_stats = (((opt_step)%stats_interval)==0) or (opt_step==(optimization_steps-1))
            if grads_interval>0 and self.record_stats:
                record_individual_grads = self.record_individual_grads
                self.record_individual_grads = ((opt_step)%grads_interval)==0 or (opt_step==(optimization_steps-1))
            
            try:
                self.optimization_step(opt_step, train_sim_steps=train_sim_steps, pre_steps=pre_steps, batch_size=batch_size,
                    log_images=(opt_step==0 or opt_step==(optimization_steps-1)))
            except:
                LOG.exception("Optimization step %d failed!", opt_step)
                try:
                    self.save_model(name_suffix="_ERR")
                except:
                    LOG.exception("Saving model failed!")
                self.stop_fn.stop()
                break
            
            self.record_stats = False
            if grads_interval>0 and self.record_stats:
                self.record_individual_grads = record_individual_grads

            if eval_interval>0 and ((opt_step+1)%eval_interval)==0:
                try:
                    self.save_model()
                except:
                    LOG.exception("Saving model failed:")
                try:
                    self.save_and_plot_stats()
                except:
                    LOG.exception("Plotting failed:")
                try: #to run longer evaluations and still continue training if the model is unstable
                    self.eval(**eval_args)
                except:
                    LOG.exception("Evaluation failed:")
                    self.reset_losses()
            
            if self.stop_fn():
                break
    
    def __eval(self, iterations, start_frame=0, with_correction=True, log_images=True, compare_steps=False, log_corrections=False,
            log_frame_losses=True, log_PSD_planes=[],
            log_vape_images=False, log_vape_video=False, log_vape_npz=False):
        eval_name = "eval_%04d_%s"%(self.evals, "c" if with_correction else "nc")
        with SAMPLE(eval_name), torch.no_grad():
            self.frame_offset = start_frame
            self.reset_losses()
            self.record_loss_per_frame = log_frame_losses
            self.record_stats = log_corrections or (log_frame_losses and self._is_frame_statistics_loss_active)
            assert (iterations<=(self.num_steps-start_frame)) if (compare_steps or (log_frame_losses and self.use_velocity_loss)) else (start_frame<=self.num_steps)
            domain = self.get_reference_clone(start_frame)
            domain.PrepareSolve()
            prep_fn = {}
            
            if self.record_stats and self._is_frame_statistics_loss_active:
                for key, stats_loss in self.statistics_losses.items():
                    stats_loss["frame_losses"].append([])

            self.log.info("Running full sequence LR eval %04d %s correction", self.evals, "with" if with_correction else "without")
            eval_path = os.path.join(self.log_path, eval_name)
            if log_images:
                vel_stats = VelocityStats(domain, eval_path, self.log, u_wall=self.u_wall, PSD_planes=log_PSD_planes, record_online_steps=True,
                    use_moments_3=True, use_moments_4=True, use_temporal_correlation_dims=[2,4]) #, use_forcing_budgets=self.use_forcing
                vel_stats.plot_u_vel_max = 25
                vel_stats.plot_vel_rms_max = 3.5
                if self.ref_profiles is not None:
                    vel_stats.set_references(self.ref_profiles)
                PISOtorch_simulation.append_prep_fn(prep_fn, "POST", vel_stats.record_vel_stats)
                
                #ref_moments = PISOTCFProfile("./test_runs", "250714-220246", load_moments=True, device=cuda_device, dtype=dtype)
                #vel_stats.add_additional_reference_stats(ref_moments, linestyle="-.")
            #else:
            #    eval_path = None
            
            if with_correction:
                PISOtorch_simulation.append_prep_fn(prep_fn, "PRE", self.pfn_apply_correction)
                #PISOtorch_simulation.append_prep_fn(prep_fn, "POST", self.pfn_clear_correction)
            else: #if self.base_forcing=="DYNAMIC":
                PISOtorch_simulation.append_prep_fn(prep_fn, "PRE", self.pfn_apply_base_forcing)
            
            output_resampling_coords, output_resampling_shape = self.get_resampling_params(domain)
            
            def lfn_step_comparison(domain, out_dir, it, **kwargs):
                reference_domain = self.references_low[start_frame + it]
                vel_differences = [torch.abs(block.velocity - ref_block.velocity) for block, ref_block in zip(domain.getBlocks(), reference_domain.getBlocks())]
                vel_differences_flat = torch.cat([v.view(3,-1) for v in vel_differences], dim=-1)
                self.log.info("velocity absolute difference: mean %s, min %s, max %s",
                    torch.mean(vel_differences_flat, dim=-1).cpu().numpy(),
                    torch.min(vel_differences_flat, dim=-1)[0].cpu().numpy(),
                    torch.max(vel_differences_flat, dim=-1)[0].cpu().numpy())
                
                save_block_data_image(vel_differences, domain, out_dir, "uDiff-z-norm", it, normalize=True)
                if domain.hasVertexCoordinates():
                    save_block_data_image(vel_differences, domain, out_dir, "uDiff-r-z-norm", it, normalize=True,
                        vertex_coord_list=domain.getVertexCoordinates(), resampling_out_shape=output_resampling_shape, fill_max_steps=20)
            
            def lfn_corrections(domain, out_dir, it, **kwargs):
                num_blocks = domain.getNumBlocks()
                self.log.info("lfn_corrections it=%d", it)
                if self.use_forcing and it>=0 and (it//self.net_time_stride)<(len(self.learned_forcings)//num_blocks) and (not self.use_network or (it%self.net_time_stride)==0):
                    forcings = self.learned_forcings[it//self.net_time_stride*num_blocks:(it//self.net_time_stride+1)*num_blocks]
                    #self.log.info("it %d, blocks %d: %s[%d:%d] = %s", it, num_blocks, len(self.learned_forcings), it*num_blocks, (it+1)*num_blocks, len(forcings))
                    forcings_flat = torch.cat([v.view(3,-1) for v in forcings], dim=-1)
                    self.log.info("forcings: mean %s, min %s, max %s",
                        ttonp(torch.mean(forcings_flat, dim=-1)),
                        ttonp(torch.min(forcings_flat, dim=-1)[0]),
                        ttonp(torch.max(forcings_flat, dim=-1)[0]))
                    
                    save_block_data_image(forcings, domain, out_dir, "S-z-norm", it, normalize=True)
                    if domain.hasVertexCoordinates():
                        save_block_data_image(forcings, domain, out_dir, "S-r-z-norm", it, normalize=True,
                            vertex_coord_list=domain.getVertexCoordinates(), resampling_out_shape=output_resampling_shape, fill_max_steps=20)
                
                if self.use_viscosity and it>=0 and it<(len(self.learned_viscosities)//num_blocks):
                    visc = self.learned_viscosities[it*num_blocks:(it+1)*num_blocks]
                    visc_flat = torch.cat([v.view(-1) for v in visc], dim=-1)
                    self.log.info("visosities: mean %s, min %s, max %s",
                        ttonp(torch.mean(visc_flat, dim=-1)),
                        ttonp(torch.min(visc_flat, dim=-1)[0]),
                        ttonp(torch.max(visc_flat, dim=-1)[0]))
                    
                    save_block_data_image(visc, domain, out_dir, "visc-z-norm", it, normalize=True)
                    if domain.hasVertexCoordinates():
                        save_block_data_image(visc, domain, out_dir, "visc-r-z-norm", it, normalize=True,
                            vertex_coord_list=domain.getVertexCoordinates(), resampling_out_shape=output_resampling_shape, fill_max_steps=20)
            
            if log_images and (log_vape_images or log_vape_video or log_vape_npz):
                from vape_stuff import render_velocity
                if log_vape_video:
                    import imageio
                    vid_writer = imageio.get_writer(os.path.join(eval_path, "vape.mp4"), format="FFMPEG", fps=20,
                        codec="libx264", quality=8, pixelformat="yuv420p")
                    #    output_params=["-c:v", "libx264", "-pix_fmt", "yuv420p", "-b:v", "1M")
                if log_vape_npz:
                    vape_npz = []
                def lfn_vape(domain, it, **kwargs):
                    vel = ntonp(domain.getBlock(0).velocity)
                    img_name = "vape_%04d"%(it,)
                    if log_vape_images or log_vape_video:
                        img = render_velocity(img_name, vel)
                        if log_vape_images:
                            save_np_png(img, os.path.join(eval_path, "%s.png"%(img_name,)))
                        if log_vape_video:
                            vid_writer.append_data(img)
                    if log_vape_npz:
                        vape_npz.append(vel)
            
            def log_fn(domain, it, **kwargs):
                self.log.info("log_fn it=%d", it)
                if log_images:
                    vel_stats.log_vel_stats(domain, it, with_reference=True, **kwargs)
                    if log_vape_images or log_vape_video:
                        lfn_vape(domain=domain, it=it, **kwargs)
                if log_frame_losses:
                    self.lfn_add_loss(domain=domain, it=it, **kwargs)
                if False:
                    log_SGS_viscosity(domain=domain, it=it, **kwargs)
                if compare_steps:
                    lfn_step_comparison(domain=domain, it=it, **kwargs)
                if log_corrections:
                    lfn_corrections(domain=domain, it=it-(1 if self.use_network else 0), **kwargs)
                else:
                    self.learned_forcings = []
            
            sim = PISOtorch_simulation.Simulation(domain=domain, prep_fn=prep_fn,
                substeps=self.substeps, time_step=self.time_step, corrector_steps=2,
                advection_tol=1e-6, pressure_tol=1e-6,
                adaptive_CFL=self.adaptive_CFL,
                advect_non_ortho_steps=1, pressure_non_ortho_steps=1, pressure_return_best_result=True,
                velocity_corrector="FD", non_orthogonal=False,
                log_interval=1, norm_vel=True, log_fn=log_fn,
                log_images=log_images,
                #output_resampling_coords=output_resampling_coords, output_resampling_shape=output_resampling_shape,
                #output_resampling_fill_max_steps=20,
                log_dir=eval_path, save_domain_name="domain",
                stop_fn=stop_handler)
            
            sim.preconditionBiCG = self.preconditionBiCG
            sim.BiCG_precondition_fallback = self.BiCG_precondition_fallback
            
            sim.log_fn(domain, out_dir=eval_path, it=0)
            
            #sim.make_divergence_free()
            # run simulation
            sim.run(iterations=iterations)
            
            if log_vape_video:
                vid_writer.close()
            if log_vape_npz:
                np.savez_compressed(os.path.join(eval_path, "vape.npz"), np.concatenate(vape_npz, axis=0))
                del vape_npz
            
            if log_frame_losses and iterations<=100:
                self.log.info("Eval per-frame losses: %s", np.asarray([ttonp(loss) for loss in self.frame_losses]))
            
            self.save_model(dir=eval_path)

            if log_frame_losses:
                self.save_eval_stats(eval_path)
                
            if log_images:
                vel_stats.save_vel_stats()
                vel_stats.plot_final_stats(reference_budgets=self.ref_budgets)
                if log_frame_losses:
                    self.plot_eval_stats(eval_path)
            
                avg_u_wall = vel_stats.get_avg_u_wall()
                self.log.info("Avg u_wall %.03e -> Re_wall %.03e.", avg_u_wall, avg_u_wall / ntonp(domain.viscosity)[0])
            
            if self.record_stats and self._is_frame_statistics_loss_active:
                for key, stats_loss in self.statistics_losses.items():
                    del stats_loss["frame_losses"][-1]
            self.reset_losses()
            self.record_loss_per_frame = False
            self.record_stats = False
            #if log_images:
            self.evals += 1
    
    def eval(self, iterations, start_frame=0, with_correction=True, log_images=True, compare_steps=False, log_corrections=False,
            log_frame_losses=True, log_PSD_planes=[],
            log_vape_images=False, log_vape_video=False, log_vape_npz=False):

        if not isinstance(start_frame, (list, tuple)):
            start_frame = [start_frame]

        for sf in start_frame:
            self.__eval(iterations, sf, with_correction, log_images, compare_steps, log_corrections,
                log_frame_losses, log_PSD_planes,
                log_vape_images, log_vape_video, log_vape_npz)

    def save_eval_stats(self, eval_path):
        eval_stats = {"total_frame_losses": np.asarray([ttonp(loss) for loss in self.frame_losses])}
        if self._is_frame_statistics_loss_active:
            for k, v in self.statistics_losses.items():
                if "frame_losses" in v:
                    eval_stats[k+"_frame_losses"] = np.asarray(v["frame_losses"][-1])
        
        np.savez_compressed(os.path.join(eval_path, "eval_stats.npz"), **eval_stats)
    
    def plot_eval_stats(self, eval_path):
        nrows=1
        ncols=1 + int(self._is_frame_statistics_loss_active)
        ax_width=6.4#/2
        ax_height=4.8#/2
        
        plt.clf()
        fig, axs = plt.subplots(nrows,ncols, figsize=(ax_width*ncols, ax_height*nrows), squeeze=False)
        
        ax = axs[0][0]
        ax.set_xlabel("Frame")
        ax.set_ylabel("Loss")
        ax.tick_params(axis='y', labelcolor="tab:blue")
        ax.plot(np.asarray([ttonp(loss) for loss in self.frame_losses]), color="tab:blue")
        
        if self._is_frame_statistics_loss_active:
            ax = axs[0][1]
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Stats Loss Parts")
            ax.set_yscale("log")
            for k, v in self.statistics_losses.items():
                plot_kwargs = {"label":k}
                if "frame_losses" in v:
                    #self.log.info("frame_loss_shape %s", np.asarray(v["frame_losses"]))
                    ax.plot(v["frame_losses"][-1], linestyle=":", **plot_kwargs)
            ax.legend()
        
        fig.align_labels()
        fig.tight_layout()
        fig.savefig(os.path.join(eval_path, "eval_stats.pdf"))
        plt.close(fig)
    
    
    def save_model(self, dir=None, name_suffix=""):
        if dir is None:
            dir = self.log_path
        if self.use_network:
            torch.save(self.net.state_dict(), os.path.join(dir, "ForcingNet%s.pt"%(name_suffix,)))
            torch.save(self.optimizer.state_dict(), os.path.join(dir, "Optimizer%s.pt"%(name_suffix,)))
        if self.use_forcing and self.use_forcing_offset:
            np.savez_compressed(os.path.join(dir, "forcing_offset%s.npz"%(name_suffix,)), forcing_offset=ttonp(self.forcing_offset))
        if self.use_SMAG:
            np.savez_compressed(os.path.join(dir, "SMAG_C%s.npz"%(name_suffix,)), C=ttonp(self.SMAG_C))
    
    
    def load_model(self, run_dir, run_id, load_optimizer=True, name_suffix=""):
        import glob
        paths = glob.glob(os.path.join(run_dir, run_id + "*"))
        if len(paths)==0:
            self.log.error("Run ID %s not found in '%s'", run_id, run_dir)
            raise IOError
        if len(paths)>1:
            self.log.error("Run ID %s not unique in '%s'", run_id, run_dir)
            raise IOError
        
        if self.use_network:
            self.net.to(cpu_device)
            net_path = os.path.join(paths[0], "ForcingNet%s.pt"%(name_suffix,))
            if not os.path.isfile(net_path):
                self.log.error("Run ID %s in '%s' does not contain a network state.", run_id, run_dir)
                raise IOError
            
            self.net.load_state_dict(torch.load(net_path, weights_only=True))
            self.net.to(cuda_device)
            self.log.info("Loaded network weights from Run ID %s in '%s'", run_id, run_dir)
        
            if load_optimizer:
                opt_path = os.path.join(paths[0], "Optimizer%s.pt"%(name_suffix,))
                if not os.path.isfile(opt_path):
                    self.log.error("Run ID %s in '%s' does not contain an optimizer state.", run_id, run_dir)
                    raise IOError
                
                self.optimizer.load_state_dict(torch.load(opt_path, weights_only=False))
                self.log.info("Loaded optimizer state from Run ID %s in '%s'", run_id, run_dir)
        
        if self.use_forcing and self.use_forcing_offset:
            forcing_offset_path = os.path.join(paths[0], "forcing_offset%s.npz"%(name_suffix,))
            with np.load(forcing_offset_path) as np_file:
                self.forcing_offset = np_file["forcing_offset"]
            self.forcing_offset = torch.tensor(self.forcing_offset, device=cuda_device, dtype=torch.float32)
        
        elif self.use_SMAG:
            SMAG_C_path = os.path.join(paths[0], "SMAG_C%s.npz"%(name_suffix,))
            with np.load(SMAG_C_path) as np_file:
                C = np_file["C"]
            self.log.info("Loaded SMAG C=%s from Run ID %s in '%s'", C, run_id, run_dir)
            self.SMAG_C = torch.tensor(C, device=cuda_device, dtype=torch.float32)
        
    
    def _smooth_plot(self, x, y):
        pooler = torch.nn.AvgPool1d(kernel_size=self.plot_smoothing_window, stride=self.plot_smoothing_window)
        x = pooler(torch.tensor(x, device=cpu_device, dtype=torch.float64).reshape((1,1,len(x)))).numpy().reshape(len(x))
        y = pooler(torch.tensor(y, device=cpu_device, dtype=torch.float64).reshape((1,1,len(y)))).numpy().reshape(len(x))
        return x, y
    
    def save_and_plot_stats(self):
        steps = np.asarray(self.stats_steps)
        stats = {
            "steps": steps,
            "target_losses": np.asarray(self.losses),
        }
        
        nrows = (1
            + int(self.use_velocity)
            + int(self.use_forcing)
            + int(self.use_viscosity)
            + int(self.use_SMAG)
            + int(self.use_SMAG and self.learn_SMAG_wall_scaling) # + int(self.use_forcing and self.forcing_loss_scale>0)
            #+ int(len(self.statistics_losses)>0)
            #+ int(len(self.statistics_losses)>0 and self.record_individual_grads and (self.use_forcing or self.use_viscosity))
            )
        ncols=3
        ax_width=6.4
        ax_height=4.8
        
        plt.clf()
        fig, axs = plt.subplots(nrows,ncols, figsize=(ax_width*ncols, ax_height*nrows), squeeze=False)
        
        ax = axs[0][0]
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.tick_params(axis='y', labelcolor="tab:blue")
        ax.plot(steps, stats["target_losses"], color="tab:blue")
        if self.use_velocity and self.velocity_loss_scale>0:
            stats["velocity_losses"] = np.asarray(self.losses_velocity)
            ax = ax.twinx()
            ax.tick_params(axis='y', labelcolor="tab:green")
            ax.plot(steps, stats["velocity_losses"], linestyle=":", color="tab:green")
        if self.use_forcing and self.forcing_loss_scale>0:
            stats["forcing_losses"] = np.asarray(self.losses_forcing)
            ax = ax.twinx()
            ax.tick_params(axis='y', labelcolor="tab:red")
            ax.plot(steps, stats["forcing_losses"], linestyle=":", color="tab:red")
        
        
        ax = axs[0][1]
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        ax.tick_params(axis='y', labelcolor="tab:blue")
        ax.plot(steps, stats["target_losses"], color="tab:blue")
        if self.use_velocity and self.velocity_loss_scale>0:
            ax = ax.twinx()
            ax.set_yscale("log")
            ax.tick_params(axis='y', labelcolor="tab:green")
            ax.plot(steps, stats["velocity_losses"], linestyle=":", color="tab:green")
        if self.use_forcing and self.forcing_loss_scale>0:
            ax = ax.twinx()
            ax.set_yscale("log")
            ax.tick_params(axis='y', labelcolor="tab:red")
            ax.plot(steps, stats["forcing_losses"], linestyle=":", color="tab:red")
        
        if len(self.statistics_losses)>0:
            ax = axs[0][2]
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Stats Loss Parts")
            ax.set_yscale("log")
            for k, v in self.statistics_losses.items():
                plot_kwargs = {"label":k}
                if "losses" in v:
                    line, = ax.plot(v["steps"], v["losses"], **plot_kwargs)
                    plot_kwargs["color"] = line.get_color()
                    del plot_kwargs["label"]
                if "frame_losses" in v:
                    #self.log.info("frame_loss_shape %s", np.asarray(v["frame_losses"]))
                    ax.plot(v["steps"], [np.mean(_) for _ in v["frame_losses"]], linestyle=":", **plot_kwargs)
            ax.legend()
            
        
        row = 1
        
        if self.use_SMAG:
            stats["SMAG_Cs"] = np.asarray(self.SMAG_Cs)
            ax = axs[row][0]
            ax.set_xlabel("Iteration")
            ax.set_ylabel("SMAG C")
            ax.plot(steps, stats["SMAG_Cs"], color="tab:blue")
            
            stats["SMAG_C_grads"] = np.asarray(self.SMAG_C_grads)
            ax = axs[row][1]
            ax.set_xlabel("Iteration")
            ax.set_ylabel("SMAG C.grad")
            ax.plot(steps, stats["SMAG_C_grads"], color="tab:blue")

            row += 1

        if self.use_SMAG and self.learn_SMAG_wall_scaling:
            ax = axs[row][0]
            ax.set_xlabel("cell y")
            ax.set_ylabel("SMAG C")
            ax.plot(ttonp(self.SMAG_C[0,0,0,:,0]), color="tab:blue")
            
            ax = axs[row][1]
            ax.set_xlabel("pos y")
            ax.set_ylabel("SMAG C")
            ax.plot(self.pos_y, ttonp(self.SMAG_C[0,0,0,:,0]), color="tab:blue")

            row += 1
        
        if self.use_velocity:
            if False and self.velocity_loss_scale>0:
                stats["velocity_losses"] = np.asarray(self.losses_velocity)
                ax = axs[row][0]
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Loss velocity")
                ax.plot(steps, stats["velocity_losses"], color="tab:red")
                
                ax = axs[row][1]
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Loss velocity")
                ax.set_yscale("log")
                ax.plot(steps, stats["velocity_losses"], color="tab:red")
                
                row += 1

            stats["velocity_stats"] = np.asarray(self.velocity_stats)
            ax = axs[row][0]
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Velocity")
            ax.plot(steps, stats["velocity_stats"][:,0], label="mean")
            ax.plot(steps, stats["velocity_stats"][:,1], label="abs mean")
            #ax.plot(steps, stats["velocity_stats"][:,2], label="min")
            #ax.plot(steps, stats["velocity_stats"][:,3], label="max")
            ax.legend()
            
            stats["velocity_grad_stats"] = np.asarray(self.velocity_grad_stats)
            ax = axs[row][1]
            ax.set_xlabel("Iteration")
            ax.set_ylabel("v.grad mean")
            ax.set_yscale("log")
            ax.tick_params(axis='y', labelcolor="tab:blue")
            ax.plot(steps, stats["velocity_grad_stats"], color="tab:blue", label="total")
            if self.velocity_loss_scale>0 or self.velocity_divergence_loss_scale>0:
                ax = ax.twinx()
                ax.set_yscale("log")
                ax.tick_params(axis='y', labelcolor="tab:red")
                if self.velocity_loss_scale>0:
                    stats["steps_grad_loss_velocity"] = np.asarray(self.steps_grad_loss_velocity)
                    stats["grads_loss_velocity"] = np.asarray(self.grads_loss_velocity)
                    ax.plot(stats["steps_grad_loss_velocity"], stats["grads_loss_velocity"], linestyle=":", color="tab:red", label="from velocity loss")
                if self.velocity_divergence_loss_scale>0:
                    stats["steps_grad_div_velocity"] = np.asarray(self.steps_grad_div_velocity)
                    stats["grads_div_velocity"] = np.asarray(self.grads_div_velocity)
                    ax.plot(stats["steps_grad_div_velocity"], stats["grads_div_velocity"], linestyle=":", color="tab:orange", label="from velocity div")
            ax.legend()
            
            if len(self.statistics_losses)>0:
                ax = axs[row][2]
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Stats Loss Grad")
                ax.set_yscale("log")
                for k, v in self.statistics_losses.items():
                    plot_kwargs = {"label":k}
                    if len(v["velocity_grad_steps"])>0:
                        line, = ax.plot(v["velocity_grad_steps"], v["velocity_grads"], **plot_kwargs)
                        plot_kwargs["color"] = line.get_color()
                        del plot_kwargs["label"]
                    if len(v["velocity_frame_grad_steps"])>0:
                        ax.plot(v["velocity_frame_grad_steps"], v["velocity_frame_grads"], linestyle=":", **plot_kwargs)
                ax.legend()

            row += 1
        
        if self.use_forcing:
            if False and self.forcing_loss_scale>0:
                stats["forcing_losses"] = np.asarray(self.losses_forcing)
                ax = axs[row][0]
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Loss Forcing")
                ax.plot(steps, stats["forcing_losses"], color="tab:red")
                
                ax = axs[row][1]
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Loss Forcing")
                ax.set_yscale("log")
                ax.plot(steps, stats["forcing_losses"], color="tab:red")
                
                row += 1

            stats["forcing_stats"] = np.asarray(self.forcing_stats)
            ax = axs[row][0]
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Forcing")
            ax.plot(steps, stats["forcing_stats"][:,0], label="mean")
            ax.plot(steps, stats["forcing_stats"][:,1], label="abs mean")
            #ax.plot(steps, stats["forcing_stats"][:,2], label="min")
            #ax.plot(steps, stats["forcing_stats"][:,3], label="max")
            ax.legend()
            
            stats["forcing_grad_stats"] = np.asarray(self.forcing_grad_stats)
            ax = axs[row][1]
            ax.set_xlabel("Iteration")
            ax.set_ylabel("S.grad mean")
            ax.set_yscale("log")
            ax.tick_params(axis='y', labelcolor="tab:blue")
            ax.plot(steps, stats["forcing_grad_stats"], color="tab:blue", label="total")
            if self.forcing_loss_scale>0 or self.forcing_divergence_loss_scale>0:
                ax = ax.twinx()
                ax.set_yscale("log")
                ax.tick_params(axis='y', labelcolor="tab:red")
                if self.forcing_loss_scale>0:
                    stats["steps_grad_loss_forcing"] = np.asarray(self.steps_grad_loss_forcing)
                    stats["grads_loss_forcing"] = np.asarray(self.grads_loss_forcing)
                    ax.plot(stats["steps_grad_loss_forcing"], stats["grads_loss_forcing"], linestyle=":", color="tab:red", label="from forcing loss")
                if self.forcing_divergence_loss_scale>0:
                    stats["steps_grad_div_forcing"] = np.asarray(self.steps_grad_div_forcing)
                    stats["grads_div_forcing"] = np.asarray(self.grads_div_forcing)
                    ax.plot(stats["steps_grad_div_forcing"], stats["grads_div_forcing"], linestyle=":", color="tab:orange", label="from forcing div")
            ax.legend()
            
            if len(self.statistics_losses)>0:
                ax = axs[row][2]
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Stats Loss Grad")
                ax.set_yscale("log")
                for k, v in self.statistics_losses.items():
                    plot_kwargs = {"label":k}
                    if len(v["forcing_grad_steps"])>0:
                        line, = ax.plot(v["forcing_grad_steps"], v["forcing_grads"], **plot_kwargs)
                        plot_kwargs["color"] = line.get_color()
                        del plot_kwargs["label"]
                    if len(v["forcing_frame_grad_steps"])>0:
                        ax.plot(v["forcing_frame_grad_steps"], v["forcing_frame_grads"], linestyle=":", **plot_kwargs)
                ax.legend()

            row += 1
        
        if self.use_viscosity:
            stats["viscosity_stats"] = np.asarray(self.viscosity_stats)
            ax = axs[row][0]
            ax.set_xlabel("Iteration")
            ax.set_ylabel("visc mean")
            ax.plot(steps, stats["viscosity_stats"], color="tab:blue")
            if self.min_viscosity is not None:
                ax.axhline(y=self.min_viscosity, linewidth=0.5, linestyle="--")
            if self.max_viscosity is not None:
                ax.axhline(y=self.max_viscosity, linewidth=0.5, linestyle="--")
            
            stats["viscosity_grad_stats"] = np.asarray(self.viscosity_grad_stats)
            ax = axs[row][1]
            ax.set_xlabel("Iteration")
            ax.set_ylabel("visc.grad mean")
            ax.plot(steps, stats["viscosity_grad_stats"], color="tab:blue")
            
            if len(self.statistics_losses)>0:
                ax = axs[row][2]
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Stats Loss Grad")
                ax.set_yscale("log")
                for k, v in self.statistics_losses.items():
                    if len(v["viscosity_grad_steps"])>0:
                        ax.plot(v["viscosity_grad_steps"], v["viscosity_grads"], label=k)
                ax.legend()
            
            row += 1
        
        np.savez_compressed(os.path.join(self.log_path, "opt_stats.npz"), **stats)

        fig.align_labels()
        fig.tight_layout()
        fig.savefig(os.path.join(self.log_path, "opt_stats.pdf"))
        #fig.savefig(os.path.join(self.log_path, "opt_stats.png"))
        plt.close(fig)



if __name__=="__main__":

    train = True # True for training and inference. False to load inference results for further investigation.
    
    if train:
        run_dir = setup_run("./test_runs",
            name="TCF550_uh-netMBC-s1d1e-4c2-u-lr1e-4-UVWuvwcTF_r64N3-p0-it12-ETT.002-ss1_opt200-Ref160_nALSg-nPLSg_MEM-gc-8GB"
            #name="TCF550_eval-R251008-152313_load-f188-it200-ss25_plotLoss-F.5"
            )
        LOG = get_logger("Main")
        LOG.info("GPU #%s", cudaID)
        stop_handler = PISOtorch_simulation.StopHandler(LOG)
        stop_handler.register_signal()
        
        dp = False # can use double precision, but should not be necessary
        dtype = torch.float64 if dp else torch.float32
        
        delta = 1 # =y_size/2
        # x_size = 2*np.pi*delta
        # y_size = delta * 2
        # z_size = np.pi*delta
        
        Re_wall = 550
        Re_center = TCF_tools.Re_wall_to_cl(Re_wall)
        
        u_wall = Re_wall / Re_center #1

        viscosity = delta/Re_center # u_wall*delta/Re_wall
        forcing = "DYNAMIC"

        # training uses simulation with fixed time steps
        ETT_step = 0.002
        time_step = TCF_tools.ETT_to_t(ETT_step, u_wall, delta) #0.002 #0.1
        substeps = 1
        adaptive_CFL = 0.8 # this is only used if substeps = "ADAPTIVE"
        #load_frames = 10
        
        # Turn this on to run only inference on a loaded model
        eval_only = False
        
        if eval_only:
            eval_start_frame = 188 # the initial condition to use for inference
            #eval_start_frame = (10, 42, 93, 113, 146, 188)
            eval_iterations = 200
            substeps = 25
        else: # training mode
            eval_start_frame = 180
            eval_iterations = 30
            eval_iterations_final = 100
            eval_iterations_final_cmp = 20
        
        initial_eval = False # run inference before training?
        final_eval = False # run inference after training?
        
        ### MODEL ###
        # The run-id to load the trained SGS model from. None to initialize a new model.
        net_run_id = None
        #net_run_id = "251008-152313"

        ### DATA ###
        # The initial conditions to load for training. run-id and number of frames (saved iterations from 'channel_flow_sim.py')
        # This also determins the resolution and grid used in training.
        load_run_id, load_frames = "251008-114126", 200
        
        # the range of initial conditions to use in training. These should be from a statistically converged simulation.
        train_start_frame, train_end_frame = 0, 160
        
        # unused
        load_down = False
        down_factors = None #3 #[3,1,3], None
        down_make_div_free = False
        
        #compare_steps = True
        
        # the turbulence statistics to use for loss computation and evaluation
        ref_profiles = TorrojaProfile("./data/tcf_torroja", 550)
        ref_budgets = TorrojaBalances("./data/tcf_torroja", 550)
        
        # In our TCF SGS training regime, the non-differentiable warm-up steps are increased in intervals as indicated below.
        training_intervals = [
            {
                "opt_steps": 6000, # number of optimization iterations
                "batch_size": 1, # must be 1
                # non-differentiable warm-up steps
                "pre_steps": 0, # int, can also be an interval [min, max], or callable "lambda opt_step: pre_step"
                # differentiable steps after the warm-up steps
                "sim_steps": 12,
                "lr_net": 1e-4,
                "stats": 20, # interval to record training statistics
                "grads": 200, # interval to output gradient statistics. can only run during "stats", so should be multiple of "stats"
                "eval": 2000, # interval to run inference and output results
            },
            # {
                # "opt_steps": 20000,
                # "batch_size": 1,
                # "pre_steps": (0,12), # int, can also be an interval [min, max], or callable "lambda opt_step: pre_step"
                # "sim_steps": 12,
                # "lr_net": 1e-4,
                # "stats": 40,
                # "grads": 200, # can only run during "stats", so should be multiple of "stats"
                # "eval": 2500,
            # },
            # {
                # "opt_steps": 8000, #20000,
                # "batch_size": 1,
                # "pre_steps": (0,24), # int, can also be an interval [min, max], or callable "lambda opt_step: pre_step"
                # "sim_steps": 12,
                # "lr_net": 1e-4,
                # "stats": 40,
                # "grads": 200, # can only run during "stats", so should be multiple of "stats"
                # "eval": 4000,
            # },
            # {
                # "opt_steps": 20000,
                # "batch_size": 1,
                # "pre_steps": (0,48), # int, can also be an interval [min, max], or callable "lambda opt_step: pre_step"
                # "sim_steps": 12,
                # "lr_net": 1e-4,
                # "stats": 40,
                # "grads": 200, # can only run during "stats", so should be multiple of "stats"
                # "eval": 4000,
            # },
            # {
               # "opt_steps": 20000,
               # "batch_size": 1,
               # "pre_steps": (0,96), # int, can also be an interval [min, max], or callable "lambda opt_step: pre_step"
               # "sim_steps": 12,
               # "lr_net": 1e-4, #/2,
               # "stats": 40,
               # "grads": 200, # can only run during "stats", so should be multiple of "stats"
               # "eval": 4000,
            # },
        ]
        
        if eval_only:
            LOG.info("Evaluating channel flow corrector.")
        else:
            LOG.info("Training channel flow corrector.")
        LOG.info("Sim setup: reference R%s, visc %0.3e, forcing %s, time_step %0.3e (ETT %.03e), substeps %d, CFL<%0.3e, double %s", load_run_id, viscosity, forcing, time_step, ETT_step, substeps, adaptive_CFL, dp)
        
        if load_down:
            domain = load_domain_from_run(load_run_id, dtype=dtype, with_scalar=False, frame=0, name="domain_down_frame_%06d")
        else:
            domain = load_domain_from_run(load_run_id, dtype=dtype, with_scalar=False, frame=0)
            if down_factors is not None:
                domain = coarsen_Re550_domain(domain, down_factors, make_div_free=down_make_div_free)
        
        velocity_limit = 2
        min_velocity = torch.tensor([-velocity_limit]*3, device=cuda_device, dtype=dtype).reshape((1,3,1,1,1))
        max_velocity = torch.tensor([ velocity_limit]*3, device=cuda_device, dtype=dtype).reshape((1,3,1,1,1))
        
        forcing_limit = 2
        min_forcing = torch.tensor([-forcing_limit]*3, device=cuda_device, dtype=dtype).reshape((1,3,1,1,1))
        max_forcing = torch.tensor([ forcing_limit]*3, device=cuda_device, dtype=dtype).reshape((1,3,1,1,1))
        
        trainer = TCFcorrectionTrainer(run_dir, load_frames, domain, forcing, u_wall=u_wall, #viscosity,
            time_step=time_step, substeps=substeps, adaptive_CFL=adaptive_CFL,
            fixed_frame=None,
            use_network=True, lr_net=4e-4, # 1e-4 for forcing (4e-4 ok for first 1k opt steps), <1e-5 for viscosity. overwritten by training_intervals.
            network_time_stride=1,
            use_velocity=False, velocity_residual=True, velocity_loss_scale=1,
            velocity_constraint="CLAMP", min_velocity=min_velocity, max_velocity=max_velocity,
            velocity_divergence_loss_scale=1e-4,
            use_forcing=True, lr_forcing=5e+8, forcing_loss_scale=1,
            forcing_constraint="CLAMP", min_forcing=min_forcing, max_forcing=max_forcing,
            forcing_divergence_loss_scale=1e-4,
            use_viscosity=False, lr_viscosity=0.4, min_viscosity=1e-5,
             #default (SGD): 1e-3, without driest 1e-6, learn_wall_sclaing 1e-5 (1e-4 Adam)
            use_SMAG=False, lr_SMAG_C=1e-4, use_van_driest=True, learn_wall_scaling=False,
            input_velocity=True, input_transformation=False, input_wall_distance=True,
            loss_scale_velocity=0.0, ref_profiles=ref_profiles,
            ref_budgets=ref_budgets,
            #total_statistics_loss_scale=0.0, frame_statistics_loss_scale=0.0,
            total_statistics_loss_scale=1.0, frame_statistics_loss_scale=0.5,
            loss_scale_vel_mean=1, loss_scale_vel_mean_v=0.5, loss_scale_vel_mean_w=0.5,
            #loss_scale_vel_mean=0.2, loss_scale_vel_mean_v=0, loss_scale_vel_mean_w=0, #for eval
            loss_scale_vel_u_var=1, # u'+ has about 1/10 loss and gradient as U+
            loss_scale_vel_v_var=1, loss_scale_vel_w_var=1, loss_scale_vel_uv_cov=1,
            record_individual_grads=False, stop_fn=stop_handler)
        
        trainer.exclude_advection_solve_gradients = True
        trainer.exclude_pressure_solve_gradients = True
        trainer.preconditionBiCG = False
        trainer.BiCG_precondition_fallback = True
        
        if net_run_id is not None:
            LOG.info("Loading model from R%s", net_run_id)
            trainer.load_model("./test_runs", net_run_id)
        
        trainer.load_reference(load_run_id, down_factors, make_div_free=down_make_div_free, load_down=load_down,
            train_start=train_start_frame, train_end=train_end_frame)
        
        LOG.info("Training setup: frames (fixed %s, total %d, start %d, end %d), use network %s (vel %s, T %s, y+ %s, stride %s), velocity %s (res %s, MSE %.03e, c %s, min %s, max %s), forcing %s (MSE %.03e, c %s, min %s, max %s), viscosity %s, loss (vel %.03e, %s, total scale %s, per frame scale %s), gradients (exclude AdvLS %s, exclulde PLS %s), eval %d %d steps, reference profiles %s, intervals:\n%s",
            trainer.fixed_frame, trainer.num_steps, trainer.train_start, trainer.train_end,
            trainer.use_network, trainer.input_velocity, trainer.input_transformation, trainer.input_wall_distance is not None, trainer.net_time_stride,
            trainer.use_velocity, trainer.velocity_residual, trainer.velocity_loss_scale, trainer.velocity_constraint, trainer.min_velocity, trainer.max_velocity,
            trainer.use_forcing, trainer.forcing_loss_scale, trainer.forcing_constraint, trainer.min_forcing, trainer.max_forcing,
            trainer.use_viscosity,
            trainer.loss_scale_velocity, ", ".join("%s %.03e"%(k, v["scale"]) for k,v in trainer.statistics_losses.items()),
            trainer.total_statistics_loss_scale, trainer.frame_statistics_loss_scale,
            trainer.exclude_advection_solve_gradients, trainer.exclude_pressure_solve_gradients,
            eval_start_frame, eval_iterations,
            ("Torroja Re%d"%ref_profiles.Re_wall if isinstance(ref_profiles, TorrojaProfile) else "PISO %s F-%s"%(ref_profiles.run_id, ref_profiles.frames)),
            training_intervals)
        
        if not eval_only:
            if initial_eval:
                try:
                    trainer.eval(eval_iterations, start_frame=eval_start_frame, with_correction=False, log_images=True, compare_steps=False, log_frame_losses=False)
                except:
                    LOG.exception("Initial evaluation failed:")
                try:
                    trainer.eval(eval_iterations, start_frame=eval_start_frame, with_correction=True, log_images=True, compare_steps=False, log_corrections=True, log_frame_losses=False)
                except:
                    LOG.exception("Initial evaluation failed:")
            
            for interval in training_intervals:
                if "lr_net" in interval:
                    trainer.set_lr_net(interval["lr_net"])
                if "forcing_loss_scale" in interval:
                    trainer.forcing_loss_scale = interval["forcing_loss_scale"]
                trainer.run_optimization(interval["opt_steps"], interval["sim_steps"], pre_steps=interval.get("pre_steps", 0), batch_size=interval.get("batch_size",1),
                    stats_interval=interval.get("stats",1), grads_interval=interval.get("grads", 0), eval_interval=interval.get("eval",0),
                    iterations=eval_iterations, start_frame=eval_start_frame, with_correction=True, log_images=True, compare_steps=False, log_corrections=True, log_frame_losses=False)
                
                if stop_handler():
                    break
            
            if not stop_handler() and final_eval:
                try:
                    trainer.eval(eval_iterations_final, start_frame=eval_start_frame, with_correction=True, log_images=True, compare_steps=False, log_corrections=True, log_frame_losses=not trainer.use_velocity_loss)
                except:
                    LOG.exception("Final evaluation failed:")
                
                try:
                    trainer.eval(eval_iterations_final_cmp, start_frame=eval_start_frame, with_correction=True, log_images=True, compare_steps=True, log_corrections=True, log_frame_losses=True)
                except:
                    LOG.exception("Final evaluation failed:")
            
            try:
                trainer.save_and_plot_stats()
            except:
                LOG.exception("Final plotting failed:")
        else:
            try:
                trainer.eval(eval_iterations, start_frame=eval_start_frame, with_correction=False, log_images=True, compare_steps=False,
                    log_frame_losses=not trainer.use_velocity_loss)#, log_PSD_planes=[0,1,2,5])
            except:
                LOG.exception("Evaluation failed:")
            try:
               trainer.eval(eval_iterations, start_frame=eval_start_frame, with_correction=True, log_images=True, compare_steps=False,
                log_frame_losses=not trainer.use_velocity_loss, log_corrections=False,
                log_vape_images=False, log_vape_video=False, log_vape_npz=False)#, log_PSD_planes=[0,1,2,5])
            except:
               LOG.exception("Evaluation failed:")
        
        trainer.mem_usage.print_max_memory()
        stop_handler.unregister_signal()
    
    
    elif True: # load sim for further investigation
        
        
        def get_run_path(run_id, base_path="./test_runs", sub_dir=None):
            load_path = os.path.join(base_path, "%s_*"%(run_id,))
            if sub_dir is not None:
                load_path = os.path.join(load_path, sub_dir)
            path = glob.glob(load_path)
            if len(path)==0:
                raise IOError("No run found for path %s"%load_path)
            if len(path)>1:
                raise IOError("No unique run found for path %s, found %s"%load_path, path)
            return path[0]
        
        def load_run_domain(run_dict, device=cuda_device, dtype=torch.float32):
            sub_dir = None
            if run_dict["sub_dir"] is not None and run_dict["domain_dir"] is None:
                sub_dir = run_dict["sub_dir"]
            elif run_dict["sub_dir"] is None and run_dict["domain_dir"] is not None:
                sub_dir = run_dict["domain_dir"]
            elif run_dict["sub_dir"] is not None and run_dict["domain_dir"] is not None:
                sub_dir = os.path.join(run_dict["sub_dir"], run_dict["domain_dir"])
            
            load_path = get_run_path(run_dict["run_id"], sub_dir=sub_dir)
            
            domain = domain_io.load_domain(os.path.join(load_path,"domain"), dtype=dtype, device=device)
            return domain
        
        def load_run_vel_stats(run_dict, u_wall=None, out_dir=None, device=cuda_device, dtype=torch.float32):
            domain = load_run_domain(run_dict, dtype=dtype, device=device)
            
            sub_dir = None
            if run_dict["sub_dir"] is not None and run_dict["stats_dir"] is None:
                sub_dir = run_dict["sub_dir"]
            elif run_dict["sub_dir"] is None and run_dict["stats_dir"] is not None:
                sub_dir = run_dict["stats_dir"]
            elif run_dict["sub_dir"] is not None and run_dict["stats_dir"] is not None:
                sub_dir = os.path.join(run_dict["sub_dir"], run_dict["stats_dir"])
            
            load_path = get_run_path(run_dict["run_id"], base_path=run_dict.get("base_path", "./test_runs"), sub_dir=sub_dir)
            
            vel_stats = VelocityStats(domain, out_dir, LOG, u_wall=u_wall)
            
            if run_dict["frames"]=="ALL":
                end_steps = 0
            elif run_dict["avg_steps"]=="PARSE":
                
                sim_frames, sim_substeps = parse_adaptive_substeps(get_run_path(run_dict["run_id"], sub_dir="log"))
                if len(sim_substeps)<run_dict["frames"]:
                    raise ValueError("The simulation does not have that many frames.")
                end_steps = sum(sim_substeps[-run_dict["frames"]:])
                avg_end_step = end_steps//run_dict["frames"]
                LOG.info("parsed log: %d frames, end steps: %d, avg: %d", run_dict["frames"], end_steps, avg_end_step)
                
            else:
                end_steps = run_dict["frames"]*run_dict["avg_steps"]
            
            vel_stats.load_vel_stats(load_path, start=-end_steps, load_energy_budgets=run_dict["energy_budgets"],
                dtype=dtype, device=cuda_device, _load_old=run_dict.get("has_old", False))
            return vel_stats
        
        dtype = torch.float32
        
        runs = [
            {
                "run_id": "250403-094103", # it400-ss25, frame 388
                "Re_wall": 550,
                "sub_dir": "eval_0000_c", # sub-directory containing the inference run to load
                "stats_dir": None, # optional different directory containing the statistics
                "domain_dir": None, # optional different directory containing the domain
                "frames": "ALL",
                "short_name": "Re550_r64N3_learned-corrector-tuned", # sub-directory for the evaluation
                "energy_budgets": True,
                "foam_ref": None, # optional OpenFOAM reference simulation to compare statistics to
            },
        ]
        
        run_dir = setup_run("./test_runs",
            name="Re550_r64N3_f388_learned-corrector_half-plot_stats-errors-sqr-norm"
            )
        LOG = get_logger("Main")
        
        for run_idx, run in enumerate(runs):
        
            
            LOG.info("Stats for run '%s' (R%s)", run["short_name"], run["run_id"])
            sub_dir = os.path.join(run_dir, "%04d_R%s_%s_f%s"%(run_idx, run["run_id"], run["short_name"], run["frames"]))
            
            
            # plot final state stats
            Re_wall = run.get("Re_wall", 550)
            Re_center = TCF_tools.Re_wall_to_cl(Re_wall)
            u_wall = Re_wall / Re_center #1
            
            LOG.info("load test stats")
            vel_stats = load_run_vel_stats(run, u_wall, out_dir=sub_dir, dtype=dtype)
            
            LOG.info("load reference stats")
            vel_stats.plot_u_vel_max = 25
            vel_stats.plot_vel_rms_max = 3.5
            profiles_path = "./data/tcf_torroja"
            profiles = TorrojaProfile(profiles_path, Re_wall)
            vel_stats.set_references(profiles)
            #LOG.info("plot channel flux")
            #vel_stats.plot_bulk_umean(name="")
            if run.get("foam_ref", None):
                from lib.data.OpenFOAM_profile import OpenFOAMProfile
                foam_ref = OpenFOAMProfile(run["foam_ref"])
                vel_stats.add_additional_reference_stats(foam_ref, linestyle="--")
            
            
            vel_stats.log_stats_errors_half()
            
            reference_budgets = None
            if "reference_budgets" in run:
                reference_vel_stats = load_run_vel_stats(run["reference_budgets"], out_dir=os.path.join(sub_dir, "reference_budgets"), dtype=dtype)
                reference_budgets = reference_vel_stats
                
            LOG.info("plot stats")
            vel_stats.plot_final_stats(reference_budgets=reference_budgets)
    
    elif False:
        
        run_dir = setup_run("./test_runs",
            name="TCF_Reference_profiles"
            )
        LOG = get_logger("Main")
        
        profiles_path = "./data/tcf_torroja"
        
        profile_550 = TorrojaProfile(profiles_path, 550)
        
        #LOG.info("data fields: %s", profile_550.field_names)
        #for name in profile_550.field_names:
        #    LOG.info("%s: %d\n%s", name, len(profile_550.profiles[name]), profile_550.profiles[name])
        
        profile_550.plot_stats(run_dir)
        profile_550.plot_full_stats(run_dir)
        
        profile_180 = TorrojaProfile(profiles_path, 180)
        profile_180.plot_stats(run_dir)
        profile_180.plot_full_stats(run_dir)

    close_logging()
