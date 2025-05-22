import os
# use GPU with lowest memory used, if available
from lib.util.GPU_info import get_available_GPU_id
os.environ["CUDA_VISIBLE_DEVICES"] = str(get_available_GPU_id(active_mem_threshold=0.8, default=None))

import torch
assert torch.cuda.is_available()
cuda_device = torch.device("cuda")
cpu_device = torch.device("cpu")
dtype = torch.float32

import PISOtorch # core CUDA module
import PISOtorch_simulation # helper for PISO loop
import lib.data.shapes as shapes # mesh generation
from lib.util.output import plot_grids
from lib.util.logging import setup_run, get_logger, close_logging # logging and output

# setup logging and output
run_dir = setup_run("./test_runs", name="buoyancy_rand_g0.1_visc2e-4_16-r4_rand-fix"
    )
LOG = get_logger("Main")
stop_handler = PISOtorch_simulation.StopHandler(LOG)
stop_handler.register_signal()

# some settings
dims = 2
x, y = 16, 32 # must be divisible by 4
res_factor = 4
refine = False
viscosity = 2e-4
#velocity = 1
buoyancy_factor = 0.1
time_step=0.25
iterations = 200

x_base = x
y_base = y
x *= res_factor
y *= res_factor

# create the mesh
grid = shapes.make_wall_refined_ortho_grid(x, y, corner_upper=(1,y/x), wall_refinement=["-x", "+x", "-y", "+y"] if refine else [], base=1.10)
plot_grids([grid], path=run_dir)
grid = grid.cuda().contiguous()

# create domain
viscosity = torch.tensor([viscosity], dtype=dtype)
domain = PISOtorch.Domain(dims, viscosity, passiveScalarChannels=1, name="CavityDomain", device=cuda_device, dtype=dtype)

# create block from the mesh on the domain (missing settings, fields and transformation metrics are created automatically)
block = domain.CreateBlock(vertexCoordinates=grid, name="CavityBlock")
block.CloseAllBoundaries()

# finalize domain
domain.PrepareSolve()

# density inflow
torch.manual_seed(83342666)
density_inflow = torch.rand((1,1,y_base//16,x_base//4), dtype=dtype, device=cuda_device)
if res_factor>1:
    density_inflow = torch.repeat_interleave(density_inflow, res_factor, dim=2)
    density_inflow = torch.repeat_interleave(density_inflow, res_factor, dim=3)
density_inflow = torch.nn.functional.pad(density_inflow, ((x-x//4)//2, (x-x//4)//2, 2*y//16, y-(3*y//16)))

# set density in "inflow" region in each step
def density_inflow_fn(domain, **kwargs):
    block = domain.getBlock(0)
    block.setPassiveScalar(torch.maximum(block.passiveScalar, density_inflow).contiguous())
    domain.UpdateDomainData()

# buoyancy
vel_src_velX_pad = torch.zeros_like(block.passiveScalar)
def buoyancy_fn(domain, time_step, **kwargs):
    block = domain.getBlock(0)
    # use density as buoyancy force in y
    block.setVelocitySource(torch.cat([vel_src_velX_pad, block.passiveScalar * buoyancy_factor], dim=1))
    domain.UpdateDomainData()

prep_fn = {
    "PRE": [density_inflow_fn,],
    "PRE_VELOCITY_SETUP": [buoyancy_fn,],
}

# setup simulation
sim = PISOtorch_simulation.Simulation(domain=domain, prep_fn=prep_fn,
    substeps="ADAPTIVE", time_step=time_step, corrector_steps=2, pressure_tol=1e-5,
    advect_non_ortho_steps=1, pressure_non_ortho_steps=1, pressure_return_best_result=True,
    velocity_corrector="FD", non_orthogonal=False,
    log_interval=1, norm_vel=True,
    #output_resampling_coords=[grid], output_resampling_shape=[x,y],
    log_dir=run_dir, save_domain_name="domain",
    stop_fn=stop_handler)

# run simulation
sim.run(iterations=iterations)

stop_handler.unregister_signal()
close_logging()
