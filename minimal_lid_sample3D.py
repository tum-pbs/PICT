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
run_dir = setup_run("./test_runs", name="minimal-lid-sample_Re-1000_refine-1.1-3D"
    )
LOG = get_logger("Main")
stop_handler = PISOtorch_simulation.StopHandler(LOG)
stop_handler.register_signal()

# some settings
dims = 3 # dimensionality of the setup
x, y, z = 32, 32, 32 # spatial resolution, number of cells
refine = True # if the grid should be refined towards the walls, usefull for higher Reynolds numbers
viscosity = 1e-3
velocity = 1 # the tangential velocity of the lid boundary
time_step = 0.25 # the output time step
iterations = 100


# create the mesh
grid = shapes.make_wall_refined_ortho_grid(x, y, wall_refinement=["-x", "+x", "-y", "+y"] if refine else [], base=1.10)
#plot_grids([grid], path=run_dir)
grid = shapes.extrude_grid_z(grid, z, 0, 2)
grid = grid.cuda().contiguous()

# create domain
viscosity = torch.tensor([viscosity], dtype=dtype)
domain = PISOtorch.Domain(
    dims, # number of spatial dimensions, 1-3
    viscosity, # global viscosity value as CPU-tensor with shape [1]
    name="CavityDomain", # a name for the domain
    device=cuda_device, # the cuda device to use
    dtype=dtype, # data type, used to check for consistency
    passiveScalarChannels=0) # optional passively advected scalar channels, not used here

# create block from the mesh on the domain (missing settings, fields and transformation metrics are created automatically)
block = domain.CreateBlock(vertexCoordinates=grid, name="CavityBlock")
block.CloseAllBoundaries()

# set lid velocity (new boundary), here a single value for the whole boundary, a tensor with shape [1,dims] (1 is the batch dimension, which has to be 1)
lid_vel = torch.tensor([[velocity,0, 0]], device=domain.getDevice(), dtype=dtype)
# it is also possible to set a spatially varying boundary velocity using a tensor with additional spatial dimensions (shape NCDHW), here it would be shape [1,2,1,32]
block.CloseBoundary("-y", lid_vel) # could also use: block.getBoundary("-y").setVelocity(lid_vel)

# finalize domain
domain.PrepareSolve()


# setup simulation
sim = PISOtorch_simulation.Simulation(domain=domain,
    time_step=time_step, # the output time step
    substeps="ADAPTIVE", # split the output time step in smaller simulation time steps if necessary to respect a CFL condition of 0.8
    corrector_steps=2, # number of pressure correction steps, PISO default is 2
    non_orthogonal=False, advect_non_ortho_steps=1, pressure_non_ortho_steps=1, # the setup is orthogonal, so no non-orthogonal corrector steps are needed
    pressure_tol=1e-5, # the numerical tolerance for the pressure CG solve
    pressure_return_best_result=False, # if the pressure solve should return its result with the lowest residual if the tolerance could not be reached
    velocity_corrector="FD", # The scheme used to compute pressure gradients for velocity correction, default is Finite Differences
    log_interval=1, norm_vel=True,
    log_images=True,
    log_vtk=False,
    output_resampling_coords=[grid], output_resampling_shape=[x,y,z],
    log_dir=run_dir, save_domain_name="domain",
    stop_fn=stop_handler)

# run simulation
sim.run(iterations=iterations)

stop_handler.unregister_signal()
close_logging()
