import os, signal
from lib.util.logging import setup_run, get_logger, close_logging
from lib.util.profiling import SAMPLE

import numpy as np

# choose which GPU to use
if __name__=="__main__":
    from lib.util.GPU_info import get_available_GPU_id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(get_available_GPU_id(active_mem_threshold=0.8, default=None))


import torch
import PISOtorch # domain data structures and core PISO functions. check /extensions/PISOtorch.cpp to see what is available
#import PISOtorch_diff # wrappers to make core PISO functions work with torch autodiff
import PISOtorch_simulation # uses core PISO functions to make full simulation steps

# PISOtorch is only implemented for GPU

assert torch.cuda.is_available()
cuda_device = torch.device("cuda")
cpu_device = torch.device("cpu")

from lib.util.output import ttonp, ntonp # tensor-to-numpy, numerical-to-numpy

from matplotlib import pyplot as plt

# some loss functions
def MSE(a, b):
    return torch.mean((a - b)**2)
def SSE(a, b):
    return torch.sum((a - b)**2)

def make_forcing_domain(x:int, y:int, viscosity:float=1, dtype=torch.float32):
    # a domain consists of at least 1 Block
    # each block has a velocity, a pressure and a passive scalar field
    dims = 2 #spatial dimensions

    v = torch.tensor([viscosity], dtype=dtype) # viscosity needs to be a tensor with shape [1] on cpu
    domain = PISOtorch.Domain(dims, v, name="DomainForcingSample", dtype=dtype, device=cuda_device)
    
    # a block can be created from just its resolution if no mesh is required. This lead to a physical cell size of 1 in all dimensions.
    block_size = PISOtorch.Int4(x=x, y=y)
    # fields/tensors (velocity, pressure, etc.) are automatically created with 0 values.
    block = domain.CreateBlockWithSize(block_size, name="Block")
    # fields are torch tensors in "channels first" format: NCDHW for 3D, NCHW for 2D
    # N=batch (must be 1), C=channels, D=depth (z-dimension), H=height (y), W=width (x)
    # shapes for fields always follow the zyx order
    # vectors (e.g. the channel dimension for velocity) use xyz order

    # by default block boundaries are set to periodic
    # we add closed no-slip boundaries at the upper and lower side of the block (parallel to the forcing)
    block.CloseBoundary("-y")
    # since the y-boundaries are periodic (by default), closing "-y" also closes "+y" to keep the block consistent
    # here, with the periodic x boundaries, this creates a periodic 2D channel

    # needs to be called before any simulation function to initialize some helper fields and check the validity of the created domain.
    domain.PrepareSolve()

    return domain

# simple 2D conv-net to learn velocity -> forcing
class ForcingNet(torch.nn.Module):
    def __init__(self):
        super(ForcingNet, self).__init__()
        kernel_size = 1
        self.conv1 = torch.nn.Conv2d(2, 8, kernel_size, 1)
        self.conv2 = torch.nn.Conv2d(8, 4, kernel_size, 1)
        self.conv3 = torch.nn.Conv2d(4, 2, kernel_size, 1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.conv3(x)
        return x


class ForcingTrainer:
    def __init__(self, log_path:str, domain:PISOtorch.Domain, forcing_type:str, target_type:str, lr:float, sim_steps=1, time_step=0.1, stop_fn=lambda: False, log=None):
        self.log = log
        self.stop_fn = stop_fn
        self.log_path = log_path
        os.makedirs(self.log_path, exist_ok=True)
        self.domain = domain
        #self.dtype = domain.getDtype()
        self.sim_steps = sim_steps
        self.time_step = time_step
        
        self.target_type = target_type
        if target_type=="FORCING":
            # use a constant forcing to simulate a target flow
            self.target_forcing = torch.tensor([[1,0]], dtype=domain.getDtype(), device=domain.getDevice()) #NC
        elif target_type=="HIGHRES":
            raise NotImplementedError
        else:
            raise ValueError("Unknown target type")
        
        self.forcing_type = forcing_type
        if forcing_type=="CONSTANT":
            # optimize a constant scalar forcing, no network training
            self.var_forcing = torch.zeros([1,domain.getSpatialDims()], device=domain.getDevice(), dtype=domain.getDtype()) #NC
            self.var_forcing.requires_grad_(True)
            self.var_list = [self.var_forcing]
            self.optimizer = torch.optim.SGD(self.var_list, lr=lr, momentum=0)
        elif forcing_type=="NETWORK":
            # train a network to infer a forcing from the velocity field at each simulation step
            self.forcing_net = ForcingNet().to(cuda_device)
            self.var_list = list(self.forcing_net.parameters())
            self.optimizer = torch.optim.Adam(self.var_list, lr=lr)
        else:
            raise ValueError("Unknown forcing type")
        
        
        
        self.losses = []
        self.gradients = []
    
    def make_base_divergence_free(self):
        sim = PISOtorch_simulation.Simulation(self.domain,
            substeps=1, time_step=self.time_step,
            non_orthogonal=False,
            log_dir=None, stop_fn=self.stop_fn)
        
        sim.make_divergence_free()
    
    def simulate_target(self):
        # the sub-directory to write the target simulation to
        target_path = os.path.join(self.log_path, "target")
        
        # get a copy to simulate the target
        # here we use Clone() to create a deep copy, including the underlying tensors, to not affect the base domain
        target_domain = self.domain.Clone()
        target_domain.PrepareSolve()
        
        if self.target_type=="FORCING":
            target_domain.getBlock(0).setVelocitySource(self.target_forcing)
            target_domain.UpdateDomainData()
        
        sim = PISOtorch_simulation.Simulation(target_domain,
            substeps=1, time_step=self.time_step,
            non_orthogonal=False,
            log_interval=1,
            log_dir=target_path, stop_fn=self.stop_fn)
        
        sim.run(self.sim_steps)
        
        self.target_domain = target_domain

    def pfn_set_forcing(self, domain, time_step, **kwargs):
        # callback to set the forcing during the simulation
        if self.forcing_type=="CONSTANT":
            forcing = self.var_forcing
        if self.forcing_type=="NETWORK":
            forcing = self.forcing_net(domain.getBlock(0).velocity)
        domain.getBlock(0).setVelocitySource(forcing)
        domain.UpdateDomainData()
    
    def pfn_clear_forcing(self, domain, **kwargs):
        # callback to remove the forcing during the simulation
        domain.getBlock(0).clearVelocitySource()
        domain.UpdateDomainData()
    
    
    def compute_loss(self, domain):
        loss = 0
        for block, target_block in zip(domain.getBlocks(), self.target_domain.getBlocks()):
            loss = loss + SSE(block.velocity, target_block.velocity)
        
        self.losses.append(ntonp(loss))
        return loss
    
    def optimization_step(self, opt_step, log_images=True):
        with SAMPLE("opt-step"):
            step_path = os.path.join(self.log_path, "step_%04d"%opt_step) if log_images else None
            prep_fn = {}
            
            with SAMPLE("copy-domain"):
                
                # Copy the domain object.
                opt_domain = self.domain.Copy()
                # N.B. Copy() creates a shallow copy with new domain, block, and boundary object that retain the references to the original tensors.
                # Thus, domain and opt_domain share the underlying tensors.
                # Running sim.run() with differentiable=False will overwrite the tensors and thus affect both domain and opt_domain.
                # Running sim.run() with differentiable=True, as we do for training, will create new tensors and thus affect only one domain, so we can safely re-use 'domain'.
                # Setting new tensors to a domain, e.g. domain.getBlock(0).setVelocity(new_velocity) will only set the new tensor reference for that domain.

                if self.forcing_type=="CONSTANT":
                    opt_domain.getBlock(0).setVelocitySource(self.var_forcing)
                elif self.forcing_type=="NETWORK":
                    # register the callback: before each simulation step ("PRE") the simulation calls 'pfn_set_forcing'
                    PISOtorch_simulation.append_prep_fn(prep_fn, "PRE", self.pfn_set_forcing)
                opt_domain.PrepareSolve()
            


            with SAMPLE("FWD"):
                sim = PISOtorch_simulation.Simulation(opt_domain,
                    prep_fn=prep_fn, # a collection of callback functions
                    substeps=1, time_step=self.time_step,
                    non_orthogonal=False, differentiable=True, 
                    log_images=log_images,
                    log_interval=1 if log_images else 0,
                    log_dir=step_path, stop_fn=self.stop_fn)
                
                
                sim.run(self.sim_steps, log_domain=False)
            
            with SAMPLE("loss"):
                loss = self.compute_loss(opt_domain)
            
            with SAMPLE("BWD"):
                loss.backward()
            
            if self.forcing_type=="CONSTANT":
                self.log.info("forcing %s: loss %s, grad %s", ttonp(self.var_forcing), ntonp(loss), ttonp(self.var_forcing.grad))
            else:
                self.log.info("loss %s", ntonp(loss))
            
            with SAMPLE("update"):
                self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # opt_domain.Detach() has to be called after every optimization step.
            # This is necessary to break cyclic references between the domain and the backprop graph.
            # If you forget this it will cause a memory leak and eventually crash training.
            opt_domain.Detach()
    
    def run_optimization(self, optimization_steps):
        
        for opt_step in range(optimization_steps):
            self.optimization_step(opt_step,
                log_images=(opt_step==0 or opt_step==(optimization_steps-1))) # only log the first and last optimization step
            if self.stop_fn():
                break
        
        self.save()
    
    def save(self):
        if self.forcing_type=="CONSTANT":
            np.savez_compressed(os.path.join(self.log_path, "forcing.npz"), forcing=ttonp(self.var_forcing))
        elif self.forcing_type=="NETWORK":
            torch.save(self.forcing_net.state_dict(), os.path.join(self.log_path, "forcingNet.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join(self.log_path, "optimizer.pt"))


    def plot_stats(self):
        nrows=1
        ncols=1
        ax_width=6.4
        ax_height=4.8
        
        fig, axs = plt.subplots(nrows,ncols, figsize=(ax_width*ncols, ax_height*nrows), squeeze=False)

        ax = axs[0][0]
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.plot(np.asarray(self.losses))
        
        fig.align_labels()
        fig.tight_layout()
        fig.savefig(os.path.join(self.log_path, "opt_stats.pdf"))
        plt.close(fig)


def learned_forcing_sample(log_dir, 
        sim_iterations:int=3, opt_steps:int=5,
        use_forcing_network=False, stop_handler=lambda: False):
    LOG = get_logger("Forcing")
    optim_steps = opt_steps #iterations for the optimization loop
    iterations = sim_iterations #iterations for the fluid simulation
    time_step = 0.1
    x = 16
    y = 16
    v = 1 # viscosity
    dtype = torch.float32
    
    # create a base domain
    domain = make_forcing_domain(x,y,v,dtype)
    
    trainer = ForcingTrainer(log_dir, domain,
        forcing_type="NETWORK" if use_forcing_network else "CONSTANT", target_type="FORCING",
        lr=4e-4 if use_forcing_network else 1e-2,
        sim_steps=iterations, time_step=time_step, stop_fn=stop_handler, log=LOG)

    # prepare the target
    trainer.make_base_divergence_free()
    trainer.simulate_target()
    
    trainer.run_optimization(optim_steps)

    trainer.plot_stats()



if __name__=='__main__':
    
    # create a new directory <time-step>_learning_sample in ./test_runs to use as base output directory
    run_dir = setup_run("./test_runs",
        name="learning-forcing-sample_ortho_constant"
    )
    LOG = get_logger("Main")
    stop_handler = PISOtorch_simulation.StopHandler()
    stop_handler.register_signal() # sets a flag when the process is interrupted to allow the simulation to stop gracefully

    # optimize a scalar forcing with GD
    learned_forcing_sample(os.path.join(run_dir, "forcing-constant"), sim_iterations=3, opt_steps=10, stop_handler=stop_handler)

    # train a forcing network with Adam
    learned_forcing_sample(os.path.join(run_dir, "forcing_network"), sim_iterations=3, opt_steps=100, use_forcing_network=True, stop_handler=stop_handler)

    
    stop_handler.unregister_signal()
    close_logging()
