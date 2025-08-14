import os
import argparse
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# PICT imports
import PISOtorch
import PISOtorch_simulation
import lib.data.shapes as shapes
from lib.util.logging import setup_run, get_logger, close_logging
from lib.util.GPU_info import get_available_GPU_id

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = str(get_available_GPU_id(active_mem_threshold=0.8, default=None))

assert torch.cuda.is_available()
cuda_device = torch.device("cuda")
cpu_device = torch.device("cpu")


class TurbulenceDataGenerator:
    def __init__(self, args):
        self.args = args
        self.dtype = torch.float32
        self.logger = logging.getLogger("TurbulenceGen")
        
    def create_domain(self, resolution):
        """Create a 3D periodic domain for turbulence simulation"""
        dims = 3 if self.args.dims == 3 else 2
        
        # Physical domain size
        domain_length = 2 * np.pi * self.args.domain_scale
        
        # Create viscosity tensor
        viscosity = torch.tensor([self.args.viscosity], dtype=self.dtype)
        
        # Create domain
        domain = PISOtorch.Domain(
            dims,
            viscosity,
            name=f"TurbulenceDomain_{resolution}",
            device=cuda_device,
            dtype=self.dtype,
            passiveScalarChannels=0
        )
        
        # Create block with specified resolution
        if dims == 3:
            size = PISOtorch.Int4(x=resolution, y=resolution, z=resolution)
        else:
            size = PISOtorch.Int4(x=resolution, y=resolution)
            
        block = domain.CreateBlockWithSize(size, name=f"TurbulenceBlock_{resolution}")
        
        # Set all boundaries to periodic
        if dims == 3:
            block.MakePeriodic("x")
            block.MakePeriodic("y") 
            block.MakePeriodic("z")
        else:
            block.MakePeriodic("x")
            block.MakePeriodic("y")
        
        return domain, block
    
    def _generate_divergence_free_field(self, shape, peak_wavenumber):
        """Generate divergence-free velocity field using proper spectral method"""
        # Create wavenumber grids
        if len(shape) == 5:  # 3D
            nz, ny, nx = shape[2], shape[3], shape[4]
            kz = torch.fft.fftfreq(nz, device=cuda_device)
            ky = torch.fft.fftfreq(ny, device=cuda_device)
            kx = torch.fft.fftfreq(nx, device=cuda_device)
            KZ, KY, KX = torch.meshgrid(kz, ky, kx, indexing='ij')
            k_mag = torch.sqrt(KX**2 + KY**2 + KZ**2)
            
            # Create random potential in Fourier space
            potential_fft = torch.complex(
                torch.randn(nz, ny, nx, device=cuda_device),
                torch.randn(nz, ny, nx, device=cuda_device)
            )
            
        else:  # 2D
            ny, nx = shape[2], shape[3]
            ky = torch.fft.fftfreq(ny, device=cuda_device)
            kx = torch.fft.fftfreq(nx, device=cuda_device)
            KY, KX = torch.meshgrid(ky, kx, indexing='ij')
            k_mag = torch.sqrt(KX**2 + KY**2)
            
            # For 2D, use streamfunction to ensure divergence-free field
            streamfunction_fft = torch.complex(
                torch.randn(ny, nx, device=cuda_device),
                torch.randn(ny, nx, device=cuda_device)
            )
        
        # Apply realistic turbulence spectrum (Kolmogorov-like)
        # E(k) ~ k^(-5/3), so velocity ~ k^(-5/6)
        k_scaled = k_mag / (peak_wavenumber / 4.0)  # Scale relative to peak
        
        # Von Karman-like spectrum with proper energy distribution
        energy_spectrum = (k_scaled**4) / (1 + k_scaled**2)**(17/6)
        
        # Add exponential cutoff for high wavenumbers
        energy_spectrum *= torch.exp(-(k_scaled / 2.0)**2)
        
        # Remove DC component
        energy_spectrum[k_mag < 1e-10] = 0
        
        # Apply spectrum to create realistic turbulence
        if len(shape) == 5:  # 3D
            # Generate velocity from vector potential curl
            # u = ∇ × A ensures ∇ · u = 0
            potential_fft *= torch.sqrt(energy_spectrum)
            
            # Apply different random phases for each component
            Ax_fft = potential_fft * torch.exp(1j * 2 * np.pi * torch.rand_like(k_mag))
            Ay_fft = potential_fft * torch.exp(1j * 2 * np.pi * torch.rand_like(k_mag))
            Az_fft = potential_fft * torch.exp(1j * 2 * np.pi * torch.rand_like(k_mag))
            
            # Compute velocity as curl of vector potential: u = ∇ × A
            # ux = ∂Az/∂y - ∂Ay/∂z
            # uy = ∂Ax/∂z - ∂Az/∂x  
            # uz = ∂Ay/∂x - ∂Ax/∂y
            ux_fft = 1j * (2*np.pi) * (KY * Az_fft - KZ * Ay_fft)
            uy_fft = 1j * (2*np.pi) * (KZ * Ax_fft - KX * Az_fft)
            uz_fft = 1j * (2*np.pi) * (KX * Ay_fft - KY * Ax_fft)
            
            # Convert back to physical space
            ux = torch.fft.ifftn(ux_fft).real
            uy = torch.fft.ifftn(uy_fft).real
            uz = torch.fft.ifftn(uz_fft).real
            
            velocity = torch.stack([ux, uy, uz], dim=0).unsqueeze(0)
            
        else:  # 2D
            # For 2D: u = (-∂ψ/∂y, ∂ψ/∂x) ensures ∇ · u = 0
            streamfunction_fft *= torch.sqrt(energy_spectrum)
            
            # Compute velocity components from streamfunction
            ux_fft = -1j * (2*np.pi) * KY * streamfunction_fft
            uy_fft = 1j * (2*np.pi) * KX * streamfunction_fft
            
            # Convert back to physical space
            ux = torch.fft.ifftn(ux_fft).real
            uy = torch.fft.ifftn(uy_fft).real
            
            velocity = torch.stack([ux, uy], dim=0).unsqueeze(0)
        
        return velocity.to(dtype=self.dtype)
    
    def load_initial_velocity_from_training_data(self, resolution):
        """Load initial velocity field from existing training data"""
        # Construct path to training data file
        training_data_dir = Path(self.args.training_data_dir)
        data_file = training_data_dir / f"decaying_turbulence_v2_{resolution}x{resolution}_index_1.npz"
        
        if not data_file.exists():
            self.logger.warning(f"Training data file not found: {data_file}")
            self.logger.info("Falling back to generated initial conditions")
            return None, None
            
        self.logger.info(f"Loading initial velocity from: {data_file}")
        
        # Load the data
        data = np.load(data_file)
        u_data = data['u']  # Shape: [time, y, x]
        v_data = data['v']  # Shape: [time, y, x]
        
        # Extract timestep information from training data
        training_timestep = None
        
        # Check for delta_t (the actual timestep field in our training data)
        if 'delta_t' in data.keys():
            training_timestep = float(data['delta_t'])
            self.logger.info(f"Found delta_t timestep in training data: {training_timestep}")
        elif 'timestep' in data.keys():
            # Alternative timestep field name
            training_timestep = float(data['timestep'])
            self.logger.info(f"Found explicit timestep in training data: {training_timestep}")
        elif 'dt' in data.keys():
            # Another alternative timestep field name
            training_timestep = float(data['dt'])
            self.logger.info(f"Found dt timestep in training data: {training_timestep}")
        elif 'time_array' in data.keys():
            # If time_array is stored, calculate timestep from it
            time_array = data['time_array']
            if len(time_array) > 1:
                training_timestep = float(time_array[1] - time_array[0])
                self.logger.info(f"Calculated timestep from time_array: {training_timestep}")
        elif 'time' in data.keys():
            # Fallback to 'time' field
            time_array = data['time']
            if len(time_array) > 1:
                training_timestep = float(time_array[1] - time_array[0])
                self.logger.info(f"Calculated timestep from time array: {training_timestep}")
        else:
            # Try to infer timestep from number of time steps and total simulation time
            num_timesteps = u_data.shape[0]
            if 'total_time' in data.keys():
                total_time = float(data['total_time'])
                training_timestep = total_time / (num_timesteps - 1)
                self.logger.info(f"Inferred timestep from total time: {training_timestep}")
            else:
                # Last resort: use default or computed timestep 
                self.logger.warning("Could not extract timestep from training data, will use computed timestep")
                
        # Log additional information about the training data
        if 'time_array' in data.keys():
            time_array = data['time_array']
            total_time = time_array[-1] - time_array[0]
            self.logger.info(f"Training data time range: {time_array[0]:.6f} to {time_array[-1]:.6f} (total: {total_time:.6f})")
            
        if 'outer_steps' in data.keys():
            self.logger.info(f"Training data outer steps: {data['outer_steps']}")
        
        # Extract t=0 velocity field
        u_t0 = u_data[0, :, :]  # [y, x]
        v_t0 = v_data[0, :, :]  # [y, x]
        
        # Convert to PICT format: [1, channels, y, x]
        if self.args.dims == 3:
            # For 3D, we need to add a z-component (set to zero for now)
            w_t0 = np.zeros_like(u_t0)
            velocity = np.stack([u_t0, v_t0, w_t0], axis=0)  # [3, y, x]
            velocity = velocity[np.newaxis, :]  # [1, 3, y, x]
        else:
            # For 2D
            velocity = np.stack([u_t0, v_t0], axis=0)  # [2, y, x]
            velocity = velocity[np.newaxis, :]  # [1, 2, y, x]
        
        # Convert to torch tensor
        velocity_tensor = torch.from_numpy(velocity).to(dtype=self.dtype, device=cuda_device)
        
        # Log statistics
        max_vel = torch.max(torch.sqrt(torch.sum(velocity_tensor**2, dim=1))).item()
        mean_vel = torch.mean(torch.sqrt(torch.sum(velocity_tensor**2, dim=1))).item()
        self.logger.info(f"Loaded velocity statistics - Max: {max_vel:.3f}, Mean: {mean_vel:.3f}")
        
        # Verify divergence if 2D
        if self.args.dims == 2:
            div_rms = self._verify_divergence_free(velocity_tensor, resolution)
            self.logger.info(f"Loaded velocity field RMS divergence: {div_rms:.2e}")
        
        return velocity_tensor, training_timestep
    
    def _verify_divergence_free(self, velocity, resolution):
        """Verify that the velocity field is divergence-free"""
        if velocity.shape[1] < 2:  # Need at least 2D
            return 0.0
            
        # Extract velocity components
        u = velocity[0, 0, :, :] if len(velocity.shape) == 4 else velocity[0, 0, :, :]
        v = velocity[0, 1, :, :] if len(velocity.shape) == 4 else velocity[0, 1, :, :]
        
        # Compute derivatives using finite differences (periodic boundaries)
        dx = 2 * np.pi * self.args.domain_scale / resolution
        
        # Central differences with periodic boundary conditions
        du_dx = (torch.roll(u, -1, dims=1) - torch.roll(u, 1, dims=1)) / (2 * dx)
        dv_dy = (torch.roll(v, -1, dims=0) - torch.roll(v, 1, dims=0)) / (2 * dx)
        
        # Compute divergence
        divergence = du_dx + dv_dy
        
        # Return RMS divergence
        div_rms = torch.sqrt(torch.mean(divergence**2)).item()
        
        return div_rms
    
    def generate_initial_turbulence(self, domain, block):
        """Generate initial turbulent velocity field with proper divergence-free constraint"""
        dims = domain.getSpatialDims()
        block_size = block.getSizes()
        
        # Get resolution for verification
        resolution = block_size.x if dims == 2 else block_size.x
        
        self.logger.info(f"Generating divergence-free turbulent field for {resolution}^{dims} domain")
        
        # Generate random velocity field
        if dims == 3:
            shape = [1, dims, block_size.z, block_size.y, block_size.x]  # [1, 3, z, y, x]
        else:
            shape = [1, dims, block_size.y, block_size.x]  # [1, 2, y, x]
        
        # Create divergence-free velocity field using vector potential method
        velocity = self._generate_divergence_free_field(shape, self.args.peak_wavenumber)
        
        # Verify divergence-free property
        if dims == 2:
            div_rms = self._verify_divergence_free(velocity, resolution)
            self.logger.info(f"Initial velocity field RMS divergence: {div_rms:.2e}")
        
        # Scale to desired maximum velocity
        velocity_magnitude = torch.sqrt(torch.sum(velocity**2, dim=1, keepdim=True))
        max_vel = torch.max(velocity_magnitude).item()
        mean_vel = torch.mean(velocity_magnitude).item()
        
        self.logger.info(f"Velocity statistics before scaling - Max: {max_vel:.3f}, Mean: {mean_vel:.3f}")
        
        velocity = velocity * (self.args.max_velocity / max_vel)
        
        # Log final statistics
        final_max = torch.max(torch.sqrt(torch.sum(velocity**2, dim=1, keepdim=True))).item()
        final_mean = torch.mean(torch.sqrt(torch.sum(velocity**2, dim=1, keepdim=True))).item()
        self.logger.info(f"Velocity statistics after scaling - Max: {final_max:.3f}, Mean: {final_mean:.3f}")
        
        # Set velocity field
        block.setVelocity(velocity)
        
        return velocity
    
    def run_simulation(self, domain, resolution, steps, save_interval, training_timestep=None):
        """Run simulation and collect velocity trajectory data"""
        if training_timestep is not None:
            time_step = training_timestep
            self.logger.info(f"Using training data timestep: {time_step}")
        else:
            time_step = self.get_time_step(resolution)
            self.logger.info(f"Using computed timestep: {time_step}")
        
        self.logger.info(f"Running simulation at {resolution}^{domain.getSpatialDims()} resolution for {steps} steps")
        
        # Create log directory for main simulation
        log_dir = Path(self.args.save_dir) / f"simulation_logs_{resolution}"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create simulation
        sim = PISOtorch_simulation.Simulation(
            domain=domain,
            time_step=time_step,
            substeps=1,
            corrector_steps=2,
            non_orthogonal=False,
            pressure_tol=1e-6,
            velocity_corrector="FD",
            log_interval=save_interval,
            log_dir=str(log_dir),
            stop_fn=lambda: False
        )
        
        # Storage for trajectory data
        trajectory_data = []
        velocity = domain.getBlock(0).velocity.detach().cpu().numpy()
        trajectory_data.append(velocity.copy())
        
        # Run simulation and collect data
        for step in range(0, steps, save_interval):
            sim.run(iterations=save_interval)
            
            # Get current velocity field
            velocity = domain.getBlock(0).velocity.detach().cpu().numpy()
            trajectory_data.append(velocity.copy())
            
            if step % (save_interval * 10) == 0:
                self.logger.info(f"Completed {step}/{steps} steps at resolution {resolution}")
        
        return np.array(trajectory_data)
    
    def warmup_simulation(self, domain, resolution):
        """Run warmup simulation to reach statistically steady state"""
        warmup_time = self.args.warmup_time
        time_step = self.get_time_step(resolution)
        warmup_steps = int(warmup_time / time_step)
        
        self.logger.info(f"Running warmup for {warmup_steps} steps at resolution {resolution}")
        
        # Create log directory for warmup simulation
        log_dir = Path(self.args.save_dir) / "warmup_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        sim = PISOtorch_simulation.Simulation(
            domain=domain,
            time_step=time_step,
            substeps=1,
            corrector_steps=2,
            non_orthogonal=False,
            pressure_tol=1e-6,
            velocity_corrector="FD",
            log_interval=max(warmup_steps // 10, 1),
            log_dir=str(log_dir),
            stop_fn=lambda: False
        )
        
        sim.run(iterations=warmup_steps)
        self.logger.info(f"Warmup completed at resolution {resolution}")
    
    def get_time_step(self, resolution):
        """Calculate stable time step for given resolution"""
        dx = (2 * np.pi * self.args.domain_scale) / resolution
        return self.args.cfl_safety_factor * dx / self.args.max_velocity
    
    def downsample_velocity(self, velocity_hr, target_resolution, source_resolution):
        """Downsample high-resolution velocity to target resolution"""
        # Simple downsampling by taking every nth point
        factor = source_resolution // target_resolution
        
        if len(velocity_hr.shape) == 5:  # 3D: [1, 3, z, y, x]
            downsampled = velocity_hr[:, :, ::factor, ::factor, ::factor]
        else:  # 2D: [1, 2, y, x]
            downsampled = velocity_hr[:, :, ::factor, ::factor]
        
        # Make tensor contiguous in memory
        return downsampled.contiguous()
    
    def generate_data(self):
        """Main data generation pipeline"""
        self.logger.info("Starting turbulence data generation with PICT")
        
        # Check if we should use training data for initialization
        use_training_data_init = getattr(self.args, 'use_training_data_init', False)
        
        if use_training_data_init:
            self.logger.info("Using training data for initialization (skipping warmup)")
        else:
            self.logger.info("Using generated initial conditions with warmup")
        
        # Create high-resolution domain for initial conditions
        hr_domain, hr_block = self.create_domain(self.args.high_res)
        hr_training_timestep = None
        
        if use_training_data_init:
            # Try to load initial velocity from training data
            initial_velocity, hr_training_timestep = self.load_initial_velocity_from_training_data(self.args.high_res)
            
            if initial_velocity is not None:
                # Set the loaded velocity field
                hr_block.setVelocity(initial_velocity)
                hr_domain.PrepareSolve()
                hr_domain.UpdateDomainData()
                self.logger.info("Successfully initialized from training data")
                if hr_training_timestep is not None:
                    self.logger.info(f"Will use training data timestep: {hr_training_timestep}")
            else:
                # Fallback to generated initial conditions
                self.logger.info("Falling back to generated initial conditions")
                initial_velocity = self.generate_initial_turbulence(hr_domain, hr_block)
                hr_domain.PrepareSolve()
                
                # Run warmup since we're using generated conditions
                self.warmup_simulation(hr_domain, self.args.high_res)
        else:
            # Original approach: generate initial turbulent field
            initial_velocity = self.generate_initial_turbulence(hr_domain, hr_block)
            hr_domain.PrepareSolve()
            
            # Run warmup at high resolution
            self.warmup_simulation(hr_domain, self.args.high_res)
        
        # Get resolutions to generate
        resolution_list = []
        res = self.args.low_res
        while res <= self.args.high_res:
            resolution_list.append(res)
            res *= 2
        
        # Generate data for each resolution
        for resolution in resolution_list:
            self.logger.info(f"Generating data for resolution {resolution}")
            current_training_timestep = None
            
            if resolution == self.args.high_res:
                domain = hr_domain
                current_training_timestep = hr_training_timestep
            else:
                domain, block = self.create_domain(resolution)
                
                if use_training_data_init:
                    # Try to load velocity for this resolution
                    target_velocity, current_training_timestep = self.load_initial_velocity_from_training_data(resolution)
                    
                    if target_velocity is not None:
                        # Use loaded velocity for this resolution
                        self.logger.info(f"Using loaded velocity for resolution {resolution}")
                        block.setVelocity(target_velocity)
                        domain.PrepareSolve()
                        domain.UpdateDomainData()
                        if current_training_timestep is not None:
                            self.logger.info(f"Using training timestep {current_training_timestep} for resolution {resolution}")
                    else:
                        # Fallback to downsampling from high-res
                        hr_velocity = hr_domain.getBlock(0).velocity
                        downsampled_velocity = self.downsample_velocity(
                            hr_velocity, resolution, self.args.high_res
                        )
                        block.setVelocity(downsampled_velocity)
                        domain.PrepareSolve()
                        domain.UpdateDomainData()
                        # Use high-res timestep if available, otherwise computed
                        current_training_timestep = hr_training_timestep
                else:
                    # Original approach: downsample from high-res
                    hr_velocity = hr_domain.getBlock(0).velocity
                    downsampled_velocity = self.downsample_velocity(
                        hr_velocity, resolution, self.args.high_res
                    )
                    block.setVelocity(downsampled_velocity)
                    domain.PrepareSolve() # 会分配/构建稀疏结构、缓冲区，并在内部调用一次 GPU 指针同步（SetupDomainGPU）。调用完后域已初始化且指针已对齐。
                    domain.UpdateDomainData() # 只能在域已初始化后调用；它用来在你“改变了张量内容”（如 setVelocity / setVelocitySource / 设置 result 向量）之后刷新 GPU 侧指针/元数据。
                    # 第一次/拓扑或边界结构变化后：
                    # 先做所有结构与初值设置（如 block.setVelocity(...)）
                    # 调用 domain.PrepareSolve()（完成初始化与一次同步）
                    # 之后若又修改了任何张量，再调用 domain.UpdateDomainData()
                    # 已初始化的域、仅改动场数据时：
                    # 直接改（如 block.setVelocity(...)）
                    # 然后 domain.UpdateDomainData()；不需要再 PrepareSolve()
            
            trajectory = self.run_simulation(
                domain, resolution, self.args.generate_steps, 
                save_interval=self.args.save_interval,
                training_timestep=current_training_timestep
            )
            
            self.save_trajectory_data(trajectory, resolution, current_training_timestep)
    
    def save_trajectory_data(self, trajectory, resolution, timestep=None):
        """Save trajectory data in numpy format"""
        save_dir = Path(self.args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        data_file = save_dir / f"{self.args.save_file}_{resolution}x{resolution}_index_{self.args.save_index}.npz"
        
        self.logger.info(f"Saving training data to: {data_file}")
        
        # Calculate timestep if not provided
        if timestep is None:
            timestep = self.get_time_step(resolution)
            self.logger.info(f"Using computed timestep for saving: {timestep}")
        else:
            self.logger.info(f"Using training data timestep for saving: {timestep}")
        
        num_timesteps = trajectory.shape[0]
        time_array = np.arange(num_timesteps) * timestep
        
        # Extract velocity components
        if trajectory.shape[2] == 3:  # 3D
            u_data = trajectory[:, 0, 0, :, :, :]  # x-velocity
            v_data = trajectory[:, 0, 1, :, :, :]  # y-velocity  
            w_data = trajectory[:, 0, 2, :, :, :]  # z-velocity
            
            np.savez_compressed(
                data_file,
                u=u_data,
                v=v_data,
                w=w_data,
                time_array=time_array,  # Use consistent field name with training data
                delta_t=timestep,       # Use consistent field name with training data
                outer_steps=num_timesteps,
                resolution=resolution,
                steps=self.args.generate_steps,
                warmup_time=self.args.warmup_time,
                max_velocity=self.args.max_velocity,
                viscosity=self.args.viscosity,
                decay=self.args.decay,
                seed=self.args.seed,
                dims=3,
                domain_scale=self.args.domain_scale,
                cfl_safety_factor=self.args.cfl_safety_factor,
                peak_wavenumber=self.args.peak_wavenumber
            )
        else:  # 2D
            u_data = trajectory[:, 0, 0, :, :]  # x-velocity
            v_data = trajectory[:, 0, 1, :, :]  # y-velocity
            
            np.savez_compressed(
                data_file,
                u=u_data,
                v=v_data,
                time_array=time_array,  # Use consistent field name with training data
                delta_t=timestep,       # Use consistent field name with training data
                outer_steps=num_timesteps,
                resolution=resolution,
                steps=self.args.generate_steps,
                warmup_time=self.args.warmup_time,
                max_velocity=self.args.max_velocity,
                viscosity=self.args.viscosity,
                decay=self.args.decay,
                seed=self.args.seed,
                dims=2,
                domain_scale=self.args.domain_scale,
                cfl_safety_factor=self.args.cfl_safety_factor,
                peak_wavenumber=self.args.peak_wavenumber
            )
        
        self.logger.info(f"Saved trajectory shape: {trajectory.shape}")
        self.logger.info(f"Resolution: {resolution}x{resolution}, Steps: {self.args.generate_steps}")
        self.logger.info(f"Timestep: {timestep}, Total time: {time_array[-1]:.6f}")


def main():
    """
    Main function for PICT turbulence data generation.
    
    NEW FEATURE: The code now automatically extracts and uses timestep information 
    from training data when --use_training_data_init is enabled. This ensures 
    that PICT simulations use the same temporal resolution as the training data.
    
    Usage examples:
    
    1. Generate data using training data for initialization and timestep (NEW):
    python generate_turbulence_data_pict.py --use_training_data_init --training_data_dir "./training_data" --generate_steps 5000 --save_file "pict_from_training"
    
    2. Generate training data with original method:
    python generate_turbulence_data_pict.py --generate_steps 5000 --high_res 512 --save_file "turbulence_training"
    
    3. Quick test with training data initialization and timestep:
    python generate_turbulence_data_pict.py --use_training_data_init --generate_steps 100 --high_res 256 --low_res 64
    
    4. Use custom training data directory:
    python generate_turbulence_data_pict.py --use_training_data_init --training_data_dir "/path/to/your/training_data" --generate_steps 1000
    
    Training data timestep extraction:
    - Looks for 'timestep' field in training data files
    - Falls back to calculating from 'time' array if available
    - Uses computed timestep if training data timestep cannot be extracted
    - Saves timestep information in generated data files for consistency
    """
    parser = argparse.ArgumentParser(description='Generate turbulence training data using PICT')
    
    # Simulation parameters
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dims', type=int, default=2, choices=[2, 3], 
                       help='Spatial dimensions (2D or 3D)')
    parser.add_argument('--generate_steps', type=int, default=5)
    parser.add_argument('--save_interval', type=int, default=1,
                       help='Save data every N simulation steps')
    parser.add_argument('--warmup_time', type=float, default=0.0)
    
    # Physical parameters
    parser.add_argument('--max_velocity', type=float, default=4.2)
    parser.add_argument('--cfl_safety_factor', type=float, default=0.5)
    parser.add_argument('--viscosity', type=float, default=1e-3)
    parser.add_argument('--peak_wavenumber', type=int, default=4)
    parser.add_argument('--domain_scale', type=float, default=1.0)
    parser.add_argument('--decay', action='store_true', default=True,
                       help='Generate decaying turbulence (no forcing)')
    
    # Resolution parameters
    parser.add_argument('--low_res', type=int, default=64)
    parser.add_argument('--high_res', type=int, default=1024,  # Reduced from 2048 for PICT
                       help='Highest resolution (limited by GPU memory)')
    
    # Output parameters
    parser.add_argument('--save_dir', type=str, default='./data/pict_turbulence')
    parser.add_argument('--save_file', type=str, default="decaying_turbulence")
    parser.add_argument('--save_index', type=int, default=1)
    
    # Training data initialization
    parser.add_argument('--use_training_data_init', action='store_true', default=True,
                       help='Use training data t=0 velocity for initialization (skips warmup)')
    parser.add_argument('--training_data_dir', type=str, default='./training_data',
                       help='Directory containing training data files')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Main")
    
    logger.info(f"Starting PICT turbulence data generation")
    logger.info(f"Parameters: {vars(args)}")
    
    # Generate data
    generator = TurbulenceDataGenerator(args)
    generator.generate_data()
    
    logger.info("Data generation completed!")


if __name__ == "__main__":
    main() 