import numbers, signal, warnings
from lib.util.profiling import SAMPLE
import torch

import PISOtorch
import PISOtorch_diff
#import PISOtorch_diff_old as PISOtorch_diff
import numpy as np

assert torch.cuda.is_available()
cuda_device = torch.device("cuda")
cpu_device = torch.device("cpu")

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import lib.data.shapes as shapes
from lib.util.output import *
from lib.util.domain_io import save_domain

from lib.util.logging import get_logger
_LOG = get_logger("PISOsim")

from contextlib import nullcontext

from lib.util.outputVtk import save_vtk

class StopHandler:
    def __init__(self, log=None):
        self._log = log
        self.reset()
    #def __del__(self):
    #    self.unregister_signal()
    def stop(self):
        if self._log is not None:
            if self._stop:
                self._log.info('Simulation still stopping...')
            else:
                self._log.warning('Simulation interrupted, stopping...')
        self._stop = True
    def signal_stop(self, sig, frame):
        self.stop()
    def __call__(self):
        return self._stop
    def register_signal(self):
        signal.signal(signal.SIGINT, self.signal_stop)
    def unregister_signal(self):
        signal.signal(signal.SIGINT, signal.SIG_DFL)
    def reset(self):
        self._stop = False


def tensor_as_np(tensor):
    return tensor.detach().cpu().numpy()

def get_max_time_step(domain, time_step_target, CFL_cond=0.8, with_transformations=True):
    max_vel = domain.getMaxVelocity(True, with_transformations).cpu().numpy()
    max_time_step = CFL_cond/max_vel
    if max_time_step>=time_step_target:
        ss = 1
        ts = time_step_target
    else:
        ss = int(np.ceil(time_step_target / max_time_step))
        ts = time_step_target / ss
    return ts, ss


def getVelocityResultMaxMag(domain:PISOtorch.Domain):
    vel = domain.velocityResult
    vel = torch.reshape(vel, (domain.getSpatialDims(), domain.getTotalSize()))
    vel_max = torch.max(torch.abs(vel)).cpu().numpy().tolist()
    if vel_max>0:
        return torch.max(torch.linalg.vector_norm(vel, dim=0)).cpu().numpy().tolist()
    return 0

def getVelocityResultMaxVel(domain:PISOtorch.Domain):
    vel = domain.velocityResult
    vel_max = torch.max(torch.abs(vel)).cpu().numpy().tolist()
    return vel_max

def get_fixed_boundary_flux(bound, bound_idx):
    assert isinstance(bound, PISOtorch.FixedBoundary)
    dims = bound.getSpatialDims()
    bound_axis = bound_idx//2
    if bound.velocityType == PISOtorch.BoundaryConditionType.DIRICHLET:
        fluxes = bound.GetFluxes()
        
        return torch.sum(fluxes)

    else:
        raise ValueError("Only DIRICHLET boundaries are supportet")

def get_fixed_boundary_fluxes(list_idx_bound):
    domain = list_idx_bound[0][1].getParentDomain()
    boundary_flux = torch.zeros([1], dtype=domain.getDtype(), device=domain.getDevice())
    for boundIdx, bound in list_idx_bound:

        if isinstance(bound, PISOtorch.FixedBoundary):
            flux = get_fixed_boundary_flux(bound, boundIdx)
        else:
            raise TypeError("boundary type not supported.")
        #_LOG.info("fixed bound flux: %s", boundary_flux)
        
        if boundIdx%2==0:
            boundary_flux -= flux
        else:
            boundary_flux += flux
    return boundary_flux

def get_varying_boundary_flux(bound, bound_idx):
    assert isinstance(bound, PISOtorch.VaryingDirichletBoundary)
    dims = bound.getSpatialDims()
    bound_axis = bound_idx//2
    if bound.hasTransform:
        shape = bound.getSizes() #x,y,z
        transform = bound.transform # NDHWC
        
        inv_row_start = dims*dims + bound_axis*dims
        inv_row_end = inv_row_start + dims
        t_inv_row = transform[...,inv_row_start:inv_row_end].view(-1,1,dims) # (NDHW)1C
        J = transform[...,-1].view(-1, 1, 1) # (NDHW)11
        bound_vel = torch.moveaxis(bound.boundaryVelocity, 1, -1).view(-1,dims,1) # NCDHW -> NDHWC -> (NDHW)C1
        flux = J * torch.bmm(t_inv_row, bound_vel)
        
        return torch.sum(flux)
    else:
        return torch.sum(bound.boundaryVelocity[:,bound_axis])

# this breaks the outflow
def restrict_inflow(bound_vel, bound_idx):
    # clamp boundary flux s.t. there is no inflow
    bound_axis = bound_idx>>1
    bound_dir = bound_idx&1
    # only for orthogonal transforms
    dims = bound_vel.dim()-2
    channels = list(bound_vel.split(dims, dim=1))
    flux = channels[bound_axis]
    zero = torch.tensor(0, dtype=bound_vel.dtype, device=bound_vel.device)
    if bound_dir==0: # lower bound, inlfow is positive
        flux = torch.mimimum(flux, zero)
    else:
        flux = torch.maximum(flux, zero)
    channels[bound_axis] = flux
    return torch.cat(channels, dim=1)

def get_advective_velocity(velms, velm_idx, bound, bound_idx):
    velm = velms if isinstance(velms, torch.Tensor) else velms[velm_idx]

    dims = bound.getSpatialDims()
    velm_static = velm.dim()==2 # NC or NCDHW (1,dims,[[z,]y,]x)

    bound_axis = (bound_idx>>1)
    bound_size = bound.getSizes() # x,y,z,w
    bound_spatial_shape = [bound_size[dims-1-d] for d in range(dims)] # z,y,x
    hasTransform = bound.hasTransform() if isinstance(bound, PISOtorch.FixedBoundary) else bound.hasTransform

    if not velm_static:
        assert all( velm.size(2+d)==bound_spatial_shape[d] for d in range(dims) ), "varying velm size must match boundary"
    
    if hasTransform:
        if velm_static: # broadcast to boundary shape
            velm = velm.reshape([1,dims]+[1]*dims).repeat(1,1,*bound_spatial_shape)
        transform = bound.transform # NDHWC
        inv_row_start = dims*dims + bound_axis*dims
        inv_row_end = inv_row_start + dims
        t_inv_row = transform[...,inv_row_start:inv_row_end].view(-1,1,dims) # (NDHW)1C
        velm = torch.moveaxis(velm, 1, -1).view(-1,dims,1) # NCDHW -> NDHWC -> (NDHW)C1
        advective_vel = torch.bmm(t_inv_row, velm) # (NDHW) 1 1
        advective_vel = advective_vel.reshape([1,1] + bound_spatial_shape) # NCDHW
    else:
        advective_vel = velm[:,bound_axis:bound_axis+1] # N1DHW

    return advective_vel

def balance_boundary_fluxes(domain, free_bounds, tol=None):
    scale_all = True
    
    with torch.no_grad():
        boundaries = []
        for block in domain.getBlocks():
            boundaries.extend(block.getFixedBoundaries()) #list((boundIdx, bound),)

        fixed_boundaries = [_ for _ in boundaries if _[1] not in free_bounds]
        fixed_boundary_flux = get_fixed_boundary_fluxes(fixed_boundaries)
        variable_boundaries = [_ for _ in boundaries if _[1] in free_bounds]
        variable_boundary_flux = get_fixed_boundary_fluxes(variable_boundaries)

    #compensation_additive = False
    #_LOG.info("bounds %d: fixed %d, var %d", len(boundaries), len(fixed_boundaries), len(variable_boundaries))
    #_LOG.info("Fluxes: fixed %s, var %s", fixed_boundary_flux, variable_boundary_flux)
    #_LOG.info("Boundary flux balance %s", domain.GetBoundaryFluxBalance().cpu().numpy())
    if not torch.allclose(fixed_boundary_flux+variable_boundary_flux, torch.zeros_like(fixed_boundary_flux),
                          atol=PISOtorch_diff._get_solver_tolerance(tol, dtype=domain.getDtype()) * 0.01):
        flux_scale = -fixed_boundary_flux/variable_boundary_flux
        #d_flux = fixed_boundary_flux - variable_boundary_flux
        #_LOG.info("FluxScale: %s", flux_scale)

        for boundIdx, bound in variable_boundaries:
            bound_vel = bound.velocity
            # TODO: scale everyting or only the boundary normal component?
            if scale_all:
                bound.setVelocity(bound_vel*flux_scale)
            else:
                # TODO: non-ortho transform
                vel_comps = list(torch.split(bound_vel, domain.getSpatialDims(), dim=1))
                vel_comps[boundIdx//2] = vel_comps[boundIdx//2]*flux_scale
                bound.setVelocity(torch.cat(vel_comps, axis=1))
    #_LOG.info("Boundary flux balance %s", domain.GetBoundaryFluxBalance().cpu().numpy())

# see also: https://www.tfd.chalmers.se/~hani/kurser/OS_CFD_2022/LeandroLucchese/Report_Lucchese.pdf
def update_advective_boundaries(domain, bounds, velms, dt, tol=None):
    # bounds: list of boundaries that can be updated
    # velms: velocity tensors to advect the boundary with. one for each boundary or a global value
    #_LOG.info("adective bound update: %d boundaries", len(bounds))
    with torch.no_grad(), SAMPLE("advect bounds"):
        assert isinstance(bounds, list) and len(bounds)>0
        assert (isinstance(velms, torch.Tensor) and velms.dim()==2 and velms.size(0)==1 and velms.size(1)==domain.getSpatialDims()) \
            or (isinstance(velms, list) and len(velms)==len(bounds) and all( \
                isinstance(velm, torch.Tensor) and (velm.dim()==2 or velm.dim()==(domain.getSpatialDims()+2)) and velm.size(0)==1 and velm.size(1)==domain.getSpatialDims() \
                for velm in velms)), "velms tensors have wrong shape"

        #if any([block.hasTransform for block in domain.getBlocks()]):
        #    raise RuntimeError("update_advective_boundaries does not support transformations.")

        dims = domain.getSpatialDims()
        boundaries = []
        for blockIdx in range(domain.getNumBlocks()):
            block = domain.getBlock(blockIdx)
            for boundIdx in range(dims*2):
                bound = block.getBoundary(boundIdx)
                if isinstance(bound, (PISOtorch.VaryingDirichletBoundary, PISOtorch.StaticDirichletBoundary)):
                    boundaries.append((block, boundIdx, bound))
                    warnings.warn("DirichletBoundary is deprecated.")
                if isinstance(bound, PISOtorch.FixedBoundary):
                    boundaries.append((block, boundIdx, bound))
        
        #fixed_boundaries = [_ for _ in boundaries if _[2] not in bounds]
        variable_boundaries = [_ for _ in boundaries if _[2] in bounds]

        if len(bounds)!=len(variable_boundaries):
            raise ValueError("Bounds passed as advective outflow are not (all) part of the domain.")
        
        #_LOG.info("%d fixed, %d variable", len(fixed_boundaries), len(variable_boundaries))

            
        for block, boundIdx, bound in variable_boundaries:
            if isinstance(bound, (PISOtorch.VaryingDirichletBoundary, PISOtorch.FixedBoundary)):
                if boundIdx==0:
                    vel_slice = block.velocity[...,:1]
                    scal_slice = block.passiveScalar[...,:1] if block.hasPassiveScalar() else None
                elif boundIdx==1:
                    vel_slice = block.velocity[...,-1:]
                    scal_slice = block.passiveScalar[...,-1:] if block.hasPassiveScalar() else None
                elif boundIdx==2:
                    vel_slice = block.velocity[...,:1,:]
                    scal_slice = block.passiveScalar[...,:1,:] if block.hasPassiveScalar() else None
                elif boundIdx==3:
                    vel_slice = block.velocity[...,-1:,:]
                    scal_slice = block.passiveScalar[...,-1:,:] if block.hasPassiveScalar() else None
                elif boundIdx==4:
                    vel_slice = block.velocity[...,:1,:,:]
                    scal_slice = block.passiveScalar[...,:1,:,:] if block.hasPassiveScalar() else None
                elif boundIdx==5:
                    vel_slice = block.velocity[...,-1:,:,:]
                    scal_slice = block.passiveScalar[...,-1:,:,:] if block.hasPassiveScalar() else None
                else:
                    raise RuntimeError
                
                if True:
                    alpha = dt*2*get_advective_velocity(velms, bounds.index(bound), bound, boundIdx) # N1DHW
                else:
                    vel_m = torch.abs(velms[bounds.index(bound)])
                    alpha = dt*2*vel_m # dt * flow speed / distance(center, face)
                    hasTransform = bound.hasTransform() if isinstance(bound, PISOtorch.FixedBoundary) else bound.hasTransform
                    if hasTransform:
                        bound_axis = (boundIdx>>1)
                        matrix_element = dims*dims + bound_axis*dims + bound_axis
                        alpha = alpha * bound.transform[...,matrix_element] # inverse cell size in boundary-normal direction, assumes orthogonal transformation


                t = 1 - 1/(1+alpha) # interpolation weight
                #_LOG.info("advective bound interpolation weight: %s", t)

                if isinstance(bound, PISOtorch.FixedBoundary):
                    if bound.isVelocityStatic:
                        raise ValueError("outflow boundaries must be varying. Use FixedBoundary.makeVelocityVarying() to make the velocity varying.")
                    vel_bound = bound.velocity
                    if block.hasPassiveScalar() and bound.hasPassiveScalar():
                        if bound.isPassiveScalarStatic():
                            raise ValueError("outflow boundary passive scalar must be varying.")
                        if any(cond!=PISOtorch.BoundaryConditionType.DIRICHLET for cond in bound.passiveScalarTypes):
                            raise ValueError("passive scalar outflow boundary condition must be Dirichlet.")
                        scal_bound = bound.passiveScalar
                    else:
                        scal_bound = None
                else:
                    vel_bound = bound.boundaryVelocity
                    scal_bound = bound.boundaryScalar
                
                #_LOG.info("advective bound vel: %s", vel_bound)
                vel_bound_update = vel_bound - t*(vel_bound - vel_slice)
                bound.setVelocity(vel_bound_update)

                if scal_bound is not None:
                    scal_bound_update = scal_bound - t*(scal_bound - scal_slice)
                    #scal_bound.copy_(scal_bound - t*(scal_bound - scal_slice))
                    bound.setPassiveScalar(scal_bound_update)
            else:
                raise TypeError
    
    balance_boundary_fluxes(domain, bounds, tol=tol)
        

def update_advective_boundaries_static(domain, bounds, velms, dt):

    boundaries = []
    for blockIdx in range(domain.getNumBlocks()):
        block = domain.getBlock(blockIdx)
        for boundIdx in range(domain.getSpatialDims()*2):
            bound = block.getBoundary(boundIdx)
            if isinstance(bound, (PISOtorch.VaryingDirichletBoundary, PISOtorch.StaticDirichletBoundary)):
                boundaries.append((boundIdx, bound))
    
    variable_boundaries = [_ for _ in boundaries if _[1] in bounds]

        
    for boundIdx, bound in variable_boundaries:
        if isinstance(bound, PISOtorch.VaryingDirichletBoundary):
            if boundIdx==0:
                scal_slice = block.passiveScalar[...,:1]
            elif boundIdx==1:
                scal_slice = block.passiveScalar[...,-1:]
            elif boundIdx==2:
                scal_slice = block.passiveScalar[...,:1,:]
            elif boundIdx==3:
                scal_slice = block.passiveScalar[...,-1:,:]
            elif boundIdx==4:
                scal_slice = block.passiveScalar[...,:1,:,:]
            elif boundIdx==5:
                scal_slice = block.passiveScalar[...,-1:,:,:]
            else:
                raise RuntimeError
            vel_m = torch.abs(velms[bounds.index(bound)])
            scal_bound = bound.boundaryScalar
            scal_bound.copy_(scal_bound - (dt*2*vel_m)*(scal_bound - scal_slice))
        else:
            raise TypeError


def append_prep_fn(prep_fn, name, fn):
    assert isinstance(prep_fn, dict)
    if name not in prep_fn:
        prep_fn[name] = [fn]
    else:
        if isinstance(prep_fn[name], tuple):
            prep_fn[name] = list(prep_fn[name])
        
        if not isinstance(prep_fn[name], list):
            prep_fn[name] = [prep_fn[name], fn]
        else:
            prep_fn[name].append(fn)
    return prep_fn

def prepend_prep_fn(prep_fn, name, fn):
    assert isinstance(prep_fn, dict)
    if name not in prep_fn:
        prep_fn[name] = [fn]
    else:
        if isinstance(prep_fn[name], tuple):
            prep_fn[name] = list(prep_fn[name])
        
        if not isinstance(prep_fn[name], list):
            prep_fn[name] = [fn, prep_fn[name]]
        else:
            prep_fn[name].insert(0, fn)
    return prep_fn

def add_pressure_ts_norm(prep_fn):
    #_LOG.info("Using normalized pressure!")
    def pressure_ts_norm_1(domain, time_step, **kwargs):
        domain.setPressureRHSdiv(domain.pressureRHSdiv*time_step.cuda())
        domain.UpdateDomainData()
    def pressure_ts_norm_2(domain, time_step, **kwargs):
        domain.setPressureResult(domain.pressureResult/time_step.cuda())
        domain.UpdateDomainData()
    prepend_prep_fn(prep_fn, "POST_PRESSURE_SETUP", pressure_ts_norm_1)
    prepend_prep_fn(prep_fn, "POST_PRESSURE_NON_ORTHO", pressure_ts_norm_2)

class Simulation:
    # these must match the definition in 'PISO_multiblock_cuda.h'
    NON_ORTHO_DIRECT_MATRIX = 1
    NON_ORTHO_DIRECT_RHS = 2 # less stable than NON_ORTHO_DIRECT_MATRIX
    NON_ORTHO_DIAGONAL_MATRIX = 4 # not implemented
    NON_ORTHO_DIAGONAL_RHS = 8
    NON_ORTHO_CENTER_MATRIX = 16

    __NON_ORTHO_MODE = NON_ORTHO_CENTER_MATRIX | NON_ORTHO_DIRECT_MATRIX | NON_ORTHO_DIAGONAL_RHS # Bit flags
    
    def __init__(self, domain:PISOtorch.Domain=None, *, time_step:float=1.0, substeps:int=1, corrector_steps:int=2, density_viscosity:float=None,
           adaptive_CFL=0.8,
           prep_fn=None, advection_use_BiCG:bool=True, pressure_use_BiCG:bool=False, scipy_solve_advection:bool=False, scipy_solve_pressure:bool=False,
           preconditionBiCG:bool=False, BiCG_precondition_fallback:bool=True, 
           advection_tol:float=None, pressure_tol:float=None, convergence_tol:float=None, solver_double_fallback:bool=False,
           advect_non_ortho_steps:int=1, pressure_non_ortho_steps:int=1,
           normalize_pressure_result:bool=True, pressure_return_best_result:bool=False,
           advect_passive_scalar:bool=True, pressure_time_step_normalized:bool=False, #advect_velocity:bool=True,
           velocity_corrector="FD", non_orthogonal:bool=True,
           differentiable:bool=False, exclude_advection_solve_gradients:bool=False, exclude_pressure_solve_gradients:bool=False,
           log_dir:str=None, log_interval:int=0, log_images:bool=True, log_vtk:bool=False, norm_vel:bool=False, block_layout=None, output_mode3D:str="slice", log_fn=None,
           output_resampling_coords=None, output_resampling_shape=None, output_resampling_fill_max_steps=0,
           save_domain_name:str=None, stop_fn=lambda: False):
        
        self.__LOG = get_logger("PISOsim")
        self.__differentiable = False
        
        self.domain = domain
        self.density_viscosity = density_viscosity
        self.prep_fn = prep_fn
        
        self.non_orthogonal = non_orthogonal
        
        self.time_step = time_step
        self.substeps = substeps
        self.corrector_steps = corrector_steps
        self.convergence_tol = convergence_tol
        self.adaptive_CFL = adaptive_CFL
        
        self.scipy_solve_advection = scipy_solve_advection
        self.advection_use_BiCG = advection_use_BiCG
        self.advect_non_ortho_steps = advect_non_ortho_steps
        self.advection_tol = advection_tol
        self.advect_passive_scalar = advect_passive_scalar
        
        self.scipy_solve_pressure = scipy_solve_pressure
        self.pressure_use_BiCG = pressure_use_BiCG
        self.pressure_non_ortho_steps = pressure_non_ortho_steps
        self.pressure_tol = pressure_tol
        self.normalize_pressure_result = normalize_pressure_result
        self.pressure_return_best_result = pressure_return_best_result
        self.pressure_time_step_normalized = pressure_time_step_normalized
        
        self.solver_double_fallback = solver_double_fallback
        self.linear_solve_max_iterations = 5000
        self.preconditionBiCG = preconditionBiCG
        self.BiCG_precondition_fallback = BiCG_precondition_fallback
        
        self._velocity_corrector_versions = {"FD": 1, "FVM_CENTER": 5, "FVM_FACE": 6}
        self.velocity_corrector = velocity_corrector
        
        self.log_dir = log_dir
        self.log_interval = log_interval
        self.log_images = log_images
        self.log_vtk = log_vtk
        self.norm_vel = norm_vel
        self.block_layout = block_layout
        
        self.output_mode3D = output_mode3D
        self.log_fn = log_fn
        self.output_resampling_coords = output_resampling_coords
        self.output_resampling_shape = output_resampling_shape
        self.output_resampling_fill_max_steps = output_resampling_fill_max_steps
        self.save_domain_name = save_domain_name
        
        self.differentiable = differentiable

        self.exclude_advection_solve_gradients = exclude_advection_solve_gradients
        self.exclude_pressure_solve_gradients = exclude_pressure_solve_gradients
        #self.exclude_all_linsolve_gradients = False
        
        self.print_adaptive_step_info = False
        
        self.stop_fn = stop_fn
        
        self.reset_step_counters()
    
    @property
    def domain(self):
        return self.__domain
    @domain.setter
    def domain(self, domain):
        if domain is not None:
            if not isinstance(domain, PISOtorch.Domain): raise TypeError("domain must be a PISOtorch.Domain object or None.")
            if not domain.IsInitialized(): raise RuntimeError("domain must be initilized. Call domain.PrepareSolve() before assignment.")
        self.__domain = domain
    
    def _check_domain(self):
        if self.__domain is None: raise ValueError("no domain set.")
    
    def __get_dtype(self):
        self._check_domain()
        if self.domain.getNumBlocks()==0:
            raise RuntimeError("Block required to infer dtype")
        return self.domain.getBlock(0).velocity.dtype

    def save_domain(self, name="domain"):
        self._check_domain()
        if self.log_dir is None: raise RuntimeError("no log_dir set.")
        domain_path = os.path.join(self.log_dir, name)
        try:
            save_domain(self.domain, domain_path)
        except:
            self.__LOG.exception("FAILED to save sim:")
        else:
            self.__LOG.info("sim saved as: %s", name)
    
    
    def reset_image_out_idx(self):
        self.img_out_idx = 0
        
    def reset_vtk_out_idx(self):
        self.vtk_out_idx = 0
    
    def save_domain_images(self, idx=None, max_mag=1, vel_exr=False):
        self._check_domain()
        if self.log_dir is None: raise RuntimeError("no log_dir set.")
        _idx = self.img_out_idx if idx is None else idx
        try:
            save_domain_images(self.domain, self.log_dir, _idx, layout=self.block_layout, norm_p=True, max_mag=max_mag, mode3D=self.output_mode3D, vel_exr=vel_exr,
                            vertex_coord_list=self.output_resampling_coords, resampling_out_shape=self.output_resampling_shape,
                            fill_max_steps=self.output_resampling_fill_max_steps)
        except:
            self.__LOG.exception("FAILED to save images %s:", _idx)
        if idx is None: self.img_out_idx += 1
        
    def save_domain_vtk(self, idx=None):
        self._check_domain()
        if self.log_dir is None: raise RuntimeError("no log_dir set.")
        _idx = self.vtk_out_idx if idx is None else idx
        try:
            save_vtk(self.domain, self.log_dir, self.vtk_out_idx, vertex_coord_list=self.output_resampling_coords)
        except:
            self.__LOG.exception("FAILED to save vtk files %d:", idx)
        if idx is None: self.vtk_out_idx += 1
    

    @property
    def output_resampling_coords(self):
        if self.__output_resampling_coords is not None:
            return self.__output_resampling_coords
        elif self.domain.hasVertexCoordinates():
            return self.domain.getVertexCoordinates()
        else:
            return None
    @output_resampling_coords.setter
    def output_resampling_coords(self, output_resampling_coords):
        self.__output_resampling_coords = output_resampling_coords

    @property
    def density_viscosity(self):
        return self.__density_viscosity
    @density_viscosity.setter
    def density_viscosity(self, density_viscosity):
        if density_viscosity is not None: raise NotImplementedError("Simulation.density_viscosity is superseded by Domain.scalarViscosity.")
        if not (density_viscosity is None or isinstance(density_viscosity, numbers.Real)): raise TypeError("density_viscosity must be float or None.")
        if density_viscosity is not None and density_viscosity<0: raise ValueError("density_viscosity must not be negative.")
        self.__density_viscosity = density_viscosity
    
    
    def __check_differentiable(self):
        if self.differentiable:
            pass
            # if self.non_orthogonal: raise NotImplementedError("Differentiability is only implemented for orthogonal mode.")
            # if not self.pressure_time_step_normalized: raise NotImplementedError("Differentiable mode expects pressure to the time step normalized.")
    
    @property
    def differentiable(self):
        return self.__differentiable
    @differentiable.setter
    def differentiable(self, differentiable):
        if not isinstance(differentiable, bool): raise TypeError("differentiable must be bool.")
        self.__differentiable = differentiable
        self.__backend = PISOtorch_diff if differentiable else PISOtorch
        self.__check_differentiable()
    
    @property
    def non_orthogonal(self):
        return self.__non_orthogonal
    @non_orthogonal.setter
    def non_orthogonal(self, non_orthogonal):
        if not isinstance(non_orthogonal, bool): raise TypeError("non_orthogonal must be bool.")
        self.__non_orthogonal = non_orthogonal
        self.__non_ortho_flags = Simulation.__NON_ORTHO_MODE if non_orthogonal else 0
        self.__check_differentiable()
    
    @property
    def time_step(self):
        return self.__time_step
    @time_step.setter
    def time_step(self, time_step):
        if not isinstance(time_step, numbers.Real): raise TypeError("time_step must be float.")
        self.__time_step = time_step
        
    def __get_time_step_torch(self):
        self._check_domain()
        return torch.tensor([self.__time_step], device=cpu_device, dtype=self.__get_dtype())
    
    @property
    def substeps(self):
        return self.__substeps
    @substeps.setter
    def substeps(self, substeps):
        if isinstance(substeps, str) and substeps.upper()=="ADAPTIVE":
            substeps = -1
        else:
            if not isinstance(substeps, numbers.Integral): raise TypeError("substeps must be integral type or 'ADAPTIVE'.")
            if not substeps>0: raise ValueError("substeps must be positive.")
        self.__substeps = substeps
    
    @property
    def adaptive_CFL(self):
        return self.__adaptive_CFL
    @adaptive_CFL.setter
    def adaptive_CFL(self, adaptive_CFL):
        if not isinstance(adaptive_CFL, numbers.Real): raise TypeError("adaptive_CFL must be float.")
        if not adaptive_CFL>0: raise ValueError("adaptive_CFL must be positive.")
        self.__adaptive_CFL = adaptive_CFL
    
    @property
    def corrector_steps(self):
        return self.__corrector_steps
    @corrector_steps.setter
    def corrector_steps(self, corrector_steps):
        if not isinstance(corrector_steps, numbers.Integral): raise TypeError("corrector_steps must be integral type.")
        if corrector_steps<0: raise ValueError("corrector_steps must not be negative.")
        self.__corrector_steps = corrector_steps
    
    @property
    def convergence_tol(self):
        return self.__convergence_tol
    @convergence_tol.setter
    def convergence_tol(self, convergence_tol):
        if convergence_tol is not None:
            if not isinstance(convergence_tol, numbers.Real): raise TypeError("convergence_tol must be float or None.")
            if not convergence_tol>0: raise ValueError("convergence_tol must be positive.")
        self.__convergence_tol = convergence_tol
    
    # Advection
    
    @property
    def scipy_solve_advection(self):
        return self.__scipy_solve_advection
    @scipy_solve_advection.setter
    def scipy_solve_advection(self, scipy_solve_advection):
        if not isinstance(scipy_solve_advection, bool): raise TypeError("scipy_solve_advection must be bool.")
        self.__scipy_solve_advection = scipy_solve_advection
    
    @property
    def advection_use_BiCG(self):
        return self.__advection_use_BiCG
    @advection_use_BiCG.setter
    def advection_use_BiCG(self, advection_use_BiCG):
        if not isinstance(advection_use_BiCG, bool): raise TypeError("advection_use_BiCG must be bool.")
        self.__advection_use_BiCG = advection_use_BiCG
    
    @property
    def advect_non_ortho_steps(self):
        return self.__advect_non_ortho_steps
    @advect_non_ortho_steps.setter
    def advect_non_ortho_steps(self, advect_non_ortho_steps):
        if not isinstance(advect_non_ortho_steps, numbers.Integral): raise TypeError("advect_non_ortho_steps must be integral type.")
        if advect_non_ortho_steps<0: raise ValueError("advect_non_ortho_steps must not be negative.")
        self.__advect_non_ortho_steps = advect_non_ortho_steps
    
    @property
    def advection_tol(self):
        if self.__advection_tol is None:
            return PISOtorch_diff._get_solver_tolerance(dtype=self.__get_dtype())
        else:
            return self.__advection_tol
    @advection_tol.setter
    def advection_tol(self, advection_tol):
        if advection_tol is not None:
            if not isinstance(advection_tol, numbers.Real): raise TypeError("advection_tol must be float or None.")
            if not advection_tol>0: raise ValueError("advection_tol must be positive.")
        self.__advection_tol = advection_tol
    
    # Pressure
    
    @property
    def scipy_solve_pressure(self):
        return self.__scipy_solve_pressure
    @scipy_solve_pressure.setter
    def scipy_solve_pressure(self, scipy_solve_pressure):
        if not isinstance(scipy_solve_pressure, bool): raise TypeError("scipy_solve_pressure must be bool.")
        self.__scipy_solve_pressure = scipy_solve_pressure
    
    @property
    def pressure_use_BiCG(self):
        return self.__pressure_use_BiCG
    @pressure_use_BiCG.setter
    def pressure_use_BiCG(self, pressure_use_BiCG):
        if not isinstance(pressure_use_BiCG, bool): raise TypeError("pressure_use_BiCG must be bool.")
        self.__pressure_use_BiCG = pressure_use_BiCG
    
    @property
    def pressure_non_ortho_steps(self):
        return self.__pressure_non_ortho_steps
    @pressure_non_ortho_steps.setter
    def pressure_non_ortho_steps(self, pressure_non_ortho_steps):
        if not isinstance(pressure_non_ortho_steps, numbers.Integral): raise TypeError("pressure_non_ortho_steps must be integral type.")
        if pressure_non_ortho_steps<0: raise ValueError("pressure_non_ortho_steps must not be negative.")
        self.__pressure_non_ortho_steps = pressure_non_ortho_steps
    
    @property
    def pressure_tol(self):
        if self.__pressure_tol is None:
            return PISOtorch_diff._get_solver_tolerance_torch(dtype=self.__get_dtype())
        else:
            return self.__pressure_tol
    @pressure_tol.setter
    def pressure_tol(self, pressure_tol):
        if pressure_tol is not None:
            if not isinstance(pressure_tol, numbers.Real): raise TypeError("pressure_tol must be float or None.")
            if not pressure_tol>0: raise ValueError("pressure_tol must be positive.")
        self.__pressure_tol = pressure_tol
    
    @property
    def normalize_pressure_result(self):
        return self.__normalize_pressure_result
    @normalize_pressure_result.setter
    def normalize_pressure_result(self, normalize_pressure_result):
        if not isinstance(normalize_pressure_result, bool): raise TypeError("normalize_pressure_result must be bool.")
        self.__normalize_pressure_result = normalize_pressure_result
    
    @property
    def pressure_return_best_result(self):
        return self.__pressure_return_best_result
    @pressure_return_best_result.setter
    def pressure_return_best_result(self, pressure_return_best_result):
        if not isinstance(pressure_return_best_result, bool): raise TypeError("pressure_return_best_result must be bool.")
        self.__pressure_return_best_result = pressure_return_best_result
    
    @property
    def velocity_corrector(self):
        return self.__velocity_corrector
    @velocity_corrector.setter
    def velocity_corrector(self, velocity_corrector):
        if not velocity_corrector in self._velocity_corrector_versions:
            raise ValueError("velocity_corrector must be one of: %s"%(list(self._velocity_corrector_versions.keys()),))
        self.__velocity_corrector = velocity_corrector
        self._velocity_corrector_version = self._velocity_corrector_versions[velocity_corrector]
    
    # Logging/Output
    
    @property
    def log_dir(self):
        return self.__log_dir
    @log_dir.setter
    def log_dir(self, log_dir):
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self.__log_dir = log_dir
        else: self.__log_dir = None
    
    @property
    def log_interval(self):
        return self.__log_interval
    @log_interval.setter
    def log_interval(self, log_interval):
        if log_interval is None: log_interval=0
        if not isinstance(log_interval, numbers.Integral): raise TypeError("log_interval must be integral type or None.")
        if log_interval<0: raise ValueError("log_interval must not be negative.")
        self.__log_interval = log_interval
    
    @property
    def log_images(self):
        return self.__log_images
    @log_images.setter
    def log_images(self, log_images):
        if not isinstance(log_images, bool): raise TypeError("log_images must be bool.")
        self.__log_images = log_images
        
    @property
    def log_vtk(self):
        return self.__log_vtk
    @log_vtk.setter
    def log_vtk(self, log_vtk):
        if not isinstance(log_vtk, bool): raise TypeError("log_vtk must be bool.")
        self.__log_vtk = log_vtk
    
    @property
    def norm_vel(self):
        return self.__norm_vel
    @norm_vel.setter
    def norm_vel(self, norm_vel):
        if not isinstance(norm_vel, bool): raise TypeError("norm_vel must be bool.")
        self.__norm_vel = norm_vel
    
    @property
    def block_layout(self):
        return self.__block_layout
    @block_layout.setter
    def block_layout(self, block_layout):
        if not (block_layout is None or isinstance(block_layout, list)): raise TypeError("block_layout must be list or None.")
        self.__block_layout = block_layout
    
    @property
    def stop_fn(self):
        return self.__stop_fn
    @stop_fn.setter
    def stop_fn(self, stop_fn):
        if stop_fn is None: stop_fn = lambda: False
        if not callable(stop_fn): raise TypeError("stop_fn must be callable or None.")
        self.__stop_fn = stop_fn
    
    def _check_stop(self):
        return self.stop_fn is not None and self.stop_fn()
    
    @property
    def prep_fn(self):
        return self.__prep_fn
    @prep_fn.setter
    def prep_fn(self, prep_fn):
        if not (prep_fn is None or isinstance(prep_fn, dict)): raise TypeError("prep_fn must be dict or None.")
        self.__prep_fn = prep_fn
    
    def _run_prep_fn(self, name, **kwargs):
        if (self.__prep_fn is not None) and (name in self.__prep_fn) and (self.__prep_fn[name] is not None):
            fns = self.__prep_fn[name]
            if not isinstance(fns, (list, tuple)):
                fns = [fns]
            for fn in fns:
                if fn is not None: fn(**kwargs)
    
    def reset_total_step(self):
        self.total_step = 0
    
    def reset_total_time(self):
        self.total_time = 0
    
    def end_step(self, time_step=0):
        self.total_step += 1
        self.total_time += ntonp(time_step)[0]
    
    
    def reset_step_counters(self):
        self.reset_total_step()
        self.reset_total_time()
        self.reset_image_out_idx()
        self.reset_vtk_out_idx()
    
    def linear_solve_scipy(self, csrMat:PISOtorch.CSRmatrix, rhs:torch.Tensor, x=None):
        with SAMPLE("scipy linear solve"):
            A = csr_matrix((csrMat.value.cpu(), csrMat.index.cpu(), csrMat.row.cpu()), shape=[csrMat.getRows()]*2)
            with SAMPLE("linear solve"):
                x = spsolve(A, torch.flatten(rhs.cpu()))
            return torch.reshape(torch.tensor(x, dtype=rhs.dtype), rhs.size()).cuda()

    def linear_solve_GPU(self, csrMat:PISOtorch.CSRmatrix, rhs:torch.Tensor, x:torch.Tensor=None, use_BiCG=False, tol=None, max_iter=5000,
                         matrix_rank_deficient=False, residual_reset_step=0, return_best_result=False, double_fallback:bool=False,
                         BiCG_with_preconditioner=True, BiCG_precondition_fallback=False):
        if self.__backend==PISOtorch:
            convergence_criterion = PISOtorch.ConvergenceCriterion.NORM2_NORMALIZED # RMSE of residual vector
            transpose = False # used for backprop
            print_residual = False # Print final solver residual information. Use for debugging. Only for CG.
            #return_best_result = False # saves best intermediate result and returns it instead of final result after max_iter if the solve does not converge. Only for CG.
            maxit_torch = torch.IntTensor([max_iter])
            tol_torch = PISOtorch_diff._get_solver_tolerance_torch(tol, rhs.dtype)#torch.ones([1], dtype = rhs.dtype)*tol
            with SAMPLE("GPU linear solve"):
                x_given = x is not None
                if not x_given:
                    x = torch.zeros_like(rhs)
                
                PISOtorch_diff._linear_solve_wrapper(csrMat, rhs, x, maxit_torch, tol_torch, convergence_criterion,
                    use_BiCG, matrix_rank_deficient, residual_reset_step, transpose,
                    print_residual, return_best_result, double_fallback=double_fallback, debug_out=False,
                    BiCG_with_preconditioner=BiCG_with_preconditioner, BiCG_precondition_fallback=BiCG_precondition_fallback)
                
                return x, True
        else:
            if x is not None:
                self.__LOG.warning("x is ignored when using PISOtorch_diff.")
            return self.__backend.linear_solve_GPU(csrMat, rhs, transpose=False, use_BiCG=use_BiCG, tol=tol, max_iter=max_iter,
                return_best_result=return_best_result, double_fallback=double_fallback,
                BiCG_with_preconditioner=BiCG_with_preconditioner, BiCG_precondition_fallback=BiCG_precondition_fallback), True

    def linear_solve(self, csrMat:PISOtorch.CSRmatrix, rhs:torch.Tensor, x:torch.Tensor=None, use_BiCG=False, tol=None, max_iter=None,
                     matrix_rank_deficient=False, residual_reset_step=0, use_scipy=False, return_best_result=False): #, double_fallback:bool=False):
        if use_scipy:
            return self.linear_solve_scipy(csrMat, rhs, x), True
        else:
            if max_iter==None: max_iter = self.linear_solve_max_iterations
            return self.linear_solve_GPU(csrMat, rhs, x, use_BiCG, tol, max_iter, matrix_rank_deficient, residual_reset_step, return_best_result,
                double_fallback=self.solver_double_fallback, BiCG_with_preconditioner=self.preconditionBiCG, BiCG_precondition_fallback=self.BiCG_precondition_fallback)
        
    
    # advect only the passive scalar
    def advect_static(self, iterations, time_step=None):
        self._check_domain()
        solve_ok = True
        advect_non_ortho_reuse_result = True and not self.differentiable
    
        if isinstance(iterations, torch.Tensor):
            iterations = iterations.numpy()[0]
        
        if time_step is None:
            time_step = self.__get_time_step_torch()
        domain = self.domain
        non_ortho_flags = self.__non_ortho_flags
        
        if not domain.hasPassiveScalar():
            raise ValueError("Domain has no passive scalar to advect.")
        
        split_scalar_channels = not (domain.isPassiveScalarViscosityStatic() and domain.isAllFixedBoundariesPassiveScalarTypeStatic())
        scalar_channels = domain.getPassiveScalarChannels()
        RHS_single_size = domain.getTotalSize()
        
        for step in range(iterations):
            with SAMPLE("advect static"):
                self._run_prep_fn("PRE", domain=domain, local_step=step, time_step=time_step, total_step=self.total_step, total_time=self.total_time)
                
                domain.UpdateDomainData()
                
                if split_scalar_channels:
                    matrices = []
                    for channel in range(scalar_channels):
                        self.__backend.SetupAdvectionMatrix(domain, time_step, non_ortho_flags, forPassiveScalar=True, passiveScalarChannel=channel)
                        matrices.append(domain.C.clone())
                else:
                    self.__backend.SetupAdvectionMatrix(domain, time_step, non_ortho_flags, forPassiveScalar=True, passiveScalarChannel=0)
                self.__backend.CopyScalarResultFromBlocks(domain) # needed for non-ortho components on RHS
                
                for no_step in range(self.advect_non_ortho_steps):
                    self.__backend.SetupAdvectionScalar(domain, time_step, non_ortho_flags)# creates RHS for all channels
                    
                    self._run_prep_fn("POST_SCALAR_SETUP", domain=domain, no_step=no_step, local_step=step, time_step=time_step, total_step=self.total_step, total_time=self.total_time)
                    
                    with torch.no_grad() if self.exclude_advection_solve_gradients else nullcontext():
                        if split_scalar_channels:
                            if (no_step==0 or not advect_non_ortho_reuse_result):
                                x = [None] * RHS_single_size
                            else:
                                x = torch.split(domain.scalarResult, RHS_single_size)
                            RHS = torch.split(domain.scalarRHS, RHS_single_size)
                            scalarResult = []
                            for channel in range(scalar_channels):
                                sR, solve_ok = self.linear_solve(matrices[channel], RHS[channel], x=x[channel], use_BiCG=self.advection_use_BiCG, use_scipy=self.scipy_solve_advection, tol=self.advection_tol)
                                scalarResult.append(sR)
                            del x
                            del RHS
                            scalarResult = torch.cat(scalarResult)
                        else:
                            x = None if (no_step==0 or not advect_non_ortho_reuse_result) else domain.scalarResult
                            scalarResult, solve_ok = self.linear_solve(domain.C, domain.scalarRHS, x=x, use_BiCG=self.advection_use_BiCG, use_scipy=self.scipy_solve_advection, tol=self.advection_tol)
                            del x
                    
                    domain.setScalarResult(scalarResult)
                    domain.UpdateDomainData()
                    
                    if not solve_ok or self._check_stop():
                        return solve_ok
                
                if split_scalar_channels:
                    del matrices
                
                self.__backend.CopyScalarResultToBlocks(domain) #set final result to blocks for next iteration/step
                
                if not solve_ok or self._check_stop():
                    return solve_ok
                
                self.end_step(time_step)
        
        return solve_ok
    
    # run pressure correction to make the velocity divergence free
    def make_divergence_free(self, iterations=1, max_iter=1000):
        self._check_domain()
        corrector_steps = 1
        pressure_reuse_result = True and not self.differentiable # speeds up pressure_non_ortho_steps, no difference in result noticed.
        pressure_use_face_transform = False
        vcv = self._velocity_corrector_version

        if isinstance(iterations, torch.Tensor):
            iterations = iterations.numpy()[0]
        
        domain = self.domain
        non_ortho_flags = self.__non_ortho_flags
        # overwrite time step and A
        time_step = torch.tensor([1], device=cpu_device, dtype=domain.A.dtype)
        domain.setA(torch.ones_like(domain.A))
        
        
        for step in range(iterations):
            with SAMPLE("PISO step"):
                self._run_prep_fn("PRE", domain=domain, local_step=step, time_step=time_step, total_step=self.total_step, total_time=self.total_time)
                
                domain.UpdateDomainData()
                self.__backend.CopyVelocityResultFromBlocks(domain)

                for cstep in range(corrector_steps):
                    with SAMPLE("corrector step"):
                        # update the existing velocity directly
                        domain.setPressureRHS(domain.velocityResult)
                        domain.UpdateDomainData()
                        
                        self.__backend.SetupPressureMatrix(domain, time_step, non_ortho_flags, pressure_use_face_transform)
                        
                        last_pressure_result = 0
                        for pstep in range(self.pressure_non_ortho_steps):
                            # build only div(rhs) + non-ortho from existing rhs vector field
                            self.__backend.SetupPressureRHSdiv(domain, time_step, non_ortho_flags, pressure_use_face_transform, timeStepNorm=self.pressure_time_step_normalized) 
                            
                            with torch.no_grad() if self.exclude_pressure_solve_gradients else nullcontext():
                                x = None if (pstep==0 or not pressure_reuse_result) else domain.pressureResult
                                pressureResult, solve_ok = self.linear_solve(domain.P, domain.pressureRHSdiv, x=x, #matrix_rank_deficient=False, residual_reset_step=100,
                                    use_BiCG=self.pressure_use_BiCG, use_scipy=self.scipy_solve_pressure, tol=self.pressure_tol, return_best_result=self.pressure_return_best_result, max_iter=max_iter)
                                del x
                            
                            if self.normalize_pressure_result:
                                pressureResult = pressureResult - torch.mean(pressureResult) # for numerical and backwards stability
                            domain.setPressureResult(pressureResult)
                            domain.UpdateDomainData()
                            
                            if not solve_ok:
                                return solve_ok
                            
                            if self._check_stop():
                                break
                        
                        del last_pressure_result
                        
                        self.__backend.CopyPressureResultToBlocks(domain)
                        
                        self.__backend.CorrectVelocity(domain, time_step, version=vcv, timeStepNorm=self.pressure_time_step_normalized)
                
                self.__backend.CopyVelocityResultToBlocks(domain)
                
                if not solve_ok or self._check_stop():
                    return solve_ok
                
                self.end_step(time_step)
        
        return solve_ok
    
    def _PISO_split_step(self, iterations, time_step=None):
        self._check_domain()
        solve_ok = True
        if time_step is None:
            time_step = self.__get_time_step_torch()
        advect_use_prev_result = True and not self.differentiable
        advect_non_ortho_reuse_result = True and not self.differentiable
        pressure_reuse_result = True and not self.differentiable # speeds up pressure_non_ortho_steps, no difference in result noticed.
        pressure_use_face_transform = False
        # velocity corrector version. use different pressure gradients: 0 default (finite volume), 1 for finite differences, 4 for correcting fluxes (orthogonal), 5 for finite volume, 6 FVM with face transformations
        vcv = self._velocity_corrector_version 
        pressure_dp = False
        
        if isinstance(iterations, torch.Tensor):
            iterations = iterations.numpy()[0]
        assert iterations>0
        
        domain = self.domain
        non_ortho_flags = self.__non_ortho_flags
        
        for step in range(iterations):
            with SAMPLE("PISO step"):
                #_LOG.info("Substep %d", step)
                if self.convergence_tol is not None:
                    self.__backend.CopyVelocityResultFromBlocks(domain)
                    last_vel = domain.velocityResult.clone().detach()
                
                self._run_prep_fn("PRE", domain=domain, local_step=step, time_step=time_step, total_step=self.total_step, total_time=self.total_time)
                domain.UpdateDomainData()
                
                advection_matrix_for_velocity = False
                if self.advect_passive_scalar and domain.hasPassiveScalar():
                    with SAMPLE("Advect scalar"):
                        
                        split_scalar_channels = not (domain.isPassiveScalarViscosityStatic() and domain.isAllFixedBoundariesPassiveScalarTypeStatic())
                        scalar_channels = domain.getPassiveScalarChannels()
                        RHS_single_size = domain.getTotalSize()
                        
                        # if the scalar advection matrix can be reused for velocity advection
                        advection_matrix_for_velocity = not (domain.hasPassiveScalarViscosity() or domain.hasBlockViscosity() or
                            domain.hasPassiveScalarBlockViscosity() or split_scalar_channels)
                        #if self.density_viscosity is not None:
                        #    viscosity = domain.viscosity
                        #    domain.setViscosity(self.density_viscosity)
                        #    advection_matrix_for_velocity = False
                        
                        #self.__backend.SetupAdvectionMatrix(domain, time_step, non_ortho_flags)
                        if split_scalar_channels:
                            matrices = [] # pre-compute all matrices to avoid re-computation during non-ortho steps
                            for channel in range(scalar_channels):
                                self.__backend.SetupAdvectionMatrix(domain, time_step, non_ortho_flags, forPassiveScalar=True, passiveScalarChannel=channel)
                                matrices.append(domain.C.clone())
                        else:
                            self.__backend.SetupAdvectionMatrix(domain, time_step, non_ortho_flags, forPassiveScalar=True, passiveScalarChannel=0)
                        
                        if not self.non_orthogonal: # orthogonal version with gradient/backprop support
                            
                            self.__backend.SetupAdvectionScalar(domain, time_step, 0)
                            
                            self._run_prep_fn("POST_SCALAR_SETUP", domain=domain, no_step=0, local_step=step, time_step=time_step, total_step=self.total_step, total_time=self.total_time)
                            
                            with torch.no_grad() if self.exclude_advection_solve_gradients else nullcontext():
                                if split_scalar_channels:
                                    RHS = torch.split(domain.scalarRHS, RHS_single_size)
                                    scalarResult = []
                                    for channel in range(scalar_channels):
                                        sR, solve_ok = self.linear_solve(matrices[channel], RHS[channel], x=None, use_BiCG=self.advection_use_BiCG, use_scipy=self.scipy_solve_advection, tol=self.advection_tol)
                                        scalarResult.append(sR)
                                        del sR
                                    del RHS
                                    scalarResult = torch.cat(scalarResult)
                                else:
                                    scalarResult, solve_ok = self.linear_solve(domain.C, domain.scalarRHS, x=None, use_BiCG=self.advection_use_BiCG, use_scipy=self.scipy_solve_advection, tol=self.advection_tol)
                            
                            domain.setScalarResult(scalarResult)
                            domain.UpdateDomainData()
                            
                        else:
                            self.__backend.CopyScalarResultFromBlocks(domain) # needed for non-ortho components on RHS
                            
                            for no_step in range(self.advect_non_ortho_steps):
                                self.__backend.SetupAdvectionScalar(domain, time_step, non_ortho_flags)
                                
                                self._run_prep_fn("POST_SCALAR_SETUP", domain=domain, no_step=no_step, local_step=step, time_step=time_step, total_step=self.total_step, total_time=self.total_time)
                                
                                with torch.no_grad() if self.exclude_advection_solve_gradients else nullcontext():
                                    if split_scalar_channels:
                                        if (no_step==0 or not advect_non_ortho_reuse_result):
                                            x = [None] * RHS_single_size
                                        else:
                                            x = torch.split(domain.scalarResult, RHS_single_size)
                                        RHS = torch.split(domain.scalarRHS, RHS_single_size)
                                        scalarResult = []
                                        for channel in range(scalar_channels):
                                            sR, solve_ok = self.linear_solve(matrices[channel], RHS[channel], x=x[channel], use_BiCG=self.advection_use_BiCG, use_scipy=self.scipy_solve_advection, tol=self.advection_tol)
                                            scalarResult.append(sR)
                                        del x
                                        del RHS
                                        scalarResult = torch.cat(scalarResult)
                                    else:
                                        x = None if (no_step==0 or not advect_non_ortho_reuse_result) else domain.scalarResult
                                        scalarResult, solve_ok = self.linear_solve(domain.C, domain.scalarRHS, x=x, use_BiCG=self.advection_use_BiCG, use_scipy=self.scipy_solve_advection, tol=self.advection_tol)
                                    del x
                                
                                domain.setScalarResult(scalarResult)
                                domain.UpdateDomainData()
                                
                                if not solve_ok or self._check_stop():
                                    return solve_ok
                        
                        if split_scalar_channels:
                            del matrices
                        
                        self.__backend.CopyScalarResultToBlocks(domain)
                

                with SAMPLE("Advect velocity"):
                    
                    # DON'T use pressure from previous corrector steps, otherwise it's applied twice
                    apply_pressure_gradient = False
                    
                    self._run_prep_fn("PRE_VELOCITY_SETUP", domain=domain, local_step=step, time_step=time_step, total_step=self.total_step, total_time=self.total_time)
                    
                    #if self.density_viscosity is not None:
                    #    domain.setViscosity(viscosity)
                    
                    if not advection_matrix_for_velocity:
                        self.__backend.SetupAdvectionMatrix(domain, time_step, non_ortho_flags)
                    
                    if not self.non_orthogonal: # orthogonal version with gradient/backprop support
                        self.__backend.SetupAdvectionVelocity(domain, time_step, 0, apply_pressure_gradient)

                        self._run_prep_fn("POST_VELOCITY_SETUP", domain=domain, no_step=0, local_step=step, time_step=time_step, total_step=self.total_step, total_time=self.total_time)
                        
                        with torch.no_grad() if self.exclude_advection_solve_gradients else nullcontext():
                            x = None if (not advect_use_prev_result) else domain.velocityResult
                            velocityResult, solve_ok = self.linear_solve(domain.C, domain.velocityRHS, x=x, use_BiCG=self.advection_use_BiCG, use_scipy=self.scipy_solve_advection, tol=self.advection_tol)
                            del x
                        
                        domain.setVelocityResult(velocityResult)
                        domain.UpdateDomainData()
                        
                    else:
                        self.__backend.CopyVelocityResultFromBlocks(domain) # needed for non-ortho components on RHS
                        
                        for no_step in range(self.advect_non_ortho_steps):
                            self.__backend.SetupAdvectionVelocity(domain, time_step, non_ortho_flags, apply_pressure_gradient)

                            self._run_prep_fn("POST_VELOCITY_SETUP", domain=domain, no_step=no_step, local_step=step, time_step=time_step, total_step=self.total_step, total_time=self.total_time)
                            
                            with torch.no_grad() if self.exclude_advection_solve_gradients else nullcontext():
                                x = None if (no_step==0 or not advect_non_ortho_reuse_result) else domain.velocityResult
                                velocityResult, solve_ok = self.linear_solve(domain.C, domain.velocityRHS, x=x, use_BiCG=self.advection_use_BiCG, use_scipy=self.scipy_solve_advection, tol=self.advection_tol)
                                del x
                            
                            domain.setVelocityResult(velocityResult)
                            domain.UpdateDomainData()
                            
                            if not solve_ok or self._check_stop():
                                return solve_ok
                            
                        
                    
                    #CopyVelocityResultToBlocks(domain) not yet, the original vel on blocks is still needed for pressure rhs
                    
                    if not solve_ok:
                        return solve_ok

                self._run_prep_fn("POST_PREDICTION", domain=domain, local_step=step, time_step=time_step, total_step=self.total_step, total_time=self.total_time)
                
                # can use stop handler in POST_PREDICTION to stop sim after advection.
                if self._check_stop():
                    break
                
                for cstep in range(self.corrector_steps):
                    with SAMPLE("corrector step"):
                        
                        if not self.non_orthogonal: # orthogonal version with gradient/backprop support
                            self.__backend.SetupPressureCorrection(domain, time_step, 0, pressure_use_face_transform, timeStepNorm=self.pressure_time_step_normalized)
                            
                            self._run_prep_fn("POST_PRESSURE_SETUP", domain=domain, local_step=step, time_step=time_step, total_step=self.total_step, total_time=self.total_time)
                            
                            with torch.no_grad() if self.exclude_pressure_solve_gradients else nullcontext():
                                pressureResult, solve_ok = self.linear_solve(domain.P, domain.pressureRHSdiv, x=None, #matrix_rank_deficient=False, residual_reset_step=100,
                                    use_BiCG=self.pressure_use_BiCG, use_scipy=self.scipy_solve_pressure, tol=self.pressure_tol, return_best_result=self.pressure_return_best_result)
                            
                            if not solve_ok:
                                return solve_ok
                            
                            if self.normalize_pressure_result:
                                pressureResult = pressureResult - torch.mean(pressureResult) # for numerical and backwards stability
                            domain.setPressureResult(pressureResult)
                            domain.UpdateDomainData()
                            
                            self._run_prep_fn("POST_PRESSURE_RESULT", domain=domain, local_step=step, time_step=time_step, total_step=self.total_step, total_time=self.total_time)
                            
                        else: # non-ortho version
                            self.__backend.SetupPressureMatrix(domain, time_step, non_ortho_flags, pressure_use_face_transform)
                            
                            for pstep in range(self.pressure_non_ortho_steps):
                                #self.__backend.SetupPressureCorrection(domain, time_step, non_ortho_flags)
                                if pstep==0:
                                    # build rhs (vector field) (and div(rhs) + non-ortho)
                                    self.__backend.SetupPressureRHS(domain, time_step, non_ortho_flags, pressure_use_face_transform, timeStepNorm=self.pressure_time_step_normalized)
                                    #self._run_prep_fn("POST_PRESSURE_RHS", domain=domain, local_step=step, time_step=time_step, total_step=self.total_step, total_time=self.total_time)
                                else:
                                    # build only div(rhs) + non-ortho from existing rhs vector field
                                    self.__backend.SetupPressureRHSdiv(domain, time_step, non_ortho_flags, pressure_use_face_transform, timeStepNorm=self.pressure_time_step_normalized)

                                self._run_prep_fn("POST_PRESSURE_SETUP", domain=domain, local_step=step, time_step=time_step, total_step=self.total_step, total_time=self.total_time)
                                
                                with torch.no_grad() if self.exclude_pressure_solve_gradients else nullcontext():
                                    #_LOG.info("Start pressure solve #%d", cstep)
                                    x = None if (pstep==0 or not pressure_reuse_result) else domain.pressureResult
                                    if pressure_dp:
                                        self.__LOG.info("Pressure solve is double precision.")
                                        P = domain.P.toType(torch.float64)
                                        pressureRHSdiv = domain.pressureRHSdiv.to(torch.float64)
                                        if x is not None: x = x.to(torch.float64)
                                        pressureResult, solve_ok = self.linear_solve(P, pressureRHSdiv, x=x,
                                            use_BiCG=self.pressure_use_BiCG, use_scipy=self.scipy_solve_pressure, tol=self.pressure_tol,
                                            return_best_result=self.pressure_return_best_result) #, x=domain.pressureResult
                                        pressureResult = pressureResult.to(domain.pressureRHSdiv.dtype)
                                        del P
                                        del pressureRHSdiv
                                    else:
                                        pressureResult, solve_ok = self.linear_solve(domain.P, domain.pressureRHSdiv, x=x, matrix_rank_deficient=False, residual_reset_step=100,
                                            use_BiCG=self.pressure_use_BiCG, use_scipy=self.scipy_solve_pressure, tol=self.pressure_tol,
                                            return_best_result=self.pressure_return_best_result)
                                    del x
                                #solve_ok = True #DEBUG
                                
                                if self.normalize_pressure_result:
                                    pressureResult = pressureResult - torch.mean(pressureResult) # for numerical and backwards stability
                                domain.setPressureResult(pressureResult)
                                domain.UpdateDomainData()

                                self._run_prep_fn("POST_PRESSURE_RESULT", domain=domain, local_step=step, time_step=time_step, total_step=self.total_step, total_time=self.total_time)
                                
                                if not solve_ok:
                                    return solve_ok
                                
                                if self._check_stop():
                                    break
                            

                        self._run_prep_fn("POST_PRESSURE_NON_ORTHO", domain=domain, local_step=step, time_step=time_step, total_step=self.total_step, total_time=self.total_time)
                        

                        self.__backend.CopyPressureResultToBlocks(domain)

                        self.__backend.CorrectVelocity(domain, time_step, version=vcv, timeStepNorm=self.pressure_time_step_normalized) #vcv

                        self._run_prep_fn("POST_VELOCITY_CORRECTION", domain=domain, local_step=step, time_step=time_step, total_step=self.total_step, total_time=self.total_time)
                            
                        if self._check_stop():
                            break
                
                self.__backend.CopyVelocityResultToBlocks(domain)

                self._run_prep_fn("POST", domain=domain, local_step=step, time_step=time_step, total_step=self.total_step, total_time=self.total_time)
                
                if self.convergence_tol is not None:
                    step_max_diff = torch.max(torch.abs(last_vel - domain.velocityResult)).cpu().numpy()
                    if step_max_diff<self.convergence_tol:
                        self.__LOG.info("Simulation step max difference is under convergence tolerance.")
                        solve_ok = False # to stop sim
                
                if not solve_ok or self._check_stop():
                    break

                self.end_step(time_step)
        
        return solve_ok
    
    def _PISO_adaptive_step(self, CFL_cond=None, max_subsetps=1000):
        self._check_domain()
        time_step_target = self.time_step
        domain = self.domain
        CFL_cond = CFL_cond if CFL_cond is not None else self.adaptive_CFL
        substep = 0
        warned = False
        while time_step_target>0 and not np.isclose(time_step_target, 0):
            with SAMPLE("adaptive step"):
                
                max_vel = domain.getMaxVelocity(True, True)
                max_vel_np = max_vel.detach().cpu().numpy()
                
                if np.isclose(max_vel_np, 0):
                    max_time_step = time_step_target
                else:
                    max_time_step = CFL_cond / max_vel_np
                
                if max_time_step>=time_step_target:
                    substeps = 1
                    ts = time_step_target
                else:
                    substeps = int(np.ceil(time_step_target / max_time_step))
                    ts = time_step_target / substeps
                
                time_step_target -= ts
                ts = torch.tensor([ts], dtype=domain.getBlock(0).velocity.dtype, device=cpu_device)
                
                #_LOG.info("Adaptive step v2: maxVel %f, substep %d, timestep %f, remaining time %f", max_vel_np, substep, ts, time_step_target)
                
                if substeps>max_subsetps and not warned:
                    self.__LOG.warning("adaptive step (CFL=%.02f) results in more than %d substeps (%d).", CFL_cond, max_subsetps, substeps)
                    warned = True
                elif substep==0 and self.print_adaptive_step_info: #
                    self.__LOG.info("Adaptive step %d substeps: %d. From CFL = %.02f, max vel = %.03e, time step = %.03e.", substep,
                        substeps, CFL_cond, max_vel_np, time_step_target)
                
                sim_ok = self._PISO_split_step(iterations=1, time_step=ts)
                substep += 1
                
                if not sim_ok or self._check_stop():
                    return False
        self.__LOG.info("Adaptive time step %.03e (CFL=%.02f) used %d substeps.", self.time_step, CFL_cond, substep)
        return True
    
    def run(self, iterations, static=False, log_domain=True):
        self._check_domain()
        # time_step: physical time to pass per iteration and substep
        # substeps: how many piso steps to make per iteration
        # corrector_steps: number of corrector steps in the PISO algorithm
        # static: only advect the passive scalar
        
        # tolerances:
        # - advection_tol: tolerance for advection BiCG convergence (residual)
        # - pressure_tol: tolerance for pressure CG convergence (residual)
        # - convergence_tol: tolerance for simulation convergence (difference between consecutive steps)
        
        domain = self.domain

        if self.log_interval>0 and (self.log_images or self.log_vtk) and self.log_dir is None:
            raise ValueError("need to specify log/output directory")
        self.__LOG.info("Starting sim with %d iterations, output in %s.", iterations, self.log_dir or "NONE")
        if log_domain:
            self.__LOG.info(str(domain))
            for blockIdx in range(domain.getNumBlocks()):
                self.__LOG.info(str(domain.getBlock(blockIdx)))
        
        domain_orientation = domain.GetCoordinateOrientation()
        if domain_orientation==0:
            self.__LOG.warning("Domain coordinate systems have mixed orientation. This can lead to issues with the simulation.")
        domain_flux_balance = domain.GetBoundaryFluxBalance()
        if ntonp(torch.abs(domain_flux_balance))>ntonp(self.pressure_tol):
            self.__LOG.warning("Domain boundary is not divergence free (flux balance: %.03e). This can prevent pressure solve convergence.", domain_flux_balance)
            return False
        #self.__LOG.info("Domain handedness %d, boundary flux balance: %s", domain_orientation, domain_flux_balance.cpu().numpy())
        
        with SAMPLE("runSim"):
            
            sim_ok = True
            time_step_target = self.time_step
            substeps = self.substeps
            if self.norm_vel: # or True:
                max_mag_temp = tensor_as_np(domain.getMaxVelocityMagnitude(False))
                max_mag = max_mag_temp * 1.05
            else:
                max_mag = 1
                max_mag_temp = max_mag
            CFL_cond = 0.8
            adaptive_step = False
            time_step = None
            
            if substeps>0:
                pass # just fixed substeps. 1 iteration with have physical time = time_step*substeps.
            elif substeps==-1:
                # compute max time step for each iteration/substep based on current velocity. 1 iteration with have physical time = time_step.
                adaptive_step = True
            elif substeps==-2:
                # compute max time step based on initial conditions, then keep it constant. 1 iteration with have physical time = time_step.
                time_step, substeps = get_max_time_step(domain, time_step_target, CFL_cond, with_transformations=True)
                self.__LOG.info("Setting time step to %.02e, substeps to %d based on initial conditions.", time_step, substeps)
                time_step = torch.tensor([time_step], dtype=domain.getBlock(0).velocity.dtype, device=cpu_device)
            else:
                raise ValueError("Invalid substeps")
            
            
            
            #DEBUG pressure stability timestep influence
            if False:
                add_pressure_ts_norm(prep_fn)
            

            out_dir = self.log_dir
            
            vel_exr=False #not static
            
            if self.log_images:
                if domain.getSpatialDims()<3 and domain.hasVertexCoordinates():
                    plot_grids(domain.getVertexCoordinates(), path=out_dir, linewidth=0.5, type="pdf")
                self.save_domain_images(max_mag=max_mag, vel_exr=vel_exr)
            for it in range(1, iterations+1):
                with SAMPLE("Iteration"):
                    #LOG.info("It: %d/%d", it, iterations)
                    log = self.log_interval>0 and (it%self.log_interval)==0
                    
                    self.__LOG.info("It: %d/%d, substeps:%s, timestep:%f", it, iterations, "adaptive" if adaptive_step else substeps, time_step_target)
                    with SAMPLE("simIt"):
                        try:
                            if static:
                                sim_ok = self.advect_static(iterations=substeps, time_step=time_step)
                            elif adaptive_step:
                                sim_ok = self._PISO_adaptive_step()
                            else:
                                sim_ok = self._PISO_split_step(iterations=substeps, time_step=time_step)
                            # if not sim_ok:
                                # break
                        except PISOtorch_diff.LinsolveError as e:
                            #self.__LOG.error("Simulation failed in major step %d (total step %d):\n%s", it, self.total_step, str(e))
                            self.__LOG.exception("Simulation failed in major step %d (total step %d):", it, self.total_step)
                            break
                    
                    with SAMPLE("vel mag"):
                        max_mag_temp = tensor_as_np(domain.getMaxVelocityMagnitude(False))
                        #_LOG.info("Max vel magnitude: %.03e ", max_mag_temp, max_mag_temp_old)
                        if log: max_vel_temp = tensor_as_np(domain.getMaxVelocity(False))
                        #max_vel_transformed = domain.getMaxVelocity(False, True).cpu().numpy()
                        #_LOG.info("Max vel: %.03e, with bounds %.03e, transformed block 0 %.03e", max_vel_temp, domain.getMaxVelocity(True).cpu().numpy(), max_vel_transformed)
                        if np.isnan(max_mag_temp):
                            self.__LOG.warning("NaN encountered in velocity, stopping")
                            sim_ok = False
                            break
                        elif False and max_mag_temp*time_step_target>CFL_cond:
                            self.__LOG.warning("CFL condition violation.")
                            sim_ok = False or not advect
                            #break
                        if self.norm_vel: # or True:
                            max_mag = max_mag_temp * 1.05
                    
                    
                    if log:
                        with SAMPLE("vel div"):
                            vel_div = PISOtorch.ComputeVelocityDivergence(domain).detach()
                            vel_div_abs = torch.abs(vel_div)
                        with SAMPLE("p stats"):
                            p = domain.pressureResult.detach()
                            p_mean = torch.mean(p).cpu().numpy()
                            p_min = torch.min(p).cpu().numpy()
                            p_max = torch.max(p).cpu().numpy()
                            del p
                        with SAMPLE("output"):
                            self.__LOG.info("%d/%d Stats:\nVelocity: max=%.03e, max mag=%.03e\nPressure: mean=%.03e, min=%.03e, max=%.03e\nDivergence: mean=%.03e, min=%.03e, max=%.03e, total=%.03e", it, iterations,
                                max_vel_temp, max_mag_temp,
                                p_mean, p_min, p_max,
                                torch.mean(vel_div_abs).cpu().numpy(),torch.min(vel_div_abs).cpu().numpy(),
                                torch.max(vel_div_abs).cpu().numpy(), torch.sum(vel_div_abs).cpu().numpy())
                            if self.log_images: self.save_domain_images(max_mag=max_mag, vel_exr=vel_exr)
                            if self.log_vtk: self.save_domain_vtk()
                    if self.log_fn is not None:
                        with SAMPLE("log fn"):
                            self.log_fn(domain=domain, out_dir=out_dir, it=it, out_it=self.img_out_idx, total_step=self.total_step)
                            
                    
                    if self._check_stop() or (not sim_ok):
                        break
        
        if self.save_domain_name is not None:
            self.save_domain(self.save_domain_name)
        
        self.__LOG.info("sim finished after %d total steps", self.total_step)
        
        return sim_ok

    
    
