# testing the backwards pass of individual PISO functions using torch's gradcheck

from lib.util.logging import setup_run, get_logger, close_logging
from lib.util.profiling import SAMPLE

#import os, signal, itertools
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
#import domain_structs
import PISOtorch
import PISOtorch_diff
import PISOtorch_simulation
#import numpy as np

assert torch.cuda.is_available()
cuda_device = torch.device("cuda")
cpu_device = torch.device("cpu")

import lib.data.shapes as shapes

LOG = get_logger("test_gradcheck")

DTYPE = torch.float64
GRADCHECK_EPS = 1e-6
GRADCHECK_ATOL = 1e-5
GRADCHECK_NONDET_TOL = 1e-7

USE_RANDOM_INPUTS = True

USE_SCALAR_VISCOSITY = False
USE_BLOCK_VISCOSITY = False


# these must match the definition in 'PISO_multiblock_cuda.h'
NON_ORTHO_DIRECT_MATRIX = 1
NON_ORTHO_DIRECT_RHS = 2 # less stable than NON_ORTHO_DIRECT_MATRIX
NON_ORTHO_DIAGONAL_MATRIX = 4 # not implemented
NON_ORTHO_DIAGONAL_RHS = 8
NON_ORTHO_CENTER_MATRIX = 16

NON_ORTHO_FLAGS = NON_ORTHO_CENTER_MATRIX | NON_ORTHO_DIRECT_MATRIX | NON_ORTHO_DIAGONAL_RHS # Bit flags
PRESSURE_FACE_TRANSFORM = False
PRESSURE_TIME_STEP_NORM = False
VELOCITY_CORRECTOR_VERSION = 1 # finite differencing

def gradcheck_LinearSolve(mat, mat_value, rhs, use_BiCG):
    x = PISOtorch_diff.linear_solve_GPU(mat, rhs, False, use_BiCG, return_best_result=not use_BiCG)
    #if not use_BiCG:
    #    x = x - torch.mean(x)
    return x

def test_LinearSolve_Scalar(domain_setup_fn, time_step):
    time_step = torch.tensor([time_step], dtype=DTYPE, device=cpu_device)
    domain, is_non_ortho, prep_fn = domain_setup_fn()
    
    # make divergence free first for stability
    sim = PISOtorch_simulation.Simulation(domain=domain, prep_fn=prep_fn, non_orthogonal=is_non_ortho,
        pressure_non_ortho_steps = 2 if is_non_ortho else 1)
    sim.make_divergence_free()
    
    PISOtorch_diff.SetupAdvectionMatrix(domain, time_step, NON_ORTHO_FLAGS, forPassiveScalar=True, passiveScalarChannel=0)
    PISOtorch_diff.SetupAdvectionScalar(domain, time_step, NON_ORTHO_FLAGS)
    domain.C.value.requires_grad_(True)
    domain.scalarRHS.requires_grad_(True)
    
    inputs = (domain.C, domain.C.value, domain.scalarRHS, True)
    test = torch.autograd.gradcheck(gradcheck_LinearSolve, inputs, eps=GRADCHECK_EPS, atol=GRADCHECK_ATOL, nondet_tol=GRADCHECK_NONDET_TOL)
    LOG.info("gradcheck LinearSolve_Scalar: %s", test)

def test_LinearSolve_Velocity(domain_setup_fn, time_step):
    time_step = torch.tensor([time_step], dtype=DTYPE, device=cpu_device)
    domain, is_non_ortho, prep_fn = domain_setup_fn()
    
    # make divergence free first for stability
    sim = PISOtorch_simulation.Simulation(domain=domain, prep_fn=prep_fn, non_orthogonal=is_non_ortho,
        pressure_return_best_result=True,
        pressure_non_ortho_steps = 2 if is_non_ortho else 1)
    sim.make_divergence_free()
    
    PISOtorch_diff.SetupAdvectionMatrix(domain, time_step, NON_ORTHO_FLAGS)
    PISOtorch_diff.SetupAdvectionVelocity(domain, time_step, NON_ORTHO_FLAGS)
    domain.C.value.requires_grad_(True)
    domain.velocityRHS.requires_grad_(True)
    
    inputs = (domain.C, domain.C.value, domain.velocityRHS, True)
    test = torch.autograd.gradcheck(gradcheck_LinearSolve, inputs, eps=GRADCHECK_EPS, atol=GRADCHECK_ATOL, nondet_tol=GRADCHECK_NONDET_TOL)
    LOG.info("gradcheck LinearSolve_Velocity: %s", test)

def test_LinearSolve_Pressure(domain_setup_fn, time_step):
    time_step = torch.tensor([time_step], dtype=DTYPE, device=cpu_device)
    domain, is_non_ortho, prep_fn = domain_setup_fn()
    
    # make divergence free first for stability
    sim = PISOtorch_simulation.Simulation(domain=domain, prep_fn=prep_fn, non_orthogonal=is_non_ortho,
        pressure_return_best_result=True,
        pressure_non_ortho_steps = 2 if is_non_ortho else 1)
    #sim.differentiable = True
    sim.make_divergence_free(4)
    
    if True:
        # run a few sim steps
        sim._PISO_split_step(10, time_step)
    
    if True:
        # run one advection step, then normal pressure
        stop_handler = PISOtorch_simulation.StopHandler()
        PISOtorch_simulation.append_prep_fn(prep_fn, "POST_PREDICTION", lambda **kwargs: stop_handler.stop())
        sim.stop_fn = stop_handler
        sim._PISO_split_step(1, time_step)
        
        LOG.info("LinearSolve_Pressure normal pressure setup")
        time_step = torch.tensor([1.0], dtype=DTYPE, device=cpu_device)
        PISOtorch_diff.SetupPressureCorrection(domain, time_step, NON_ORTHO_FLAGS, PRESSURE_FACE_TRANSFORM, PRESSURE_TIME_STEP_NORM)
    
    else:
        # like make_divergence_free
        time_step = torch.tensor([1.0], dtype=DTYPE, device=cpu_device)

        LOG.info("LinearSolve_Pressure make_divergence_free variant")
        PISOtorch.CopyVelocityResultFromBlocks(domain)
        domain.setA(torch.ones_like(domain.A))
        domain.setPressureRHS(domain.velocityResult)
        domain.UpdateDomainData()
        
        PISOtorch.SetupPressureMatrix(domain, time_step, NON_ORTHO_FLAGS, PRESSURE_FACE_TRANSFORM)
        PISOtorch.SetupPressureRHSdiv(domain, time_step, NON_ORTHO_FLAGS, PRESSURE_FACE_TRANSFORM, PRESSURE_TIME_STEP_NORM) 
    
    #domain.P.value.requires_grad_(True)
    domain.pressureRHSdiv.requires_grad_(True)
    
    if False:
        PISOtorch_diff.linear_solve_GPU(domain.P, domain.pressureRHSdiv, False, False, return_best_result=False)
        LOG.info("LinearSolve_Pressure without gradcheck done")
    
    else:
        #PISOtorch_diff._LOG_DEBUG=True
        inputs = (domain.P, domain.P.value, domain.pressureRHSdiv, False)
        test = torch.autograd.gradcheck(gradcheck_LinearSolve, inputs, eps=GRADCHECK_EPS if GRADCHECK_EPS<1e-8 else 1e-8, # should not be above solver tolerance?
            atol=GRADCHECK_ATOL, nondet_tol=GRADCHECK_NONDET_TOL)
        LOG.info("gradcheck LinearSolve_Pressure: %s", test)
        #PISOtorch_diff._LOG_DEBUG=False

def gradcheck_SetupAdvectionMatrix(domain, time_step, for_passive_scalar, *block_tensors):
    return PISOtorch_diff.SetupAdvectionMatrix(domain, time_step, NON_ORTHO_FLAGS, for_passive_scalar)

def test_SetupAdvectionMatrix_base(domain_setup_fn, time_step, for_passive_scalar):
    time_step = torch.tensor([time_step], dtype=DTYPE, device=cpu_device)
    domain, is_non_ortho, prep_fn = domain_setup_fn()
    
    if USE_RANDOM_INPUTS:
        LOG.info("test_SetupAdvectionMatrix: using random inputs")
        for block in domain.getBlocks():
            block.setVelocity(torch.randn_like(block.velocity))
            for bound_idx, bound in block.getFixedBoundaries():
                #LOG.info("randomize boundary[%d] velocity", bound_idx)
                bound.setVelocity(torch.randn_like(bound.velocity))
    
    use_block_viscosity = USE_BLOCK_VISCOSITY
    if use_block_viscosity:
        LOG.info("test_SetupAdvectionMatrix: using random varying block viscosity")
        block_viscosity = domain.viscosity.to(cuda_device)
        for block in domain.getBlocks():
            block_viscosity = block_viscosity + block_viscosity*0.5*torch.randn_like(block.pressure) #pressure always exists and has channel=1
            block.setViscosity(block_viscosity)
    
    use_scalar_viscosity = USE_SCALAR_VISCOSITY
    if use_scalar_viscosity:
        LOG.info("test_SetupAdvectionMatrix: using random scalar viscosity")
        scalar_viscosity = domain.viscosity.to(cuda_device)
        scalar_viscosity = scalar_viscosity + scalar_viscosity*0.5*torch.randn([domain.getPassiveScalarChannels()], dtype=DTYPE, device=cuda_device)
        domain.setScalarViscosity(scalar_viscosity)
    
    tensor_filter = ["VELOCITY", "BOUNDARY_VELOCITY", "VISCOSITY"]
    if use_block_viscosity:
        tensor_filter.append("VISCOSITY_BLOCK")
    if use_scalar_viscosity:
        tensor_filter.append("PASSIVE_SCALAR_VISCOSITY")
    domain_dict, tensors = PISOtorch_diff.flatten_domain(domain, tensor_filter)
    
    for t in tensors: t.requires_grad_(True)
    inputs = (domain, time_step, for_passive_scalar, *tensors)
    test = torch.autograd.gradcheck(gradcheck_SetupAdvectionMatrix, inputs, eps=GRADCHECK_EPS, atol=GRADCHECK_ATOL, nondet_tol=GRADCHECK_NONDET_TOL)
    
    domain.Detach() # needed to free the domain after combination with torch autograd
    LOG.info("gradcheck SetupAdvectionMatrix: %s", test)

def test_SetupAdvectionMatrix(domain_setup_fn, time_step):
    test_SetupAdvectionMatrix_base(domain_setup_fn, time_step, False)

def test_SetupAdvectionMatrix_scalar(domain_setup_fn, time_step):
    test_SetupAdvectionMatrix_base(domain_setup_fn, time_step, True)

def gradcheck_SetupAdvectionScalar(domain, time_step, *block_tensors):
    #domain.setScalarResult(block_tensors[-1])
    #domain.UpdateDomainData()
    return PISOtorch_diff.SetupAdvectionScalar(domain, time_step, NON_ORTHO_FLAGS)

def test_SetupAdvectionScalar(domain_setup_fn, time_step):
    time_step = torch.tensor([time_step], dtype=DTYPE, device=cpu_device)
    domain, is_non_ortho, prep_fn = domain_setup_fn()
    PISOtorch.CopyScalarResultFromBlocks(domain)
    
    if USE_RANDOM_INPUTS:
        LOG.info("SetupAdvectionScalar: using random inputs")
        for block in domain.getBlocks():
            block.setPassiveScalar(torch.randn_like(block.passiveScalar))
            for bound_idx, bound in block.getFixedBoundaries():
                bound.setPassiveScalar(torch.randn_like(bound.passiveScalar))
                bound.setVelocity(torch.randn_like(bound.velocity))
        domain.setScalarResult(torch.randn_like(domain.scalarResult))
    
    # tensors = [block.passiveScalar for block in domain.getBlocks()]
    # if is_non_ortho:
        # tensors.append(domain.scalarResult)

    use_scalar_viscosity = USE_SCALAR_VISCOSITY
    if use_scalar_viscosity:
        LOG.info("SetupAdvectionScalar: using random scalar viscosity")
        scalar_viscosity = domain.viscosity.to(cuda_device)
        scalar_viscosity = scalar_viscosity + scalar_viscosity*0.5*torch.randn([domain.getPassiveScalarChannels()], dtype=DTYPE, device=cuda_device)
        domain.setScalarViscosity(scalar_viscosity)
    
    tensor_filter = ["PASSIVE_SCALAR", "BOUNDARY_PASSIVE_SCALAR", "BOUNDARY_VELOCITY", "VISCOSITY"] #, "VISCOSITY"]
    if is_non_ortho:
        tensor_filter.append("PASSIVE_SCALAR_RESULT")
    if use_scalar_viscosity:
        tensor_filter.append("PASSIVE_SCALAR_VISCOSITY")
    domain_dict, tensors = PISOtorch_diff.flatten_domain(domain, tensor_filter)
    
    
    
    for t in tensors: t.requires_grad_(True)
    inputs = (domain, time_step, *tensors)
    test = torch.autograd.gradcheck(gradcheck_SetupAdvectionScalar, inputs, eps=GRADCHECK_EPS, atol=GRADCHECK_ATOL, nondet_tol=GRADCHECK_NONDET_TOL)
    
    domain.Detach() # needed to free the domain after combination with torch autograd
    LOG.info("gradcheck SetupAdvectionScalar: %s", test)

def gradcheck_SetupAdvectionVelocity(domain, time_step, *block_tensors):
    return PISOtorch_diff.SetupAdvectionVelocity(domain, time_step, NON_ORTHO_FLAGS, False)

def test_SetupAdvectionVelocity(domain_setup_fn, time_step):
    time_step = torch.tensor([time_step], dtype=DTYPE, device=cpu_device)
    domain, is_non_ortho, prep_fn = domain_setup_fn()
    PISOtorch.CopyVelocityResultFromBlocks(domain)
    
    tensor_filter = ["VELOCITY", "BOUNDARY_VELOCITY", "VISCOSITY"] #, "VISCOSITY"]
    
    if True:
        LOG.info("test_SetupAdvectionVelocity: using varying boundaries")
        for block in domain.getBlocks():
            for bound_idx, bound in block.getFixedBoundaries():
                #LOG.info("varying boundary[%d] velocity", bound_idx)
                bound.makeVelocityVarying()
    
    if False:
        LOG.info("test_SetupAdvectionVelocity: using random velocity source")
        tensor_filter.append("VELOCITY_SOURCE")
        for block in domain.getBlocks():
            block.setVelocitySource(torch.randn_like(block.velocity))
    
    if USE_RANDOM_INPUTS:
        LOG.info("test_SetupAdvectionVelocity: using random inputs")
        for block in domain.getBlocks():
            block.setVelocity(torch.randn_like(block.velocity))
            for bound_idx, bound in block.getFixedBoundaries():
                #LOG.info("randomize boundary[%d] velocity", bound_idx)
                bound.setVelocity(torch.randn_like(bound.velocity))
        domain.setVelocityResult(torch.randn_like(domain.velocityResult))
    
    # tensors = [block.velocity for block in domain.getBlocks()]
    # if is_non_ortho:
        # tensors.append(domain.velocityResult)
    
    use_block_viscosity = USE_BLOCK_VISCOSITY
    if use_block_viscosity:
        LOG.info("test_SetupAdvectionVelocity: using random varying block viscosity")
        block_viscosity = domain.viscosity.to(cuda_device)
        for block in domain.getBlocks():
            block_viscosity = block_viscosity + block_viscosity*0.5*torch.randn_like(block.pressure) #pressure always exists and has channel=1
            block.setViscosity(block_viscosity)
    
    if is_non_ortho:
        tensor_filter.append("VELOCITY_RESULT")
    if use_block_viscosity:
        tensor_filter.append("VISCOSITY_BLOCK")
    domain_dict, tensors = PISOtorch_diff.flatten_domain(domain, tensor_filter)
    
    for t in tensors: t.requires_grad_(True)
    inputs = (domain, time_step, *tensors)
    test = torch.autograd.gradcheck(gradcheck_SetupAdvectionVelocity, inputs, eps=GRADCHECK_EPS, atol=GRADCHECK_ATOL, nondet_tol=GRADCHECK_NONDET_TOL)
    
    domain.Detach() # needed to free the domain after combination with torch autograd
    LOG.info("gradcheck SetupAdvectionVelocity: %s", test)

def gradcheck_CopyScalarResultToBlocks(domain, *block_tensors):
    PISOtorch_diff.CopyScalarResultToBlocks(domain)
    return domain.scalarResult

def gradcheck_SetupPressureCorrection(domain, time_step, *tensors):
    PISOtorch_diff.SetupPressureCorrection(domain, time_step, NON_ORTHO_FLAGS, PRESSURE_FACE_TRANSFORM, PRESSURE_TIME_STEP_NORM)
    return domain.P.value, domain.pressureRHS, domain.pressureRHSdiv

def test_SetupPressureCorrection(domain_setup_fn, time_step):
    time_step = torch.tensor([time_step], dtype=DTYPE, device=cpu_device)
    domain, is_non_ortho, prep_fn = domain_setup_fn()
    PISOtorch.CopyPressureResultFromBlocks(domain)
    tensor_filter = ["VELOCITY", "VELOCITY_RESULT", "BOUNDARY_VELOCITY", "VISCOSITY", "A", "C"] #"C"
    
    if False:
        LOG.info("test_SetupPressureCorrection: using random velocity source")
        tensor_filter.append("VELOCITY_SOURCE")
        for block in domain.getBlocks():
            block.setVelocitySource(torch.randn_like(block.velocity))

    if USE_RANDOM_INPUTS:
        LOG.info("test_SetupPressureCorrection: using random inputs")
        domain.C.setValue(torch.randn_like(domain.C.value))
        domain.setA(torch.randn_like(domain.A)) # should be the diagonal of C, but should not matter for the gradcheck
        domain.setVelocityResult(torch.randn_like(domain.velocityResult))
        #domain.setPressureRHS(torch.randn_like(domain.pressureRHS))
        domain.setPressureResult(torch.randn_like(domain.pressureResult))
        for block in domain.getBlocks():
            block.setVelocity(torch.randn_like(block.velocity))
    else:
        PISOtorch_diff.CopyVelocityResultFromBlocks(domain)
        for blockIdx in range(0, domain.getNumBlocks()):
            domain.getBlock(blockIdx).CreateVelocity() #velocity.zero_()
            domain.getBlock(blockIdx).CreatePressure() #pressure.zero_()
        domain.UpdateDomainData()
        PISOtorch_diff.SetupAdvectionMatrix(domain, time_step, NON_ORTHO_FLAGS)
        PISOtorch_diff.CopyVelocityResultToBlocks(domain)

    # tensors = [block.velocity for block in domain.getBlocks()]
    # tensors.append(domain.velocityResult)
    # if is_non_ortho:
        # tensors.append(domain.pressureResult)
    
    use_block_viscosity = USE_BLOCK_VISCOSITY
    if use_block_viscosity:
        LOG.info("test_SetupAdvectionVelocity: using random varying block viscosity")
        block_viscosity = domain.viscosity.to(cuda_device)
        for block in domain.getBlocks():
            block_viscosity = block_viscosity + block_viscosity*0.5*torch.randn_like(block.pressure) #pressure always exists and has channel=1
            block.setViscosity(block_viscosity)
    
    if is_non_ortho:
        tensor_filter.append("PRESSURE_RESULT")
    if use_block_viscosity:
        tensor_filter.append("VISCOSITY_BLOCK")
    domain_dict, tensors = PISOtorch_diff.flatten_domain(domain, tensor_filter)
    
    for t in tensors: t.requires_grad_(True)
    inputs = (domain, time_step, *tensors)
    test = torch.autograd.gradcheck(gradcheck_SetupPressureCorrection, inputs, eps=GRADCHECK_EPS, atol=GRADCHECK_ATOL, nondet_tol=GRADCHECK_NONDET_TOL)
    
    domain.Detach() # needed to free the domain after combination with torch autograd
    LOG.info("gradcheck SetupPressureCorrection: %s", test)

def gradcheck_SetupPressureMatrix(domain, time_step, *tensors):
    PISOtorch_diff.SetupPressureMatrix(domain, time_step, NON_ORTHO_FLAGS, PRESSURE_FACE_TRANSFORM)
    return domain.P.value

def test_SetupPressureMatrix(domain_setup_fn, time_step):
    time_step = torch.tensor([time_step], dtype=DTYPE, device=cpu_device)
    domain, is_non_ortho, prep_fn = domain_setup_fn()
    PISOtorch.CopyPressureResultFromBlocks(domain)
    tensor_filter = ["A"]
    
    if USE_RANDOM_INPUTS:
        LOG.info("test_SetupPressureMatrix: using random inputs")
        domain.setA(torch.randn_like(domain.A))
    else:
        PISOtorch_diff.CopyVelocityResultFromBlocks(domain)
        for blockIdx in range(0, domain.getNumBlocks()):
            domain.getBlock(blockIdx).CreateVelocity() #velocity.zero_()
            domain.getBlock(blockIdx).CreatePressure() #pressure.zero_()
        domain.UpdateDomainData()
        PISOtorch_diff.SetupAdvectionMatrix(domain, time_step, NON_ORTHO_FLAGS)
        PISOtorch_diff.CopyVelocityResultToBlocks(domain)
    
    domain_dict, tensors = PISOtorch_diff.flatten_domain(domain, tensor_filter)
    
    for t in tensors: t.requires_grad_(True)
    inputs = (domain, time_step, *tensors)
    test = torch.autograd.gradcheck(gradcheck_SetupPressureMatrix, inputs, eps=GRADCHECK_EPS, atol=GRADCHECK_ATOL, nondet_tol=GRADCHECK_NONDET_TOL)
    
    domain.Detach() # needed to free the domain after combination with torch autograd
    LOG.info("gradcheck SetupPressureMatrix: %s", test)

def gradcheck_SetupPressureRHS(domain, time_step, *tensors):
    PISOtorch_diff.SetupPressureRHS(domain, time_step, NON_ORTHO_FLAGS, PRESSURE_FACE_TRANSFORM, PRESSURE_TIME_STEP_NORM)
    return domain.pressureRHS, domain.pressureRHSdiv

def test_SetupPressureRHS(domain_setup_fn, time_step):
    time_step = torch.tensor([time_step], dtype=DTYPE, device=cpu_device)
    domain, is_non_ortho, prep_fn = domain_setup_fn()
    PISOtorch.CopyPressureResultFromBlocks(domain)
    tensor_filter = ["VELOCITY", "VELOCITY_RESULT", "BOUNDARY_VELOCITY", "VISCOSITY", "C", "A"] #
    
    if False:
        LOG.info("test_SetupPressureCorrection: using random static velocity source")
        tensor_filter.append("VELOCITY_SOURCE")
        for block in domain.getBlocks():
            block.setVelocitySource(torch.randn([1,domain.getSpatialDims()], dtype=domain.getBlock(0).velocity.dtype, device=domain.getDevice()))

    if USE_RANDOM_INPUTS:
        LOG.info("test_SetupPressureRHS: using random inputs")
        domain.C.setValue(torch.randn_like(domain.C.value))
        domain.setA(torch.randn_like(domain.A)) # should be the diagonal of C, but should not matter for the gradcheck
        domain.setVelocityResult(torch.randn_like(domain.velocityResult))
        #domain.setPressureRHS(torch.randn_like(domain.pressureRHS))
        domain.setPressureResult(torch.randn_like(domain.pressureResult))
        for block in domain.getBlocks():
            block.setVelocity(torch.randn_like(block.velocity))
    else:
        PISOtorch_diff.CopyVelocityResultFromBlocks(domain)
        for blockIdx in range(0, domain.getNumBlocks()):
            domain.getBlock(blockIdx).CreateVelocity() #velocity.zero_()
            domain.getBlock(blockIdx).CreatePressure() #pressure.zero_()
        domain.UpdateDomainData()
        PISOtorch_diff.SetupAdvectionMatrix(domain, time_step, NON_ORTHO_FLAGS)
        PISOtorch_diff.CopyVelocityResultToBlocks(domain)

    # tensors = [block.velocity for block in domain.getBlocks()]
    # tensors.append(domain.velocityResult)
    # if is_non_ortho:
        # tensors.append(domain.pressureResult)
    
    use_block_viscosity = USE_BLOCK_VISCOSITY
    if use_block_viscosity:
        LOG.info("test_SetupAdvectionVelocity: using random varying block viscosity")
        block_viscosity = domain.viscosity.to(cuda_device)
        for block in domain.getBlocks():
            block_viscosity = block_viscosity + block_viscosity*0.5*torch.randn_like(block.pressure) #pressure always exists and has channel=1
            block.setViscosity(block_viscosity)
    
    if is_non_ortho:
        tensor_filter.append("PRESSURE_RESULT")
    if use_block_viscosity:
        tensor_filter.append("VISCOSITY_BLOCK")
    domain_dict, tensors = PISOtorch_diff.flatten_domain(domain, tensor_filter)
        
    for t in tensors:
        if t is not None:
            t.requires_grad_(True)
    inputs = (domain, time_step, *tensors)
    test = torch.autograd.gradcheck(gradcheck_SetupPressureRHS, inputs, eps=GRADCHECK_EPS, atol=GRADCHECK_ATOL, nondet_tol=GRADCHECK_NONDET_TOL)
    
    domain.Detach() # needed to free the domain after combination with torch autograd
    LOG.info("gradcheck SetupPressureRHS: %s", test)

def gradcheck_SetupPressureRHSdiv(domain, time_step, *tensors):
    PISOtorch_diff.SetupPressureRHSdiv(domain, time_step, NON_ORTHO_FLAGS, PRESSURE_FACE_TRANSFORM, PRESSURE_TIME_STEP_NORM)
    return domain.pressureRHSdiv

def test_SetupPressureRHSdiv(domain_setup_fn, time_step):
    time_step = torch.tensor([time_step], dtype=DTYPE, device=cpu_device)
    domain, is_non_ortho, prep_fn = domain_setup_fn()
    PISOtorch.CopyPressureResultFromBlocks(domain)

    if USE_RANDOM_INPUTS:
        LOG.info("test_SetupPressureRHSdiv: using random inputs")
        domain.setA(torch.randn_like(domain.A))
        domain.setPressureRHS(torch.randn_like(domain.pressureRHS))
        domain.setPressureResult(torch.randn_like(domain.pressureResult))
    else:
        PISOtorch_diff.CopyVelocityResultFromBlocks(domain)
        for blockIdx in range(0, domain.getNumBlocks()):
            domain.getBlock(blockIdx).CreateVelocity() #velocity.zero_()
            domain.getBlock(blockIdx).CreatePressure() #pressure.zero_()
        domain.UpdateDomainData()
        PISOtorch_diff.SetupAdvectionMatrix(domain, time_step, NON_ORTHO_FLAGS)
        PISOtorch_diff.CopyVelocityResultToBlocks(domain)
        
        domain.setPressureRHS(domain.velocityResult.clone())
        domain.UpdateDomainData()

    # tensors = []
    # tensors.append(domain.pressureRHS)
    # if is_non_ortho:
        # tensors.append(domain.pressureResult)
    
    tensor_filter = ["PRESSURE_RHS", "BOUNDARY_VELOCITY", "A"] #
    if is_non_ortho:
        tensor_filter.append("PRESSURE_RESULT")
        tensor_filter.append("A")
    domain_dict, tensors = PISOtorch_diff.flatten_domain(domain, tensor_filter)

    for t in tensors: t.requires_grad_(True)
    inputs = (domain, time_step, *tensors)
    test = torch.autograd.gradcheck(gradcheck_SetupPressureRHSdiv, inputs, eps=GRADCHECK_EPS, atol=GRADCHECK_ATOL, nondet_tol=GRADCHECK_NONDET_TOL)
    
    domain.Detach() # needed to free the domain after combination with torch autograd
    LOG.info("gradcheck SetupPressureRHSdiv: %s", test)

def gradcheck_CorrectVelocity(domain, time_step, *tensors):
    PISOtorch_diff.CorrectVelocity(domain, time_step, VELOCITY_CORRECTOR_VERSION, PRESSURE_TIME_STEP_NORM)
    return domain.velocityResult

def test_CorrectVelocity(domain_setup_fn, time_step):
    time_step = torch.tensor([time_step], dtype=DTYPE, device=cpu_device)
    domain, is_non_ortho, prep_fn = domain_setup_fn()

    
    
    if USE_RANDOM_INPUTS:
        LOG.info("test_CorrectVelocity: using random inputs")
        #domain.C.setValue(torch.randn_like(domain.C.value))
        domain.setA(torch.randn_like(domain.A)) # should be the diagonal of C, but should not matter for the gradcheck
        #domain.setVelocityResult(torch.randn_like(domain.velocityResult))
        domain.setPressureRHS(torch.randn_like(domain.pressureRHS))
        for block in domain.getBlocks():
        #    block.setVelocity(torch.randn_like(block.velocity))
            block.setPressure(torch.randn_like(block.pressure))
    else:
        PISOtorch_diff.CopyVelocityResultFromBlocks(domain)
        for blockIdx in range(0, domain.getNumBlocks()):
            domain.getBlock(blockIdx).CreateVelocity() #velocity.zero_()
            domain.getBlock(blockIdx).CreatePressure() #pressure.zero_()
        domain.UpdateDomainData()
        PISOtorch_diff.SetupAdvectionMatrix(domain, time_step, NON_ORTHO_FLAGS)
        PISOtorch_diff.CopyVelocityResultToBlocks(domain)

        PISOtorch_diff.SetupPressureCorrection(domain, time_step, NON_ORTHO_FLAGS, PRESSURE_FACE_TRANSFORM, PRESSURE_TIME_STEP_NORM)
        pressureResult = PISOtorch_diff.linear_solve_GPU(domain.P, domain.pressureRHSdiv, use_BiCG=False)
        domain.setPressureResult(pressureResult)
        domain.UpdateDomainData()
        PISOtorch_diff.CopyPressureResultToBlocks(domain)
    
    tensor_filter = ["PRESSURE", "PRESSURE_RHS", "A"] #
    domain_dict, tensors = PISOtorch_diff.flatten_domain(domain, tensor_filter)
    
    for t in tensors: t.requires_grad_(True)
    inputs = (domain, time_step, *tensors)
    
    #PISOtorch_diff.exclude_gradient("CorrectVelocity", "PRESSURE_GRAD", False)
    
    test = torch.autograd.gradcheck(gradcheck_CorrectVelocity, inputs, eps=GRADCHECK_EPS, atol=GRADCHECK_ATOL, nondet_tol=GRADCHECK_NONDET_TOL)
    
    #PISOtorch_diff.clear_excluded_gradients()
    
    domain.Detach() # needed to free the domain after combination with torch autograd
    LOG.info("gradcheck CorrectVelocity: %s", test)


## FUSED TESTS

def gradcheck_dRHS_dA_dvisc(domain, time_step, *tensors):
    for_passive_scalar = False
    PISOtorch_diff.SetupAdvectionMatrix(domain, time_step, NON_ORTHO_FLAGS, for_passive_scalar)
    PISOtorch_diff.SetupPressureRHS(domain, time_step, NON_ORTHO_FLAGS, PRESSURE_FACE_TRANSFORM, PRESSURE_TIME_STEP_NORM)
    return domain.pressureRHS, domain.pressureRHSdiv

def test_dRHS_dA_dvisc(domain_setup_fn, time_step):
    time_step = torch.tensor([time_step], dtype=DTYPE, device=cpu_device)
    domain, is_non_ortho, prep_fn = domain_setup_fn()
    PISOtorch.CopyPressureResultFromBlocks(domain)

    if USE_RANDOM_INPUTS:
        LOG.info("test_dRHS_dA_dvisc: using random inputs")
        for block in domain.getBlocks():
            block.setVelocity(torch.randn_like(block.velocity))
            for bound_idx, bound in block.getFixedBoundaries():
                #LOG.info("randomize boundary[%d] velocity", bound_idx)
                bound.setVelocity(torch.randn_like(bound.velocity))
        
        domain.setVelocityResult(torch.randn_like(domain.velocityResult))
        domain.setPressureResult(torch.randn_like(domain.pressureResult))
    else:
        raise NotImplementedError
    
    #tensor_filter = ["VELOCITY", "BOUNDARY_VELOCITY", "VISCOSITY"]
    #tensor_filter = ["VELOCITY"]
    tensor_filter = ["VISCOSITY"]
    domain_dict, tensors = PISOtorch_diff.flatten_domain(domain, tensor_filter)
    
    for t in tensors:
        if t is not None:
            t.requires_grad_(True)
    inputs = (domain, time_step, *tensors)
    test = torch.autograd.gradcheck(gradcheck_dRHS_dA_dvisc, inputs, eps=GRADCHECK_EPS, atol=GRADCHECK_ATOL, nondet_tol=GRADCHECK_NONDET_TOL)
    
    domain.Detach() # needed to free the domain after combination with torch autograd
    LOG.info("gradcheck dRHS_dA_dvisc: %s", test)


def gradcheck_advect_static(domain, time_step, *block_tensors):
    raise NotImplementedError

def gradcheck_div_free(domain, time_step, *velocities):
    raise NotImplementedError



def get_tests():
    import tests.test_setups as test_setups
    
    setups = test_setups.setups_simple_combined
    #setups = test_setups.setups_simple_2D_combined
    #setups = {_: test_setups.setups_simple_combined[_] for _ in ["1Block2D_T-n_v_velX+const_periodic"]}
    #setups.update({_: test_setups.setups_simple_combined[_] for _ in ["1Block2D_v_velY-blob_periodic"]})
    # setups.update({_: test_setups.setups_simple_combined[_] for _ in ["1Block2D_v_velX+blob_closed"]})
    # #setups.update({_: test_setups.setups_simple_combined[_] for _ in ["1Block2D_T-u_v_velY-blob_periodic"]})
    # setups.update({_: test_setups.setups_simple_combined[_] for _ in ["1Block2D_T-n_v_velY-blob_periodic"]})
    # #setups = {_: test_setups.setups_nonortho[_] for _ in ["1Block2D_T-n-rot30_v_velY-blob_periodic"]}
    # setups.update({_: test_setups.setups_nonortho[_] for _ in ["1Block2D_T-n-rot30_v_velX+blob_closed"]})
    # setups.update({_: test_setups.setups_nonortho[_] for _ in ["1Block2D_T-n-rot30_v_velY-blob_periodic"]})
    # #setups = {_: test_setups.setups_nonortho[_] for _ in ["4Torus2D_T_v_velIn_inOut"]}
    # #setups = {_: test_setups.setups_nonortho[_] for _ in ["1Block3D_T-e-rot30_v_velZ+blob_periodic"]}
    # #setups = {_: test_setups.setups_nonortho[_] for _ in ["1Block2D_T-n-rot30_v_velY-blob_periodic", "1Block2D_T-n-rot30_v_velX+blob_closed"]}
    
    tests = {
        "SetupAdvectionMatrix": (test_SetupAdvectionMatrix, setups),
        "SetupAdvectionMatrixScalar": (test_SetupAdvectionMatrix_scalar, setups),
        "SetupAdvectionScalar": (test_SetupAdvectionScalar, setups),
        "SetupAdvectionVelocity": (test_SetupAdvectionVelocity, setups),
        "SetupPressureCorrection": (test_SetupPressureCorrection, setups),
        "SetupPressureMatrix": (test_SetupPressureMatrix, setups),
        "SetupPressureRHS": (test_SetupPressureRHS, setups),
        "SetupPressureRHSdiv": (test_SetupPressureRHSdiv, setups),
        "CorrectVelocity": (test_CorrectVelocity, setups),
        
        # "dRHS_dA_dvisc": (test_dRHS_dA_dvisc, setups),
        
        # "LinearSolve_Velocity": (test_LinearSolve_Velocity, setups),
        # "LinearSolve_Pressure": (test_LinearSolve_Pressure, setups),
        # "LinearSolve_Scalar": (test_LinearSolve_Scalar, {
           # "1Block2D_velX+const_periodic": lambda: test_setups.make1BlockSetup2D(res, res, [1,0], False, False, 0, False, False, DTYPE),
           # "1Block2D_velY-blob_periodic": lambda: test_setups.make1BlockSetup2D(res, res, [0,-1], False, False, 0, False, True, DTYPE),
           # "1Block2D_velX+blob_closed": lambda: test_setups.make1BlockSetup2D(res, res, [1,0], False, True, 0, False, True, DTYPE),
           # "2Block2D_velX+constY+_periodic": lambda: test_setups.make2BlockSetup2D(res, res, res, closed_bounds=False, vel_blob=False, dtype=DTYPE),
           # "2Block2D_velX+constY+_closed": lambda: test_setups.make2BlockSetup2D(res, res, res, closed_bounds=True, vel_blob=False, dtype=DTYPE),
        # }),
        # "LinearSolve_Velocity": (test_LinearSolve_Velocity, {
           # "1Block2D_velX+const_periodic": lambda: test_setups.make1BlockSetup2D(res, res, [1,0], False, False, 0, False, False, DTYPE),
           # "1Block2D_velY-blob_periodic": lambda: test_setups.make1BlockSetup2D(res, res, [0,-1], False, False, 0, False, True, DTYPE),
           # "1Block2D_velX+blob_closed": lambda: test_setups.make1BlockSetup2D(res, res, [1,0], False, True, 0, False, True, DTYPE),
           # "2Block2D_velX+constY+_periodic": lambda: test_setups.make2BlockSetup2D(res, res, res, closed_bounds=False, vel_blob=False, dtype=DTYPE),
           # #"2Block2D_velX+constY+_closed": lambda: test_setups.make2BlockSetup2D(res, res, res, closed_bounds=True, vel_blob=False, dtype=DTYPE), # convergence issues
        # }),
        # "LinearSolve_Pressure": (test_LinearSolve_Pressure, {
            # "1Block2D_velX+const_periodic": lambda: test_setups.make1BlockSetup2D(res, res, [1,0], False, False, 0, False, False, DTYPE),
            # "1Block2D_velY-blob_periodic": lambda: test_setups.make1BlockSetup2D(res, res, [0,-1], False, False, 0, False, True, DTYPE),
            # "1Block2D_velX+blob_closed": lambda: test_setups.make1BlockSetup2D(res, res, [1,0], False, True, 0, False, True, DTYPE),
            # "2Block2D_velX+constY+_periodic": lambda: test_setups.make2BlockSetup2D(res, res, res, closed_bounds=False, vel_blob=False, dtype=DTYPE),
            # "2Block2D_velX+constY+_closed": lambda: test_setups.make2BlockSetup2D(res, res, res, closed_bounds=True, vel_blob=False, dtype=DTYPE),
        # }),
    }

    return tests

def get_test_params(domain_setup_fn):
    #domain, is_non_ortho, prep_fn = domain_setup_fn()
    return {"domain_setup_fn": domain_setup_fn, "time_step": 0.5}
