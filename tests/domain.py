# test domain copy and IO
import os
import torch
import PISOtorch
import PISOtorch_simulation
import lib.util.domain_io as domain_io
from lib.util.logging import get_logger
_LOG = get_logger("DomainTest")


assert torch.cuda.is_available()
cuda_device = torch.device("cuda")
cpu_device = torch.device("cpu")

DTYPE = torch.float64

def cmp_tensors(t1, t2, with_data=True, tensor_object_equal=False, name="tensor"):

    if (t1 is None) != (t2 is None):
        raise ValueError(name+": only 1 tensor exists: T1 %s, T2, %s"%(t1 is not None, t2 is not None))
    elif (t1 is None) and (t2 is None):
        return # both are not set, thus equal

    if tensor_object_equal:
        if not t1 is t2: #t1.data_ptr()==t2.data_ptr():
            raise ValueError(name+": tensor object not equal")
    else:
        if t1 is t2: #t1.data_ptr()==t2.data_ptr():
            raise ValueError(name+": tensor object equal (should not be equal).")
    
    if not t1.shape==t2.shape:
        raise ValueError(name+": tensor shape not equal")
    
    if with_data:
        if not torch.allclose(t1,t2):
            raise ValueError(name+": tensor data not close")
    
def cmp_boundaries(d1, bound1, d2, bound2, with_data=True, tensor_object_equal=False, name="bound"):
    if not bound1.type==bound2.type:
        raise ValueError(name+": type not equal: %s, %s"%(bound1.type, bound2.type))


    if bound1.type==PISOtorch.DIRICHLET or bound1.type==PISOtorch.DIRICHLET_VARYING or bound1.type==PISOtorch.NEUMANN:
        raise TypeError(name+": %s boundaries are deprecated"%(bound1.type))
    elif bound1.type==PISOtorch.FIXED:
        name += "FIXED"
        if not bound1.velocityType==bound2.velocityType:
            raise ValueError(name+": velocityType not equal: %s, %s"%(bound1.velocityType, bound2.velocityType))
        cmp_tensors(bound1.velocity, bound2.velocity, with_data=with_data, tensor_object_equal=tensor_object_equal, name=name+".velocity")
        
        if not bound1.getPassiveScalarChannels()==bound2.getPassiveScalarChannels():
            raise ValueError("%s passive scalar channels not equal: %d, %d"%(name, bound1.getPassiveScalarChannels(), bound2.getPassiveScalarChannels()))
        if bound1.hasPassiveScalar():
            if (not len(bound1.passiveScalarTypes)==len(bound2.passiveScalarTypes)) or any(not t1==t2 for t1, t2 in zip(bound1.passiveScalarTypes, bound2.passiveScalarTypes)):
                raise ValueError(name+": passiveScalarType not equal: %s, %s"%(bound1.passiveScalarTypes, bound2.passiveScalarTypes))
            cmp_tensors(bound1.passiveScalar, bound2.passiveScalar, with_data=with_data, tensor_object_equal=tensor_object_equal, name=name+".passiveScalar")
        
        cmp_tensors(bound1.transform, bound2.transform, with_data=with_data, tensor_object_equal=tensor_object_equal, name=name+".transform")
    elif bound1.type==PISOtorch.CONNECTED:
        name += "CONNECTED"
        if not d1.getBlocks().index(bound1.getConnectedBlock())==d2.getBlocks().index(bound2.getConnectedBlock()):
            raise ValueError(name+": connection index not equal: %d, %d"%(bound1.getConnectedBlock(), bound2.getConnectedBlock()))
        for a_idx, (a1, a2) in enumerate(zip(bound1.axes, bound2.axes)):
            if not a1==a2:
                raise ValueError(name+": axes[%d] not equal: %d, %d"%(a_idx, a1, a2))
    elif bound1.type==PISOtorch.PERIODIC:
        name += "PERIODIC"
        pass
    else:
        raise TypeError(name+": unknown boundary type.")

def cmp_blocks(d1, b1, d2, b2, with_name=True, with_data=True, tensor_object_equal=False, name="block"):
    if with_name and not b1.name==b2.name:
        raise ValueError(name+": names not equal: '%s', '%s'"%(b1.name, b2.name))
    
    # spatial dims, dtype, device must match via domain
    
    if not b1.hasViscosity()==b2.hasViscosity():
        raise ValueError("%s use of block viscosity not equal: %s, %s"%(name, b1.hasViscosity(), b2.hasViscosity()))
    if b1.hasViscosity():
        cmp_tensors(b1.viscosity, b2.viscosity, with_data=with_data, tensor_object_equal=tensor_object_equal, name=name+".viscosity")
    
    cmp_tensors(b1.velocity, b2.velocity, with_data=with_data, tensor_object_equal=tensor_object_equal, name=name+".velocity")
    cmp_tensors(b1.pressure, b2.pressure, with_data=with_data, tensor_object_equal=tensor_object_equal, name=name+".pressure")
    if not b1.getPassiveScalarChannels()==b2.getPassiveScalarChannels():
        raise ValueError("%s passive scalar channels not equal: %d, %d"%(name, b1.getPassiveScalarChannels(), b2.getPassiveScalarChannels()))
    if b1.hasPassiveScalar():
        cmp_tensors(b1.passiveScalar, b2.passiveScalar, with_data=with_data, tensor_object_equal=tensor_object_equal, name=name+".passiveScalar")
    
    if not b1.hasVelocitySource()==b2.hasVelocitySource():
        raise ValueError("%s use velocity source not equal: %s, %s"%(name, b1.hasVelocitySource(), b2.hasVelocitySource()))
    if b1.hasVelocitySource():
        cmp_tensors(b1.velocitySource, b2.velocitySource, with_data=with_data, tensor_object_equal=tensor_object_equal, name=name+".velocitySource")
    
    cmp_tensors(b1.vertexCoordinates, b2.vertexCoordinates, with_data=with_data, tensor_object_equal=tensor_object_equal, name=name+".vertexCoordinates")
    cmp_tensors(b1.transform, b2.transform, with_data=with_data, tensor_object_equal=not b1.hasVertexCoordinates(), name=name+".transform")
    cmp_tensors(b1.faceTransform, b2.faceTransform, with_data=with_data, tensor_object_equal=not b1.hasVertexCoordinates(), name=name+".faceTransform")

    #bound_names = ["-x","+x","-y","+y","-z","+z"]
    for bound_idx in range(b1.getSpatialDims()*2):
        #bound_name = bound_names[bound_idx]
        bound1 = b1.getBoundary(bound_idx)
        bound2 = b2.getBoundary(bound_idx)
        cmp_boundaries(d1, bound1, d2, bound2, with_data=with_data, tensor_object_equal=tensor_object_equal, name=name+".bound[%d]"%(bound_idx, ))

def cmp_domains(d1, d2, with_name=True, with_data=True, tensor_object_equal=False):
    # compares to domains for equality
    # considers blocks, boundaries and primary tensors (velocity, pressure, passiveScalar).
    # does not check matrices, RHS, Result, and Gradient tensors
    # with_name: compare also the names of blocks and the domain
    # with data: compare also the data/contents of tensors. otherwise only the size is checked.
    # tensor_object_equal: whether tensor objects should be equal. tensor objects/references should be equal after copy, but not after clone or save/load.
    try:
        if with_name and not d1.name==d2.name:
            raise ValueError("domain names not equal: '%s', '%s'"%(d1.name, d2.name))
        
        if not d1.getSpatialDims()==d2.getSpatialDims():
            raise ValueError("domain spatial dimensions not equal: %d, %d"%(d1.getSpatialDims(), d2.getSpatialDims()))
        
        cmp_tensors(d1.viscosity, d2.viscosity, with_data=with_data, tensor_object_equal=tensor_object_equal, name="domain.viscosity")
        
        if not d1.hasPassiveScalarViscosity()==d2.hasPassiveScalarViscosity():
            raise ValueError("domain use of passive scalar viscosity not equal: %s, %s"%(d1.hasPassiveScalarViscosity(), d2.hasPassiveScalarViscosity()))
        if d1.hasPassiveScalarViscosity():
            cmp_tensors(d1.passiveScalarViscosity, d2.passiveScalarViscosity, with_data=with_data, tensor_object_equal=tensor_object_equal, name="domain.passiveScalarViscosity")
        
        if not d1.getPassiveScalarChannels()==d2.getPassiveScalarChannels():
            raise ValueError("domain passive channels not equal: %d, %d"%(d1.getPassiveScalarChannels(), d2.getPassiveScalarChannels()))
        
        if not d1.getNumBlocks()==d2.getNumBlocks():
            raise ValueError("domain number of blocks not equal: %d, %d"%(d1.getNumBlocks(), d2.getNumBlocks()))
        
        for idx, (b1, b2) in enumerate(zip(d1.getBlocks(), d2.getBlocks())):
            cmp_blocks(d1, b1, d2, b2, with_name=with_name, with_data=with_data, tensor_object_equal=tensor_object_equal, name="block[%d]"%(idx,))
    
    except Exception as e:
        #_LOG.info("Domains not equal:\n%s", str(e))
        _LOG.exception("Domains not equal:")
        return False
    else:
        return True

def make_domain():
    pass

    # variations:
    # 2D and 3D
    # with vertexCoordiantes, with only transformations, without any
    # multiple blocks
    # periodic and connected boundaries
    # fixed boundary with static and varying vel/dens (inherit transformations)
    # some varying data (inflow profile, after a few simulation steps)
    
    # TODO:
    # without scalar channels
    # with multiple scalar channels
    # with Neumann scalar boundaries
    # with passive scalar global viscosity
    # with block viscosity

    # candidates:
    # 8 Block Channel Flow: 2D, 3D, w/ & w/o transformations, periodic z
    # Torus Vortex Street: non-ortho transformations, periodic z

    from vortex_street_sample import make8BlockChannelFlowSetup

    res_scale = 8
    use_3D = True
    closed_bounds = True
    closed_z = False
    scalar_channels = 2
    domain, prep_fn, layout = make8BlockChannelFlowSetup(x=16*res_scale, y=3*res_scale, z=3*res_scale if use_3D else None, x_in=1*res_scale, y_in=1*res_scale, x_pos=2*res_scale,
                                                         in_vel=4.0 if closed_bounds else 1, in_var=0.4, closed_y=closed_bounds, closed_z=closed_z, viscosity=1e-3, scale=1/res_scale, dtype=DTYPE)
    
    if domain.hasPassiveScalar():
        domain.setScalarViscosity((torch.rand(domain.getPassiveScalarChannels(), dtype=DTYPE, device=cuda_device)+0.1).contiguous())
    
    for block in domain.getBlocks():
        block_size = block.getSizes()
        block.setViscosity((torch.rand(1,1,block_size.z, block_size.y, block_size.x, dtype=DTYPE, device=cuda_device)+0.1).contiguous())
    
    # advect 1 adaptive step for interesting data
    sim = PISOtorch_simulation.Simulation(domain=domain, block_layout=layout, prep_fn=prep_fn,
            substeps="ADAPTIVE", time_step=0.1, corrector_steps=2, pressure_tol=1e-5,
            advect_non_ortho_steps=1, pressure_non_ortho_steps=1, pressure_return_best_result=True,
            velocity_corrector="FD", non_orthogonal=False, log_images=False)
    
    sim.make_divergence_free()
    
    sim.run(1)

    return domain, False, prep_fn

def test_copy(domain_setup_fn):
    domain1, is_non_ortho, prep_fn = domain_setup_fn()
    domain2 = domain1.Copy()

    if not cmp_domains(domain1, domain2, 
                       with_name=False, # copy() appends '_copy' to names
                       with_data=True,
                       tensor_object_equal=True): # copy() should retain tensor references
        raise RuntimeError("Domain Copy test failed.")
    else:
        _LOG.info("Domain Copy test OK")

def test_clone(domain_setup_fn):
    #raise NotImplementedError("TODO: Implement Clone() for domain.")
    domain1, is_non_ortho, prep_fn = domain_setup_fn()
    domain2 = domain1.Clone()

    if not cmp_domains(domain1, domain2, 
                       with_name=False, # Clone() appends '_clone' to names
                       with_data=True,
                       tensor_object_equal=False): # Clone() should create new tensors
        raise RuntimeError("Domain Clone test failed.")
    else:
        _LOG.info("Domain Clone test OK")

def test_save_load(domain_setup_fn, path="./temp"):
    domain1, is_non_ortho, prep_fn = domain_setup_fn()
    os.makedirs(path, exist_ok=True)
    domain_path = os.path.join(path, "save-load_test-domain")
    if os.path.exists(domain_path+".npz") or os.path.exists(domain_path+".json"):
        raise IOError("domain files already exist.")
    try:
        domain_io.save_domain(domain1, os.path.join(path, "save-load_test-domain"))
    except:
        raise RuntimeError("Domain save/load test failed: could not save domain.")

    try:
        domain2 = domain_io.load_domain(domain_path, dtype=DTYPE, device=cuda_device)
    except:
        os.remove(domain_path+".npz")
        os.remove(domain_path+".json")
        raise RuntimeError("Domain save/load test failed: could not load domain.")
    else:
        os.remove(domain_path+".npz")
        os.remove(domain_path+".json")

    if not cmp_domains(domain1, domain2, 
                       with_name=True,
                       with_data=True,
                       tensor_object_equal=False):
        raise RuntimeError("Domain save/load test failed: domains not equal.")
    else:
        _LOG.info("Domain save/load test OK")


def get_tests():
    #import tests.test_setups as test_setups
    setups = {"3D8BlockChannelFlow": make_domain}
    tests = {
        "Copy": (test_copy, setups),
        "Clone": (test_clone, setups),
        "SaveLoad": (test_save_load, setups),
    }
    return tests

def get_test_params(domain_setup_fn):
    #PISOtorch_sim.set_backend(diff=False)
    return {"domain_setup_fn":domain_setup_fn}