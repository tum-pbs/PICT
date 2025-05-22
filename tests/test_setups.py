
from lib.util.logging import get_logger

import numpy as np
import torch
import PISOtorch
import PISOtorch_simulation


assert torch.cuda.is_available()
cuda_device = torch.device("cuda:0")
cpu_device = torch.device("cpu")

import lib.data.shapes as shapes

LOG = get_logger("test_setups")

def scale_transform(transform, scaling):
    dims = len(scaling)
    M_scale = [scaling[row] if row==col else 1 for row in range(dims) for col in range(dims)]
    M_inv_scale = [1/s for s in M_scale]
    J_scale = np.prod(scaling)
    scalings = M_scale + M_inv_scale + [J_scale]
    #LOG.info("transform scalings: %s", scalings)
    return transform * torch.tensor(scalings, dtype=transform.dtype, device=transform.device)




def make1BlockSetup(x:int, y:int, z:int=0, vel=[1,0], closed_bounds=False, viscosity=0.0, 
        domain_scale=None, transform_strength=None, rot_distortion_max_angle=None, vel_blob=True,
        bound_vel=False, bound_scalar=False,
        dtype=torch.float32) -> PISOtorch.Domain:
    if z>0:
        dims = 3
        res1 = [z,y,x]
    else:
        dims = 2
        z = 0
        res1 = [y,x]
    
    assert len(vel)==dims
    assert domain_scale is None or len(domain_scale)==dims
    assert transform_strength is None or len(transform_strength)==dims
    
    is_non_ortho = False
    prep_fn = {}
    
    LOG.info("Domain")
    viscosity = torch.tensor([viscosity], dtype=dtype, device=cpu_device)
    domain = PISOtorch.Domain(dims, viscosity, "Domain1Block2D", dtype=dtype, device=cuda_device)

    data = shapes.get_grid_normal_dist(res1, [0]*dims,[0.5]*dims)
    data = torch.reshape(data, [1,1]+res1).to(dtype).to(cuda_device).contiguous()
    
    velocity = torch.tensor(vel, dtype=dtype, device=cuda_device).reshape([1,dims]+[1]*dims).repeat(1,1,*res1).contiguous()
    if vel_blob:
        velocity = velocity * data
    
    
    grid = None
    if domain_scale is not None or transform_strength is not None or rot_distortion_max_angle is not None:
        if domain_scale is None:
            domain_scale = [1]*dims
        if transform_strength is None:
            transform_strength = [1]*dims
        
        corner_upper = (x*domain_scale[0], y*domain_scale[1])
        grid = shapes.make_wall_refined_ortho_grid(x,y, corner_upper=corner_upper,
            wall_refinement=["-x","+x","-y","+y"], base=transform_strength[:2], dtype=dtype)
        
        if rot_distortion_max_angle is not None:
            max_r = min(corner_upper)*0.5*0.9
            #distance_scaling = lambda dist: max(0, 1 - dist/max_r)
            distance_scaling = shapes.make_rotation_distance_scaling_fn_sine_half(rot_distortion_max_angle, max_r)
            grid = shapes.rotate_grid(grid, angle=rot_distortion_max_angle, distance_scaling=distance_scaling)
            is_non_ortho = True
        
        if dims==3:
            grid = shapes.extrude_grid_z(grid, z, end_z=z*domain_scale[2], weights_z="EXP", exp_base=transform_strength[2])
        
        grid = grid.to(cuda_device).contiguous()
    
    LOG.info("CreateBlock")
    block = domain.CreateBlock(velocity=velocity, passiveScalar=data, vertexCoordinates=grid, name="Block")
    

    if closed_bounds:
        LOG.info("CloseAllBoundaries")
        block.CloseAllBoundaries()
        if bound_vel:
            bound_res = [_ for _ in res1]
            bound_res[-1] = 1
            block.getBoundary("+x").setVelocity(torch.tensor([0,1], dtype=dtype, device=cuda_device).reshape([1,dims] + [1]*dims).repeat(1,1,*bound_res).contiguous)
        if bound_scalar:
            bound_res = [_ for _ in res1]
            bound_res[-1] = 1
            block.getBoundary("+x").setPassiveScalar(torch.tensor([1], dtype=dtype, device=cuda_device).reshape([1,1] + [1]*dims).repeat(1,1,*bound_res).contiguous)

    
    LOG.info("PrepareSolve")
    domain.PrepareSolve()
    
    LOG.info("make1BlockSetup DONE")
    return domain, is_non_ortho, prep_fn

def make2BlockSetup2D(x1:int, y:int, x2:int, vel1=[1,0], vel2=[0,1], vel_blob1=True, vel_blob2=True, closed_bounds=False, viscosity=0.0, dtype=torch.float32) -> PISOtorch.Domain:
    dims = 2
    res1 = [y,x1]
    res2 = [y,x2]
    
    is_non_ortho = False
    prep_fn = {}
    
    viscosity = torch.tensor([viscosity], dtype=dtype, device=cpu_device)
    domain = PISOtorch.Domain(dims, viscosity, name="Domain2Block2D", dtype=dtype, device=cuda_device)
    
    data = shapes.get_grid_normal_dist(res1, [0]*dims,[0.5]*dims)
    data = torch.reshape(data, [1,1]+res1).to(dtype).to(cuda_device)
    velocity = torch.tensor(vel1, dtype=dtype, device=cuda_device).reshape([1,dims]+[1]*dims).repeat(1,1,*res1).contiguous()
    if vel_blob1:
        velocity = velocity * data
    block = domain.CreateBlock(velocity=velocity, passiveScalar=data, name="Block 1")

    data = None # torch.zeros(size=[1,1]+res2, dtype=dtype, device=cuda_device)
    velocity = torch.tensor(vel2, dtype=dtype, device=cuda_device).reshape([1,dims]+[1]*dims).repeat(1,1,*res2).contiguous()
    if vel_blob2:
        velocity = velocity * 0
    block2 = domain.CreateBlock(velocity=velocity, passiveScalar=data, name="Block 2")

    if closed_bounds:
        block.CloseAllBoundaries()
        block2.CloseAllBoundaries()
    else:
        block.ConnectBlock("-x", block2, "+x", "-y")
    
    block.ConnectBlock("+x", block2, "-x", "-y")

    domain.PrepareSolve()

    return domain, is_non_ortho, prep_fn

def makeSplitBlockSetup2D(x:int, y:int, dInv=False, moebius=False, closed_bounds=False, viscosity=0.0, vel_blob=True, dtype=torch.float32) -> PISOtorch.Domain:
    dims = 2
    res = [y,x]
    x1 = x//2
    x2 = x - x1
    y1 = y//2
    y2 = y - y1
    #res1 = [y,x//2]
    #res2 = [y,x - res1[1]]
    
    is_non_ortho = False
    prep_fn = {}
    
    r1 = [y1,x1]
    r2 = [y1,x2]
    r3 = [y2,x1]
    r4 = [y2,x2]

    make_blob = lambda res: torch.reshape(shapes.get_grid_normal_dist(res, [0]*dims,[0.5]*dims), [1,1]+res).to(dtype).to(cuda_device)
    d1 = make_blob(r1) #torch.zeros(size=[1,1]+r1, dtype=dtype, device=cuda_device)
    d2 = make_blob(r2)
    d3 = make_blob(r3)
    d4 = make_blob(r4)

    v1 = None #torch.zeros(size=[1,dims]+r1, dtype=dtype, device=cuda_device)
    v2 = torch.tensor([0,1], dtype=dtype, device=cuda_device).reshape([1,dims]+[1]*dims).repeat(1,1,*r2).contiguous()
    v3 = torch.tensor([0,-1], dtype=dtype, device=cuda_device).reshape([1,dims]+[1]*dims).repeat(1,1,*r3).contiguous()
    v4 = None #torch.zeros(size=[1,dims]+r4, dtype=dtype, device=cuda_device)

    if vel_blob:
        v2 = v2*d2
        v3 = v3*d3

    data = torch.cat([torch.cat([d1,d2], axis=-1), torch.cat([d3,d4], axis=-1)], axis=-2)
    velocity = torch.cat([torch.cat([v1,v2], axis=-1), torch.cat([v3,v4], axis=-1)], axis=-2)
    #pressure =  torch.zeros(size=[1,1]+res, dtype=dtype, device=cuda_device)

    # need to make split contiguous
    data = [_.contiguous() for _ in torch.split(data, (y1, y2), dim=-2)]
    velocity = [_.contiguous() for _ in torch.split(velocity, (y1, y2), dim=-2)]
    #pressure = [torch.clone(_) for _ in torch.split(pressure, (y1, y2), dim=-2)]

    viscosity = torch.tensor([viscosity], dtype=dtype, device=cpu_device)
    domain = PISOtorch.Domain(dims, viscosity, name="Domain2Block2D", dtype=dtype, device=cuda_device)
    
    block = domain.CreateBlock(velocity=velocity[0], passiveScalar=data[0], name="Block 1")

    block2 = domain.CreateBlock(velocity=velocity[1], passiveScalar=data[1], name="Block 2")

    if closed_bounds:
        block.CloseAllBoundaries()
        block2.CloseAllBoundaries()
    
    block.ConnectBlock("-y", block2, "+y", "-x")
    block.ConnectBlock("+y", block2, "-y", "-x")

    domain.PrepareSolve()

    return domain, is_non_ortho, prep_fn


def make4BlockTorus_VortexStreet(x, r1=0.5, r2=1, rot_vel=0, z:int=0, z_size=1, z_closed=False, obs_vel_z=0, vel_in=0,
        rot_distortion_max_angle=None, grid_z_sine_scale=None, viscosity=0.01, scaling=None, dtype=torch.float32):
    # r_obstacle: radius of the round obstacle
    # r_torus: additional radius of the torus grid around the obstacle
    # res_torus: 1/4 of the angular resolution of the torus around the obstacle
    # r_quad: additional radius/size of the square around the torus
    # res_quad: 
    # r_border: width of the border layer
    # res_in_mul/res_out_mul: resolution multiplier for the inflow and outflow grids, based on res_torus
    dims = 3 if z>0 else 2
    
    is_non_ortho = True
    prep_fn = {}
    
    viscosity = torch.tensor([viscosity], dtype=dtype, device=cpu_device)
    domain = PISOtorch.Domain(dims, viscosity, name="DomainTorusParts", dtype=dtype, device=cuda_device)
    
    # coordinates are centered on the obstacle
    grid_rotation = rot_distortion_max_angle is not None
    angle = rot_distortion_max_angle
    grid_z_sine = grid_z_sine_scale is not None
    z_scale = grid_z_sine_scale
    
    # make the torus grids
    # torus generation: 
    # - angle starts at x-axis and goes counterclockwise
    # - grid: x goes along angle, y goes inner to outer
    # - coords: y is up
    # - return: torch.tensor with shape NCHW
    torus_tr_coords = shapes.make_torus_2D(x, r1=r1, r2=r2, start_angle=90, angle=-90, dtype=dtype) # y up, x right
    torus_br_coords = shapes.make_torus_2D(x, r1=r1, r2=r2, start_angle=0, angle=-90, dtype=dtype) # y right, x down
    torus_bl_coords = shapes.make_torus_2D(x, r1=r1, r2=r2, start_angle=-90, angle=-90, dtype=dtype) # y down, x left
    torus_tl_coords = shapes.make_torus_2D(x, r1=r1, r2=r2, start_angle=-180, angle=-90, dtype=dtype) # y left, x up
    
    
    torus_coords = [torus_tr_coords, torus_br_coords, torus_bl_coords, torus_tl_coords]
    y = torus_tr_coords.size(-2)-1
    
    if grid_rotation:
        distance_scaling = shapes.make_rotation_distance_scaling_fn_sine(angle, r1*1.1, r2*0.9)
        torus_coords = [shapes.rotate_grid(grid, angle=angle, distance_scaling=distance_scaling, center=(0,0)) for grid in torus_coords]
    
    if scaling is not None:
        if not isinstance(scaling, (list, tuple)):
            scaling = [scaling]*dims
        if not len(scaling)==dims:
            raise ValueError("Invalid scaling")
        LOG.info("Torus vortex street scaling: %s", scaling)
        
        scale_factor = torch.tensor(scaling, dtype=dtype).reshape([1,dims]+[1]*dims)
        
        torus_coords = [_*scale_factor for _ in torus_coords]
    
    if dims==3:
        torus_coords = [shapes.extrude_grid_z(grid, z, end_z=z_size, weights_z=None, exp_base=1.05) for grid in torus_coords]
        if grid_z_sine:
            if z_closed:
                raise RuntimeError("z distortion requires periodic z boundaries")
            x_max = torch.max(torch.tensor([torch.max(grid[:,0]) for grid in torus_coords]))
            x_min = torch.min(torch.tensor([torch.min(grid[:,0]) for grid in torus_coords]))
            y_max = torch.max(torch.tensor([torch.max(grid[:,1]) for grid in torus_coords]))
            y_min = torch.min(torch.tensor([torch.min(grid[:,1]) for grid in torus_coords]))
            x_size = x_max - x_min
            y_size = y_max - y_min
            x_norm = (2*np.pi)/(x_size)
            y_norm = (2*np.pi)/(y_size)
            
            for grid in torus_coords:
                x_pos, y_pos, z_pos = torch.split(grid, 1, dim=1)
                z_offset_x = (torch.cos((x_pos - x_min)*x_norm)+1)*0.5*z_scale
                z_offset_y = (torch.cos((y_pos - y_min)*y_norm)+1)*0.5*z_scale
                
                grid[:,-1:] += z_offset_x - z_offset_y
    
    torus_coords = [_.to(cuda_device).contiguous() for _ in torus_coords]
    
    # make torus segment blocks
    for t_idx, coords in enumerate(torus_coords):
        block = domain.CreateBlock(vertexCoordinates=coords, name="Torus%d"%t_idx) #make_block_with_transform(transform, name="Torus%d"%t_idx, dtype=dtype)
        block.CloseAllBoundaries()
    
    torus_blocks = domain.getBlocks()
    torus_tr, torus_br, torus_bl, torus_tl = torus_blocks
    
    
    # connect segments
    axes = ["-y", "-z"] if dims==3 else ["-y"]
    for block_idx in range(len(torus_blocks)):
        torus_blocks[block_idx].ConnectBlock("+x", torus_blocks[(block_idx+1)%(len(torus_blocks))], "-x", *axes)
    
    if rot_vel!=0 or (dims==3 and obs_vel_z!=0): # rotating obstacle
        x_total = x*len(torus_coords)
        angle = -360
        deg_step = angle/x_total
        start_angle = -90 + deg_step * 0.5
        rad_step = np.deg2rad(deg_step)
        start_rad = np.deg2rad(start_angle)

        def get_tangential_vel(i):
            vel = [-np.sin(start_rad + rad_step*i)*rot_vel, np.cos(start_rad + rad_step*i)*rot_vel]
            if dims==3:
                vel.append(obs_vel_z)
            return vel
        
        bound_vel = [get_tangential_vel(i) for i in range(x_total)]
        bound_vel = np.asarray(bound_vel) # WC
        bound_vel = np.moveaxis(bound_vel, -1, 0) # CW
        bound_vel = np.reshape(bound_vel, (1,dims,1,x_total))
        bound_vel = torch.tensor(bound_vel, device=cuda_device, dtype=dtype)
        if dims==3:
            bound_vel = bound_vel.reshape((1,dims,1,1,x_total)).repeat(1,1,z,1,1)
        bound_vel_parts = torch.split(bound_vel, x, dim=-1)

        for block, bound_vel_part in zip(torus_blocks, bound_vel_parts):
            block.getBoundary("-y").setVelocity(bound_vel_part.to(dtype).to(cuda_device).contiguous())
    
    if dims==3 and not z_closed:
        for block in torus_blocks:
            block.MakePeriodic("z")
    
    if vel_in!=0:
        # initialize velocity
        if True:
            for block in torus_blocks:
                block.velocity[:,0,...] = vel_in
        
        # make inflow
        vel_in = torch.tensor([[vel_in] + [0]*(dims-1)], dtype=dtype, device=cuda_device) #NC
        torus_tl.getBoundary("+y").setVelocity(vel_in)
        torus_bl.getBoundary("+y").setVelocity(vel_in)

        # make outflow
        out_bounds = [torus_tr.getBoundary("+y"), torus_br.getBoundary("+y")]
        for out_bound in out_bounds:
            out_bound.setVelocity(vel_in)
            out_bound.makeVelocityVarying() # new tensor
            out_bound.CreatePassiveScalar(domain.getPassiveScalarChannels(), False)
        
        prep_fn["PRE"] = lambda domain, time_step, **kwargs: PISOtorch_simulation.update_advective_boundaries(domain, out_bounds, vel_in, time_step.cuda())
        prep_fn_static = lambda it, dt: PISOtorch_simulation.update_advective_boundaries(domain, out_bounds, torch.zeros_like(vel_in), dt.cuda())
        # update once to make the boundaries already consistent/divergence-free
        prep_fn_static(0, torch.ones([1], dtype=torch.float32, device=cuda_device))
    
    domain.PrepareSolve()

    
    return domain, is_non_ortho, prep_fn

DTYPE = torch.float64

BASE_RES = 4
RES_2D = [BASE_RES, BASE_RES+1, 0] # x,y,z
RES_3D = [BASE_RES, BASE_RES, BASE_RES+2]

#BASE_RES = 4
#RES_2D = [BASE_RES, BASE_RES, 0] # x,y,z

BASE_VEL = 0.8
VISCOSITY = 0.3

S_UNIFORM_2D = [1/BASE_RES, 1/BASE_RES]
S_NORMALISED_2D = [1/r for r in RES_2D[:2]]
S_UNIFORM_3D = [1/BASE_RES, 1/BASE_RES, 1/BASE_RES]
S_NORMALISED_3D = [1/r for r in RES_3D]

setups_simple_2D = {
    "1Block2D_velX+const_periodic": lambda: make1BlockSetup(*RES_2D, vel=[BASE_VEL,0], vel_blob=False, dtype=DTYPE),
    "1Block2D_velX+blob_periodic": lambda: make1BlockSetup(*RES_2D, vel=[BASE_VEL,0], vel_blob=True, dtype=DTYPE),
    "1Block2D_velX-blob_periodic": lambda: make1BlockSetup(*RES_2D, vel=[-BASE_VEL,0], vel_blob=True, dtype=DTYPE),
    "1Block2D_velY+blob_periodic": lambda: make1BlockSetup(*RES_2D, vel=[0,BASE_VEL], vel_blob=True, dtype=DTYPE),
    "1Block2D_velY-blob_periodic": lambda: make1BlockSetup(*RES_2D, vel=[0,-BASE_VEL], vel_blob=True, dtype=DTYPE),
    "1Block2D_velX+blob_closed": lambda: make1BlockSetup(*RES_2D, vel=[BASE_VEL,0], closed_bounds=True, vel_blob=True, dtype=DTYPE),
    "2Block2D_velX+const_periodic": lambda: make2BlockSetup2D(BASE_RES, BASE_RES, BASE_RES, vel1=[BASE_VEL,0], vel2=[BASE_VEL,0], closed_bounds=False, vel_blob1=False, vel_blob2=False, dtype=DTYPE),
    "2Block2D_velX+constY+_periodic": lambda: make2BlockSetup2D(BASE_RES, BASE_RES, BASE_RES, closed_bounds=False, vel_blob1=False, vel_blob2=False, dtype=DTYPE),
    "2Block2D_velX+-blob_periodic": lambda: make2BlockSetup2D(BASE_RES, BASE_RES, BASE_RES, closed_bounds=False, vel2=[-BASE_VEL,0], vel_blob1=True, vel_blob2=True, dtype=DTYPE),
    "2Block2D_velX+constY+_closed": lambda: make2BlockSetup2D(BASE_RES, BASE_RES, BASE_RES, closed_bounds=True, vel_blob1=False, vel_blob2=False, dtype=DTYPE),
}

setups_simple_2D_viscosity = {
    "1Block2D_v_velX+const_periodic": lambda: make1BlockSetup(*RES_2D, vel=[BASE_VEL,0], viscosity=VISCOSITY, vel_blob=False, dtype=DTYPE),
    "1Block2D_v_velX+blob_periodic": lambda: make1BlockSetup(*RES_2D, vel=[BASE_VEL,0],  viscosity=VISCOSITY, vel_blob=True, dtype=DTYPE),
    "1Block2D_v_velX-blob_periodic": lambda: make1BlockSetup(*RES_2D, vel=[-BASE_VEL,0],  viscosity=VISCOSITY, vel_blob=True, dtype=DTYPE),
    "1Block2D_v_velY+blob_periodic": lambda: make1BlockSetup(*RES_2D, vel=[0,BASE_VEL],  viscosity=VISCOSITY, vel_blob=True, dtype=DTYPE),
    "1Block2D_v_velY-blob_periodic": lambda: make1BlockSetup(*RES_2D, vel=[0,-BASE_VEL],  viscosity=VISCOSITY, vel_blob=True, dtype=DTYPE),
    "1Block2D_v_velX+blob_closed": lambda: make1BlockSetup(*RES_2D, vel=[BASE_VEL,0],  viscosity=VISCOSITY, closed_bounds=True, vel_blob=True, dtype=DTYPE),
    "2Block2D_v_velX+const_periodic": lambda: make2BlockSetup2D(BASE_RES, BASE_RES, BASE_RES, vel1=[BASE_VEL,0], vel2=[BASE_VEL,0], closed_bounds=False,  viscosity=VISCOSITY, vel_blob1=False, vel_blob2=False, dtype=DTYPE),
    "2Block2D_v_velX+constY+_periodic": lambda: make2BlockSetup2D(BASE_RES, BASE_RES, BASE_RES, closed_bounds=False,  viscosity=VISCOSITY, vel_blob1=False, vel_blob2=False, dtype=DTYPE),
    "2Block2D_v_velX+-blob_periodic": lambda: make2BlockSetup2D(BASE_RES, BASE_RES, BASE_RES, closed_bounds=False,  viscosity=VISCOSITY, vel2=[-BASE_VEL,0], vel_blob1=True, vel_blob2=True, dtype=DTYPE),
    "2Block2D_v_velX+constY+_closed": lambda: make2BlockSetup2D(BASE_RES, BASE_RES, BASE_RES, closed_bounds=True,  viscosity=VISCOSITY, vel_blob1=False, vel_blob2=False, dtype=DTYPE),
}
setups_simple_2D_transform = { #I=Identity, u=uniform streching, n=normalized (non-uniform unless square resolution), s=non-uniform stretching, e=exponential scaling
    "1Block2D_T-I_velX+const_periodic": lambda: make1BlockSetup(*RES_2D, vel=[BASE_VEL,0], transform_strength=[1,1], vel_blob=False, dtype=DTYPE),
    "1Block2D_T-u_velX+const_periodic": lambda: make1BlockSetup(*RES_2D, vel=[BASE_VEL,0], domain_scale=S_UNIFORM_2D, vel_blob=False, dtype=DTYPE),
    "1Block2D_T-n_velX+const_periodic": lambda: make1BlockSetup(*RES_2D, vel=[BASE_VEL,0], domain_scale=S_NORMALISED_2D, vel_blob=False, dtype=DTYPE),
    "1Block2D_T-s_velX+const_periodic": lambda: make1BlockSetup(*RES_2D, vel=[BASE_VEL,0], domain_scale=[1,2], vel_blob=False, dtype=DTYPE),
    "1Block2D_T-e_velX+const_periodic": lambda: make1BlockSetup(*RES_2D, vel=[BASE_VEL,0], transform_strength=[1.05,1.1], vel_blob=False, dtype=DTYPE),
    "1Block2D_T-I_velY-blob_periodic": lambda: make1BlockSetup(*RES_2D, vel=[0,-BASE_VEL], transform_strength=[1,1], vel_blob=True, dtype=DTYPE),
    "1Block2D_T-u_velY-blob_periodic": lambda: make1BlockSetup(*RES_2D, vel=[0,-BASE_VEL], domain_scale=S_UNIFORM_2D, vel_blob=True, dtype=DTYPE),
    "1Block2D_T-n_velY-blob_periodic": lambda: make1BlockSetup(*RES_2D, vel=[0,-BASE_VEL], domain_scale=S_NORMALISED_2D, vel_blob=True, dtype=DTYPE),
    "1Block2D_T-s_velY-blob_periodic": lambda: make1BlockSetup(*RES_2D, vel=[0,-BASE_VEL], domain_scale=[1,2], vel_blob=True, dtype=DTYPE),
    "1Block2D_T-e_velY-blob_periodic": lambda: make1BlockSetup(*RES_2D, vel=[0,-BASE_VEL], transform_strength=[1.05,1.1], vel_blob=True, dtype=DTYPE),
    "1Block2D_T-I_velX+blob_closed": lambda: make1BlockSetup(*RES_2D, vel=[BASE_VEL,0], closed_bounds=True, transform_strength=[1,1], vel_blob=True, dtype=DTYPE),
    "1Block2D_T-u_velX+blob_closed": lambda: make1BlockSetup(*RES_2D, vel=[BASE_VEL,0], closed_bounds=True, domain_scale=S_UNIFORM_2D, vel_blob=True, dtype=DTYPE),
    "1Block2D_T-n_velX+blob_closed": lambda: make1BlockSetup(*RES_2D, vel=[BASE_VEL,0], closed_bounds=True, domain_scale=S_NORMALISED_2D, vel_blob=True, dtype=DTYPE),
    "1Block2D_T-s_velX+blob_closed": lambda: make1BlockSetup(*RES_2D, vel=[BASE_VEL,0], closed_bounds=True, domain_scale=[1,2], vel_blob=True, dtype=DTYPE),
    "1Block2D_T-e_velX+blob_closed": lambda: make1BlockSetup(*RES_2D, vel=[BASE_VEL,0], closed_bounds=True, transform_strength=[1.05,1.1], vel_blob=True, dtype=DTYPE),
}
setups_simple_2D_transform_viscosity = { #I=Identity, n=normalized, e=exponential scaling
    "1Block2D_T-I_v_velX+const_periodic": lambda: make1BlockSetup(*RES_2D, vel=[BASE_VEL,0], viscosity=VISCOSITY, transform_strength=[1,1], vel_blob=False, dtype=DTYPE),
    "1Block2D_T-u_v_velX+const_periodic": lambda: make1BlockSetup(*RES_2D, vel=[BASE_VEL,0], viscosity=VISCOSITY, domain_scale=S_UNIFORM_2D, vel_blob=False, dtype=DTYPE),
    "1Block2D_T-n_v_velX+const_periodic": lambda: make1BlockSetup(*RES_2D, vel=[BASE_VEL,0], viscosity=VISCOSITY, domain_scale=S_NORMALISED_2D, vel_blob=False, dtype=DTYPE),
    "1Block2D_T-s_v_velX+const_periodic": lambda: make1BlockSetup(*RES_2D, vel=[BASE_VEL,0], viscosity=VISCOSITY, domain_scale=[1,2], vel_blob=False, dtype=DTYPE),
    "1Block2D_T-e_v_velX+const_periodic": lambda: make1BlockSetup(*RES_2D, vel=[BASE_VEL,0], viscosity=VISCOSITY, transform_strength=[1.05,1.1], vel_blob=False, dtype=DTYPE),
    "1Block2D_T-I_v_velY-blob_periodic": lambda: make1BlockSetup(*RES_2D, vel=[0,-BASE_VEL], viscosity=VISCOSITY, transform_strength=[1,1], vel_blob=True, dtype=DTYPE),
    "1Block2D_T-u_v_velY-blob_periodic": lambda: make1BlockSetup(*RES_2D, vel=[0,-BASE_VEL], viscosity=VISCOSITY, domain_scale=S_UNIFORM_2D, vel_blob=True, dtype=DTYPE),
    "1Block2D_T-n_v_velY-blob_periodic": lambda: make1BlockSetup(*RES_2D, vel=[0,-BASE_VEL], viscosity=VISCOSITY, domain_scale=S_NORMALISED_2D, vel_blob=True, dtype=DTYPE),
    "1Block2D_T-2_v_velY-blob_periodic": lambda: make1BlockSetup(*RES_2D, vel=[0,-BASE_VEL], viscosity=VISCOSITY, domain_scale=[1,2], vel_blob=True, dtype=DTYPE),
    "1Block2D_T-e_v_velY-blob_periodic": lambda: make1BlockSetup(*RES_2D, vel=[0,-BASE_VEL], viscosity=VISCOSITY, transform_strength=[1.05,1.1], vel_blob=True, dtype=DTYPE),
    "1Block2D_T-I_v_velX+blob_closed": lambda: make1BlockSetup(*RES_2D, vel=[BASE_VEL,0], viscosity=VISCOSITY, closed_bounds=True, transform_strength=[1,1], vel_blob=True, dtype=DTYPE),
    "1Block2D_T-u_v_velX+blob_closed": lambda: make1BlockSetup(*RES_2D, vel=[BASE_VEL,0], viscosity=VISCOSITY, closed_bounds=True, domain_scale=S_UNIFORM_2D, vel_blob=True, dtype=DTYPE),
    "1Block2D_T-n_v_velX+blob_closed": lambda: make1BlockSetup(*RES_2D, vel=[BASE_VEL,0], viscosity=VISCOSITY, closed_bounds=True, domain_scale=S_NORMALISED_2D, vel_blob=True, dtype=DTYPE),
    "1Block2D_T-s_v_velX+blob_closed": lambda: make1BlockSetup(*RES_2D, vel=[BASE_VEL,0], viscosity=VISCOSITY, closed_bounds=True, domain_scale=[1,2], vel_blob=True, dtype=DTYPE),
    "1Block2D_T-e_v_velX+blob_closed": lambda: make1BlockSetup(*RES_2D, vel=[BASE_VEL,0], viscosity=VISCOSITY, closed_bounds=True, transform_strength=[1.05,1.1], vel_blob=True, dtype=DTYPE),
}
setups_simple_3D = {
    "1Block3D_velX+const_periodic": lambda: make1BlockSetup(*RES_3D, vel=[BASE_VEL,0,0], vel_blob=False, dtype=DTYPE),
    "1Block3D_velX+blob_periodic": lambda: make1BlockSetup(*RES_3D, vel=[BASE_VEL,0,0], vel_blob=True, dtype=DTYPE),
    "1Block3D_velX-blob_periodic": lambda: make1BlockSetup(*RES_3D, vel=[-BASE_VEL,0,0], vel_blob=True, dtype=DTYPE),
    "1Block3D_velY+blob_periodic": lambda: make1BlockSetup(*RES_3D, vel=[0,BASE_VEL,0], vel_blob=True, dtype=DTYPE),
    "1Block3D_velY-blob_periodic": lambda: make1BlockSetup(*RES_3D, vel=[0,-BASE_VEL,0], vel_blob=True, dtype=DTYPE),
    "1Block3D_velZ+blob_periodic": lambda: make1BlockSetup(*RES_3D, vel=[0,0,BASE_VEL], vel_blob=True, dtype=DTYPE),
    "1Block3D_velZ-blob_periodic": lambda: make1BlockSetup(*RES_3D, vel=[0,0,-BASE_VEL], vel_blob=True, dtype=DTYPE),
    "1Block3D_velX+blob_closed": lambda: make1BlockSetup(*RES_3D, vel=[BASE_VEL,0,0], closed_bounds=True, vel_blob=True, dtype=DTYPE),
}
setups_simple_3D_viscosity = {
    "1Block3D_v_velX+const_periodic": lambda: make1BlockSetup(*RES_3D, vel=[BASE_VEL,0,0], viscosity=VISCOSITY, vel_blob=False, dtype=DTYPE),
    "1Block3D_v_velX+blob_periodic": lambda: make1BlockSetup(*RES_3D, vel=[BASE_VEL,0,0], viscosity=VISCOSITY, vel_blob=True, dtype=DTYPE),
    "1Block3D_v_velX-blob_periodic": lambda: make1BlockSetup(*RES_3D, vel=[-BASE_VEL,0,0], viscosity=VISCOSITY, vel_blob=True, dtype=DTYPE),
    "1Block3D_v_velY+blob_periodic": lambda: make1BlockSetup(*RES_3D, vel=[0,BASE_VEL,0], viscosity=VISCOSITY, vel_blob=True, dtype=DTYPE),
    "1Block3D_v_velY-blob_periodic": lambda: make1BlockSetup(*RES_3D, vel=[0,-BASE_VEL,0], viscosity=VISCOSITY, vel_blob=True, dtype=DTYPE),
    "1Block3D_v_velZ+blob_periodic": lambda: make1BlockSetup(*RES_3D, vel=[0,0,BASE_VEL], viscosity=VISCOSITY, vel_blob=True, dtype=DTYPE),
    "1Block3D_v_velZ-blob_periodic": lambda: make1BlockSetup(*RES_3D, vel=[0,0,-BASE_VEL], viscosity=VISCOSITY, vel_blob=True, dtype=DTYPE),
    "1Block3D_v_velX+blob_closed": lambda: make1BlockSetup(*RES_3D, vel=[BASE_VEL,0,0], closed_bounds=True, viscosity=VISCOSITY, vel_blob=True, dtype=DTYPE),
}
setups_simple_3D_transform_viscosity = {
    "1Block3D_T-I_v_velZ+blob_periodic": lambda: make1BlockSetup(*RES_3D, vel=[0,0,BASE_VEL], viscosity=VISCOSITY, transform_strength=[1,1,1], vel_blob=True, dtype=DTYPE),
    "1Block3D_T-u_v_velZ+blob_periodic": lambda: make1BlockSetup(*RES_3D, vel=[0,0,BASE_VEL], viscosity=VISCOSITY, domain_scale=S_UNIFORM_3D, vel_blob=True, dtype=DTYPE),
    "1Block3D_T-e_v_velZ+blob_periodic": lambda: make1BlockSetup(*RES_3D, vel=[0,0,BASE_VEL], viscosity=VISCOSITY, transform_strength=[1.04,1.07, 1.1], vel_blob=True, dtype=DTYPE),
    "1Block3D_T-I_v_velX+blob_closed": lambda: make1BlockSetup(*RES_3D, vel=[BASE_VEL,0,0], closed_bounds=True, viscosity=VISCOSITY, transform_strength=[1,1,1], vel_blob=True, dtype=DTYPE),
    "1Block3D_T-u_v_velX+blob_closed": lambda: make1BlockSetup(*RES_3D, vel=[BASE_VEL,0,0], closed_bounds=True, viscosity=VISCOSITY, domain_scale=S_UNIFORM_3D, vel_blob=True, dtype=DTYPE),
    "1Block3D_T-e_v_velX+blob_closed": lambda: make1BlockSetup(*RES_3D, vel=[BASE_VEL,0,0], closed_bounds=True, viscosity=VISCOSITY, transform_strength=[1.04,1.07, 1.1], vel_blob=True, dtype=DTYPE),
}

setups_nonortho_2D = {
    "1Block2D_T-n-rot30_v_velY-blob_periodic": lambda: make1BlockSetup(*RES_2D, vel=[0,-BASE_VEL], viscosity=VISCOSITY,
        domain_scale=S_NORMALISED_2D, rot_distortion_max_angle=30, vel_blob=True, dtype=DTYPE),
    "1Block2D_T-n-rot30_v_velX+blob_closed": lambda: make1BlockSetup(*RES_2D, vel=[BASE_VEL,0], viscosity=VISCOSITY,
        closed_bounds=True, domain_scale=S_NORMALISED_2D, rot_distortion_max_angle=30, vel_blob=True, dtype=DTYPE),
    "4Torus2D_T_v_velIn_inOut": lambda: make4BlockTorus_VortexStreet(8, r1=0.5, r2=8, vel_in=BASE_VEL, viscosity=1e-3, dtype=DTYPE),
    "4Torus2D_T-rot30_v_velIn_inOut": lambda: make4BlockTorus_VortexStreet(8, r1=0.5, r2=8, vel_in=BASE_VEL, viscosity=1e-3, rot_distortion_max_angle=30, dtype=DTYPE),
}
setups_nonortho_3D = {
    "1Block3D_T-e-rot30_v_velZ+blob_periodic": lambda: make1BlockSetup(*RES_3D, vel=[0,0,BASE_VEL], viscosity=VISCOSITY,
        transform_strength=[1.04,1.07,1.1], rot_distortion_max_angle=30, vel_blob=True, dtype=DTYPE),
    "4Torus3D_T_v_velIn_inOut": lambda: make4BlockTorus_VortexStreet(16, z=16, r1=0.5, r2=8, vel_in=BASE_VEL, viscosity=1e-3, dtype=DTYPE),
    "4Torus3D_T-rot30-zSine_v_velIn_inOut": lambda: make4BlockTorus_VortexStreet(16, z=16, r1=0.5, r2=8, vel_in=BASE_VEL, viscosity=1e-3,
        grid_z_sine_scale=1, rot_distortion_max_angle=30, dtype=DTYPE),
}

setups_simple = {}
setups_simple.update(setups_simple_2D)
setups_simple.update(setups_simple_2D_viscosity)
setups_simple.update(setups_simple_3D)
setups_simple.update(setups_simple_3D_viscosity)

setups_simple_2D_combined = {}
setups_simple_2D_combined.update(setups_simple_2D)
setups_simple_2D_combined.update(setups_simple_2D_viscosity)
setups_simple_2D_combined.update(setups_simple_2D_transform)
setups_simple_2D_combined.update(setups_simple_2D_transform_viscosity)

setups_simple_3D_combined = {}
setups_simple_3D_combined.update(setups_simple_3D)
setups_simple_3D_combined.update(setups_simple_3D_viscosity)
setups_simple_3D_combined.update(setups_simple_3D_transform_viscosity)

setups_simple_combined = {}
setups_simple_combined.update(setups_simple_2D_combined)
setups_simple_combined.update(setups_simple_3D_combined)

setups_nonortho = {}
setups_nonortho.update(setups_nonortho_2D)
setups_nonortho.update(setups_nonortho_3D)
