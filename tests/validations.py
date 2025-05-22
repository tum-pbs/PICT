# Hagen P?? Flow between plates
# Lid driven cavity, 2D and 3D, comparison to references and between 2D and periodic-z 3D
# These tests take some time to converge
from lib.util.logging import get_logger
from lib.util.profiling import SAMPLE

import os#, signal, itertools
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
#import numpy as np
import PISOtorch
#import PISOtorch_sim
import PISOtorch_simulation

assert torch.cuda.is_available()
cuda_device = torch.device("cuda")
cpu_device = torch.device("cpu")

import lib.data.shapes as shapes
import lib.data.resample as resampling
from lib.util.output import *

from matplotlib import pyplot as plt 
#import matplotlib.cm as cm 
import matplotlib.ticker as ticker

# domain setups

def scale_transform(transform, scaling):
    dims = len(scaling)
    M_scale = [scaling[row] if row==col else 1 for row in range(dims) for col in range(dims)]
    M_inv_scale = [1/s for s in M_scale]
    J_scale = np.prod(scaling)
    scalings = M_scale + M_inv_scale + [J_scale]
    #LOG.info("transform scalings: %s", scalings)
    return transform * torch.tensor(scalings, dtype=transform.dtype, device=transform.device)

# def scale_block_transform(block, scaling):
    # assert block.getSpatialDims()==len(scaling)
    # assert block.hasTransform
    # # needs to be orthogonal transform
    # dims = block.getSpatialDims()
    # M_scale = [scaling[row] if row==col else 1 for row in range(dims) for col in range(dims)]
    # M_inv_scale = [1/s for s in M_scale]
    # J_scale = np.prod(scaling)
    # block.transform.copy_(block.transform * torch.tensor(M_scale + M_inv_scale + [J_scale], dtype=block.transform.dtype, device=block.transform.device))

def get_cell_size_min_max(domain):
    sizes = domain.getBlock(0).transform[...,-1]
    min_size = torch.min(sizes)
    max_size = torch.max(sizes)
    for idx in range(1, domain.getNumBlocks()):
        sizes = domain.getBlock(0).transform[...,-1]
        min_size = torch.minimum(min_size, torch.min(sizes))
        max_size = torch.maximum(max_size, torch.max(sizes))
    return min_size, max_size

def make1BlockLidCavity3DExpScaling(x,y,z, viscosity=0, lid_vel=0, periodic_z=False, scale_strength=[1,1,1], domain_scale=1, aspect_x=1, aspect_z=1, lid_permutation=(0,3), dtype=torch.float32):
    dims = 3
    # lid_permutation (boundary,flowDir): 0=-x, 1=+x, 2=-y, ...
    assert (lid_permutation[0]&6)!=(lid_permutation[1]&6), "lid-normal flows are not permitted"
    assert 0<=lid_permutation[0] and lid_permutation[0] <dims*2, "lid boundary index out of bounds"
    assert 0<=lid_permutation[1] and lid_permutation[1] <dims*2, "lid velocity axis index out of bounds"
    lid_bound_axis = lid_permutation[0]>>1
    lid_flow_axis = lid_permutation[1]>>1
    lid_tangent_axis = 3 - (lid_bound_axis + lid_flow_axis)
    #axes = [lid_bound_axis, lid_flow_axis, lid_tangent_axis]
    # TODO: shuffle resoution to match permutation?
    # TODO: aspect_x => aspect_lid_normal, aspect_z => aspect_lid_tangent

    res = [z,y,x] # !order:z,y,x!
    domain_scale = [domain_scale*aspect_x / x, domain_scale / y, domain_scale*aspect_z / z] #x,y,z
    make_transform = scale_strength is not None
    
    grid = None
    if make_transform:

        if domain_scale is None:
            domain_scale = [1]*dims
        assert isinstance(domain_scale, (list, tuple)) and len(domain_scale)==dims
        
        grid = shapes.make_wall_refined_ortho_grid(x,y, corner_upper=(x*domain_scale[0], y*domain_scale[1]),
            wall_refinement=["-x","+x","-y","+y"], base=scale_strength[:2], dtype=dtype)
        
        if dims==3:
            grid = shapes.extrude_grid_z(grid, z, end_z=z*domain_scale[2], weights_z="EXP", exp_base=scale_strength[2])
        
        grid = grid.to(cuda_device).contiguous()

        
    #velocity =  torch.zeros(size=[1,dims]+res, dtype=dtype, device=cuda_device)
    if grid is not None:
        transforms = PISOtorch.CoordsToTransforms(grid.to(cuda_device).contiguous())
        data = shapes.get_grid_normal_dist_from_ortho_transforms(transforms, [0]*dims,[0.5]*dims, dtype=dtype)
    else:
        data = shapes.get_grid_normal_dist(res, [0]*dims,[0.5]*dims, dtype=dtype)
    data = torch.reshape(data, [1,1]+res).cuda()
    
    viscosity = torch.ones([viscosity], dtype=dtype, device=cpu_device)
    domain = PISOtorch.Domain(dims, viscosity, name="Domain1Block3D", dtype=dtype, device=cuda_device)

    block = domain.CreateBlock(passiveScalar=data, vertexCoordinates=grid, name="Block")
    
    # Boundaries
    block.CloseAllBoundaries()
    if periodic_z:
        block.MakePeriodic(lid_tangent_axis)
    
    lid_vel_static = [0,0,0]
    lid_vel_static[lid_flow_axis] = lid_vel if (lid_permutation[1]&1)==1 else -lid_vel
    lid_vel_static =  torch.tensor([lid_vel_static], dtype=dtype, device=cuda_device)

    block.getBoundary(lid_permutation[0]).setVelocity(lid_vel_static)

    domain.PrepareSolve()

    # if make_transform:
        # min_size, max_size = get_cell_size_min_max(domain)
        # min_size = min_size.cpu().numpy()
        # max_size = max_size.cpu().numpy()
        # LOG.info("Transform cell size range: %s - %s (%s)", min_size, max_size, max_size/min_size)

    return domain



def split_block_2D(block, x_div, y_div):
    raise NotImplementedError("Old version, no longer supported")
    boundaries = [block.getBoundary(i) for i in range(4)]
    # split bounds: periodic becomes connected and has to be handeled after block creation. static can be copied, varying needs to be split. Connected can not be handeled by single block split.
    assert all(isinstance(b, (PISOtorch.PeriodicBoundary, PISOtorch.StaticDirichletBoundary, PISOtorch.VaryingDirichletBoundary)) for b in boundaries), "block has unsupported boundary."
    bounds = []
    for b_idx in range(4):
        div = x_div if b_idx>1 else y_div
        split_dim = 3 if b_idx>1 else 2
        n_div = len(div)
        bound = block.getBoundary(b_idx)
        if isinstance(bound, PISOtorch.PeriodicBoundary):
            bounds.append(["periodic"]*n_div)
        elif isinstance(bound, PISOtorch.StaticDirichletBoundary):
            bounds.append([bound]*n_div)
        elif isinstance(bound, PISOtorch.VaryingDirichletBoundary):
            slip = bound.slip
            v = torch.split(bound.boundaryVelocity, div, dim=split_dim)
            s = torch.split(bound.boundaryScalar, div, dim=split_dim)
            if bound.hasTransform:
                t = torch.split(bound.transform, div, dim=split_dim-1)
            new_bounds = []
            for i in range(n_div):
                b = PISOtorch.VaryingDirichletBoundary(slip, v[i].clone(), s[i].clone())
                if bound.hasTransform:
                    b.setTransform(t[i].clone())
                new_bounds.append(b)
            bounds.append(new_bounds)
        else:
            raise TypeError("block has unsupported boundary.")

    # split block
    blocks = [[None for _ in range(len(x_div))] for _ in range(len(y_div))]
    v_y = torch.split(block.velocity, y_div, dim=2)
    p_y = torch.split(block.pressure, y_div, dim=2)
    s_y = torch.split(block.passiveScalar, y_div, dim=2)
    if block.hasTransform:
        t_y = torch.split(block.transform, y_div, dim=1) #NHWC
    for y_idx, y_res in enumerate(y_div):
        v_x = torch.split(v_y[y_idx], x_div, dim=3)
        p_x = torch.split(p_y[y_idx], y_div, dim=3)
        s_x = torch.split(s_y[y_idx], y_div, dim=3)
        if block.hasTransform:
            t_x = torch.split(t_y[y_idx], y_div, dim=2)
        for x_idx, x_res in enumerate(x_div):
            v = v_x[x_idx]
            p = p_x[x_idx]
            s = s_x[x_idx]

            block_split = PISOtorch.Block(v.clone(), p.clone(), s.clone(), block.name + "_x%d-y%d"%(x_idx, y_idx))
            if block.hasTransform:
                t = t_x[x_idx]
                block_split.setTransform(t.clone())

            # make and add boundaries
            if x_idx==0:
                bound = bounds[0][y_idx]
                if bound!="periodic":
                    block_split.setBoundary(0, bound)
            if x_idx==len(x_div)-1:
                bound = bounds[1][y_idx]
                if bound!="periodic":
                    block_split.setBoundary(1, bound)
            if y_idx==0:
                bound = bounds[2][x_idx]
                if bound!="periodic":
                    block_split.setBoundary(2, bound)
            if y_idx==len(y_div)-1:
                bound = bounds[3][x_idx]
                if bound!="periodic":
                    block_split.setBoundary(3, bound)
            blocks[y_idx][x_idx] = block_split

    # connect blocks
    for y_idx in range(len(y_div)):
        for x_idx in range(len(x_div)):
            block = blocks[y_idx][x_idx]
            if x_idx<len(x_div)-1:
                upper_x = blocks[y_idx][x_idx+1]
                PISOtorch.ConnectBlocks(block, "+x", upper_x, "-x", "-y", "")
            if y_idx<len(y_div)-1:
                upper_y = blocks[y_idx+1][x_idx]
                PISOtorch.ConnectBlocks(block, "+y", upper_y, "-y", "-x", "")

    # handle originally periodic boundaries
    for y_idx, bound in enumerate(bounds[0]):
        if bound=="periodic":
            assert bounds[1][y_idx]=="periodic"
            PISOtorch.ConnectBlocks(blocks[y_idx][0], "-x", blocks[y_idx][-1], "+x", "-y", "")
    for x_idx, bound in enumerate(bounds[2]):
        if bound=="periodic":
            assert bounds[3][x_idx]=="periodic"
            PISOtorch.ConnectBlocks(blocks[0][x_idx], "-y", blocks[-1][x_idx], "+y", "-x", "")
    
    # for output
    blocks_flat = []
    layout = []
    i=0
    for y_idx in range(len(y_div)):
        row = []
        for x_idx in range(len(x_div)):
            blocks_flat.append(blocks[y_idx][x_idx])
            row.append(i)
            i +=1
        layout.append(row)

    #LOG.info("split blocks: %s", [str(b) for b in blocks_flat])

    return blocks_flat, layout

def combine_blocks_2D_velocity(blocks_flat, layout):
    # only to return a combined velocity tensor
    if layout is None:
        return blocks_flat[0].velocity, \
                blocks_flat[0].pressure, \
                blocks_flat[0].passiveScalar, \
                blocks_flat[0].getCellCoordinates() if  blocks_flat[0].hasVertexCoordinates() else None
    
    has_coords = all(block.hasVertexCoordinates() for block in blocks_flat)
    
    v = []
    p = []
    s = []
    c = []
    for row in layout:
        v_row = []
        p_row = []
        s_row = []
        c_row = []
        for idx in row:
            v_row.append(blocks_flat[idx].velocity)
            p_row.append(blocks_flat[idx].pressure)
            s_row.append(blocks_flat[idx].passiveScalar)
            if has_coords:
                c_row.append(blocks_flat[idx].getCellCoordinates())
        v.append(torch.cat(v_row, dim=3))
        p.append(torch.cat(p_row, dim=3))
        s.append(torch.cat(s_row, dim=3))
        if has_coords:
            c.append(torch.cat(c_row, dim=3))
    v = torch.cat(v, dim=2)
    p = torch.cat(p, dim=2)
    s = torch.cat(s, dim=2)
    if has_coords:
        c = torch.cat(c, dim=2)
    else:
        c = None

    return v, p, s, c


def make1BlockSetup2DExpScaling(x:int, y:int, vel=[1,0], vel_blob=True, viscosity=0.0, top_vel=0, connect=None, closed_bounds=False, slip=0,
        scale_strength=[1,1], domain_scale=None, invert_transform=False, lid_permutation=(3,1), rot_distortion_max_angle=None, dtype=torch.float32) -> PISOtorch.Domain:
    dims = 2
    # block_subdivision: size in cells of the sub-blocks for x and y direction
    if isinstance(x, (list, tuple)):
        x_div = x
        x = sum(x)
    else:
        x_div = [x]
    if isinstance(y, (list, tuple)):
        y_div = y
        y = sum(y)
    else:
        y_div = [y]
    res = [y,x]
    # lid_permutation (boundary,flowDir): 0=-x, 1=+x, 2=-y, ...
    assert (lid_permutation[0]&6)!=(lid_permutation[1]&6), "lid-normal flows are not permitted"
    assert 0<=lid_permutation[0] and lid_permutation[0] <dims*2, "lid boundary index out of bounds"
    assert 0<=lid_permutation[1] and lid_permutation[1] <dims*2, "lid velocity axis index out of bounds"
    lid_bound_axis = lid_permutation[0]>>1
    lid_flow_axis = lid_permutation[1]>>1
    #axes = [lid_bound_axis, lid_flow_axis]
    
    make_transform = scale_strength is not None
    
    grid = None
    if make_transform:
        if invert_transform:
            raise NotImplementedError
            # OLD:
            # grid_l, grid_r = torch.split(transforms, [res[0]//2, res[0] - res[0]//2], dim=-2)
            # transforms = torch.cat([grid_r, grid_l], axis=-2)
            # grid_l, grid_r = torch.split(transforms, [res[0]//2, res[0] - res[0]//2], dim=-3)
            # transforms = torch.cat([grid_r, grid_l], axis=-3)
        
        if domain_scale is None:
            domain_scale = [1]*dims
        assert isinstance(domain_scale, (list, tuple)) and len(domain_scale)==dims
        
        corner_upper = (x*domain_scale[0], y*domain_scale[1])
        grid = shapes.make_wall_refined_ortho_grid(x,y, corner_upper=corner_upper,
            wall_refinement=["-x","+x","-y","+y"], base=scale_strength[:2], dtype=dtype)
        
        if rot_distortion_max_angle is not None:
            max_r = min(corner_upper)*0.5*0.9
            #distance_scaling = lambda dist: max(0, 1 - dist/max_r)
            distance_scaling = shapes.make_rotation_distance_scaling_fn_sine_half(rot_distortion_max_angle, max_r)
            grid = shapes.rotate_grid(grid, angle=rot_distortion_max_angle, distance_scaling=distance_scaling)
        
        # if dims==3:
            # grid = shapes.extrude_grid_z(grid, z, end_z=z*domain_scale[2], weights_z="EXP", exp_base=scale_strength[2])
        
        # grid (vertexCoordinates) need split with 1 overlap
        grid_div = []
        x_div_sum = [0] + np.cumsum(x_div).tolist()
        y_div_sum = [0] + np.cumsum(y_div).tolist()
        for y_idx, y_size in enumerate(y_div):
            grid_div_row = []
            grid_row = grid[:,:,y_div_sum[y_idx]:y_div_sum[y_idx+1]+1,:]
            for x_idx, x_size in enumerate(x_div):
                grid_subblock = grid_row[:,:,:,x_div_sum[x_idx]:x_div_sum[x_idx+1]+1]
                grid_div_row.append(grid_subblock.to(cuda_device).contiguous())
            grid_div.append(grid_div_row)

        # if dims==2 and x<10 and y<10:
            # coords = grid #shapes.ortho_transform_to_coords(transforms, dims)
            # LOG.info("coords:\n%s", coords)
            # center_coords = shapes.coords_to_center_coords(coords)
            # LOG.info("center_coords:\n%s", center_coords)
    
    
    
    
    if grid is not None:
        transforms = PISOtorch.CoordsToTransforms(grid.to(cuda_device).contiguous())
        data = shapes.get_grid_normal_dist_from_ortho_transforms(transforms, [0]*dims,[0.5]*dims, dtype=dtype)
    else:
        data = shapes.get_grid_normal_dist(res, [0]*dims,[0.5]*dims, dtype=dtype)
    data = torch.reshape(data, [1,1]+res).to(cuda_device)
    
    velocity = torch.tensor(vel, dtype=dtype, device=cuda_device).reshape([1,dims]+[1]*dims).repeat(1,1,*res).contiguous()
    if vel_blob:
        velocity = velocity * data
    
    viscosity = torch.tensor([viscosity], dtype=dtype, device=cpu_device)
    domain = PISOtorch.Domain(dims, viscosity, name="Domain1Block2D", dtype=dtype, device=cuda_device)

    # create the grid of blocks
    block_idx = 0
    blocks = []
    layout = []
    v_y = torch.split(velocity, y_div, dim=2)
    s_y = torch.split(data, y_div, dim=2)
    for y_idx, y_res in enumerate(y_div):
        blocks_row = []
        layout_row = []
        v_x = torch.split(v_y[y_idx], x_div, dim=3)
        s_x = torch.split(s_y[y_idx], x_div, dim=3)
        for x_idx, x_res in enumerate(x_div):
            v = v_x[x_idx].contiguous()
            s = s_x[x_idx].contiguous()
            g = grid_div[y_idx][x_idx] if grid is not None else None

            block = domain.CreateBlock(velocity=v, passiveScalar=s, vertexCoordinates=g, name="Block_x%d_y%d"%(x_idx, y_idx))
            blocks_row.append(block)
            layout_row.append(block_idx)
            block_idx += 1
        blocks.append(blocks_row)
        layout.append(layout_row)
    
    # connect blocks and close outer boundaries if closed_bounds
    for y_idx in range(len(y_div)):
        for x_idx in range(len(x_div)):
            block = blocks[y_idx][x_idx]
            # close global lower boundaries before
            if x_idx==0 and closed_bounds:
                block.CloseBoundary("-x")
            if y_idx==0 and closed_bounds:
                block.CloseBoundary("-y")
            # connect to upper blocks
            if (x_idx+1)<len(x_div):
                block_x = blocks[y_idx][x_idx+1]
                block.ConnectBlock("+x", block_x, "-x", "-y")
            elif not closed_bounds:
                if len(x_div)>1: # otherwise the default periodic boundary will handle this
                    block_x = blocks[y_idx][0]
                    block.ConnectBlock("+x", block_x, "-x", "-y")
            else:
                block.CloseBoundary("+x")
            
            if (y_idx+1)<len(y_div):
                block_y = blocks[y_idx+1][x_idx]
                block.ConnectBlock("+y", block_y, "-y", "-x")
            elif not closed_bounds:
                if len(y_div)>1:
                    block_y = blocks[0][x_idx]
                    block.ConnectBlock("+y", block_y, "-y", "-x")
            else:
                block.CloseBoundary("+y")

    if closed_bounds:
        # set lid vel
        bound_idx = lid_permutation[0]
        
        lid_vel_static = [0,0]
        lid_vel_static[lid_flow_axis] = top_vel if (lid_permutation[1]&1)==1 else -top_vel
        lid_vel_static =  torch.tensor([lid_vel_static], dtype=dtype, device=cuda_device) # NC
        
        if bound_idx in [0,1]: # lower or upper x
            x_idx = 0 if bound_idx==0 else -1
            for y_idx in range(len(y_div)):
                blocks[y_idx][x_idx].getBoundary(bound_idx).setVelocity(lid_vel_static)
        elif bound_idx in [2,3]: # lower or upper y
            y_idx = 0 if bound_idx==2 else -1
            for x_idx in range(len(x_div)):
                blocks[y_idx][x_idx].getBoundary(bound_idx).setVelocity(lid_vel_static)
    
    if isinstance(connect, str):
        if connect[0]=="x":
            for y_idx in range(len(y_div)):
                block_start = blocks[y_idx][0]
                block_end = blocks[y_idx][-1]
                if not (connect=="xp" and len(x_div)>1):
                    block_end.ConnectBlock("+x", block_start, "-x", "+y" if connect=="xm" else "-y")
                else:
                    block.MakePeriodic("x")
        if connect[0]=="y":
            for x_idx in range(len(x_div)):
                block_start = blocks[0][x_idx]
                block_end = blocks[-1][x_idx]
                if not (connect=="yp" and len(y_div)>1):
                    block_end.ConnectBlock("+y", block_start, "-y", "+x" if connect=="xm" else "-x")
                else:
                    block.MakePeriodic("y")


    domain.PrepareSolve()

    # if make_transform:
        # min_size, max_size = get_cell_size_min_max(domain)
        # min_size = min_size.cpu().numpy()
        # max_size = max_size.cpu().numpy()
        # LOG.info("Transform cell size range: %s - %s (%s)", min_size, max_size, max_size/min_size)
    
    resample_shape = None
    if make_transform:
        resample_shape = [x*2,y*2]

    return domain, layout, resample_shape


# dir and logging setup
# test runs with plotting


def plane_poiseuille_flow(run_dir, it, time_step=1, forcing=1, viscosity=1, resolutions=[8,16,32,64],
                          domain_scale=1, max_size_diff_rel=[1,2,4,16], global_scale=True, rot_distortion_max_angle=None,
                          dp=True, STOP_FN=lambda: False, 
                          plot_format="pdf"):
    # Plane Poiseuille Flow Test: https://en.wikipedia.org/wiki/Hagen%E2%80%93Poiseuille_equation#Plane_Poiseuille_flow
        
    i = 0
    substeps = -2 #100 # -1 for adaptive
    corr = 2
    rot_angle = rot_distortion_max_angle
    
    # 1: directly added to velocity before every piso iteration. 2: added to advection RHS. 3: added to advection and pressure RHS. 4: added as velocity source term.
    forcing_version = 4

    def poiseuille(y):
        return forcing/(2*viscosity) * y * ( domain_scale - y)
    
    max_vel = poiseuille(domain_scale/2)
    
    dtype = torch.float64 if dp else torch.float32
    
    os.makedirs(run_dir, exist_ok=True)
    
    LOG = get_logger("PHPF")
    LOG.info("Plane Poiseuille Flow Test: G=%.03e (v%d), visc=%.03e, ts=%.03e (ss=%d), max vel=%.03e -> Re~%.03e", forcing, forcing_version, viscosity, time_step, substeps, max_vel, max_vel*domain_scale/viscosity)
    

    max_res = max(resolutions) #32 #higher is too many crosses in the plot ...
    reference_coords = [(i+0.5)*domain_scale/max_res for i in range(max_res)]
    reference_result = [poiseuille(i) for i in reference_coords]
    num_res = len(resolutions)
    num_sizes = len(max_size_diff_rel)
    plot_size = 5
    prof_fig, prof_ax = plt.subplots(1,num_sizes, figsize=(num_sizes*plot_size, plot_size))
    if num_sizes==1:
        prof_ax = [prof_ax]
    #prof_fig.suptitle("Velocity profiles.")
    
    if forcing_version==1:
        G = torch.tensor([forcing,0], dtype=dtype, device=cuda_device).reshape((1,2,1,1))
        def forcing_fn(domain, time_step, **kwargs):
            for block in domain.getBlocks():
                block.velocity.copy_(block.velocity + (G*time_step.to(cuda_device)))
            # differentiable version?
            #    block.setVelocity((block.velocity + (G*time_step).cuda()).contiguous())
            #    LOG.info("block forced vel max: %s", torch.max(block.velocity))
            #domain.UpdateDomainData()
        prep_fn = {"PRE": forcing_fn}
    elif forcing_version in [2,3]:
        G = torch.tensor([forcing,0], dtype=dtype, device=cuda_device).reshape((2,1)) # channel, cells flat
        def forcing_fn(domain, time_step, **kwargs):
            # RHS version for multiple blocks
            domain.velocityRHS.copy_(torch.flatten(domain.velocityRHS.reshape((2,-1)) + G))
            #domain.UpdateDomainData()
        prep_fn = {"POST_VELOCITY_SETUP": forcing_fn}
        if forcing_version==3:
            raise NotImplementedError("forcing v3 needs POST_PRESSURE_RHS.")
            def forcing_fn_pressure(domain, time_step, **kwargs):
                # RHS version for multiple blocks
                domain.pressureRHS.copy_(torch.flatten(domain.pressureRHS.reshape((2,-1)) + G / domain.A.reshape((1,-1))))
                #domain.UpdateDomainData()
            prep_fn["POST_PRESSURE_RHS"] = forcing_fn_pressure
    elif forcing_version==4:
        G = torch.tensor([[forcing,0]], dtype=dtype, device=cuda_device) # NC, static velocity source
        prep_fn = {}

            
    
    results = {}
    vcv = 1
    
    is_non_ortho = True #(rot_angle is not None)
    sim = PISOtorch_simulation.Simulation(corrector_steps=corr, velocity_corrector="FD",
            advection_tol=None, pressure_tol=None, pressure_return_best_result=True,
            advect_non_ortho_steps=2 if is_non_ortho else 1, pressure_non_ortho_steps=4 if is_non_ortho else 1, non_orthogonal=is_non_ortho,
            log_dir=None, log_interval=1, norm_vel=True, prep_fn=prep_fn, convergence_tol=1e-7 if dp else 1e-7, stop_fn=STOP_FN, save_domain_name="PHPF")
    
    for s_idx, s in enumerate(max_size_diff_rel):
        if STOP_FN(): break
        
        ax = prof_ax[s_idx]
        #ax.set_title('scale %.02e (%s)'%(s, "global" if global_scale else "local"))
        ax.set_title(('uniform' if s==1 else 'refined') + ('' if rot_angle is None else ', rot $=%d\\degree$'%(rot_angle)))
        ax.set(xlabel='y', ylabel='$u$')
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.025))
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        ax.set_ylim(0, 0.13)
        ax.grid()
        #ax.plot(reference_coords, reference_result, 'kx', label="reference", zorder=num_res)
        ax.plot(reference_coords, reference_result, 'k-', label="reference", zorder=0)
        ax.legend()
        
        prof_fig.align_labels()
        prof_fig.tight_layout()
        prof_fig.savefig(os.path.join(run_dir, "convergence.{}".format(plot_format)))

        #results[s] = {}
        for r_idx, r in enumerate(resolutions):
            if STOP_FN(): break
            sub_dir = os.path.join(run_dir, "{:02d}_scale-{}_r{}_vcv{}".format(i, s, r, vcv))
            os.makedirs(sub_dir)
            res = [r]*2
            scl = s ** (1/(r//2 - 1)) if global_scale else s
            scale_strength = [1,scl]
            ds = domain_scale
            if ds is not None:
                ds = [ds/r for r in res]
            if substeps==-2: #test
                CFL_cond = 0.8
                max_time_step = CFL_cond/(max(max_vel, forcing)/ds[0])
                if max_time_step>=time_step:
                    ss = 1
                    ts = time_step
                else:
                    ss = int(np.ceil(time_step / max_time_step))
                    ts = time_step / ss
                LOG.info("Fixed time step for r=%d: %.03e, %d substeps", r, ts, ss)
            else:
                ts = time_step
                ss = substeps


            LOG.info("scale_strength=%s", scale_strength)
            domain, layout, resample_shape = make1BlockSetup2DExpScaling(*res, vel=[0,0], vel_blob=False, viscosity=viscosity, connect='xp', closed_bounds=True, slip=0, scale_strength=scale_strength, domain_scale=ds, rot_distortion_max_angle=rot_angle, dtype=dtype)
            min_size, max_size = get_cell_size_min_max(domain)
            if max_size/min_size > 1000:
                LOG.warning("Extreme grid scaling %f from scaleStrength %s, abort.", max_size/min_size, scale_strength)
                continue
            #max_vel_raw = domain.getMaxVelocity(False).cpu().numpy()
            #block = domain.getBlock(0)
            #LOG.info("vel %s, t %s", block.velocity.shape, block.transform.shape)
            #max_vel_transformed = torch.max(PISOtorch.TransformVectors(domain.getBlock(0).velocity, domain.getBlock(0).transform, True))
            #LOG.info("raw %s, transformed %s", max_vel_raw, max_vel_transformed)
            if scale_strength is None: scale_strength=[0,0]

            if forcing_version==4:
                LOG.info("using velocity source forcing: %s", G)
                for block in domain.getBlocks():
                    block.setVelocitySource(G)
                domain.UpdateDomainData()
            
            sim.domain = domain
            sim.block_layout = layout
            sim.output_resampling_shape = resample_shape
            sim.time_step = ts
            sim.substeps = "ADAPTIVE" if ss==-1 else ss
            sim.log_dir = sub_dir
            sim.reset_step_counters()
            
            sim.run(iterations=it)
            
            i+=1
            
            if rot_angle is not None:
                if not domain.hasVertexCoordinates():
                    raise RuntimeError("vertex coordinates are required for resampling")
                resample_shape = [r*2 for r in res]
                vel_resampled = resampling.sample_multi_coords_to_uniform_grid(
                    data_list=[block.velocity for block in domain.getBlocks()],
                    coords_list=[block.vertexCoordinates for block in domain.getBlocks()],
                    out_shape=resample_shape, fill_max_steps=8)
                result_n = vel_resampled[0,0,:,res[0]//2].cpu().numpy()
                coords = [(i+0.5)*domain_scale/resample_shape[1] for i in range(resample_shape[1])]
            else:
                result_n = domain.getBlock(0).velocity[0,0,:,res[0]//2].cpu().numpy()
                LOG.info("vel profile mid:\n%s", domain.getBlock(0).velocity[...,res[0]//2])
            
                if(domain.getBlock(0).hasTransform):
                    coords = shapes.ortho_transform_to_coords(domain.getBlock(0).transform, 2)
                    coords = shapes.coords_to_center_coords(coords)[0,1,:,res[0]//2].cpu().numpy()
                else:
                    coords = [i+0.5 for i in range(res[0])]
            result_a = [poiseuille(i) for i in coords]
            #LOG.info("coords:\n%s\npoiseuille:\n%s", coords, result_a)
            
            ax.plot(coords, result_n, "x", label="$%d^{2}$"%r, zorder=r_idx+1)
            ax.legend()
            
            prof_fig.align_labels()
            prof_fig.tight_layout()
            prof_fig.savefig(os.path.join(run_dir, "convergence.{}".format(plot_format)))

            #results[s][r] = (coords, result_n, result_a)
    
    prof_fig.clf()
    plt.close(prof_fig)

def lid_driven_cavity_2D(run_dir, it, time_step=1, Re=1000, forcing=1, domain_scale=1,
                         resolutions=[16,[8,8],24], scalings=[1,(4,4,1)], lid_permutations=[(3,1)],
                         cmp_3D=False, rot_distortion_max_angle=None,
                         dp=True, STOP_FN=lambda: False,
                         plot_format="pdf"):
    # resolutions: list of resolution: always a square domain, so either a single integer or a list to make a split domain

    #u vel
    reference_coords_y = [ 0.0000,  0.0547,  0.0625,  0.0703,  0.1016,  0.1719,  0.2813,  0.4531,  0.5000,  0.6172,  0.7344,  0.8516,  0.9531,  0.9609,  0.9688,  0.9766,  1.0000]
    reference_vel_u = {
                    100 : [0.00000,-0.03717,-0.04192,-0.04775,-0.06434,-0.10150,-0.15662,-0.21090,-0.20581,-0.13641, 0.00332, 0.23151, 0.68717, 0.73722, 0.78871, 0.84123, 1.00000], #Re 100
                    1000 : [0.00000,-0.18109,-0.20196,-0.22220,-0.29730,-0.38289,-0.27805,-0.10648,-0.06080, 0.05702, 0.18719, 0.33304, 0.46604, 0.51117, 0.57492, 0.65928, 1.00000], #Re 1000
                    5000 : [0.00000,-0.41165,-0.42901,-0.43643,-0.40435,-0.33050,-0.22855,-0.07404,-0.03039, 0.08183, 0.20087, 0.33556, 0.46036, 0.45992, 0.46120, 0.48223, 1.00000], #Re 5000
                    10000 : [0.00000,-0.42735,-0.42537,-0.41657,-0.38000,-0.32709,-0.23186,-0.07540, 0.03111, 0.08344, 0.20673, 0.34635, 0.47804, 0.48070, 0.47783, 0.47221, 1.00000], #Re 10000
    }

    #v vel
    reference_coords_x = [ 0.0000,  0.0625,  0.0703,  0.0781,  0.0938,  0.1563,  0.2266,  0.2344,  0.5000,  0.8047,  0.8594,  0.9063,  0.9453,  0.9531,  0.9609,  0.9688,  1.0000]
    reference_vel_v = {
                    100 : [0.00000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507, 0.17527, 0.05454,-0.24533,-0.22445,-0.16914,-0.10313,-0.08864,-0.07391,-0.05906, 0.00000], #Re 100
                    1000 : [0.00000, 0.27485, 0.29012, 0.30353, 0.32627, 0.37095, 0.33075, 0.32235, 0.02526,-0.31966,-0.42665,-0.51550,-0.39188,-0.33714,-0.27669,-0.21388, 0.00000], #Re 1000
                    5000 : [0.00000, 0.42447, 0.43329, 0.43648, 0.42951, 0.35368, 0.28066, 0.27280, 0.00945,-0.30018,-0.36214,-0.41442,-0.52876,-0.55408,-0.55069,-0.49774, 0.00000], #Re 5000
                    10000 : [0.00000, 0.43983, 0.43733, 0.43124, 0.41487, 0.35070, 0.28003, 0.27224, 0.00831,-0.30719,-0.36737,-0.41496,-0.45863,-0.49099,-0.52987,-0.54302, 0.00000], #Re 10000
    }
    i = 0
    #it = 300 #100
    substeps = -2 #100 # -1 for adaptive, -2 for adaptive based on initialization
    corr = 2
    #time_step = 1 #/substeps
    #Re = 1000
    #forcing = 1 # boundary speed
    #domain_scale = 1
    #lid_permutation = (3,1) # default & reference data: (3,1) (lid at y+ with velocity in x+ direction)
    true_untransformed = False
    #cmp_3D = False
    rot_angle = rot_distortion_max_angle
    
    viscosity = forcing*domain_scale/Re #1e-2
    assert Re in reference_vel_u and Re in reference_vel_v, "No reference found for Re %s"%(Re,)
    #all_lid_permutations = [(b,v) for b in range(4) for v in range(4) if (b&6)!=(v&6)]
    #max_size_diff_rel = [(1, l) for l in all_lid_permutations] + [(8,l) for l in all_lid_permutations]
    #max_size_diff_rel = [(1,lid_permutation), (8,lid_permutation)] #((4,4,1), lid_permutation), 
    max_size_diff_rel = [(s,l) for l in lid_permutations for s in scalings]
    #dp = True
    vcv = 1
    
    is_non_ortho = (rot_angle is not None)
    sim = PISOtorch_simulation.Simulation(corrector_steps=corr, velocity_corrector="FD",
            advection_tol=None, pressure_tol=None,
            advect_non_ortho_steps=2 if is_non_ortho else 1, pressure_non_ortho_steps=4 if is_non_ortho else 1, non_orthogonal=is_non_ortho,
            log_dir=None, log_interval=1, norm_vel=True, save_domain_name="Lid2D",
            convergence_tol=1e-7 if dp else 1e-7, stop_fn=STOP_FN)
    
    dtype = torch.float64 if dp else torch.float32

    LOG = get_logger("Lid2D")
    #LOG.info("Lid permutations: %s", all_lid_permutations)
    LOG.info("Lid-Driven Cavity Test:\nts=%.03e, Re=%d, forcing=%.03e, domain_scale=%.03e -> viscosity=%.03e\nr=%s, s=%s, lid=%s, cmp3D=%s",
             time_step, Re, forcing, domain_scale, viscosity, resolutions, scalings, lid_permutations, cmp_3D)
    
    num_sizes = len(max_size_diff_rel)
    plot_size = 5
    plot_title = len(lid_permutations)>1
    prof_fig, prof_ax = plt.subplots(num_sizes,3 if cmp_3D else 2, figsize=(plot_size*3 if cmp_3D else plot_size*2, num_sizes*plot_size))
    if num_sizes==1:
        prof_ax = [prof_ax]
    #prof_fig.suptitle("Center velocity profiles, RE %d."%Re)
    
    num_res = len(resolutions)
    stream_fig, stream_ax = plt.subplots(len(max_size_diff_rel),num_res, figsize=(num_res*6, num_sizes*6))
    if num_res==1:
        if num_sizes==1:
            stream_ax = [[stream_ax]]
        else:
            stream_ax = [[ax] for ax in stream_ax]
    elif num_sizes==1:
        stream_ax = [stream_ax]
    #stream_fig.suptitle("Center velocity profiles.")
    
    for r_idx, resolution in enumerate(resolutions):
        if STOP_FN(): break
        for s_idx, (s, lid_permutation) in enumerate(max_size_diff_rel):
            if STOP_FN(): break
            
            if isinstance(s, (list, tuple)):
                if len(s)==2:
                    assert not cmp_3D
                    sx,sy = s
                elif len(s)==3:
                    sx,sy,sz = s
                else:
                    raise ValueError
            else:
                sx = s
                sy = s
                sz = s

            scale_name = 'scale %d-%d-%d'%(sx,sy,sz) if cmp_3D else 'scale %d-%d'%(sx,sy)

            if r_idx==0:
                if plot_title:
                    prof_ax[s_idx][0].set_title('u-vel along y, %s, lid %d-%d'%(scale_name,*lid_permutation))
                else:
                    prof_ax[s_idx][0].set_title('uniform' if sx==1 and sy==1 else 'refined')
                prof_ax[s_idx][0].set(xlabel='$u$', ylabel='y')
                prof_ax[s_idx][0].grid()
                #for r, result in results.items():
                #    prof_ax[s_idx][0].plot(result[2], result[1], label="r-%d"%r)
                prof_ax[s_idx][0].plot(reference_vel_u[Re], reference_coords_y, 'kx', label="reference", zorder=num_res*2)
                prof_ax[s_idx][0].legend()
                
                if plot_title:
                    prof_ax[s_idx][1].set_title('v-vel along x, %s, lid %d-%d'%(scale_name,*lid_permutation))
                else:
                    prof_ax[s_idx][1].set_title('uniform' if sx==1 and sy==1 else 'refined')
                prof_ax[s_idx][1].set(xlabel='x', ylabel='$v$')
                prof_ax[s_idx][1].grid()
                #for r, result in results.items():
                #    prof_ax[s_idx][1].plot(result[0], result[3], label="r-%d"%r)
                prof_ax[s_idx][1].plot(reference_coords_x, reference_vel_v[Re], 'kx', label="reference", zorder=num_res*2)
                prof_ax[s_idx][1].legend()

                if cmp_3D:
                    if plot_title:
                        prof_ax[s_idx][2].set_title('w-vel along z, %s, lid %d-%d'%(scale_name,*lid_permutation))
                    else:
                        prof_ax[s_idx][2].set_title('uniform' if sx==1 and sy==1 else 'refined')
                    prof_ax[s_idx][2].set(xlabel='z', ylabel='$w$')
                    prof_ax[s_idx][2].grid()
                
                prof_fig.align_labels()
                prof_fig.tight_layout()
                prof_fig.savefig(os.path.join(run_dir, "Re{:d}_profiles.{}".format(Re, plot_format)))
        
            #results = {}
            if STOP_FN(): break
            res = [resolution]*2
            split_block = False
            if isinstance(resolution, (list, tuple)):
                r = sum(resolution)
                split_block = True
            else:
                r = resolution
            sclx = sx ** (1/(r//2 - 1))
            scly = sy ** (1/(r//2 - 1))
            scale_strength = [sclx,scly]
            ds = domain_scale
            if ds is not None:
                ds = [ds/r for _ in res]

            if true_untransformed and sx==1 and sy==1:
                scale_strength = None
                top_vel = r*forcing
                visc = top_vel*r/Re
                LOG.info("True untransformed: r=%d, u=%.02e, visc=%.02e", r, top_vel, visc)
                norm_vel = True
            else:
                top_vel = forcing
                visc = viscosity
                norm_vel = False

            #LOG.info("TEST: scale_strength=%s", scale_strength)
            domain, layout, resample_shape = make1BlockSetup2DExpScaling(*res, vel=[0,0], vel_blob=False, viscosity=visc, connect=None, closed_bounds=True, top_vel=top_vel, slip=0,
                                                    scale_strength=scale_strength, domain_scale=ds, invert_transform=False, lid_permutation=lid_permutation, dtype=dtype,
                                                    rot_distortion_max_angle=rot_angle)
            
            #vel, _, _, cell_coordinates = combine_blocks_2D_velocity(domain.getBlocks(), layout)
            
            if cmp_3D:
                sclz = sz ** (1/(r//2 - 1))
                domain_3D = make1BlockLidCavity3DExpScaling(*res, r, viscosity=viscosity, lid_vel=top_vel, periodic_z=True,
                    scale_strength=scale_strength+[sclz], domain_scale=domain_scale, aspect_x=1, aspect_z=1, lid_permutation=lid_permutation, dtype=dtype)
            

            res = [r]*2 # in case r was split setup
            

            
            lid_bound_axis = lid_permutation[0]>>1
            lid_flow_axis = lid_permutation[1]>>1
            
            if substeps==-2:
                max_vel = domain.getMaxVelocity(True, True).cpu().numpy()
                max_time_step = 0.8/max_vel
                if max_time_step>=time_step:
                    ss = 1
                    ts = time_step
                else:
                    ss = int(np.ceil(time_step / max_time_step))
                    ts = time_step / ss
                LOG.info("initial max vel with transform and bounds %.02e -> time step %.02e, substeps %d", max_vel, ts, ss)
            else:
                ts = time_step
                ss = substeps
                LOG.info("time step %.02e, substeps %d", ts, ss)
            #max_vel_transformed = torch.max(PISOtorch.TransformVectors(domain.getBlock(0).velocity, domain.getBlock(0).transform, True))
            #LOG.info("raw %s, transformed %s", max_vel_raw, max_vel_transformed)
            #if scale_strength is None: scale_strength=[0,0]
            
            sim.domain = domain
            sim.block_layout = layout
            sim.output_resampling_shape = resample_shape if rot_angle is not None else None
            sim.time_step = ts
            sim.substeps = "ADAPTIVE" if ss==-1 else ss
            sim.log_dir = os.path.join(run_dir, "{:02d}_scale-{}-{}_lid-{}-{}_r{}{}".format(i, sx, sy, *lid_permutation, r, "_split" if split_block else ""))
            sim.reset_step_counters()
            
            i+=1
            
            sim.run(iterations=it)

            if cmp_3D:
                sim.domain = domain_3D
                sim.save_domain_name = "Lid3D_cmp"
                sim.log_dir = os.path.join(run_dir, "{:02d}_scale-{}-{}_lid-{}-{}_r{}_3D".format(i, sx, sy, *lid_permutation, r))
                sim.reset_step_counters()
                
                i+=1
                
                sim.run(iterations=it)
                
                sim.save_domain_name = "Lid2D"

            if rot_angle is None:
                vel, _, _, cell_coordinates = combine_blocks_2D_velocity(domain.getBlocks(), layout) #domain.getBlock(0).velocity
            else:
                if not domain.hasVertexCoordinates():
                    raise RuntimeError("vertex coordinates are required for resampling")
                resample_shape = res #[r*2 for r in res]
                vel = resampling.sample_multi_coords_to_uniform_grid(
                    data_list=[block.velocity for block in domain.getBlocks()],
                    coords_list=[block.vertexCoordinates for block in domain.getBlocks()],
                    out_shape=resample_shape)
                cell_coordinates = None
            
            # vel along geometric center lines. interpolate to faces if necessary
            if res[0]%2==0:
                vel_y = ((vel[...,:,res[0]//2] + vel[...,:,res[0]//2-1])*0.5)#.cpu().numpy()
            else:
                vel_y = vel[...,:,res[0]//2]#.cpu().numpy()
            if res[1]%2==0:
                vel_x = ((vel[...,res[1]//2,:] + vel[...,res[1]//2-1,:])*0.5)#.cpu().numpy()
            else:
                vel_x = vel[...,res[1]//2,:]#.cpu().numpy()
            vels = [vel_x, vel_y]
            #LOG.info("u profile mid:\n%s", result_u)
            #LOG.info("v profile mid:\n%s", result_v)

            if cmp_3D:
                assert (r%2)==0
                x=r
                y=r
                z=r
                lid_tangent_axis = 3 - (lid_bound_axis + lid_flow_axis)
                vel = domain_3D.getBlock(0).velocity
                vel_3D_x = (vel[...,z//2,y//2,:] + vel[...,z//2,y//2-1,:] + vel[...,z//2-1,y//2,:] + vel[...,z//2-1,y//2-1,:])*0.25
                vel_3D_y = (vel[...,z//2,:,x//2] + vel[...,z//2,:,x//2-1] + vel[...,z//2-1,:,x//2] + vel[...,z//2-1,:,x//2-1])*0.25
                vel_3D_z = (vel[...,:,y//2,x//2] + vel[...,:,y//2,x//2-1] + vel[...,:,y//2-1,x//2] + vel[...,:,y//2-1,x//2-1])*0.25
                vels_3D = [vel_3D_x, vel_3D_y, vel_3D_z]
            
            # result_u: originally u-vel along y-axis, with permutation (default: bound=y+,flow=+x) flow-vel along bound-axis
            result_u = vels[lid_bound_axis][0,lid_flow_axis]
            if (lid_permutation[1]&1)==0: #invert velocities if velocity's axis is inverted compared to reference
                result_u = result_u *-1
            result_u = result_u.cpu().numpy()
            result_v = vels[lid_flow_axis][0,lid_bound_axis]
            if (lid_permutation[0]&1)==0: #invert velocities if velocity's axis is inverted compared to reference
                result_v = result_v *-1
            result_v = result_v.cpu().numpy()

            if cmp_3D:
                result_3D_u = vels_3D[lid_bound_axis][0,lid_flow_axis]
                if (lid_permutation[1]&1)==0:
                    result_3D_u = result_3D_u *-1
                result_3D_u = result_3D_u.cpu().numpy()
                
                result_3D_v = vels_3D[lid_flow_axis][0,lid_bound_axis]
                if (lid_permutation[0]&1)==0:
                    result_3D_v = result_3D_v *-1
                result_3D_v = result_3D_v.cpu().numpy()

                result_3D_w = vels_3D[lid_tangent_axis][0,lid_tangent_axis]
                result_3D_w = result_3D_w.cpu().numpy()

            if norm_vel:
                result_u = result_u / r
                result_v = result_v / r
                if cmp_3D:
                    result_3D_u = result_3D_u / r
                    result_3D_v = result_3D_v / r
                    result_3D_w = result_3D_w / r
            
            if cell_coordinates is not None: #(sx!=1 or sy!=1) and domain.getBlock(0).hasTransform:
                coords = cell_coordinates
                coords_x = coords[0,0,res[1]//2,:].cpu().numpy()
                coords_y = coords[0,1,:,res[0]//2].cpu().numpy()
            else:
                coords_x = np.asarray([(i+0.5)*domain_scale/res[0] for i in range(res[0])])
                coords_y = np.asarray([(i+0.5)*domain_scale/res[1] for i in range(res[1])])
            
            if cmp_3D:
                coords_3D = cell_coordinates
                coords_3D_z = coords_3D[0,2,:,y//2,x//2].cpu().numpy()
            
            coords = [coords_x, coords_y]
            coords_x = coords[lid_flow_axis]
            if (lid_permutation[1]&1)==0:
                coords_x = domain_scale - coords_x
            coords_y = coords[lid_bound_axis]
            if (lid_permutation[0]&1)==0:
                coords_y = domain_scale - coords_y

            prof_ax[s_idx][0].plot(result_u, coords_y, "1" if split_block else ".-", label="$%d^{2}$"%r, zorder=r_idx*2)
            if cmp_3D:
                prof_ax[s_idx][0].plot(result_3D_u, coords_y, "1", label="$%d^{2}$ 3D"%r, zorder=r_idx*2+1)
            prof_ax[s_idx][0].legend()
            prof_ax[s_idx][1].plot(coords_x, result_v, "1" if split_block else ".-", label="$%d^{2}$"%r, zorder=r_idx*2)
            if cmp_3D:
                prof_ax[s_idx][1].plot(coords_x, result_3D_v, "1", label="$%d^{2}$ 3D"%r, zorder=r_idx*2+1)
            prof_ax[s_idx][1].legend()
            if cmp_3D:
                prof_ax[s_idx][2].plot(coords_3D_z, result_3D_w, "1", label="$%d^{2}$ 3D"%r, zorder=r_idx*2+1)
                prof_ax[s_idx][2].legend()
            
            prof_fig.align_labels()
            prof_fig.tight_layout()
            prof_fig.savefig(os.path.join(run_dir, "Re{:d}_profiles.{}".format(Re, plot_format)))
            #results[r] = (coords_x, coords_y, result_u, result_v)
            
            if sx==1 and sy==1: #would need resampling for grid stretching
                u = vel[0,0,:,:].cpu().numpy()
                v = vel[0,1,:,:].cpu().numpy()
                if norm_vel:
                    u = u / r
                    v = v / r
                mag = np.sqrt(u**2 + v**2)
                ax = stream_ax[s_idx][r_idx]
                ax.set_title('resolution %d, scale %d'%(r,s))
                #ax.set_aspect("equal", "box")
                ax.axis("equal")
                ax.set(xlabel='x', ylabel='y', xlim=(0,domain_scale), ylim=(0,domain_scale))
                ax.streamplot(*coords, u, v, color=mag, cmap='viridis')
                #ax.label_outer()
                #plt.savefig(os.path.join(run_dir, "Re{:d}_s{:d}_r{:d}_stream.svg".format(Re, s, r)))
                #plt.clf()
            stream_fig.savefig(os.path.join(run_dir, "Re{:d}_stream.{}".format(Re, plot_format)))
                
    
    #prof_fig.savefig(os.path.join(run_dir, "Re{:d}_profiles.svg".format(Re)))
    prof_fig.clf()
    plt.close(prof_fig)
    
    #stream_fig.savefig(os.path.join(run_dir, "Re{:d}_stream.svg".format(Re)))
    stream_fig.clf()
    plt.close(stream_fig)

def lid_driven_cavity_3D(run_dir, it, time_step=1, Re=1000, domain_scale=1, aspect_x=1, aspect_z=1, periodic_z=False,
                         resolutions=[12,24], scalings=[1,(4,4,1)], lid_permutations=[(0,3)],
                         dp=True, STOP_FN=lambda: False,
                         plot_format='pdf'):
    # Lid driven cavity 3D
    # reference data from paper "Accurate three-dimensional lid-driven cavity flow" (Albensoeder, Kuhlmann; 2004)
    #v vel (Table 5)
    reference_coords_x_T5 =      list(reversed([ 0.5000,  0.4453,  0.4375,  0.4297,  0.3984,  0.3281,  0.2187,  0.0469,  0.0000, -0.1172, -0.2344, -0.3516, -0.4531, -0.4609, -0.4688, -0.4766, -0.5000]))
    reference_coords_x = {
        (1000, 1, 1, False) : reference_coords_x_T5,
        (1000, 1, 2, False) : reference_coords_x_T5,
        (1000, 1, 3, False) : reference_coords_x_T5,
        (1000, 1, 1,  True) : reference_coords_x_T5,
        #Table 7
        (1000, 2, 1, False) : [  -1.00,   -0.97,   -0.95,   -0.90,   -0.80,   -0.50,   -0.25,   -0.10,    0.00,    0.10,    0.20,    0.50,    0.75,    1.00],
    }
    # for 96x96x96, Re 1000. Key: (aspect_x, aspect_z, periodic_z)
    reference_vel_v = { # normalized with /Re
        (1000, 1, 1, False) : list(reversed([0.00000,-0.20623,-0.22283,-0.23696,-0.27293,-0.25160,-0.10999,-0.00612, 0.00802, 0.03905, 0.07334, 0.12183, 0.33171, 0.39821, 0.48443, 0.58964, 1.00000])),
        #(1000, 1, 2, False) : [0.00000, None, 1.00000],
        (1000, 1, 3, False) : list(reversed([0.00000,-0.23000,-0.25008,-0.26813,-0.32232,-0.32475,-0.17757,-0.05571,-0.03258, 0.02709, 0.11031, 0.23687, 0.41972, 0.47304, 0.54535, 0.63664, 1.00000])),
        (1000, 1, 1,  True) : list(reversed([0.00000,-0.27863,-0.29767,-0.31332,-0.35181,-0.34355,-0.22776,-0.08615,-0.04746, 0.05504, 0.17537, 0.28501, 0.38239, 0.43104, 0.50294, 0.59926, 1.00000])),
        (1000, 1, 1,  True, 1/6) : list(reversed([0.00000,-0.12057,-0.13269,-0.14442,-0.19376,-0.32432,-0.26071,-0.08969,-0.04841, 0.05888, 0.17118, 0.29694, 0.45325, 0.50990, 0.58225, 0.66926, 1.00000])),
        #Table 7
        (1000, 2, 1, False) : [1.00000, 0.47072, 0.27384, 0.13025, 0.09232, 0.03804,-0.04294,-0.18427,-0.25051,-0.16515,-0.05775, 0.00825, 0.03429, 0.00000],
    }
    #u vel (Table 6)
    reference_coords_y_T6 =      [-0.5000, -0.4375, -0.4297, -0.4219, -0.4062, -0.3437, -0.2734, -0.2656,  0.0000,  0.3047,  0.3594,  0.4063,  0.4453,  0.4531,  0.4609,  0.4688,  0.5000]
    reference_coords_y = { # normalized with /Re
        (1000, 1, 1, False) : reference_coords_y_T6,
        (1000, 1, 2, False) : reference_coords_y_T6,
        (1000, 1, 3, False) : reference_coords_y_T6,
        (1000, 1, 1,  True) : reference_coords_y_T6,
        #Table 7
        (1000, 2, 1, False) : [  -0.50,   -0.46,   -0.41,   -0.36,   -0.30,   -0.16,   -0.07,    0.00,    0.10,    0.20,    0.30,    0.40,    0.45,    0.50],
    }
    # for 96x96x96, Re 1000. Key: (aspect_x, aspect_z, periodic_z)
    reference_vel_u = { # normalized with /Re
        (1000, 1, 1, False) : [0.00000,-0.21738,-0.22746,-0.23503,-0.24407,-0.22924,-0.17580,-0.16987,-0.03674, 0.15223, 0.31117, 0.43423, 0.33511, 0.29032, 0.24095, 0.18864, 0.00000],
        #(1000, 1, 2, False) : [0.00000, None, 0.00000],
        (1000, 1, 3, False) : [0.00000,-0.27236,-0.28596,-0.29691,-0.31255,-0.31204,-0.23636,-0.22639,-0.00969, 0.24206, 0.37527, 0.48932, 0.38044, 0.32944, 0.27272, 0.21239, 0.00000],
        (1000, 1, 1,  True) : [0.00000,-0.29706,-0.30039,-0.30185,-0.30282,-0.30586,-0.29272,-0.28899,-0.03098, 0.30973, 0.44886, 0.52418, 0.38058, 0.32738, 0.26973, 0.20935, 0.00000],
        (1000, 1, 1,  True, 1/6) : [0.00000,-0.19808,-0.20820,-0.21862,-0.23373,-0.27984,-0.32527,-0.31995,-0.02877, 0.26206, 0.34339, 0.44938, 0.37309, 0.32565, 0.27100, 0.21161, 0.00000],
        #Table 7
        (1000, 2, 1, False) : [0.00000,-0.03392,-0.04791,-0.05156,-0.04719,-0.01138, 0.02631, 0.06211, 0.11097, 0.11548, 0.05200,-0.00777,-0.01409, 0.00000],
        (1000, 2, 1, False, -0.5) : [0.00000,-0.14389,-0.22661,-0.21846,-0.18202,-0.11861,-0.08614,-0.06097,-0.02318, 0.02208, 0.12499, 0.42828, 0.32420, 0.00000], #x=-0.5
        (1000, 2, 1, False,  0.5) : [0.00000, 0.04155, 0.07601, 0.07827, 0.05326,-0.00039,-0.01566,-0.02229,-0.02791,-0.03066,-0.02995,-0.02218,-0.01342, 0.00000], #x=+0.5
    }
    
    #it = 100 #100
    #Re = 1000
    #domain_scale = 1
    #aspect_x = 1
    # paper has fix y=1
    #aspect_z = 1
    #periodic_z = False
    lid_permutation = (0,3) # default & reference: (0,3) #(boundary,flowDir): 0=-x, 1=+x, 2=-y
    #if aspect_z==1 and aspect_x==1:
    #    resolutions = [[12,12,8], [24,24,16], [34,34,26], [48,48,32], [64,64,48], ] #x,y,z (int>2)
    #    resolutions = [[12,12,8], [24,24,16], [48,48,32], ] #x,y,z (int>2)
    #    resolutions = [[12,12,12], [24,24,24], [48,48,48], ] #x,y,z (int>2)
    #else:
        #resolutions = [[12,12,12], [24,24,24], [34,34,34], [48,48,48], [64,64,64], ] #x,y,z (int>2)
    #    resolutions = [[12,12,8], [12,12,12], [24,24,16], [24,24,24], [48,48,32], [48,48,48], ] #x,y,z (int>2)
    #resolutions = [[12,12,8], [12,12,9], [12,12,12], [48,48,32], [48,48,33], [48,48,48],] #x,y,z (int>2)
    #max_size_diff_rel = [[1,1,1], [2,2,1], [2,2,2], [4,4,1], ] #x,y,z (float>=1)
   # max_size_diff_rel = [[1,1,1], [2,2,1], [2,2,2]] #x,y,z (float>=1)
    
    #all_lid_permutations = [(b,v) for b in range(6) for v in range(6) if (b&6)!=(v&6)]
    #max_size_diff_rel = [(s,l) for s in max_size_diff_rel for l in all_lid_permutations] #x,y,z (float>=1)
    max_size_diff_rel = [(s,l) for l in lid_permutations for s in scalings]
    
    # fixed:
    lid_vel = Re # boundary speed
    time_step = 1/lid_vel #/substeps
    substeps = -2 #100 # -1 for adaptive, -2 for adaptive based on initialization
    viscosity = domain_scale
    #dp = True #double precision?
    #vcv = 1
    corr = 2 #number of pressure corrector steps
    dtype = torch.float64 if dp else torch.float32
    i = 0
    ref_key = (Re, aspect_x, aspect_z, periodic_z)
    assert ref_key in reference_vel_u
    assert ref_key in reference_coords_x
    assert ref_key in reference_vel_v
    assert ref_key in reference_coords_y

    LOG = get_logger("Lid3D")
    LOG.info("lid velocity: %.02e, time_step: %.02e (adaptive substeps), viscosity: %.02e, resolution %s, scalings %s, corrector steps: %d",
        lid_vel, time_step, viscosity, resolutions, max_size_diff_rel, corr)
    #LOG.info("lid permutations: %s", all_lid_permutations)
    
    num_res = len(resolutions)
    num_sizes = len(max_size_diff_rel)
    num_figs = 3
    u_extra = False
    z_extra = False
    if (*ref_key, -0.5) in reference_vel_u and (*ref_key,  0.5) in reference_vel_u and lid_permutation==(0,3):
        num_figs+=2
        u_extra = True
    if (*ref_key,  1/6) in reference_vel_u and (*ref_key,  1/6) in reference_vel_v:
        num_figs+=2
        z_extra = True
    plot_size = 5
    prof_fig, prof_ax = plt.subplots(num_sizes,num_figs, figsize=(num_figs*plot_size, num_sizes*plot_size))
    if num_sizes==1:
        prof_ax = [prof_ax]
    #prof_fig.suptitle("Center velocity profiles, RE %d."%Re)

    def make_res_label(x,y,z):
        if x==y and x==z:
            return "$%d^3$"%(x,)
        else:
            return "%d \\times %d \\times %d"%(x,y,z)
    
    use_simple_title = len(lid_permutations)==1 and len(scalings)==2 and scalings[0]==1 and not all (s==1 for s in scalings[1])
    def is_refined(scale):
        return not all(s==1 for s in scale)
    def get_refined_title(scale):
        return "refined" if is_refined(scale) else "uniform"

    
    sim = PISOtorch_simulation.Simulation(corrector_steps=corr, velocity_corrector="FD",
            advection_tol=None, pressure_tol=None,
            advect_non_ortho_steps=1, pressure_non_ortho_steps=1, non_orthogonal=False,
            log_dir=None, log_interval=1, norm_vel=True, save_domain_name="Lid3D",
            stop_fn=STOP_FN)
    
    
    for r_idx, resolution in enumerate(resolutions):
        if STOP_FN(): break
        for s_idx, (s, lid_permutation) in enumerate(max_size_diff_rel):
            if STOP_FN(): break
            
            if isinstance(resolution, (list, tuple)):
                if not len(resolution)==3:
                    raise ValueError("resolution must be scalar or 3D")
            else:
                resolution = [resolution]*3
            x,y,z = resolution

            res_label = make_res_label(x,y,z)
            
            if isinstance(s, (list, tuple)):
                if len(s)==3:
                    pass
                elif len(s)==2:
                    s = list(s) + [1]
                else:
                    raise ValueError("scaling must be scalar, 2D or 3D")
            else:
                s = [s,s,1]
            
            scale_strength = [_ ** (1/(r//2 - 1)) for r, _ in zip(resolution, s)]
            
            if u_extra:
                ax_u_l = prof_ax[s_idx][0]
                ax_u = prof_ax[s_idx][1]
                ax_u_u = prof_ax[s_idx][2]
                ax_v = prof_ax[s_idx][3]
                ax_w = prof_ax[s_idx][4]
                plot_u_extra = all(_==1 for _ in s) #and x%4==0 and z%4==0
            elif z_extra:
                ax_u = prof_ax[s_idx][0]
                ax_u_z = prof_ax[s_idx][1]
                ax_v = prof_ax[s_idx][2]
                ax_v_z = prof_ax[s_idx][3]
                ax_w = prof_ax[s_idx][4]
                plot_u_extra = False
            else:
                ax_u = prof_ax[s_idx][0]
                ax_v = prof_ax[s_idx][1]
                ax_w = prof_ax[s_idx][2]
                plot_u_extra = False
            if r_idx==0:
                if use_simple_title:
                    ax_u.set_title(get_refined_title(s))
                else:
                    ax_u.set_title('$u$ along y-axis, scale %d-%d-%d, lid %d-%d'%(*s,*lid_permutation))
                ax_u.set(xlabel='$u/$Re', ylabel='y')
                ax_u.grid()
                ax_u.plot(reference_vel_u[ref_key], reference_coords_y[ref_key], 'kx', label="reference", zorder=num_res)
                ax_u.legend()
                if u_extra:
                    ax_u_l.set_title('$u$ along y-axis at $x=-0.5$, scale %d-%d-%d, lid %d-%d'%(*s,*lid_permutation))
                    ax_u_l.set(xlabel='$u/$Re', ylabel='y')
                    ax_u_l.grid()
                    ax_u_l.plot(reference_vel_u[(*ref_key, -0.5)], reference_coords_y[ref_key], 'kx', label="reference", zorder=num_res)
                    ax_u_l.legend()

                    ax_u_u.set_title('$u$ along y-axis at $x=+0.5$, scale %d-%d-%d, lid %d-%d'%(*s,*lid_permutation))
                    ax_u_u.set(xlabel='$u/$Re', ylabel='y')
                    ax_u_u.grid()
                    ax_u_u.plot(reference_vel_u[(*ref_key,  0.5)], reference_coords_y[ref_key], 'kx', label="reference", zorder=num_res)
                    ax_u_u.legend()
                elif z_extra:
                    ax_u_z.set_title('$u$ along y-axis (ref:$z=1/6$), scale %d-%d-%d, lid %d-%d'%(*s,*lid_permutation))
                    ax_u_z.set(xlabel='$u/$Re', ylabel='y')
                    ax_u_z.grid()
                    ax_u_z.plot(reference_vel_u[(*ref_key, 1/6)], reference_coords_y[ref_key], 'kx', label="reference", zorder=num_res)
                    ax_u_z.legend()

                    ax_v_z.set_title('$v$ along x-axis (ref:$z=1/6$), scale %d-%d-%d, lid %d-%d'%(*s,*lid_permutation))
                    ax_v_z.set(xlabel='x', ylabel='$v/$Re')
                    ax_v_z.grid()
                    ax_v_z.plot(reference_coords_x[ref_key], reference_vel_v[(*ref_key, 1/6)], 'kx', label="reference", zorder=num_res)
                    ax_v_z.legend()
                
                if use_simple_title:
                    ax_v.set_title(get_refined_title(s))
                else:
                    ax_v.set_title('$v$ along x-axis, scale %d-%d-%d, lid %d-%d'%(*s,*lid_permutation))
                ax_v.set(xlabel='x', ylabel='$v/$Re')
                ax_v.grid()
                ax_v.plot(reference_coords_x[ref_key], reference_vel_v[ref_key], 'kx', label="reference", zorder=num_res)
                ax_v.legend()
                
                if use_simple_title:
                    ax_w.set_title(get_refined_title(s))
                else:
                    ax_w.set_title('$w$ along z-axis, scale %d-%d-%d, lid %d-%d'%(*s,*lid_permutation))
                ax_w.set(xlabel='z (normalized)', ylabel='$w$')
                ax_w.grid()
                
                prof_fig.align_labels()
                prof_fig.tight_layout()
                prof_fig.savefig(os.path.join(run_dir, "vel_profiles.{}".format(plot_format)))
            
            domain = make1BlockLidCavity3DExpScaling(*resolution, viscosity=viscosity, lid_vel=lid_vel, periodic_z=periodic_z,
                scale_strength=scale_strength, domain_scale=domain_scale, aspect_x=aspect_x, aspect_z=aspect_z, lid_permutation=lid_permutation, dtype=dtype)
            
            lid_bound_axis = lid_permutation[0]>>1
            lid_flow_axis = lid_permutation[1]>>1
            lid_tangent_axis = 3 - (lid_bound_axis + lid_flow_axis)
            
            if substeps==-2:
                max_vel = domain.getMaxVelocity(True, True).cpu().numpy()
                max_time_step = 0.8/max_vel
                if max_time_step>=time_step:
                    ss = 1
                    ts = time_step
                else:
                    ss = int(np.ceil(time_step / max_time_step))
                    ts = time_step / ss
                LOG.info("initial max vel with transform and bounds %.02e -> time step %.02e, substeps %d", max_vel, ts, ss)
            else:
                ts = time_step
                ss = substeps
                LOG.info("time step %.02e, substeps %d", ts, ss)

            def log_plot(domain, out_dir, out_it, **kwargs):
                num_figs = 3
                prof_fig, prof_ax = plt.subplots(1,num_figs, figsize=(num_figs*plot_size, plot_size))
                ax_u = prof_ax[0]
                ax_v = prof_ax[1]
                ax_w = prof_ax[2]

                if use_simple_title:
                    ax_u.set_title(get_refined_title(s))
                else:
                    ax_u.set_title('$u$ along y-axis, scale %d-%d-%d, lid %d-%d'%(*s,*lid_permutation))
                ax_u.set(xlabel='$u/$Re', ylabel='y')
                ax_u.grid()
                ax_u.plot(reference_vel_u[ref_key], reference_coords_y[ref_key], 'kx', label="reference", zorder=num_res)
                ax_u.legend()
                
                if use_simple_title:
                    ax_v.set_title(get_refined_title(s))
                else:
                    ax_v.set_title('$v$ along x-axis, scale %d-%d-%d, lid %d-%d'%(*s,*lid_permutation))
                ax_v.set(xlabel='x', ylabel='$v/$Re')
                ax_v.grid()
                ax_v.plot(reference_coords_x[ref_key], reference_vel_v[ref_key], 'kx', label="reference", zorder=num_res)
                ax_v.legend()
                
                if use_simple_title:
                    ax_w.set_title(get_refined_title(s))
                else:
                    ax_w.set_title('$w$ along z-axis, scale %d-%d-%d, lid %d-%d'%(*s,*lid_permutation))
                ax_w.set(xlabel='z (normalized)', ylabel='$w$')
                ax_w.grid()

                vel = domain.getBlock(0).velocity
                vel_x = (vel[...,z//2,y//2,:] + vel[...,z//2,y//2-1,:] + vel[...,z//2-1,y//2,:] + vel[...,z//2-1,y//2-1,:])*0.25
                vel_y = (vel[...,z//2,:,x//2] + vel[...,z//2,:,x//2-1] + vel[...,z//2-1,:,x//2] + vel[...,z//2-1,:,x//2-1])*0.25
                vel_z = (vel[...,:,y//2,x//2] + vel[...,:,y//2,x//2-1] + vel[...,:,y//2-1,x//2] + vel[...,:,y//2-1,x//2-1])*0.25
                vels = [vel_x, vel_y, vel_z]
                result_u = (vels[lid_flow_axis][0,lid_bound_axis] / Re)
                if (lid_permutation[0]&1)==1: #invert velocities if velocity's axis is inverted compared to reference
                    result_u = result_u *-1
                #if (lid_permutation[1]&1)==0: #invert order if slice's axis is inverted compared to reference. Invert coords instead.
                #    result_u = result_u[::-1]
                result_u = result_u.cpu().numpy()
                result_v = (vels[lid_bound_axis][0,lid_flow_axis] / Re)
                if (lid_permutation[1]&1)==0: #invert velocities if velocity's axis is inverted compared to reference
                    result_v = result_v *-1
                #if (lid_permutation[0]&1)==1: #invert order if slice's axis is inverted compared to reference. Invert coords instead.
                #    result_v = result_v[::-1]
                result_v = result_v.cpu().numpy()
                # TODO: might have to invert w/z on certain combinations of other inversions?
                result_w = vels[lid_tangent_axis][0,lid_tangent_axis]
                #result_u = (vel_y[0,0] / Re).cpu().numpy()
                #result_v = (vel_x[0,1] / Re).cpu().numpy()
                result_w = result_w.cpu().numpy()

                coords = shapes.ortho_transform_to_coords(domain.getBlock(0).transform, 3)
                coords = shapes.coords_to_center_coords(coords)
                coords_x = coords[0,0,z//2,y//2,:].cpu().numpy()
                coords_y = coords[0,1,z//2,:,x//2].cpu().numpy()
                coords_z = coords[0,2,:,y//2,x//2].cpu().numpy()
                coords = [coords_x, coords_y, coords_z]
                coords_x = coords[lid_bound_axis] - (domain_scale*aspect_x)/2
                if (lid_permutation[0]&1)==1:
                    coords_x = coords_x*-1
                coords_y = coords[lid_flow_axis] - (domain_scale)/2
                if (lid_permutation[1]&1)==0:
                    coords_y = coords_y*-1
                coords_z = coords[lid_tangent_axis]/aspect_z - 0.5
                ax_u.plot(result_u, coords_y, ".-", label=res_label, zorder=r_idx)
                ax_u.legend()
                ax_v.plot(coords_x, result_v, ".-", label=res_label, zorder=r_idx)
                ax_v.legend()
                ax_w.plot(coords_z, result_w, ".-", label=res_label, zorder=r_idx)
                ax_w.legend()

                prof_fig.align_labels()
                prof_fig.tight_layout()
                prof_fig.savefig(os.path.join(out_dir, "profiles_{:04d}.{}".format(out_it,"png")))
                prof_fig.clf()
                plt.close(prof_fig)

            
            convergence_tol = (1e-7 if dp else 1e-4) * ts * Re # from reference paper: max(abs(u(t) - u(t-ts))) / (ts*Re) < 1e-7 ==> max(abs(u(t) - u(t-ts))) < 1e-7*ts*Re
            LOG.info("Convergence tolerance: %.02e", convergence_tol)
            sim_dir = "{:02d}_scale-{}-{}-{}_lid-{}-{}_r-{}-{}-{}".format(i, *s, *lid_permutation, *resolution)

            sim.domain = domain
            sim.time_step = ts
            sim.substeps = "ADAPTIVE" if ss==-1 else ss
            sim.convergence_tol = convergence_tol
            sim.log_dir = os.path.join(run_dir, sim_dir)
            sim.log_fn = log_plot
            sim.reset_step_counters()

            #PISOtorch_sim.runSim(domain=domain, iterations=it, time_step=ts, substeps=ss, corrector_steps=corr,
            #                     convergence_tol=convergence_tol, log_dir=os.path.join(run_dir, sim_dir), log_fn=log_plot,
            #                     log_interval=1, norm_vel=True, save_domain_name="Lid3D", STOP_FN=STOP_FN)
            i+=1

            sim.run(iterations=it)
            
            vel = domain.getBlock(0).velocity
            assert x%2==0
            assert y%2==0
            assert z%2==0
            vel_x = (vel[...,z//2,y//2,:] + vel[...,z//2,y//2-1,:] + vel[...,z//2-1,y//2,:] + vel[...,z//2-1,y//2-1,:])*0.25
            vel_y = (vel[...,z//2,:,x//2] + vel[...,z//2,:,x//2-1] + vel[...,z//2-1,:,x//2] + vel[...,z//2-1,:,x//2-1])*0.25
            vel_z = (vel[...,:,y//2,x//2] + vel[...,:,y//2,x//2-1] + vel[...,:,y//2-1,x//2] + vel[...,:,y//2-1,x//2-1])*0.25
            vel_x_v_min, coord_x_v_min = torch.min(vel_x[0,1], dim=-1)
            vel_y_u_max, coord_y_u_max = torch.max(vel_y[0,0], dim=-1)
            vel_y_u_min, coord_y_u_min = torch.min(vel_y[0,0], dim=-1)
            loc_x_v_min = (coord_x_v_min.cpu().numpy() / x - 0.5) * domain_scale * aspect_x
            loc_y_u_max = (coord_y_u_max.cpu().numpy() / y - 0.5) * domain_scale
            loc_y_u_min = (coord_y_u_min.cpu().numpy() / y - 0.5) * domain_scale
            LOG.info("vel min/max stats:\ny_u_max %.02f at %.04f (%d)\ny_u_min %.02f at %.04f (%d)\nx_v_min %.02f at %.04f (%d)",
                vel_y_u_max.cpu().numpy(), loc_y_u_max, coord_y_u_max.cpu().numpy(),
                vel_y_u_min.cpu().numpy(), loc_y_u_min, coord_y_u_min.cpu().numpy(),
                vel_x_v_min.cpu().numpy(), loc_x_v_min, coord_x_v_min.cpu().numpy())
            
            vels = [vel_x, vel_y, vel_z]
            # norm with Re
            result_u = (vels[lid_flow_axis][0,lid_bound_axis] / Re)
            if (lid_permutation[0]&1)==1: #invert velocities if velocity's axis is inverted compared to reference
                result_u = result_u *-1
            result_u = result_u.cpu().numpy()
            result_v = (vels[lid_bound_axis][0,lid_flow_axis] / Re)
            if (lid_permutation[1]&1)==0: #invert velocities if velocity's axis is inverted compared to reference
                result_v = result_v *-1
            result_v = result_v.cpu().numpy()
            # TODO: might have to invert w/z on certain combinations of other inversions?
            result_w = vels[lid_tangent_axis][0,lid_tangent_axis]
            result_w = result_w.cpu().numpy()
            # result_u = (vel_y[0,0] / Re).cpu().numpy()
            # result_v = (vel_x[0,1] / Re).cpu().numpy()
            # result_w = vel_z[0,2].cpu().numpy()
            
            if plot_u_extra:
                if x%4==0:
                    result_u_l = (vel[0,0,z//2,:,x//4] + vel[0,0,z//2,:,x//4-1] + vel[0,0,z//2-1,:,x//4] + vel[0,0,z//2-1,:,x//4-1])*0.25
                    result_u_u = (vel[0,0,z//2,:,x//4*3] + vel[0,0,z//2,:,x//4*3-1] + vel[0,0,z//2-1,:,x//4*3] + vel[0,0,z//2-1,:,x//4*3-1])*0.25
                else:
                    result_u_l = (vel[0,0,z//2,:,x//4] + vel[0,0,z//2-1,:,x//4])*0.5
                    result_u_u = (vel[0,0,z//2,:,x//4*3+1] + vel[0,0,z//2-1,:,x//4*3+1])*0.25
                result_u_l = (result_u_l / Re).cpu().numpy()
                result_u_u = (result_u_u / Re).cpu().numpy()

            if domain.getBlock(0).hasVertexCoordinates():
                # transform = domain.getBlock(0).transform
                # t_scales_x = transform[...,z//2,y//2,:,0].cpu().numpy()
                # t_scales_y = transform[...,z//2,:,x//2,4].cpu().numpy()
                # t_scales_z = transform[...,:,y//2,x//2,8].cpu().numpy()
                # LOG.info("SCALES (%s):\nx: %s\ny: %s\nz: %s", transform.shape, t_scales_x, t_scales_y, t_scales_z)
                #coords = shapes.ortho_transform_to_coords(domain.getBlock(0).transform, 3)
                coords = domain.getBlock(0).getCellCoordinates() #shapes.coords_to_center_coords(coords)
                coords_x = coords[0,0,z//2,y//2,:].cpu().numpy()
                coords_y = coords[0,1,z//2,:,x//2].cpu().numpy()
                coords_z = coords[0,2,:,y//2,x//2].cpu().numpy()
            else:
                raise NotImplementedError
                coords_x = np.asarray([(i+0.5)*domain_scale/res[0] for i in range(res[0])])
                coords_y = np.asarray([(i+0.5)*domain_scale/res[1] for i in range(res[1])])
            
            coords = [coords_x, coords_y, coords_z]
            coords_x = coords[lid_bound_axis] - (domain_scale*aspect_x)/2
            if (lid_permutation[0]&1)==1:
                coords_x = coords_x*-1
            coords_y = coords[lid_flow_axis] - (domain_scale)/2
            if (lid_permutation[1]&1)==0:
                coords_y = coords_y*-1
            coords_z = coords[lid_tangent_axis]/aspect_z - 0.5
            # coords_x = coords_x - (domain_scale*aspect_x)/2
            # coords_y = coords_y - (domain_scale)/2
            # coords_z = coords_z/aspect_z - 0.5
            
            # LOG.info("COORDS:\nx: %s\ny: %s", coords_x, coords_y)
            # c_scales_x = coords_x[1:] - coords_x[:-1]
            # c_scales_y = coords_y[1:] - coords_y[:-1]
            # LOG.info("SCALES:\nx: %s\ny: %s", c_scales_x, c_scales_y)

            ax_u.plot(result_u, coords_y, ".-", label=res_label, zorder=r_idx)
            ax_u.legend()
            if plot_u_extra:
                ax_u_l.plot(result_u_l, coords_y, ".-", label=res_label, zorder=r_idx)
                ax_u_l.legend()
                ax_u_u.plot(result_u_u, coords_y, ".-", label=res_label, zorder=r_idx)
                ax_u_u.legend()
            elif z_extra:
                ax_u_z.plot(result_u, coords_y, ".-", label=res_label, zorder=r_idx)
                ax_u_z.legend()
                ax_v_z.plot(coords_x, result_v, ".-", label=res_label, zorder=r_idx)
                ax_v_z.legend()
            ax_v.plot(coords_x, result_v, ".-", label=res_label, zorder=r_idx)
            ax_v.legend()
            ax_w.plot(coords_z, result_w, ".-", label=res_label, zorder=r_idx)
            ax_w.legend()
            
            prof_fig.align_labels()
            prof_fig.tight_layout()
            prof_fig.savefig(os.path.join(run_dir, "vel_profiles.{}".format(plot_format)))
    
    prof_fig.clf()
    plt.close(prof_fig)