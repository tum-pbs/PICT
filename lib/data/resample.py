
import numpy as np
import torch
import math

import PISOtorch
from lib.data.shapes import ortho_transform_to_coords, coords_to_center_coords

def make_matrix_translation(t):
    dims = t.size()[0]
    mat = torch.zeros([dims+1]*2, dtype=t.dtype)
    for d in range(dims):
        mat[d,-1] = t[d]
        mat[d,d] = 1
    mat[-1,-1] = 1
    return mat

def make_matrix_scaling(s):
    dims = s.size()[0]
    mat = torch.zeros([dims+1]*2, dtype=s.dtype)
    for d in range(dims):
        mat[d,d] = s[d]
    mat[-1,-1] = 1
    return mat

def make_meshgrid_AABB(vertex_coords, cells_per_unit, dims, dtype=torch.float32):
    vertex_coords = vertex_coords.view(dims, -1)
    lower, _ = vertex_coords.min(dim=-1)
    upper, _ = vertex_coords.max(dim=-1)
    size = upper - lower
    resolution = np.ceil(cells_per_unit * size.cpu().numpy()).astype(int)

    grid = torch.meshgrid(*[torch.linspace(lower[dim], upper[dim], steps=resolution[dim]) for dim in range(dims)], indexing='xy')
    grid = torch.stack(grid)
    grid = torch.unsqueeze(grid, 0)

    print("meshgrid_AABB:", grid.size())
    print("meshgrid_AABB:", grid)
    return grid

    

def make_uniform_transform_AABB_outer(vertex_coords, out_shape, dims, dtype=torch.float32):
    vertex_coords = vertex_coords.view(dims, -1)
    lower, _ = vertex_coords.min(dim=-1)
    upper, _ = vertex_coords.max(dim=-1)
    size = upper - lower
    center = lower + size*0.5
    
    # transform matrix:
    # lower maps to -0.5 (lower border of cell)
    # higher maps to out_shape-0.5
    # scaling and translation, no rotation
    out_shape_float = out_shape.to(dtype) if isinstance(out_shape, torch.Tensor) else torch.tensor(out_shape, dtype=dtype)
    scale = torch.tensor([torch.max(size.cpu()/out_shape_float)]*dims)
    translation_1 = make_matrix_translation(-out_shape_float*0.5 + 0.5) # center on origin #torch.tensor([0.5]*dims, dtype=data_list[0].dtype))
    scaling = make_matrix_scaling(scale)
    translation_2 = make_matrix_translation(center.cpu()) # center on bounding box center
    
    mat = torch.matmul(translation_2, torch.matmul(scaling, translation_1))
    mat = torch.reshape(mat, (1,dims+1,dims+1))

    return mat

def make_uniform_transform_AABB_inner(vertex_coords, out_shape, dims, dtype=torch.float32):
    vertex_coords = vertex_coords.view(dims, -1)
    lower, _ = vertex_coords.min(dim=-1)
    upper, _ = vertex_coords.max(dim=-1)
    size = upper - lower
    center = lower + size*0.5
    
    # transform matrix:
    # lower maps to -0.5 (lower border of cell)
    # higher maps to out_shape-0.5
    # scaling and translation, no rotation
    out_shape_float = torch.tensor(out_shape, dtype=dtype)
    scale = torch.tensor([torch.min(size.cpu()/out_shape_float)]*dims)
    translation_1 = make_matrix_translation(-out_shape_float*0.5 + 0.5) # center on origin #torch.tensor([0.5]*dims, dtype=data_list[0].dtype))
    scaling = make_matrix_scaling(scale)
    translation_2 = make_matrix_translation(center.cpu()) # center on bounding box center
    
    mat = torch.matmul(translation_2, torch.matmul(scaling, translation_1))
    mat = torch.reshape(mat, (1,dims+1,dims+1))

    return mat

def get_uniform_transform(transform, vertex_coords, shape, dims, dtype=torch.float32):
    if isinstance(transform, torch.Tensor):
        if not (transform.size(0)==1 and transform.size(1)==dims+1 and transform.size(2)==dims+1):
            raise ValueError("Invalid transform matrix shape. must be (1,%d,%d), is %s"%(dims+1, dims+1, transform.size()))
        return transform
    elif isinstance(transform, np.ndarray):
        transform = torch.tensor(transform, dtype=dtype)
        if not (transform.size(0)==1 and transform.size(1)==dims+1 and transform.size(2)==dims+1):
            raise ValueError("Invalid transform matrix shape. must be (1,%d,%d), is %s"%(dims+1, dims+1, transform.size()))
        return transform
    elif transform=="AABB_OUTER":
        return make_uniform_transform_AABB_outer(vertex_coords, shape, dims, dtype=dtype)
    elif transform=="AABB_INNER":
        return make_uniform_transform_AABB_inner(vertex_coords, shape, dims, dtype=dtype)
    else:
        raise ValueError("Unknown transform parameter.")

def get_output_shape(out_shape, dims):
    if isinstance(out_shape, torch.Tensor):
        # must have shape [dims] and integer type
        if not (out_shape.dim()==1 and out_shape.size(0)==dims): raise ValueError("Resampling output shape does not match dimensions.")
        if not (out_shape.dtype==torch.int32): raise TypeError("Resampling output shape must have dtype torch.int32.")
        return out_shape
    elif isinstance(out_shape, np.ndarray):
        # must have shape [dims] and integer type
        return out_shape
    elif isinstance(out_shape, (list, tuple)):
        if not len(out_shape)==dims: raise ValueError("Resampling output shape does not match dimensions.")
        if not all(isinstance(_, int) for _ in out_shape): raise TypeError("Resampling output shape must be int or list of int.")
        return out_shape
    elif isinstance(out_shape, int):
        return [out_shape] * dims
    else:
        raise TypeError("Invalid resampling output shape: %s", out_shape)

def sample_transform_to_uniform_grid(data, transform, out_shape, transform_uniform="AABB_OUTER", fill_max_steps=0):
    dims = len(data.size())-2
    # out_shape is x,y,z
    out_shape = get_output_shape(out_shape, dims)
    vertex_coords = ortho_transform_to_coords(transform, dims) #NCDHW with C=x,y,z coords
    cell_coords = coords_to_center_coords(vertex_coords)
    
    mat = get_uniform_transform(transform_uniform, vertex_coords, out_shape, dims, dtype=data.dtype)
    
    out_data, out_weights = PISOtorch.SampleTransformedGridLocalToGlobal(data, cell_coords, mat, torch.tensor(out_shape, dtype=torch.int32), fillMaxSteps=fill_max_steps)
    
    return out_data

# compatibility
sample_to_uniform_grid = sample_transform_to_uniform_grid

def sample_coords_to_uniform_grid(data, coords, out_shape, is_cell_coords=False, transform_uniform="AABB_OUTER", fill_max_steps=0):
    dims = len(data.size())-2
    # out_shape is x,y,z
    # coords: NCDHW with C=x,y,z coords
    out_shape = get_output_shape(out_shape, dims)
    if is_cell_coords:
        if isinstance(transform_uniform, str) and transform_uniform.startswith("AABB"):
            raise NotImplementedError("vertex_coords are required for bounding box.")
        else:
            vertex_coords = None
        cell_coords = coords
    else:
        vertex_coords = coords
        cell_coords = coords_to_center_coords(vertex_coords)
    
    mat = get_uniform_transform(transform_uniform, vertex_coords, out_shape, dims, dtype=data.dtype)
    
    out_data, out_weights = PISOtorch.SampleTransformedGridLocalToGlobal(data, cell_coords, mat, torch.tensor(out_shape, dtype=torch.int32), fillMaxSteps=fill_max_steps)
    
    return out_data

def sample_multi_coords_to_uniform_grid(data_list, coords_list, out_shape, is_cell_coords=False, transform_uniform="AABB_OUTER", fill_max_steps=0):
    assert len(data_list) == len(coords_list)
    assert len(data_list) > 0
    dims = len(data_list[0].size())-2
    # out_shape is x,y,z
    # coords: NCDHW with C=x,y,z coords
    out_shape = get_output_shape(out_shape, dims)
    
    vertex_coords_list = []
    cell_coords_list = []
    for coords in coords_list:
        if is_cell_coords:
            if isinstance(transform_uniform, str) and transform_uniform.startswith("AABB"):
                raise NotImplementedError("vertex_coords are required for bounding box.")
            cell_coords_list.append(coords)
        else:
            vertex_coords = coords
            vertex_coords_list.append(vertex_coords)
            cell_coords_list.append(coords_to_center_coords(vertex_coords))
    
    # get bounding box, assuming N=1
    if is_cell_coords:
        vertex_coords = None
    else:
        vertex_coords = torch.cat([_.view(dims, -1) for _ in vertex_coords_list], dim=-1)

    mat = get_uniform_transform(transform_uniform, vertex_coords, out_shape, dims, dtype=data_list[0].dtype)
    
    out_shape = out_shape.to(torch.int32) if isinstance(out_shape, torch.Tensor) else torch.tensor(out_shape, dtype=torch.int32)
    out_data, out_weights = PISOtorch.SampleTransformedGridLocalToGlobalMulti(data_list, cell_coords_list, mat, out_shape, fillMaxSteps=fill_max_steps)
    
    return out_data


def sample_transform_from_uniform_grid(data, transform, transform_uniform="AABB_OUTER", boundary_mode="CLAMP"):
    shape = data.size()
    dims = len(shape)-2
    in_shape = [shape[-(d+1)] for d in range(dims)]
    
    vertex_coords = ortho_transform_to_coords(transform, dims) #NCDHW with C=x,y,z coords
    #print("vertex_coords:", vertex_coords.size())
    cell_coords = coords_to_center_coords(vertex_coords) #block.getCellCoordinates()
    
    mat = get_uniform_transform(transform_uniform, vertex_coords, in_shape, dims, dtype=data.dtype)

    if boundary_mode=="CLAMP":
        boundary_mode = PISOtorch.BoundarySampling.CLAMP
    elif boundary_mode=="CONSTANT":
        boundary_mode = PISOtorch.BoundarySampling.CONSTANT
    
    out_data = PISOtorch.SampleTransformedGridGlobalToLocal(data, mat, cell_coords, boundary_mode, torch.zeros([1], dtype=data.dtype))
    
    return out_data

sample_from_uniform_grid = sample_transform_from_uniform_grid

def sample_coords_from_uniform_grid(data, coords, is_cell_coords=False, transform_uniform="AABB_OUTER", boundary_mode="CLAMP"):
    shape = data.size()
    dims = len(shape)-2
    in_shape = [shape[-(d+1)] for d in range(dims)]
    
    #NCDHW with C=x,y,z coords
    if is_cell_coords:
        if isinstance(transform_uniform, str) and transform_uniform.startswith("AABB"):
            raise NotImplementedError("vertex_coords are required for bounding box.")
        else:
            vertex_coords = None
        cell_coords = coords
    else:
        vertex_coords = coords
        cell_coords = coords_to_center_coords(vertex_coords)
    #print("cell_coords:", cell_coords)
    
    mat = get_uniform_transform(transform_uniform, vertex_coords, in_shape, dims, dtype=data.dtype)

    if boundary_mode=="CLAMP":
        boundary_mode = PISOtorch.BoundarySampling.CLAMP
    elif boundary_mode=="CONSTANT":
        boundary_mode = PISOtorch.BoundarySampling.CONSTANT
    
    out_data = PISOtorch.SampleTransformedGridGlobalToLocal(data, mat, cell_coords, boundary_mode, torch.zeros([1], dtype=data.dtype))
    
    return out_data


def sample_multi_coords_from_uniform_grid(data, coords_list, out_shape=None, is_cell_coords=False, transform_uniform="AABB_OUTER", boundary_mode="CLAMP"):
    assert len(coords_list) > 0
    dims = data.dim()-2
    # out_shape is x,y,z
    # coords: NCDHW with C=x,y,z coords
    if out_shape is None:
        shape = data.size()
        out_shape = [shape[-(d+1)] for d in range(dims)]
    else:
        out_shape = get_output_shape(out_shape, dims)
    
    vertex_coords_list = []
    cell_coords_list = []
    for coords in coords_list:
        if is_cell_coords:
            if isinstance(transform_uniform, str) and transform_uniform.startswith("AABB"):
                raise NotImplementedError("vertex_coords are required for bounding box.")
            cell_coords_list.append(coords)
        else:
            vertex_coords = coords
            vertex_coords_list.append(vertex_coords)
            cell_coords_list.append(coords_to_center_coords(vertex_coords))
    
    # get bounding box, assuming N=1
    if is_cell_coords:
        vertex_coords = None
    else:
        vertex_coords = torch.cat([_.view(dims, -1) for _ in vertex_coords_list], dim=-1)

    mat = get_uniform_transform(transform_uniform, vertex_coords, out_shape, dims, dtype=data.dtype)
    if boundary_mode=="CLAMP":
        boundary_mode = PISOtorch.BoundarySampling.CLAMP
    elif boundary_mode=="CONSTANT":
        boundary_mode = PISOtorch.BoundarySampling.CONSTANT
    
    data_list = []
    for cell_coords in cell_coords_list:
        out_data = PISOtorch.SampleTransformedGridGlobalToLocal(data, mat, cell_coords, boundary_mode, torch.zeros([1], dtype=data.dtype))
        data_list.append(out_data)

    return data_list