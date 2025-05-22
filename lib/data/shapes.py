import torch
import math
import numpy as np
import scipy.stats as stats
from scipy.spatial.transform import Rotation


def get_grid_coords(size, dtype=torch.int32):
    coords = []
    for dim in size:
        coords.append(torch.range(0,dim-1, dtype=dtype))
    
    coords = torch.meshgrid(*coords, indexing="xy")
    coords = torch.stack(coords)
    
    return coords #CDHW

def get_grid_striped_x(size, dtype=torch.float32):
    coords = get_grid_coords(size, dtype)
    
    #coords_x = coords[0]
    #data = coords_x%2
    
    return coords[0]%2

def make_matrix_rotation_2D(angle, degrees=True):
    if degrees:
        angle = np.deg2rad(angle)
    return np.asarray([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]],
        dtype=np.float32)

def make_matrix_rotation_3D(rotvec, degrees=True):
    return Rotation.from_rotvec(rotvec, degrees=degrees).as_dcm()

def make_rotation_distance_scaling_fn_sine(angle, r_start, r_end):
    if not r_start>0:
        raise RuntimeError("r_start must be positive.")
    r_half = (r_start+r_end)*0.5
    rad = np.deg2rad(angle)
    def displacement(dist):
        return (1 - (np.cos((dist - r_start)/(r_end - r_start)*2*np.pi)+1)*0.5)
    factor = rad*r_half / displacement(r_half)
    def distance_scaling(dist):
        if dist<r_start or r_end<dist:
            return 0
        else:
            return factor*displacement(dist)/(dist*rad)
    
    return distance_scaling

def make_rotation_distance_scaling_fn_sine_half(angle, r_end):
    
    def distance_scaling(dist):
        if r_end<dist:
            return 0
        else:
            return (np.cos(dist/r_end *np.pi) +1)*0.5
    
    return distance_scaling

def rotate_grid(grid:torch.Tensor, angle:float, axis:list=None, center:str="CENTER", distance_scaling:callable=None):
    # angle: rotation angle in degrees
    # axis: axis to rotate around for 3D
    # center: the position to rotate around. coordinate or string
    # distance_scaling: function to return a weighting factor for axis, depending on a points distance to center
    pass
    assert isinstance(grid, torch.Tensor) and (grid.dim() in [4,5]) and grid.size(1)==(grid.dim()-2), "grid must be a torch tensor with shape NCHW or NCDHW"
    dims = grid.dim()-2
    if dims==3:
        assert isinstance(axis, (list, tuple)) and len(axis)==3, "3D rotation axis is required"
        axis = np.asarray(axis, dtype=np.float32)
        axis_norm = np.linalg.norm(axis)
        if axis_norm<1e-5:
            raise ValueError("rotation axis is too short for normalization")
        axis /= axis_norm
    
    grid = torch.moveaxis(grid, 1, -1)
    grid_size = grid.size()
    grid = torch.reshape(grid, (-1,dims))
    
    if center=="ORIGIN":
        center = [0]*dims
    elif center=="CENTER":
        lower, _ = grid.min(dim=0)
        upper, _ = grid.max(dim=0)
        center = [(l+u)*0.5 for l,u in zip(lower, upper)]
    
    assert isinstance(center, (list, tuple)) and len(center)==dims
    center = torch.tensor(center, device=grid.device, dtype=grid.dtype)
    grid = grid - center # now centered on origin
    
    if distance_scaling is not None:
        distances = torch.linalg.norm(grid, dim=-1, keepdims=False).cpu().numpy()
        angle = [angle*distance_scaling(distance) for distance in distances]
        
        make_matrix_rotation = make_matrix_rotation_2D if dims==2 else lambda angle: make_matrix_rotation_3D(angle*axis)
        rotation_matrices = np.asarray([make_matrix_rotation(a) for a in angle])
        
        rotation_matrices = torch.tensor(rotation_matrices, device=grid.device, dtype=grid.dtype)
        
    else:
        if dims==2:
            rotation_matrix = make_matrix_rotation_2D(angle)
        else:
            rotation_matrix = make_matrix_rotation_3D(angle*axis)
        rotation_matrix = torch.tensor(rotation_matrix, device=grid.device, dtype=grid.dtype)
        rotation_matrices = rotation_matrix.reshape((1,dims,dims)).repeat(grid.size(0),1,1)
    
    
    grid = torch.reshape(grid, (-1,dims,1))
    grid = torch.bmm(rotation_matrices, grid)
    grid = torch.reshape(grid, (-1,dims))
    
    grid = grid + center
    grid = torch.reshape(grid, grid_size)
    grid = torch.moveaxis(grid, -1, 1)
    
    return grid

def get_grid_cube_centered(size, cube_size, dtype=torch.float32):
    data = []
    for dim, cs in zip(size, cube_size):
        assert cs<=dim
        border1 = (dim-cs)//2
        border2 = dim - cs - border1
        x = [0]*border1 + [1]*cs + [0]*border2
        data.append(torch.tensor(x, dtype=dtype))
    
    data = torch.meshgrid(*data)
    data = torch.prod(torch.stack(data), dim=0)
    
    return data

def get_grid_normal_dist(size, mean, var, dtype=torch.float32):
    #https://stackoverflow.com/questions/10138085/how-to-plot-normal-distribution
    data = []
    for dim, m, v in zip(size, mean, var):
        s = math.sqrt(v)
        #x = np.linspace(m-3*s, m+3*s, dim)
        # grid borders -1 to 1, so cell centers have some offset
        cell_coord = (dim/2 - 0.5)/(dim/2)
        x = np.linspace(-cell_coord, cell_coord, dim)
        data.append(torch.tensor(stats.norm.pdf(x, m, s), dtype=dtype))
    
    data = torch.meshgrid(*data)
    data = torch.prod(torch.stack(data), dim=0)
    
    return data


def ortho_transform_to_coords(transform, dims):
    # transform shape: NDHWC
    assert transform.shape[-1] == 2*dims*dims+1
    coords = []
    for dim in range(dims): #x,y,z
        scales = transform[...,(dims+1)*dim]
        # padding argument goes in inverse dimension order
        coord = torch.nn.functional.pad(torch.cumsum(scales, dim=-(dim+1)), [0,0]*dim + [1,0], value=0)
        coord = torch.nn.functional.pad(coord, [1,0]*dim + [0,0] + [1,0]*(dims - 1 - dim), mode="replicate")
        coords.append(coord)
    coords = torch.stack(coords, dim=1)
    return coords #NCDHW


def coords_to_center_coords(coords):
    # coords NCDHW
    dims = coords.shape[1]
    if dims==2:
        pool = torch.nn.AvgPool2d(2, stride=1)
    elif dims==3:
        pool = torch.nn.AvgPool3d(2, stride=1)
    else:
        raise ValueError()
    return pool(coords)

def get_grid_normal_dist_from_ortho_transforms(transforms, mean, var, dtype=torch.float32, normalize=True):
    # transforms, Coords: NCDHW
    dims = len(transforms.shape)-2
    vertex_coords = ortho_transform_to_coords(transforms, dims)
    coords = coords_to_center_coords(vertex_coords)
    size = [transforms.shape[-(i+1)] for i in range(dims)]
    data = []
    for dim, (res, m, v) in enumerate(zip(size, mean, var)):
        s = math.sqrt(v)
        #x = np.linspace(m-3*s, m+3*s, dim)
        slicing = tuple([0,dim] + [slice(None) if dim==(dims-1-d) else 0 for d in range(dims)])
        #print(slicing)
        x = coords[slicing].cpu().numpy()
        #print(x)
        if normalize:
            slicing = tuple([0,dim] + [0]*dims)
            dim_min = vertex_coords[slicing].cpu().numpy()
            slicing = tuple([0,dim] + [-1]*dims)
            dim_max = vertex_coords[slicing].cpu().numpy()
            #print(dim_min, dim_max)
            #x = x / x[-1] * 2 -1 # -> [-1,1]
            x = (x-dim_min)/(dim_max - dim_min) *2 -1
            #print(x)
        data.append(torch.tensor(stats.norm.pdf(x, m, s), dtype=dtype))
    
    data = data[::-1]
    
    data = torch.meshgrid(*data) # C-DHW
    data = torch.prod(torch.stack(data), dim=0) # DHW
    
    return data

def interpolate_vertices_from_borders_2D(borders, x_weights=None, y_weights=None, dtype=torch.float32):
    # borders: 2D: [-x,+x,-y,+y]
    assert len(borders)==4, "only 2D for now"
    dims=2
    res = [len(borders[0]), len(borders[2])] # y,x
    assert len(borders[1])==res[0]
    assert len(borders[3])==res[1]
    
    borders = [np.asarray(border) for border in borders]

    grid = torch.zeros((1,dims,res[0],res[1]), dtype=dtype)
    # set borders of grid
    # for y_idx in range(0,res[0]):
        # grid[0,0,y_idx, 0] = borders[0][y_idx][0]
        # grid[0,1,y_idx, 0] = borders[0][y_idx][1]
        # grid[0,0,y_idx, -1] = borders[1][y_idx][0]
        # grid[0,1,y_idx, -1] = borders[1][y_idx][1]
    # for x_idx in range(0,res[1]):
        # grid[0,0,0, x_idx] = borders[2][x_idx][0]
        # grid[0,1,0, x_idx] = borders[2][x_idx][1]
        # grid[0,0,-1, x_idx] = borders[3][x_idx][0]
        # grid[0,1,-1, x_idx] = borders[3][x_idx][1]
    
    
    if x_weights is None:
        x_weights = [i/(res[0]-1) for i in range(res[0])]
    else:
        assert len(x_weights) == res[0]
    
    if y_weights is None:
        y_weights = [i/(res[1]-1) for i in range(res[1])]
    else:
        assert len(y_weights) == res[1]
    

    # interpolate inner
    # for y_idx in range(1,res[0]-1):
        # for x_idx in range(1,res[1]-1):
            # y_weight = x_weights[x_idx] * (1 - x_weights[x_idx])
            # y_weight = y_weight**2
            # y_weight_upper = y_weights[y_idx] * y_weight #* 0.5
            # y_weight_lower = (1 - y_weights[y_idx]) * y_weight #* 0.5
            # x_weight = y_weights[y_idx] * (1 - y_weights[y_idx])
            # x_weight = x_weight**2
            # x_weight_upper = x_weights[x_idx] * x_weight #* 0.5
            # x_weight_lower = (1 - x_weights[x_idx]) * x_weight #* 0.5
            
            # #y_weight_upper = y_weight_upper**2
            # #y_weight_lower = y_weight_lower**2
            # #x_weight_upper = x_weight_upper**2
            # #x_weight_lower = x_weight_lower**2
            
            # weight_norm = 1/(y_weight_upper + y_weight_lower + x_weight_upper + x_weight_lower)
            # y_weight_upper *= weight_norm
            # y_weight_lower *= weight_norm
            # x_weight_upper *= weight_norm
            # x_weight_lower *= weight_norm
            
            # grid[0,0,y_idx, x_idx] = borders[0][y_idx][0]*x_weight_lower + borders[1][y_idx][0]*x_weight_upper + borders[2][x_idx][0]*y_weight_lower + borders[3][x_idx][0]*y_weight_upper
            # grid[0,1,y_idx, x_idx] = borders[0][y_idx][1]*x_weight_lower + borders[1][y_idx][1]*x_weight_upper + borders[2][x_idx][1]*y_weight_lower + borders[3][x_idx][1]*y_weight_upper
            
    #for y_idx in range(1,res[0]-1):
    for y_idx in range(res[0]):
        y_weight_upper = x_weights[y_idx]
        y_weight_lower = (1 - x_weights[y_idx])
        x_start = borders[2][0]*y_weight_lower + borders[3][0]*y_weight_upper
        x_end   = borders[2][-1]*y_weight_lower + borders[3][-1]*y_weight_upper
        x_size = x_end - x_start
            
        target_size = borders[1][y_idx] - borders[0][y_idx]
        size_diff = target_size - x_size
        size_fac = target_size/x_size
        
        #for x_idx in range(1,res[1]-1):
        for x_idx in range(res[1]):
            x_val = borders[2][x_idx]*y_weight_lower + borders[3][x_idx]*y_weight_upper
            #x_frac = x_val/x_size
            x_frac = x_idx/(res[1]-1)#y_weights[y_idx]
            #x_val = x_val * (borders[1][y_idx][0] - borders[0][y_idx][0]) + borders[0][y_idx]
            if np.any(np.isclose(x_size,0)):
                x_val = x_val - x_start + size_diff * x_frac + borders[0][y_idx]
            else:
                x_val = (x_val - x_start)*size_fac + borders[0][y_idx]
            
            grid[0,0,y_idx, x_idx] = x_val[0]
            grid[0,1,y_idx, x_idx] = x_val[1]
    
    
    return grid

def _check_weights(weights, res, name="weights"):
    if not (len(weights) == res): raise ValueError("Invalid %s: length must match resolution."%(name,))
    if not (isinstance(weights, (list, tuple)) and all(isinstance(w, (int, float)) for w in weights)):
        raise TypeError("Invalid %s: weights must be a list of float."%(name,))
    if not all((0-1e-5)<=w and w<=(1+1e-5) for w in weights): raise ValueError("Invalid %s: weights must be in [0,1]: %s"%(name,weights))
    if not np.isclose(weights[0], 0): raise ValueError("Invalid %s: start weight must be 0, is %s."%(name,weights[0]))
    if not np.isclose(weights[-1], 1): raise ValueError("Invalid %s: end weight must be 1, is %s."%(name,weights[-1]))
    if not all(weights[i]<weights[i+1] for i in range(len(weights)-1)): raise ValueError("Invalid %s: weights must be strictly increasing."%(name,))

def invert_weights(weights):
    _check_weights(weights, len(weights))
    
    sizes = [weights[i+1] - weights[i] for i in range(len(weights)-1)]
    inv_weights = [0]
    size = 0
    # inverse cumulative sum of the sizes
    for i in range(len(sizes)-1,-1,-1):
        size = size + sizes[i]
        inv_weights.append(size)
    
    return inv_weights

def make_weights_linear(res):
    return [x/(res) for x in range(res+1)]

def make_weights_exp(res, base, refinement):
    # refinement: "START", "END", "BOTH"
    
    exponents = [e for e in range(res)]
    if refinement=="END":
        exponents.reverse()
    elif refinement=="BOTH":
        exponents = exponents[:res//2] + list(reversed(exponents))[res//2:]
    
    sizes = [base**e for e in exponents]
    total_size = np.sum(sizes)
    weights = [0] + [w/total_size for w in np.cumsum(sizes)]
    
    return weights

def make_weights_exp_global(res, global_scale, refinement, log_fn=None):
    resolution = res//2 if refinement=="BOTH" else res
    base = global_scale ** (1/(resolution - 1))
    
    if log_fn is not None:
        log_fn("r %d exp weights global %.03e -> base %.03e", res, global_scale, base)
    
    return make_weights_exp(res, base, refinement)

def make_weights_cos(res, refinement):
    
    if refinement=="START":
        c_start = 0
        c_end = np.pi/2
        n_mul = -1
        n_add = 1
    elif refinement=="END":
        c_start = np.pi/2
        c_end = np.pi
        n_mul = -1
        n_add = 0
    elif refinement=="BOTH":
        c_start = 0
        c_end = np.pi
        n_mul = -0.5
        n_add = 0.5
    else:
        raise ValueError("Unkown refinement side.")
    
    def c_val_lerp(t):
        return c_start*(1-t) + c_end*t
    
    weights = [np.cos(c_val_lerp(x/res))*n_mul+n_add for x in range(res+1)]
    return weights

def generate_grid_vertices_2D(res, corner_vertices, border_vertices=None, x_weights=None, y_weights=None, dtype=torch.float32):
    # res: grid resolution in [y,x]
    # corner_vertices: [-x-y,+x-y,-x+y,+x+y], (x,y)-tuple
    # border_vertices: [-x,+x,-y,+y], lists of (x,y)-tuple. will be interpolated from corners if None
    # boundaries: list of str, boundaries to generate an additional layer for
    border_to_corners = {0:(0,2),1:(1,3),2:(0,1),3:(2,3)}
    assert isinstance(corner_vertices, (tuple, list))
    assert len(corner_vertices)==4

    if border_vertices is None:
        border_vertices = [None] * 4
    
    if x_weights is None:
        # weights for x-boundaries, so based on y coordinate
        x_weights = [i/(res[0]-1) for i in range(res[0])]
    else:
        _check_weights(x_weights, res[0], "x_weights")
    
    if y_weights is None:
        y_weights = [i/(res[1]-1) for i in range(res[1])]
    else:
        _check_weights(y_weights, res[1], "y_weights")

    for border_idx in range(len(border_vertices)):
        r = res[border_idx//2]
        if border_vertices[border_idx] is None:
            lower_corner = corner_vertices[border_to_corners[border_idx][0]]
            upper_corner = corner_vertices[border_to_corners[border_idx][1]]
            weights = x_weights if border_idx<2 else y_weights
            border_vertices[border_idx] = []
            for idx in range(r):
                weight_upper = weights[idx]
                weight_lower = 1 - weight_upper
                border_vertices[border_idx].append((lower_corner[0]*weight_lower + upper_corner[0]*weight_upper, lower_corner[1]*weight_lower + upper_corner[1]*weight_upper))
        else:
            assert len(border_vertices[border_idx]) == r, "is %d, expected %d"%(len(border_vertices[border_idx]), r)
            # TODO: check that corners match
    
    #print(border_vertices)

    return interpolate_vertices_from_borders_2D(border_vertices, x_weights=x_weights, y_weights=y_weights, dtype=dtype)

def extrapolate_boundary_layers(grid, boundaries=[]):
    # linear extrapolation of the boundaries
    # boundaries: list of tuple: boundary and extrapolation scale [("-x", 0.5),]
    assert isinstance(grid, torch.Tensor) and grid.dim()==4, "grid must be a 2D torch tensor of shape NCHW"
    assert grid.size(-1)>1 and grid.size(-2)>1, "grid must be at least 2x2"
    assert all(bound in ["-x","+x","-y","+y"] and scale>0 for bound, scale in boundaries)
    
    for bound, scale in boundaries:
        if bound=="-x":
            layer_1 = grid[...,:1]
            layer_2 = grid[...,1:2]
        if bound=="+x":
            layer_1 = grid[...,-1:]
            layer_2 = grid[...,-2:-1]
        if bound=="-y":
            layer_1 = grid[...,:1,:]
            layer_2 = grid[...,1:2,:]
        if bound=="+y":
            layer_1 = grid[...,-1:,:]
            layer_2 = grid[...,-2:-1,:]
        
        bound_layer = layer_1 + (layer_1 - layer_2)*scale
        
        if bound[0]=="-":
            grid = [bound_layer, grid]
        else:
            grid = [grid, bound_layer]
        if bound[1]=="x":
            dim = -1
        else:
            dim = -2
        
        grid = torch.cat(grid, dim=dim)
    
    return grid

def get_extrapolated_boundary_layer(grid, bound, scale):
    assert isinstance(grid, torch.Tensor) and grid.dim()==4, "grid must be a 2D torch tensor of shape NCHW"
    assert grid.size(-1)>1 and grid.size(-2)>1, "grid must be at least 2x2"
    assert bound in ["-x","+x","-y","+y"]
    
    
    if bound=="-x":
        layer_1 = grid[...,:1]
        layer_2 = grid[...,1:2]
    if bound=="+x":
        layer_1 = grid[...,-1:]
        layer_2 = grid[...,-2:-1]
    if bound=="-y":
        layer_1 = grid[...,:1,:]
        layer_2 = grid[...,1:2,:]
    if bound=="+y":
        layer_1 = grid[...,-1:,:]
        layer_2 = grid[...,-2:-1,:]
    
    bound_layer = layer_1 + (layer_1 - layer_2)*scale
    
    if bound[0]=="-":
        grid = [bound_layer, layer_1]
    else:
        grid = [layer_1, bound_layer]
    if bound[1]=="x":
        dim = -1
    else:
        dim = -2
    
    return torch.cat(grid, dim=dim)


def make_wall_refined_ortho_grid(res_x, res_y, corner_lower=(0,0), corner_upper=(1,1), wall_refinement=[], base=1.05, dtype=torch.float32):
    dims=2
    assert isinstance(corner_lower, (list, tuple)) and len(corner_lower)==2
    assert isinstance(corner_upper, (list, tuple)) and len(corner_upper)==2
    corners = [tuple(corner_lower), (corner_upper[0], corner_lower[1]), (corner_lower[0], corner_upper[1]), tuple(corner_upper)]
    #y_weights = [(i/(res_x))**exponent for i in range(res_x+1)]
    #y_weights_inv = [1 - (1 - i/(res_x))**exponent for i in range(res_x+1)]
    #x_weights = [(i/(res_y))**exponent for i in range(res_y+1)]
    #x_weights_inv = [1 - (1 - i/(res_y))**exponent for i in range(res_y+1)]
    
    if not isinstance(base, (list, tuple)):
        base = [base]*dims

    #transformation of block
    y_w = None
    if "-x" in wall_refinement:
        if "+x" in wall_refinement:
            y_w = make_weights_exp(res_x, base=base[0], refinement="BOTH") # y_weights[:res_x//2] + y_weights_inv[res_x//2:]
        else:
            y_w = make_weights_exp(res_x, base=base[0], refinement="START") # y_weights
    elif "+x" in wall_refinement:
        y_w = make_weights_exp(res_x, base=base[0], refinement="END") # y_weights_inv
    
    x_w = None
    if "-y" in wall_refinement:
        if "+y" in wall_refinement:
            x_w = make_weights_exp(res_y, base=base[1], refinement="BOTH") # x_weights[:res_x//2] + x_weights_inv[res_x//2:]
        else:
            x_w = make_weights_exp(res_y, base=base[1], refinement="START") # x_weights
    elif "+y" in wall_refinement:
        x_w = make_weights_exp(res_y, base=base[1], refinement="END") # x_weights_inv
    
    grid = generate_grid_vertices_2D([res_y+1,res_x+1],corners, None, x_weights=x_w, y_weights=y_w, dtype=dtype)
    
    return grid

def extrude_grid_z(grid, res_z, start_z=0, end_z=1, weights_z=None, exp_base=1.05):
    # res_z z resolution of the cell grid. coordinates grid will have res_z+1.

    assert grid.dim() == 4 and grid.size(1)==2
    res_x = grid.size(-1)-1
    res_y = grid.size(-2)-1

    if isinstance(weights_z, list):
        assert len(weights_z)==(res_z+1)
    elif weights_z is None or weights_z=="LINEAR":
        weights_z = make_weights_linear(res_z)
    elif weights_z=="EXP" or weights_z=="EXP_BOTH":
        weights_z = make_weights_exp(res_z, base=exp_base, refinement="BOTH")
    elif weights_z=="EXP_START":
        weights_z = make_weights_exp(res_z, base=exp_base, refinement="START")
    elif weights_z=="EXP_END":
        weights_z = make_weights_exp(res_z, base=exp_base, refinement="EXP_END")
    else:
        raise ValueError("Unknown weights specification")
    
    def lerp(a,b,t):
        return a*(1-t) + b*t
    coords_z = torch.tensor([lerp(start_z, end_z, w) for w in weights_z], device=grid.device, dtype=grid.dtype)

    coords_z = coords_z.reshape((1,1,res_z+1,1,1)).repeat(1,1,1,res_y+1, res_x+1)
    grid = grid.reshape((1,2,1,res_y+1, res_x+1)).repeat(1,1,res_z+1,1,1)
    grid = torch.cat([grid, coords_z], dim=1)

    return grid
    


def make_torus_2D(res:int, r1:float, r2:float, start_angle:float, angle:float, offset=None, dtype=torch.float32):
        # res: x-resolution along angle, y is computed to result in approx. square cells
        # r1: inner radius. r2: outer radius
        # angles in degrees, start_angle=0 is x-axis, angle goes counterclockwise
        # offset: 2-tuple of float (x,y), "CENTER", None
        # returns: grid of coordinate with layout NCHW with C=(x,y)
        # x goes along angle, y along radius
        assert res>1
        assert r1>0
        assert r2>r1
        start_angle = start_angle%360
        x = res+1
        deg_step = angle/(x-1)
        rad_step = np.deg2rad(deg_step)
        start_rad = np.deg2rad(start_angle)
        end_rad = start_rad + np.deg2rad(angle)
        corners = [
            (np.cos(start_rad)*r1, np.sin(start_rad)*r1),
            (np.cos(end_rad)*r1, np.sin(end_rad)*r1),
            (np.cos(start_rad)*r2, np.sin(start_rad)*r2),
            (np.cos(end_rad)*r2, np.sin(end_rad)*r2)
        ]
        lower_border = [(np.cos(start_rad + rad_step*i)*r1, np.sin(start_rad + rad_step*i)*r1) for i in range(x)] # -y
        upper_border = [(np.cos(start_rad + rad_step*i)*r2, np.sin(start_rad + rad_step*i)*r2) for i in range(x)] # -y
        
        # roughly square cells, growing linearly with radius
        r = r2-r1
        sizes = []
        d = r1
        y = 1
        width_scale = 2 * np.pi / x * (abs(angle)/360)
        while d<r2:
            width = d * width_scale
            sizes.append(width)
            d += width
            y += 1
        scale = (d-r1) / r
        sizes = [w/scale for w in sizes]
        # interpolation weights in [0,1]
        x_weights = [0] + [w/r for w in np.cumsum(sizes)]

        #print("square cells: (x=%d, y=%d),\ns=%s,\nw=%s"%(x,y,sizes,x_weights))

        #l_border = [() for i in range(res)]
        #print(lower_border)
        grid = generate_grid_vertices_2D([y,x],corners, [None, None, lower_border, upper_border], x_weights=x_weights, dtype=dtype)

        if offset=="CENTER":
            # via AABB
            vertex_corners = torch.tensor(corners, dtype=dtype)
            lower, _ = vertex_corners.min(dim=0)
            upper, _ = vertex_corners.max(dim=0)
            size = upper - lower
            center = lower + size*0.5
            offset = -center
            print(offset)
        
        if isinstance(offset, (list, tuple, np.ndarray)):
            offset = torch.tensor(offset, dtype=dtype)
        
        if isinstance(offset, torch.Tensor):
            assert offset.dtype==dtype
            assert offset.dim()==1
            assert offset.size(0)==2
            offset = offset.view(1,2,1,1)
            grid = grid + offset

        
        return grid

def make_cosX_grid(x, y, x_scale=1, y_scale=1, strength=1):
    
    baseline = np.asarray([[i*x_scale, 0] for i in range(x+1)])
    
    x_norm = (2*np.pi)/(y)
    offsets = np.asarray([((np.cos(x_norm*_)+1)*0.5*x_scale*strength, _*y_scale) for _ in range(y+1)])
    rows = [baseline + offsets[i] for i in range(y+1)]
    
    vertex_coords = np.asarray(rows) # HWC
    vertex_coords = np.moveaxis(vertex_coords, -1, 0) # CHW
    vertex_coords = np.expand_dims(vertex_coords, 0) #NCHW
    #vertex_coords = torch.ones((1,2,y,x), dtype=dtype, device=cuda_device) * vertex_coords #torch.tensor(vertex_coords, dtype=dtype, device=cuda_device)
    vertex_coords = torch.tensor(vertex_coords, dtype=dtype, device=cuda_device).contiguous()
    
    return vertex_coords
