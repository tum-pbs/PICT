
import imageio, os
import numpy as np
import torch
import PISOtorch

from matplotlib import pyplot as plt

from lib.data.resample import sample_multi_coords_to_uniform_grid


def tensor_to_numpy(tensor):
    return tensor.detach().cpu().numpy()
ttonp = tensor_to_numpy

def numerical_to_numpy(data):
    if isinstance(data, (int, float, list, tuple)):
        return np.asarray(data)
    if isinstance(data, torch.Tensor):
        return tensor_to_numpy(data)
    if isinstance(data, np.ndarray):
        return data
    raise TypeError
ntonp = numerical_to_numpy

class StringWriter:
    def __init__(self):
        self.__s = []
    
    def write(self, s, *fmt):
        if fmt:
            s = s%(*fmt,)
        self.__s.append(s)
    
    def write_line(self, s='', *fmt, newline='\n'):
        if s:
            if fmt:
                s = s%(*fmt,)
            self.__s.append(s)
        self.__s.append(newline)
    
    def flush(self):
        pass
    
    def reset(self):
        self.__s = []
    
    def get_string(self):
        return ''.join(self.__s)
    
    def __str__(self):
        return self.get_string()

def print_CSR_obj(csrMat):
    return print_CSR(csrMat.row.cpu(), csrMat.index.cpu(), csrMat.value.cpu(), csrMat.getRows())

def print_CSR(row, index, value, size):
    sb = StringWriter()
    assert len(row) == size+1
    
    for rowI in range(size):
        row_start = row[rowI]
        row_size = row[rowI+1] - row_start
        col_step = 0
        for colI in range(size):
            if (row_start + col_step)<len(index) and col_step<row_size and colI==index[row_start + col_step]:
                sb.write("%5.2f"%value[row_start + col_step])
                col_step += 1
            else:
                sb.write("  -  ")
        sb.write_line()
    
    #print_fn("Matrix from CSR:\n%s", sb.get_string())
    return sb.get_string()

def print_grid(grid, fmt="{:s}" , string_buffer=None):
    grid = grid.cpu()
    if not string_buffer:
        string_buffer = StringWriter()
    if grid.dim()>3:
        raise ValueError
    if grid.dim()==3:
        for idx in range(grid.size()[0]):
            print_grid(grid[idx], fmt=fmt, string_buffer=string_buffer)
            if idx<(grid.size()[0]-1):
                string_buffer.write_line("---")
        
    else:
        for row in range(grid.size()[0]):
            for col in range(grid.size()[1]):
                string_buffer.write(fmt.format(grid[row,col].numpy()))
                string_buffer.write(", ")
            string_buffer.write_line()
    
    return string_buffer.get_string()

def plot_grid(grid, color="tab:blue", ax=None, linewidth=1.0):
    if ax is None:
        ax = plt
    grid = grid.cpu()
    shape = [grid.size(-2), grid.size(-1)]
    for y in range(shape[0]):
        # plot x edges
        vx, vy = torch.unbind(torch.reshape(grid[0,:,y,:], (2,-1)), dim=0)
        vx = vx.numpy()
        vy = vy.numpy()
        ax.plot(vx, vy, linestyle="-", color=color, linewidth=linewidth)
    for x in range(shape[1]):
        # plot x edges
        vx, vy = torch.unbind(torch.reshape(grid[0,:,:,x], (2,-1)), dim=0)
        vx = vx.numpy()
        vy = vy.numpy()
        ax.plot(vx, vy, linestyle="-", color=color, linewidth=linewidth)

def get_grid_AABB(grid):
    assert isinstance(grid, torch.Tensor)
    
    grid_dim = grid.size(1)
    assert grid.dim()==(grid_dim+2)
    
    grid_flat = torch.movedim(grid, 1, 0).reshape(grid_dim, -1)
    
    min_coords = torch.min(grid_flat, dim=1).values
    max_coords = torch.max(grid_flat, dim=1).values
    
    return min_coords, max_coords

def get_grids_AABB(grids):
    min_coords, max_coords = get_grid_AABB(grids[0])
    for grid in grids[1:]:
        min_new, max_new = get_grid_AABB(grid)
        min_coords = torch.minimum(min_coords, min_new)
        max_coords = torch.maximum(max_coords, max_new)
    return min_coords, max_coords

def save_plotted_grids(path, name="grid", type="svg"):
    plt.gca().set_aspect('equal')
    plt.savefig(os.path.join(path, "%s.%s"%(name, type)))
    plt.close()
    plt.clf()

def plot_grids(grids, color="tab:blue", path=None, name="grid", type="svg", linewidth=1.0, fig_scale=5):
    if not isinstance(grids, (list, tuple)):
        grids = [grids]
    if not isinstance(color, list):
        color = [color]*len(grids)
    
    if not all(isinstance(grid, torch.Tensor) and grid.dim()==4 and grid.size(0)==1 and grid.size(1)==2 for grid in grids):
        raise ValueError("grids must be a list of tensors with shape CNHW with N=1 and C=2")
    
    if not len(color)==len(grids):
        raise ValueError("Need 1 color per grid or a global color")
    
    # set figure aspect ratio to fit grid(s)
    min_coords, max_coords = get_grids_AABB(grids)
    min_coords = min_coords.cpu().numpy()
    max_coords = max_coords.cpu().numpy()
    grids_size = max_coords - min_coords
    
    min_dim = np.argmin(grids_size)
    grids_size = grids_size / grids_size[min_dim] * fig_scale
    
    fig, ax = plt.subplots(1,1, figsize=(grids_size[0], grids_size[1]))
    #fig.suptitle(name)
    
    for grid, c in zip(grids, color):
        plot_grid(grid, color=c, ax=ax, linewidth=linewidth)
    
    ax.axis("equal")
    fig.tight_layout()
    
    if path is not None:
        os.makedirs(path, exist_ok=True)
        fig.savefig(os.path.join(path, "%s.%s"%(name, type)))
        plt.close(fig)
    else:
        return fig, ax

def vel_to_color(vel, mag):
    h = torch.atan2(vel[0,1], vel[0,0]) + np.pi # [0, 2pi]
    #hue/(np.pi/3) = hue/np.pi *3 -> [0,6]
    R = torch.clamp(torch.abs(h/np.pi*3-3)-1, 0,1)
    G = torch.clamp(torch.abs((h+4/3*np.pi)%(2*np.pi)/np.pi*3-3)-1, 0,1)
    B = torch.clamp(torch.abs((h+2/3*np.pi)%(2*np.pi)/np.pi*3-3)-1, 0,1)
    color = torch.stack([R,G,B], axis=-1)
    return color*torch.unsqueeze(mag[0], -1)

def save_np_png_channels(data, path):
    channels = torch.split(data, 1, dim=-1)
    for i, img in enumerate(channels):
        save_np_png(img, path.format(channel=i))

def save_np_png(data, path):
    data = np.clip(data, 0,1)
    try:
        imageio.imwrite(path, (data*255.0).astype(np.uint8), "png")
    except ValueError as e:
        if data.shape[-1]==1 and repr(e).find("Can't write images with one color channel.")>0:
            # newer pillow versions only work with 3-channel images
            data = np.repeat(data, 3, axis=-1)
            imageio.imwrite(path, (data*255.0).astype(np.uint8), "png")
        else:
            raise e

def save_np_exr(data, path):
    imageio.imwrite(path, data, "exr")

def pad_to_size(data, size_x, size_y, padding=0, pad_col=(0,)):
    assert isinstance(data, torch.Tensor) and data.dim()==3
    
    padded_x = size_x
    padded_y = size_y
    
    size_x = data.size(1)
    size_y = data.size(0)
    channels = data.size(-1)
    
    assert len(pad_col)==channels
    
    pad_left = (padded_x - size_x)//2 
    pad_right = padded_x - size_x - pad_left
    pad_top = (padded_y - size_y)//2
    pad_bot = padded_y - size_y - pad_top
    paddings = [pad_left, pad_right, pad_top, pad_bot]
    paddings = [_+padding for _ in paddings]
    
    data = [torch.nn.functional.pad(data[...,i], paddings, value=pad_col[i]) for i in range(channels)]
    data = torch.stack(data, axis=-1)
    
    return data

def arrange_blocks(blocks, layout="H", padding=2, pad_col=0):
    # blocks: list of HWC
    if layout in ["H", "h", None]:
        layout = [[i for i in range(len(blocks))]]
    elif layout in ["V", "v"]:
        layout = [[i] for i in range(len(blocks))]
    elif isinstance(layout, (np.ndarray, list)):
        layout = np.asarray(layout, dtype=np.int32)
        if layout.ndim==1:
            layout = np.expaind_dims(layout, 0)
        assert layout.ndim==2
    else:
        raise ValueError("invalid layout")
    
    channels = blocks[0].size(-1)
    if not isinstance(pad_col, (list, tuple)):
        pad_col = [pad_col]*channels
    pad_col = np.asarray(pad_col, dtype=np.float32)
    pad_col = torch.FloatTensor(pad_col).cuda()
    
    row_heights = [0]*len(layout)
    col_widths = [0]*len(layout[0])
    for row_idx, row in enumerate(layout):
        for col_idx, block_idx in enumerate(row):
            if block_idx>=0:
                h = blocks[block_idx].size(0)
                w = blocks[block_idx].size(1)
                row_heights[row_idx] = max(row_heights[row_idx], h)
                col_widths[col_idx] = max(col_widths[col_idx], w)

    rows = []
    for row_idx, row in enumerate(layout):
        row_data = []
        for col_idx, block_idx in enumerate(row):
            size_y = row_heights[row_idx]
            size_x = col_widths[col_idx]
            if block_idx==-1:
                row_data.append(torch.ones((size_y+padding*2, size_x+padding*2, channels), dtype=torch.float32, device=torch.device("cuda")) * pad_col)
            else:
                row_data.append(pad_to_size(blocks[block_idx], size_x, size_y, padding, pad_col))
        rows.append(torch.cat(row_data, axis=1))
    return torch.cat(rows, axis=0)

def reduce_3D(data, size, axis3D=0, mode3D="slice"):
    if mode3D=="slice":
        if axis3D==0:
            data = data[size.z//2]
        elif axis3D==1:
            data = data[:,size.y//2]
        elif axis3D==2:
            data = data[:,:,size.x//2]
        else:
            raise ValueError
    elif mode3D=="mean":
        data = torch.mean(data, dim=axis3D)
    elif mode3D=="max":
        data = torch.amax(data, dim=axis3D)
    else:
        raise ValueError
    return data

def _resample_block_data(data_list, vertex_coord_list, resampling_out_shape, ndims, fill_max_steps=0):
    if isinstance(resampling_out_shape, (list, tuple)):
        out_shape = torch.tensor(resampling_out_shape, dtype=torch.int32)
    elif isinstance(resampling_out_shape, torch.Tensor):
        out_shape = resampling_out_shape
    elif isinstance(resampling_out_shape, (int,)):
        out_shape = torch.tensor([resampling_out_shape]*ndims, dtype=torch.int32)
    else:
        raise TypeError("resampling_out_shape must be list, tensor, or int")
    data = sample_multi_coords_to_uniform_grid(data_list, vertex_coord_list, out_shape, is_cell_coords=False, fill_max_steps=fill_max_steps)
    
    #if ndims==3:
    #    raise NotImplementedError("TODO: implement 3D reduction.")
    
    return data

def save_block_data_image(block_data, domain, path, name, id, min_val=0, max_val=1, normalize=False, pad_col=1, layout="H", axis3D=0, mode3D="slice",
        vertex_coord_list=None, resampling_out_shape=10, fill_max_steps=0):
    
    if not (isinstance(block_data, (list, tuple)) and all(isinstance(_, torch.Tensor) for _ in block_data)):
        raise TypeError("block_data must be a list of tensors.")
    if not domain.getNumBlocks()==len(block_data):
        raise ValueError("block_data must match numbers of blocks on domain.")
    
    def cmp_data_block_size(data, block):
        dims = block.getSpatialDims()
        if not data.dim()==(dims+2):
            return False
        for dim in range(dims):
            if not data.size(dim+2)==block.getDim(dim+2):
                return False
        return True
    if not all(cmp_data_block_size(data, block) for data, block in zip(block_data, domain.getBlocks())):
        raise ValueError("data dimensionality and spatial shape must match blocks.")
    
    channels = block_data[0].size(1)
    if not all(data.size(1)==channels for data in block_data):
        raise ValueError("inconsistent use of channels (dim 1).")
    
    num_blocks = domain.getNumBlocks()
    ndims = domain.getSpatialDims()
    
    if vertex_coord_list is None or resampling_out_shape is None:
        if normalize:
            min_val=torch.min(block_data[0])
            max_val=torch.max(block_data[0])
            for blockIdx in range(1, num_blocks):
                min_val=torch.minimum(min_val, torch.min(block_data[blockIdx]))
                max_val=torch.maximum(max_val, torch.max(block_data[blockIdx]))
        
        
        padding = 1
        
        block_data_out = []
        for blockIdx in range(0, num_blocks):
            block = domain.getBlock(blockIdx)
            size = block.getSizes()
            #data = block.passiveScalar[0,0]
            data = block_data[blockIdx][0]
            if ndims==2:
                data = torch.permute(data, (1,2,0)) #CHW -> HWC
            elif ndims==3:
                data = torch.permute(data, (1,2,3,0))
                data = reduce_3D(data, size, axis3D, mode3D)
            data = (data-min_val) / (max_val - min_val)
            #block_data.append(torch.stack([data]*3, axis=-1))
            block_data_out.append(data)
        
        block_data = arrange_blocks(block_data_out, layout=layout, padding=padding, pad_col=pad_col)
    
    else:

        data = _resample_block_data(block_data, vertex_coord_list, resampling_out_shape, ndims, fill_max_steps=fill_max_steps)
        data = data[0] # NC[D]HW -> C[D]HW
        if ndims==2:
            data = torch.permute(data, (1,2,0)) # CHW -> HWC
        elif ndims==3:
            size = PISOtorch.Int4(x=data.size(-1),y=data.size(-2),z=data.size(-3))
            data = torch.permute(data, (1,2,3,0)) # CDHW -> DHWC
            data = reduce_3D(data, size, axis3D, mode3D)
    
        if normalize:
            min_val = torch.min(data)
            max_val = torch.max(data)
        data = (data-min_val) / (max_val - min_val)
        
        #block_data = torch.stack([data]*3, axis=-1)
        block_data = data
    
    if channels not in [1,3]:
        imgs = torch.split(block_data, 1, dim=-1)
        for i, img in enumerate(imgs):
            save_np_png(img.detach().cpu().numpy(), os.path.join(path, "%s_c%d_%04d.png"%(name, i, id,)))
    else:
        save_np_png(block_data.cpu().detach().numpy(), os.path.join(path, "%s_%04d.png"%(name, id,)))

def save_scalar_image(domain, path, name, id, min_val=0, max_val=1, normalize=False, pad_col=1, layout="H", axis3D=0, mode3D="slice",
        vertex_coord_list=None, resampling_out_shape=10, fill_max_steps=0):
    # axis: 0=z, 1=y, 2=x
    assert domain.getNumBlocks()>0
    ndims = domain.getSpatialDims()
    channels = domain.getPassiveScalarChannels()
    
    if not domain.hasPassiveScalar():
        return
    
    if vertex_coord_list is None or resampling_out_shape is None:
        if normalize:
            block = domain.getBlock(0)
            min_val=torch.min(block.passiveScalar)
            max_val=torch.max(block.passiveScalar)
            for blockIdx in range(1, domain.getNumBlocks()):
                block = domain.getBlock(blockIdx)
                min_val=torch.minimum(min_val, torch.min(block.passiveScalar))
                max_val=torch.maximum(max_val, torch.max(block.passiveScalar))
        
        
        padding = 1
        
        block_data = []
        for blockIdx in range(0, domain.getNumBlocks()):
            block = domain.getBlock(blockIdx)
            size = block.getSizes()
            #data = block.passiveScalar[0,0]
            data = block.passiveScalar[0]
            if ndims==2:
                data = torch.permute(data, (1,2,0)) #CHW -> HWC
            elif ndims==3:
                data = torch.permute(data, (1,2,3,0))
                data = reduce_3D(data, size, axis3D, mode3D)
            data = (data-min_val) / (max_val - min_val)
            #block_data.append(torch.stack([data]*3, axis=-1))
            block_data.append(data)
        
        block_data = arrange_blocks(block_data, layout=layout, padding=padding, pad_col=pad_col)
    
    else:
        data_list = [block.passiveScalar for block in domain.getBlocks()]

        data = _resample_block_data(data_list, vertex_coord_list, resampling_out_shape, ndims, fill_max_steps=fill_max_steps)
        data = data[0] # NC[D]HW -> C[D]HW
        if ndims==2:
            data = torch.permute(data, (1,2,0)) # CHW -> HWC
        elif ndims==3:
            size = PISOtorch.Int4(x=data.size(-1),y=data.size(-2),z=data.size(-3))
            data = torch.permute(data, (1,2,3,0)) # CDHW -> DHWC
            data = reduce_3D(data, size, axis3D, mode3D)
    
        if normalize:
            min_val = torch.min(data)
            max_val = torch.max(data)
        data = (data-min_val) / (max_val - min_val)
        
        #block_data = torch.stack([data]*3, axis=-1)
        block_data = data
    
    if channels not in [1,3]:
        imgs = torch.split(block_data, 1, dim=-1)
        for i, img in enumerate(imgs):
            save_np_png(img.detach().cpu().numpy(), os.path.join(path, "%s_c%d_%04d.png"%(name, i, id,)))
    else:
        save_np_png(block_data.cpu().detach().numpy(), os.path.join(path, "%s_%04d.png"%(name, id,)))

def save_pressure_image(domain, path, name, id, min_val=0, max_val=1, normalize=False, pad_col=(0,0,0.5), layout="H", axis3D=0, mode3D="slice",
        vertex_coord_list=None, resampling_out_shape=10, fill_max_steps=0):
    assert domain.getNumBlocks()>0
    #assert domain.getSpatialDims()==2
    ndims = domain.getSpatialDims()
    
    if vertex_coord_list is None or resampling_out_shape is None:
        if normalize:
            block = domain.getBlock(0)
            min_val=torch.min(block.pressure)
            max_val=torch.max(block.pressure)
            for blockIdx in range(1, domain.getNumBlocks()):
                block = domain.getBlock(blockIdx)
                min_val=torch.minimum(min_val, torch.min(block.pressure))
                max_val=torch.maximum(max_val, torch.max(block.pressure))
        
        padding = 1
        
        block_data = []
        for blockIdx in range(0, domain.getNumBlocks()):
            block = domain.getBlock(blockIdx)
            size = block.getSizes()
            data = block.pressure[0,0]
            if ndims==3:
                data = reduce_3D(data, size, axis3D, mode3D)
            data = (data-min_val) / (max_val - min_val)
            block_data.append(torch.stack([data]*3, axis=-1))
        
        block_data = arrange_blocks(block_data, layout=layout, padding=padding, pad_col=pad_col)

    else:
        data_list = [block.pressure for block in domain.getBlocks()]

        data = _resample_block_data(data_list, vertex_coord_list, resampling_out_shape, ndims, fill_max_steps=fill_max_steps)
        data = data[0,0]
        if ndims==3:
            size = PISOtorch.Int4(x=data.size(-1),y=data.size(-2),z=data.size(-3))
            data = reduce_3D(data, size, axis3D, mode3D)
    
        if normalize:
            min_val = torch.min(data)
            max_val = torch.max(data)
        data = (data-min_val) / (max_val - min_val)
        
        block_data = torch.stack([data]*3, axis=-1)
    
    save_np_png(block_data.cpu().detach().numpy(), os.path.join(path, "%s_%04d.png"%(name, id,)))

def save_velocity_image(domain, path, name, id, max_mag=1, normalize=False, pad_col=(1,1,1), layout="H", axis3D=0, mode3D="slice",
        vertex_coord_list=None, resampling_out_shape=10, fill_max_steps=0):
    assert domain.getNumBlocks()>0
    ndims = domain.getSpatialDims()
    
    if vertex_coord_list is None or resampling_out_shape is None:
        blocks = [domain.getBlock(blockIdx) for blockIdx in range(0, domain.getNumBlocks())]
        mags = [torch.linalg.vector_norm(block.velocity, dim=1) for block in blocks]
        
        if normalize:
            max_mag = torch.max(mags[0])
            for mag in mags:
                max_mag = torch.maximum(max_mag, torch.max(mag))
        
        #vel /= max_mag
        mags = [mag/max_mag for mag in mags]
        
        padding = 1
        
        block_data = []
        for block, mag in zip(blocks, mags):
            size = block.getSizes()
            if ndims==2:
                data = vel_to_color(block.velocity, mag)
            else:
                data = torch.abs(block.velocity[0])
                data = torch.permute(data, (1,2,3,0))
                data = reduce_3D(data, size, axis3D, mode3D)
                #data = torch.permute(block.velocity[0,:,size.z//2,:,:], (1,2,0)) / max_mag
                data = data / max_mag
            block_data.append(data)
        
        block_data = arrange_blocks(block_data, layout=layout, padding=padding, pad_col=pad_col)

    else:
        data_list = [block.velocity for block in domain.getBlocks()]

        data = _resample_block_data(data_list, vertex_coord_list, resampling_out_shape, ndims, fill_max_steps=fill_max_steps)
        mag = torch.linalg.vector_norm(data, dim=1)

        #print(data.size(), mag.size())

        if normalize:
            max_mag = torch.max(mag)
        mag = mag/max_mag

        if ndims==2:
            data =  vel_to_color(data, mag)
        else:
            data = torch.abs(data[0])
            size = PISOtorch.Int4(x=data.size(-1),y=data.size(-2),z=data.size(-3))
            data = torch.permute(data, (1,2,3,0)) #DHWC
            data = reduce_3D(data, size, axis3D, mode3D)
            data = data / max_mag
        
        block_data = data

        #print(block_data.size())
        
    
    save_np_png(block_data.cpu().detach().numpy(), os.path.join(path, "%s_%04d.png"%(name, id,)))

def save_velocity_source_image(domain, path, name, id, max_mag=1, normalize=False, pad_col=(1,1,1), layout="H", axis3D=0, mode3D="slice",
        vertex_coord_list=None, resampling_out_shape=10, fill_max_steps=0):
    assert domain.getNumBlocks()>0
    ndims = domain.getSpatialDims()
    
    if not any(block.hasVelocitySource() and not block.isVelocitySourceStatic for block in domain.getBlocks()):
        return
    
    blocks = domain.getBlocks()
    vel_sources = [block.velocitySource if (block.hasVelocitySource() and not block.isVelocitySourceStatic) else torch.zeros_like(block.velocity) for block in blocks]
    
    if vertex_coord_list is None or resampling_out_shape is None:
        #blocks = [domain.getBlock(blockIdx) for blockIdx in range(0, domain.getNumBlocks())]
        mags = [torch.linalg.vector_norm(vel_source, dim=1) for vel_source in vel_sources]
        
        if normalize:
            max_mag = torch.max(mags[0])
            for mag in mags:
                max_mag = torch.maximum(max_mag, torch.max(mag))
        
        #vel /= max_mag
        mags = [mag/max_mag for mag in mags]
        
        padding = 1
        
        block_data = []
        for block, vel_source, mag in zip(blocks, vel_sources, mags):
            size = block.getSizes()
            if ndims==2:
                data = vel_to_color(vel_source, mag)
            else:
                data = torch.abs(vel_source[0])
                data = torch.permute(data, (1,2,3,0))
                data = reduce_3D(data, size, axis3D, mode3D)
                #data = torch.permute(vel_source[0,:,size.z//2,:,:], (1,2,0)) / max_mag
                data = data / max_mag
            block_data.append(data)
        
        block_data = arrange_blocks(block_data, layout=layout, padding=padding, pad_col=pad_col)

    else:
        data_list = vel_sources #[block.velocitySource for block in domain.getBlocks()]

        data = _resample_block_data(data_list, vertex_coord_list, resampling_out_shape, ndims, fill_max_steps=fill_max_steps)
        mag = torch.linalg.vector_norm(data, dim=1)

        #print(data.size(), mag.size())

        if normalize:
            max_mag = torch.max(mag)
        mag = mag/max_mag

        if ndims==2:
            data =  vel_to_color(data, mag)
        else:
            data = torch.abs(data[0])
            size = PISOtorch.Int4(x=data.size(-1),y=data.size(-2),z=data.size(-3))
            data = torch.permute(data, (1,2,3,0)) #DHWC
            data = reduce_3D(data, size, axis3D, mode3D)
            data = data / max_mag
        
        block_data = data

        #print(block_data.size())
        
    
    save_np_png(block_data.cpu().detach().numpy(), os.path.join(path, "%s_%04d.png"%(name, id,)))

def save_velocity_exr(domain, path, name, id, scale=1, normalize=False, pad_col=(1,1,1), layout="H", axis3D=0, mode3D="slice",
        vertex_coord_list=None, resampling_out_shape=10, fill_max_steps=0):
    assert domain.getNumBlocks()>0
    ndims = domain.getSpatialDims()
    
    blocks = [domain.getBlock(blockIdx) for blockIdx in range(0, domain.getNumBlocks())]
    
    if normalize:
        mags = [torch.linalg.vector_norm(block.velocity, dim=1) for block in blocks]
        max_mag = torch.max(mags[0])
        for mag in mags:
            max_mag = torch.maximum(max_mag, torch.max(mag))
        scale = 1 / max_mag
    
    padding = 2
    
    block_data = []
    for block in blocks:
        size = block.getSizes()
        if ndims==2:
            data = torch.permute(block.velocity[0], (1,2,0))
            data = torch.nn.functional.pad(data, (0,1,0,0,0,0), value=0)
        else:
            data = torch.permute(block.velocity[0], (1,2,3,0))
            data = reduce_3D(data, size, axis3D, mode3D)
            #data = torch.permute(block.velocity[0,:,size.z//2,:,:], (1,2,0)) / max_mag
        data = data * scale
        block_data.append(data)
    
    block_data = arrange_blocks(block_data, layout=layout, padding=padding, pad_col=pad_col)
    
    save_np_exr(block_data.cpu().detach().numpy(), os.path.join(path, "%s_%04d.exr"%(name, id,)))

def save_transform_exr(domain, path, name, id, scale=1, pad_col=(1,1,1), layout="H", axis3D=0, mode3D="slice", vertex_coord_list=None, resampling_out_shape=10):
    assert domain.getNumBlocks()>0
    ndims = domain.getSpatialDims()
    assert ndims in [2,3]
    
    blocks = [domain.getBlock(blockIdx) for blockIdx in range(0, domain.getNumBlocks())]
    
    
    padding = 2
    
    if ndims==2:
        block_data = ["Tx","Ty","Tix","Tiy","det"] + ["Nx", "Ny", "Nxd", "Nyd", "Nd"]
    else:
        block_data = ["Tx","Ty","Tz","Tix","Tiy","Tiz","det"]
    block_data = {_:[] for _ in block_data}
    for block in blocks:
        size = block.getSizes()
        if block.hasTransform:
            if ndims==2:
                t = block.transform
                block_data["Tx"].append(scale * torch.nn.functional.pad(t[0,...,:ndims], (0,1,0,0,0,0), value=0))
                block_data["Ty"].append(scale * torch.nn.functional.pad(t[0,...,ndims:ndims*2], (0,1,0,0,0,0), value=0))
                block_data["Tix"].append(scale * torch.nn.functional.pad(t[0,...,ndims*2:ndims*3], (0,1,0,0,0,0), value=0))
                block_data["Tiy"].append(scale * torch.nn.functional.pad(t[0,...,ndims*3:ndims*4], (0,1,0,0,0,0), value=0))
                det =t[0,...,ndims*4:]
                block_data["det"].append(scale *  torch.nn.functional.pad(det, (0,2,0,0,0,0), value=0))
                
                block_data["Nx"].append(block_data["Tix"][-1] * det)
                block_data["Ny"].append(block_data["Tiy"][-1] * det)
                
                block_data["Nxd"].append(block_data["Nx"][-1][:,2:,:] - block_data["Nx"][-1][:,:-2,:])
                block_data["Nyd"].append(block_data["Ny"][-1][2:,:,:] - block_data["Ny"][-1][:-2,:,:])
                
                block_data["Nd"].append(block_data["Nxd"][-1][1:-1,:,:] + block_data["Nyd"][-1][:,1:-1,:])
                
            else:
                raise NotImplementedError("3D transformations.")
                data = torch.permute(block.velocity[0], (1,2,3,0))
                data = reduce_3D(data, size, axis3D, mode3D)
                #data = torch.permute(block.velocity[0,:,size.z//2,:,:], (1,2,0)) / max_mag
            
        else:
            raise NotImplementedError("Block has no transformation set.")
        #data = data * scale
        #block_data.append(data)
    
    for key, data in block_data.items():
        data = arrange_blocks(data, layout=layout, padding=padding, pad_col=pad_col)
        
        save_np_exr(data.cpu().detach().numpy(), os.path.join(path, "%s_%s_%04d.exr"%(name, key, id,)))

def save_domain_images(domain, out_dir, it, layout="H", norm_p=True, max_mag=1, mode3D="mean", vel_exr=False,
        vertex_coord_list=None, resampling_out_shape=10, fill_max_steps=0):
    ndims = domain.getSpatialDims()
    if ndims==2:
        save_scalar_image(domain, out_dir, "d", it, layout=layout, mode3D=mode3D)
        save_pressure_image(domain, out_dir, "p", it, normalize=norm_p, layout=layout)
        save_velocity_image(domain, out_dir, "v", it, max_mag=max_mag, layout=layout)
        save_velocity_source_image(domain, out_dir, "vs", it, max_mag=max_mag, layout=layout)
        if vertex_coord_list is not None and resampling_out_shape is not None:
            save_scalar_image(domain, out_dir, "d-r", it, layout=layout, mode3D=mode3D,
                vertex_coord_list=vertex_coord_list, resampling_out_shape=resampling_out_shape, fill_max_steps=fill_max_steps)
            save_pressure_image(domain, out_dir, "p-r", it, normalize=norm_p, layout=layout, mode3D=mode3D,
                vertex_coord_list=vertex_coord_list, resampling_out_shape=resampling_out_shape, fill_max_steps=fill_max_steps)
            save_velocity_image(domain, out_dir, "v-r", it, max_mag=max_mag, layout=layout, mode3D=mode3D,
                vertex_coord_list=vertex_coord_list, resampling_out_shape=resampling_out_shape, fill_max_steps=fill_max_steps)
            save_velocity_source_image(domain, out_dir, "vs-r", it, max_mag=max_mag, layout=layout, mode3D=mode3D,
                vertex_coord_list=vertex_coord_list, resampling_out_shape=resampling_out_shape, fill_max_steps=fill_max_steps)
            if vel_exr: save_velocity_exr(domain, out_dir, "vx-r", it, layout=layout, mode3D=mode3D,
                vertex_coord_list=vertex_coord_list, resampling_out_shape=resampling_out_shape, fill_max_steps=fill_max_steps)
    elif ndims==3:
        if not isinstance(mode3D, (list, tuple)):
            mode3D = [mode3D]
        for m3D in mode3D:
            for dim, name in ((0,"z"),(1,"y"),(2,"x")):
                save_scalar_image(domain, out_dir, "d-%s-%s"%(name, m3D), it, layout=layout, axis3D=dim, mode3D=m3D)
                save_pressure_image(domain, out_dir, "p-%s-%s"%(name, m3D), it, normalize=norm_p, layout=layout, axis3D=dim, mode3D=m3D)
                save_velocity_image(domain, out_dir, "v-%s-%s"%(name, m3D), it, max_mag=max_mag, layout=layout, axis3D=dim, mode3D=m3D)
                if vertex_coord_list is not None and resampling_out_shape is not None:
                    save_scalar_image(domain, out_dir, "d-r-%s-%s"%(name, m3D), it, layout=layout, axis3D=dim, mode3D=m3D,
                        vertex_coord_list=vertex_coord_list, resampling_out_shape=resampling_out_shape, fill_max_steps=fill_max_steps)
                    save_pressure_image(domain, out_dir, "p-r-%s-%s"%(name, m3D), it, normalize=norm_p, layout=layout, axis3D=dim, mode3D=m3D,
                        vertex_coord_list=vertex_coord_list, resampling_out_shape=resampling_out_shape, fill_max_steps=fill_max_steps)
                    save_velocity_image(domain, out_dir, "v-r-%s-%s"%(name, m3D), it, max_mag=max_mag, layout=layout, axis3D=dim, mode3D=m3D,
                        vertex_coord_list=vertex_coord_list, resampling_out_shape=resampling_out_shape, fill_max_steps=fill_max_steps)
                    if vel_exr: save_velocity_exr(domain, out_dir, "vx-r-%s-%s"%(name, m3D), it, layout=layout, mode3D=m3D,
                        vertex_coord_list=vertex_coord_list, resampling_out_shape=resampling_out_shape, fill_max_steps=fill_max_steps)