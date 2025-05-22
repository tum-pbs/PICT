import torch
import PISOtorch


def _swap_xyz_zyx(idx, dims):
    # works both ways
    return dims-1-idx
    
def _get_block_spatial_shape(block):
    dims = block.getSpatialDims() # in [1,2,3]
    sizes = block.getSizes() # xyzw, always length 4
    return [sizes[_swap_xyz_zyx(dim, dims)] for dim in range(dims)] # ((z)y)x

def _direction_to_axis(direction):
    return direction>>1

def _direction_is_upper(direction):
    return direction&1

class AxesMapping:
    def __init__(self, axis_mapping):
        if not (isinstance(axis_mapping, list) and all(isinstance(x, int) for x in axis_mapping)):
            raise TypeError("axis_mapping must be list of int.")
        dims = len(axis_mapping)
        if not all(0<=i and i<(dims*2) for i in axis_mapping):
            raise ValueError("axis_mapping entries mut be valid target axes.")
        axes = set(_direction_to_axis(direction) for direction in axis_mapping)
        if not dims==len(axes):
            raise ValueError("Target axes must be unique: %s"%(axis_mapping,))
        self.__axis_mapping = axis_mapping[:]
    
    def __len__(self):
        return len(self.__axis_mapping)
    
    def __getitem__(self, idx):
        return self.__axis_mapping[idx]
    
    def __str__(self):
        return str(self.__axis_mapping)
    
    def get_axis(self, idx):
        return _direction_to_axis(self.__axis_mapping[idx])
    
    def is_inverted(self, idx):
        return _direction_is_upper(self.__axis_mapping[idx])
    
    def copy(self):
        return self.__class__(self.__axis_mapping)
    
    @classmethod
    def make_identity(cls, dims):
        return cls([dim*2 for dim in range(dims)])
    
    @classmethod
    def from_bound(cls, bound_dir, bound_axes, dims):
        bound_axis = _direction_to_axis(bound_dir)
        bound_is_upper = _direction_is_upper(bound_dir)
        axes_mapping = [None]*dims
        axes_mapping[bound_axis] = _direction_to_axis(bound_axes[0])*2 + int(bound_is_upper==_direction_is_upper(bound_axes[0]))
        for i in range(1, dims):
            axes_mapping[(bound_axis+i)%dims] = bound_axes[i]
        return cls(axes_mapping)
    
    def make_inverted(self):
        axes_mapping = [None]*len(self)
        for i in range(len(self)):
            axes_mapping[self.get_axis(i)] = i*2 + self.is_inverted(i)
        return self.__class__(axes_mapping)
    
    def make_fused_with(self, other_mapping):
        # is this commutative?
        axes_mapping = [None]*len(self)
        for i in range(len(self)):
            own_target_axis = self.get_axis(i)
            axes_mapping[i] = other_mapping.get_axis(own_target_axis)*2 + int(other_mapping.is_inverted(own_target_axis)!=self.is_inverted(i))
        
        return self.__class__(axes_mapping)
    
    def transform_direction(self, direction):
        axis = _direction_to_axis(direction)
        is_upper = _direction_is_upper(direction)
        return self.get_axis(axis)*2 + int(is_upper!=self.is_inverted(axis))
    
    def get_permute_flip_args(self):
        # call torch.permute first
        dims = len(self)
        permute_args = [0,1] + [_swap_xyz_zyx(self.get_axis(_swap_xyz_zyx(dim, dims)), dims)+2 for dim in range(dims)]
        flip_args = [_swap_xyz_zyx(dim, dims)+2 for dim in range(dims) if self.is_inverted(dim)]
        return permute_args, flip_args


class MultiblockConv(torch.nn.Module):
    def __init__(self, domain:PISOtorch.Domain, fixed_boundary_padding:list,
            in_channels:int, out_channels:int, kernel_size:int, stride:int=1, padding:int='same', dilation:int=1,
            groups:int=1, bias:bool=True, padding_mode:str='zeros', device=None, dtype=None):
        #raise NotImplementedError
        super().__init__()
        
        dims = domain.getSpatialDims()
        if dims==1:
            conv = torch.nn.Conv1d
        elif dims==2:
            conv = torch.nn.Conv2d
        elif dims==3:
            conv = torch.nn.Conv3d
        else:
            raise ValueError
        
        self.dims = dims
        if not((isinstance(fixed_boundary_padding, (list, tuple))
                    and len(fixed_boundary_padding)==in_channels
                    and all(pad in ["VEL_U", "VEL_V", "VEL_W", "S", "ZERO"] for pad in fixed_boundary_padding)
                ) or (
                    fixed_boundary_padding in ["ZERO"]
                )
            ):
            raise ValueError
        self.fixed_boundary_padding = fixed_boundary_padding
        self.neighbor_precedent = "BLOCK" # "BORDER", "BLOCK"
        
        self.in_channels = in_channels
        if not kernel_size in [3, 5, 7]: #, [3]*dims]:
            raise NotImplementedError("kernel_size must be 3, 5, or 7.")
        self.kernel_size = kernel_size
        if not padding in [kernel_size//2, "same"]:
            raise NotImplementedError("padding must be 1 or 'same'")
        self.padding = kernel_size//2 #if padding=="same" else padding
        if not dilation==1:
            raise NotImplementedError("dilation is not supported")
        
        self.device = device #if device is not None else domain.getDevice()
        self.dtype = dtype #if dtype is not None else domain.getDtype()
        
        blocks = domain.getBlocks()
        padding_infos = []
        for block in blocks:
            padding_infos.append(self._make_padding_info(block, blocks))
        self.padding_infos = padding_infos
        
        self.conv_layer = conv(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation,
            groups=groups, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)
    
    def _flatten_neighbor_index(self, pos):
        assert len(pos)==self.dims
        flat_pos = 0
        for axis in range(self.dims):
            flat_pos += pos[axis]*(3**axis)
        return flat_pos
    
    def _unflatten_neighbor_index(self, flat_pos):
        pos = [0]*self.dims
        strides = [3**dim for dim in range(self.dims)]
        for axis in range(self.dims):
            pos[axis] = (flat_pos//strides[axis])%3
        return pos
    
    def _add_neighbor_info(self, block, pos, axes_mapping, axes, neighbor_infos, blocks):
        
        
        flat_index = self._flatten_neighbor_index(pos)
        
        for axis in axes:
            remaining_axes = axes[:] #copy
            remaining_axes.remove(axis)
            for is_upper in range(2):
                n_pos = pos[:]
                n_pos[axis] += is_upper*2 - 1 #[0,1] -> [-1,1]
                n_pos_flat = self._flatten_neighbor_index(n_pos)
                # !need to adjust axes and directions based on axes_mapping!
                n_dir = axes_mapping.transform_direction(axis*2 + is_upper)
                bound = block.getBoundaries()[n_dir]
                bound_type = bound.type
                if bound_type==PISOtorch.BoundaryType.FIXED:
                    if neighbor_infos[n_pos_flat] is None or (neighbor_infos[n_pos_flat][0]=="BLOCK" and self.neighbor_precedent=="BORDER"):
                        neighbor_infos[n_pos_flat] = ("BORDER", )
                        # TODO: info about the boundary for padding with boundary values?
                    #raise NotImplementedError
                elif bound_type==PISOtorch.BoundaryType.CONNECTED:
                    # !need to adjust axes and directions based on bound.axes! -> axis mapping object (similarly done in C++)
                    if neighbor_infos[n_pos_flat] is None or (neighbor_infos[n_pos_flat][0]=="BORDER" and self.neighbor_precedent=="BLOCK"):
                        n_block = bound.getConnectedBlock()
                        n_block_idx = blocks.index(n_block)
                        n_axes_mapping = AxesMapping.from_bound(n_dir, bound.axes, self.dims)
                        n_axes_mapping = axes_mapping.make_fused_with(n_axes_mapping)
                        
                        neighbor_infos[n_pos_flat] = ("BLOCK", n_block_idx, n_axes_mapping.copy())
                        
                        self._add_neighbor_info(n_block, n_pos, n_axes_mapping, remaining_axes, neighbor_infos, blocks)
                    #raise NotImplementedError
                elif bound_type==PISOtorch.BoundaryType.PERIODIC:
                    if neighbor_infos[n_pos_flat] is None or (neighbor_infos[n_pos_flat][0]=="BORDER" and self.neighbor_precedent=="BLOCK"):
                        n_block_idx = blocks.index(block)
                        
                        neighbor_infos[n_pos_flat] = ("BLOCK", n_block_idx, axes_mapping.copy())
                        
                        self._add_neighbor_info(block, n_pos, axes_mapping, remaining_axes, neighbor_infos, blocks)
                    #raise NotImplementedError
                else:
                    raise TypeError("Boundary type not supported.")
                
    
    def _make_padding_info(self, block, blocks):
        # can be precomputed
        
        # how to index neighbors?
        # 3^dims grid, (lower, center, upper) -> flat index from coords
        neighbor_infos = [None]*(3**self.dims)
        
        # iterate neighbors
        # axes (up and down)
        axes = [axis for axis in range(self.dims)]
        
        self._add_neighbor_info(block, pos=[1]*self.dims, axes_mapping=AxesMapping.make_identity(self.dims), axes=axes,
            neighbor_infos=neighbor_infos, blocks=blocks)# starting at center with coords (1,1,1)
        
        # unreachable corners:
        for idx in range(len(neighbor_infos)):
            if neighbor_infos[idx] is None:
                neighbor_infos[idx] = ("BORDER", )
        
        # center:
        neighbor_infos[self._flatten_neighbor_index([1]*self.dims)] = ("CENTER", )
        
        block_shape = _get_block_spatial_shape(block) # zyx
        
        # slicing info or zero padding size
        padding_infos = [None]*(3**self.dims)
        for idx in range(3**self.dims):
            pos = self._unflatten_neighbor_index(idx) # a 1 is center, 0 lower, 2 upper
            # if pos is not 1 for an axis, this dim has padding size; otherwise it has the center block's size in that dimension.
            if neighbor_infos[idx][0] == "BORDER":
                padding_infos[idx] = ("ZERO", [self.padding if pos[_swap_xyz_zyx(dim, self.dims)]!=1 else block_shape[dim] for dim in range(self.dims)]) # spatial size (z,y,x) of the padding
            elif neighbor_infos[idx][0] == "BLOCK":
                _, block_idx, axes_mapping = neighbor_infos[idx]
                inv_axes_mapping = axes_mapping.make_inverted()
                permute_args, flip_args = inv_axes_mapping.get_permute_flip_args()
                
                # slicing is applied to neighbor block, so use its size
                n_block_shape = _get_block_spatial_shape(blocks[block_idx])
                # slicing is applied to the permuted block, so permute shape axes accordingly
                n_block_shape_permuted = [None]*self.dims
                for i in range(self.dims):
                    n_block_shape_permuted[i] = n_block_shape[permute_args[i+2]-2]
                
                slice_args = [slice(None)]*2 # batch and channel dimension are not sliced
                for dim in range(self.dims):
                    dim_inv = _swap_xyz_zyx(dim, self.dims) # slicing zyx-order tensors
                    if pos[dim_inv]==0: # center's lower boundary, so get slice from neighbor's upper part
                        slice_args.append(slice(n_block_shape_permuted[dim]-self.padding, n_block_shape_permuted[dim]))
                    elif pos[dim_inv]==2: #upper
                        slice_args.append(slice(self.padding))
                    else: #==1, center
                        slice_args.append(slice(None))
                    
                padding_infos[idx] = ("INPUT", block_idx, flip_args, permute_args, slice_args) # to be handled in the same order: get block data, flip, permute, slice.
                
            elif neighbor_infos[idx][0] == "CENTER":
                padding_infos[idx] = ("CENTER", )
        
        return padding_infos
    
    def _pad_tensor(self, idx, tensor_list):
        # padding 1 or 2 is relatively easy since the minimum block size is 2
        # larger padding must potentially handle multiple block connections
        
        batch_size = tensor_list[0].size(0)
        
        padding_infos = self.padding_infos[idx]
        tensor_grid = [None]*(3**self.dims)
        for flat_pos in range(3**self.dims):
            padding_info = padding_infos[flat_pos]
            if padding_info[0]=="ZERO":
                tensor_grid[flat_pos] = torch.zeros([batch_size, self.in_channels] + padding_info[1], device=self.device, dtype=self.dtype)
            elif padding_info[0]=="CENTER":
                tensor_grid[flat_pos] = tensor_list[idx]
            elif padding_info[0]=="INPUT":
                _, block_idx, flip_args, permute_args, slice_args = padding_info
                tensor = tensor_list[block_idx]
                if flip_args:
                    tensor = torch.flip(tensor, flip_args)
                if permute_args:
                    tensor = torch.permute(tensor, permute_args)
                if slice_args:
                    tensor = tensor[slice_args]
                tensor_grid[flat_pos] = tensor
        
        for cat_dim in range(self.dims-1, -1, -1): # tensor shape is zyx, cat in xyz order
            new_tensor_grid = []
            for i in range(len(tensor_grid)//3):
                new_tensor_grid.append(torch.cat(tensor_grid[i*3:i*3+3], dim = cat_dim+2))
            tensor_grid = new_tensor_grid
        
        return tensor_grid[0]
    
    def forward(self, tensor_list, domain=None):
        if domain is not None:
            if not domain.getSpatialDims()==self.dims:
                raise ValueError("domain dimensionality does not match this layer.")
            
            blocks = domain.getBlocks()
            num_blocks = len(blocks)
            
            # check tensor list
            if not (isinstance(tensor_list, (list, tuple)) and all(isinstance(tensor, torch.Tensor) for tensor in tensor_list)):
                raise TypeError("tensor_list must be a list of torch.Tensor.")
            if not len(tensor_list)==num_blocks:
                raise ValueError("length of tensor_list must match domain blocks.")
            if num_blocks==0:
                raise ValueError("no input tensors.")
                
            # each tensor must have shape NCDHW; channels C must match self.conv_layer in_channels; spatial dimensions DHW must match those of the correspoinding block of domain.
            rank = self.dims + 2
            if any(not tensor.dim()==rank for tensor in tensor_list):
                raise ValueError("tensors must have shape NCDHW.")
            batch_size = tensor_list[0].size(0)
            if any(not tensor.size(0)==batch_size for tensor in tensor_list):
                raise ValueError("tensors have inconsistent batch size.")
            if any(not tensor.size(1)==self.in_channels for tensor in tensor_list):
                raise ValueError("channel dimension of tensors must match in_channels.")
            block_sizes = [block.getSizes() for block in blocks]
            if any(any(not tensor.size(dim+2)==block_size[self.dims-1-dim] for dim in range(self.dims)) for tensor, block_size in zip(tensor_list, block_sizes)):
                raise ValueError("tensor spatial dimensions must match block sizes")
            
            # now tensors should be consistent and match domain blocks, with a variable amount of channels.
        
        out_tensor_list = []
        
        for idx in range(len(tensor_list)):
            tensor_padded = self._pad_tensor(idx, tensor_list)
            
            tensor_out = self.conv_layer(tensor_padded)
            out_tensor_list.append(tensor_out)
        
        return out_tensor_list

class MultiblockConv_naive(torch.nn.Module):
    # version with default padding per block
    # for testing and comparison, or 1x1 convolutions
    def __init__(self, dims,  *conv_args, **conv_kwargs):
        super().__init__()
        if dims==1:
            conv = torch.nn.Conv1d
        elif dims==2:
            conv = torch.nn.Conv2d
        elif dims==3:
            conv = torch.nn.Conv3d
        else:
            raise ValueError
        
        self.conv_layer = conv(*conv_args, **conv_kwargs)
    
    def forward(self, tensor_list, domain=None):
        out_tensor_list = []
        
        for idx in range(len(tensor_list)):
            tensor_out = self.conv_layer(tensor_list[idx])
            out_tensor_list.append(tensor_out)
        
        return out_tensor_list

class MultiblockReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor_list, domain=None):
        out_tensor_list = []
        for tensor in tensor_list:
            out_tensor_list.append(torch.nn.functional.relu(tensor))
        return out_tensor_list
