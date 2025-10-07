import os
import numpy as np
import torch
import PISOtorch
import lib.data.shapes as shapes

try: # for compatibility with existing setups that do not include pyvista
    import pyvista as pv
except ImportError:
    def save_vtk(domain : PISOtorch.Domain, out_dir, it, vertex_coord_list=None):
        raise ImportError("pyvista not available")
else:
    def save_vtk(domain : PISOtorch.Domain, out_dir, it, vertex_coord_list=None):
        assert domain.getNumBlocks() == len(vertex_coord_list)
        assert domain.getNumBlocks() > 0
        assert not vertex_coord_list is None
        multi_block = pv.MultiBlock()
        ndims = vertex_coord_list[0].shape[1]
        assert ndims in (2, 3)


        points_list = []
        index_tensor_list = []
        start_index = 0
        channel_names = []
        cells_list = []
        for blockIdx, vertex_coords in enumerate(vertex_coord_list):
            vertex_coords = vertex_coords.cpu()
            points_shape = vertex_coords[0, 0].shape
            numel = int(vertex_coords.numel() / vertex_coords.shape[1])
            index_tensor = torch.linspace(start_index, start_index + numel - 1, numel).reshape(points_shape).to(torch.int)
            index_tensor_list.append(index_tensor)
            start_index += numel

            # channel dimension last
            points_list.append(vertex_coords.reshape(vertex_coords.shape[1], -1).permute(1, 0))

            
            cells_shape = [d - 1 for d in points_shape]
            # different number of corners for 2d and 3d grids
            quad_start = torch.ones(cells_shape) * (4 if ndims==2 else 8)
            cell_indices = torch.stack(torch.meshgrid(*[torch.linspace(0, c - 1, c) for c in cells_shape], indexing="ij")).to(torch.int32).to("cuda").flip(0).unsqueeze(0).contiguous()
            
            corner_indices = PISOtorch.GetElementCornerValues(index_tensor.unsqueeze(0).unsqueeze(0).to(torch.float32).to("cuda"), cell_indices).cpu().squeeze(0)
            order = [0, 1, 3, 2] if ndims == 2 else [0, 1, 3, 2, 4, 5, 7, 6]
            corner_indices = torch.stack([corner_indices[i] for i in order], dim=0)

            cells = torch.cat([quad_start.unsqueeze(0), corner_indices], dim=0)
            cells_flat = cells.reshape(cells.shape[0], -1).permute(1, 0).flatten()
            cells_list.append(cells_flat)

        # get cell vertices in neighbor index order, need to be rearranged
        all_vertices = torch.cat(points_list, dim=0)


        if ndims == 2:
            all_vertices = torch.cat([all_vertices, torch.zeros(all_vertices.shape[0], 1)], dim=-1)
        all_vertices = all_vertices.numpy()
        
        all_cells = torch.cat(cells_list, dim=0).to(torch.int32).numpy()

        cell_types = None
        if ndims == 2:
            cell_types = np.array([pv.CellType.QUAD for i in range(int(all_cells.shape[0] / 5))], dtype=np.uint8)
        else:
            cell_types = np.array([pv.CellType.HEXAHEDRON for i in range(int(all_cells.shape[0] / 9))], dtype=np.uint8)

        grid = pv.UnstructuredGrid(all_cells, cell_types, all_vertices.flatten())
        p_list = []
        v_list = []
        for blockIdx, vertex_coords in enumerate(vertex_coord_list):
            block = domain.getBlock(blockIdx)
            pressure_tensor = block.pressure.cpu().squeeze(0).squeeze(0)
            if ndims == 2:
                pressure_tensor.permute(1, 0)
            else:
                pressure_tensor.permute(2, 1, 0)
            p_list.append(pressure_tensor.flatten())
            velocity_tensor = None
            if ndims == 2:
                velocity_tensor = block.velocity.cpu().squeeze(0).squeeze(0).swapaxes(0, -1).swapaxes(0, 1).reshape(-1, ndims)
                # append 0 z dimension to make ParaView happy
                velocity_tensor = torch.cat([velocity_tensor, torch.zeros(velocity_tensor.shape[0], 1)], dim=-1)
            else:
                velocity_tensor = block.velocity.cpu().squeeze(0).squeeze(0).swapaxes(0, -1).permute(1, 2, 0, 3).reshape(-1, ndims)
            v_list.append(velocity_tensor)
        
        all_pressures = torch.cat(p_list)
        all_velocities = torch.cat(v_list)
        grid.cell_data["p"] = all_pressures
        grid.cell_data["v"] = all_velocities
        filename = os.path.join(out_dir, "vtk_out_%04d.vtu"%(it))
        grid.save(filename)
        return grid


        

    