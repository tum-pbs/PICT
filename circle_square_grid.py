import os
import numpy as np
import torch


import lib.data.shapes as shapes
from lib.util.output import plot_grids
from lib.util.logging import setup_run, get_logger, close_logging

def makeCircleSquareGrid(r_obstacle:float=1, r_circle:float=0.2, res_circle=24, r_quad:float=0.8, res_quad:int="AUTO", viscosity=0.01, dtype=torch.float32, path=None):
    
    if path:
        os.makedirs(path, exist_ok=True)
    
    circle_cell_width = 2 * np.pi * r_obstacle / (res_circle*4) # likely smallest cell size
    grid_width = (r_obstacle + r_circle + r_quad) * 2
    resample_res = [int(np.ceil(grid_width / circle_cell_width))]*2
    
    # coordinates are centered on the obstacle
    
    # --- INNER CIRCLE ---
    
    # make the circular grids
    # circle generation: 
    # - angle starts at x-axis and goes counterclockwise
    # - grid: x goes along angle, y goes inner to outer
    # - coords: y is up
    # - return: torch.tensor with shape NCHW
    t_r1 = r_obstacle
    t_r2 = t_r1 + r_circle
    x = res_circle
    circle_top_coords = shapes.make_torus_2D(x, r1=t_r1, r2=t_r2, start_angle=135, angle=-90, dtype=dtype) # y up, x right
    #circle_top_coords = torch.flip(torus_top_coords, (-2,)) # y down, x right
    circle_right_coords = shapes.make_torus_2D(x, r1=t_r1, r2=t_r2, start_angle=45, angle=-90, dtype=dtype) # y right, x down
    #circle_right_coords = torch.movedim(torus_right_coords, -1, -2) # y down, x right
    circle_bot_coords = shapes.make_torus_2D(x, r1=t_r1, r2=t_r2, start_angle=-45, angle=-90, dtype=dtype) # y down, x left
    #circle_bot_coords = torch.flip(torus_bot_coords, (-1,)) # y down, x right
    circle_left_coords = shapes.make_torus_2D(x, r1=t_r1, r2=t_r2, start_angle=-135, angle=-90, dtype=dtype) # y left, x up
    #circle_left_coords = torch.movedim(torus_left_coords, -1, -2) # y up, x left
    #circle_left_coords = torch.flip(torus_left_coords, (-2,-1)) # y down, x right
    
    y = circle_top_coords.size(-2)-1
    
    
    # --- QUAD ---
    
    # make the padding-to-square grids (could be combined with circle segments)
    # corners, inner and outer
    # inner edge, from circle outer transforms/coords
    #generate_grid_vertices_2D(res, corner_vertices, border_vertices=None, x_weights=None, y_weights=None, dtype=torch.float32)
        # res: grid resolution in [y,x]
        # corner_vertices: [-x-y,+x-y,-x+y,+x+y], (x,y)-tuple
        # border_vertices: [-x,+x,-y,+y], lists of (x,y)-tuple. will be interpolated from corners if None
    
    # corners
    quad_r_outer = r_obstacle + r_circle + r_quad
    quad_r_inner = np.sin(np.deg2rad(45)) * t_r2
    quad_corners_top = [(-quad_r_inner,quad_r_inner), (quad_r_inner,quad_r_inner), (-quad_r_outer,quad_r_outer), (quad_r_outer,quad_r_outer)]
    quad_corners_right = [(quad_r_inner,quad_r_inner), (quad_r_inner,-quad_r_inner), (quad_r_outer,quad_r_outer), (quad_r_outer,-quad_r_outer)]
    quad_corners_bot = [(quad_r_inner,-quad_r_inner), (-quad_r_inner,-quad_r_inner), (quad_r_outer,-quad_r_outer), (-quad_r_outer,-quad_r_outer)]
    quad_corners_left = [(-quad_r_inner,-quad_r_inner), (-quad_r_inner,quad_r_inner), (-quad_r_outer,-quad_r_outer), (-quad_r_outer,quad_r_outer)]
    
    # borders, specify round circle borders, rest is linear interpolated from corners
    #LOG.info("circle_top_coords: %s", circle_top_coords)
    def make_border(t):
        return torch.movedim(t, 0, 1).cpu().clone().numpy().tolist()
    quad_border_top = [None, None, make_border(circle_top_coords[0,:,-1,:]), None]
    quad_border_right = [None, None, make_border(circle_right_coords[0,:,-1,:]), None]
    quad_border_bot = [None, None, make_border(circle_bot_coords[0,:,-1,:]), None]
    quad_border_left = [None, None, make_border(circle_left_coords[0,:,-1,:]), None]
    
    # generate the coordinates
    quad_res_angular = x + 1
    quad_res_radial = int(np.ceil(r_quad/r_circle)) * y + 1 if res_quad=="AUTO" else res_quad
    quad_coords_top = shapes.generate_grid_vertices_2D([quad_res_radial, quad_res_angular], quad_corners_top, quad_border_top, dtype=torch.float32)
    quad_coords_right = shapes.generate_grid_vertices_2D([quad_res_radial, quad_res_angular], quad_corners_right, quad_border_right, dtype=torch.float32)
    quad_coords_bot = shapes.generate_grid_vertices_2D([quad_res_radial, quad_res_angular], quad_corners_bot, quad_border_bot, dtype=torch.float32)
    quad_coords_left = shapes.generate_grid_vertices_2D([quad_res_radial, quad_res_angular], quad_corners_left, quad_border_left, dtype=torch.float32)
    
    # fuse grids
    # coords = torch.cat([
    #     torch.cat([circle_top_coords[:,:,:-1,:-1], circle_right_coords[:,:,:-1,:-1], circle_bot_coords[:,:,:-1,:-1], circle_left_coords[:,:,:-1,:]], dim=-1),
    #     torch.cat([quad_coords_top[:,:,:,:-1], quad_coords_right[:,:,:,:-1], quad_coords_bot[:,:,:,:-1], quad_coords_left], dim=-1)
    #     ], dim=-2)
    
    #coords = shapes.extrapolate_boundary_layers(coords, [("-y", 2)])
    
    if path:
        plot_grids([circle_top_coords, circle_right_coords, circle_bot_coords, circle_left_coords, quad_coords_top, quad_coords_right, quad_coords_bot, quad_coords_left],
                   color=["r","b","r","b","g","y","g","y"], path=path, type="pdf")

if __name__=="__main__":
    run_dir = setup_run("./test_runs", 
        name="circle-quad-grid"
    )

    makeCircleSquareGrid(path=run_dir)

    close_logging()