
import numpy as np
import lib.data.TCF_tools as TCF_tools

class OpenFOAMProfile:
    # should be compatible with TorrojaProfile
    def __init__(self, data_path, viscosity=2e-5, u_wall=0.010301294851713142):
        
        self.viscosity = viscosity
        self.u_wall = u_wall
        
        def to_wall_vel(vel, order=1):
            return TCF_tools.vel_to_vel_wall(vel, u_wall, order)
        
        def to_wall_pos(coords):
            return TCF_tools.pos_to_pos_wall(coords, viscosity, u_wall)
        
        self.profiles = {}
        self.data = {}
        
        with np.load(data_path) as np_file:
            for key, value in np_file.items():
                self.data[key] = value
        
        self.Re_wall = self.data["ret"]
        self.profiles["U+"] = to_wall_vel(self.data["U"])
        self.profiles["u'+"] = to_wall_vel(np.sqrt(self.data["urms"]))
        self.profiles["v'+"] = to_wall_vel(np.sqrt(self.data["vrms"]))
        self.profiles["w'+"] = to_wall_vel(np.sqrt(self.data["wrms"]))
        self.profiles["uv'+"] = to_wall_vel(self.data["uv"], order=2)
        self.profiles["y/h"] = self.data["y"]
        self.profiles["y+"] = to_wall_pos(self.data["y"])
    
    def get_full_pos_y(self):
        key = "y/h"
        if not key in self.profiles:
            raise KeyError("y position data not found.")
        return np.concatenate((self.profiles[key] - 1, 1 - self.profiles[key][::-1]))
    
    def get_full_data(self, key):
        if key not in ["U+", "u'+", "v'+", "w'+", "uv'+"]:
            raise NotImplementedError("Unsupported profile: %s"%(key,))
        if key not in self.profiles:
            raise KeyError("%s data not found."%(key,))
        
        if key in ["U+", "u'+", "v'+", "w'+"]:
            return np.concatenate((self.profiles[key], self.profiles[key][::-1]))
        if key in ["uv'+"]:
            return np.concatenate((-self.profiles[key], self.profiles[key][::-1]))