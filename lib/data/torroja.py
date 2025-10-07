import os, struct
import numpy as np
from matplotlib import pyplot as plt

class TorrojaProfile:
    # https://torroja.dmt.upm.es/channels/data/
    def __init__(self, base_path, Re):
        path = os.path.join(base_path, "Re%d.prof"%Re)
        
        self.Re_wall = Re
        
        with open(path, "r") as profiles_file:
            line = profiles_file.readline()
            while "End of Header" not in line:
                line = profiles_file.readline()
            line = profiles_file.readline() # one empty line
            line = profiles_file.readline() # ny and Re
            self.ny = int(line.split()[3][:-1])
            line = profiles_file.readline() # empty line
            line = profiles_file.readline() # field names
            self.field_names = line.split()[1:]
            line = profiles_file.readline() # empty line
            line = profiles_file.readline() # ----- line
            stats = [[] for _ in self.field_names]
            
            #i = 0
            for line in profiles_file:
                #i += 1
                for idx, value in enumerate(line.split()):
                    stats[idx].append(float(value))
            #print("read", i, "lines")
            self.profiles = {name: np.asarray(values) for name, values in zip(self.field_names, stats)}
    
    def get_full_pos_y(self):
        key = "y/h"
        if not key in self.profiles:
            raise KeyError("y position data not found.")
        return np.concatenate((self.profiles[key] - 1, 1 - self.profiles[key][::-1]))
    
    def get_full_data(self, key):
        if key not in ["U+", "u'+", "v'+", "w'+", "uv'+", "p'"]:
            raise NotImplementedError("Unsupported profile: %s"%(key,))
        if key not in self.profiles:
            raise KeyError("%s data not found."%(key,))
        
        if key in ["U+", "u'+", "v'+", "w'+", "p'"]:
            return np.concatenate((self.profiles[key], self.profiles[key][::-1]))
        if key in ["uv'+"]:
            return np.concatenate((-self.profiles[key], self.profiles[key][::-1]))
    
    def plot_full_stats(self, path, file_type="svg"):
        keys = ["U+", "u'+", "v'+", "w'+", "uv'+", "p'"]
        nrows=1
        ncols=len(keys)
        ax_width=6.4/2
        ax_height=4.8/2
        fig, axs = plt.subplots(nrows,ncols, figsize=(ax_width*ncols, ax_height*nrows))
        
        x_label = "y/h"
        y_pos = self.get_full_pos_y()
        
        for idx, key in enumerate(keys):
            ax = axs[idx]
            data = self.get_full_data(key)
            
            ax.set_xlabel(x_label)
            ax.set_ylabel(key)
            ax.plot(y_pos, data)
            
        fig.align_labels()
        fig.tight_layout()
        fig.savefig(os.path.join(path, "Reference_profiles_full_Re%d.%s"%(self.Re_wall, file_type)))
        plt.close(fig)
        plt.clf()
    
    def plot_stats(self, path, use_y_wall_units=False, file_type="svg"):
        nrows=1
        ncols=len(self.profiles)-2
        ax_width=6.4/2
        ax_height=4.8/2
        fig, axs = plt.subplots(nrows,ncols, figsize=(ax_width*ncols, ax_height*nrows))
        
        if not use_y_wall_units:
            x_label = "y/h"
        else:
            x_label = "y+"
        y_pos = self.profiles[x_label]
        
        for idx in range(ncols):
            ax = axs[idx]
            name = self.field_names[idx+2]
            data = self.profiles[name]
            
            ax.set_xlabel(x_label)
            ax.set_ylabel(name)
            ax.plot(y_pos, data)
        
        fig.align_labels()
        fig.tight_layout()
        fig.savefig(os.path.join(path, "Reference_profiles_Re%d.%s"%(self.Re_wall, file_type)))
        plt.close(fig)
        plt.clf()

class TorrojaBalances:
    # https://torroja.dmt.upm.es/channels/data/
    class TorrojaBalance:
        _components = ["u","v","w"]
        def __init__(self, base_path, Re, i, j):
            path = os.path.join(base_path, "Re%d.bal.%s%s"%(Re, self._components[i], self._components[j]))
            
            self.Re_wall = Re
            
            with open(path, "r") as balances_file:
                line = balances_file.readline()
                while "End of Header" not in line:
                    line = balances_file.readline()
                line = balances_file.readline() # one empty line
                line = balances_file.readline() # ny and Re
                self.ny = int(line.split()[3][:-1])
                line = balances_file.readline() # empty line
                line = balances_file.readline() # field names
                self.field_names = line.split()[1:]
                #line = balances_file.readline() # empty line
                line = balances_file.readline() # ----- line
                stats = [[] for _ in self.field_names]
                
                for line in balances_file:
                    for idx, value in enumerate(line.split()):
                        stats[idx].append(float(value))
                self.balances = {name: np.asarray(values) for name, values in zip(self.field_names, stats)}
            
        def get_full_pos_y(self):
            key = "y/h"
            if not key in self.balances:
                raise KeyError("y position data not found.")
            return np.concatenate((self.balances[key] - 1, 1 - self.balances[key][::-1]))
        
        def get_full_data(self, key):
            if key not in ["dissip", "produc", "p-strain", "p-diff", "t-diff", "v-diff", "bal"]:
                raise NotImplementedError("Unsupported profile: %s"%(key,))
            if key not in self.balances:
                raise KeyError("%s data not found."%(key,))
            
            return np.concatenate((self.balances[key], self.balances[key][::-1]))
    
    def __init__(self, base_path, Re):
        self.balances = {
            (0,0): TorrojaBalances.TorrojaBalance(base_path, Re, 0, 0),
            (1,1): TorrojaBalances.TorrojaBalance(base_path, Re, 1, 1),
            (2,2): TorrojaBalances.TorrojaBalance(base_path, Re, 2, 2),
            (0,1): TorrojaBalances.TorrojaBalance(base_path, Re, 0, 1),
        }
        
        self.Re_wall = Re
    
    # for plotting
    @property
    def has_pos_y(self):
        return True
        
    def get_pos_y(self, i, j, as_wall):
        self.__check_key((i,j))
        
        if as_wall:
            return self.balances[(i,j)].balances["y+"]
        else:
            return self.balances[(i,j)].get_full_pos_y()
    
    # compatible as drop-in for TurbulentEnergyBudgetsOnlineParallel_Torch
    @property
    def use_forcing(self):
        return False
    def __check_wall(self, as_wall):
        if not as_wall:
            raise ValueError("TorrojaBalances only supports non-dimentionalized stats.")
    def __check_key(self, key):
        if key not in self.balances:
            raise KeyError("Balances not available for %s."%(key,))
    
    def production(self, i, j, as_wall=True):
        self.__check_wall(as_wall)
        self.__check_key((i,j))
        
        return self.balances[(i,j)].get_full_data("produc")
    
    def dissipation(self, i, j, as_wall=True):
        self.__check_wall(as_wall)
        self.__check_key((i,j))
        
        return - self.balances[(i,j)].get_full_data("dissip")
    
    def turbulent_transport(self, i, j, as_wall=True):
        self.__check_wall(as_wall)
        self.__check_key((i,j))
        
        return self.balances[(i,j)].get_full_data("t-diff")
    
    def viscous_diffusion(self, i, j, as_wall=True):
        self.__check_wall(as_wall)
        self.__check_key((i,j))
        
        return self.balances[(i,j)].get_full_data("v-diff")
    
    def velocity_pressure_gradient(self, i, j, as_wall=True):
        self.__check_wall(as_wall)
        self.__check_key((i,j))
        
        return self.balances[(i,j)].get_full_data("p-diff")

class TorrojaSpectra:
    FMT_int = "<i" # little endian 32 bit integer
    FMT_float = "<f"
    def __init__(self, base_path, Re):
        base_path = os.path.join(base_path, "spectra/re%s/2D"%Re)
        assert Re in [180, 550, 950, 2000]
        self.Re_wall = Re
        
        with open(os.path.join(base_path, "Re%s.spe.j%02d"%(Re, 1)), "rb") as file:
            self._load_header_1(file)
            self._load_header_2(file)
        
        self.data_types = ["uu", "vv", "ww", "Re(u*v)", "Ox*Ox", "Oy*Oy", "Oz*Oz"]
        if Re==2000:
            self.data_types.append("Im(u*v)")
        assert len(self.data_types)==self.nvar
        
        self.spectra = {data_type:[None]*self.nplan for data_type in self.data_types}
        for jind in range(self.nplan):
            with open(os.path.join(base_path, "Re%s.spe.j%02d"%(Re, jind+1)), "rb") as file:
                self._skip_header_1(file)
                self._skip_header_2(file)
                for data_type in self.data_types:
                    self._load_data_record(file, jind, data_type)

    def __read_int(self, file):
        return struct.unpack(TorrojaSpectra.FMT_int, file.read(4))[0]
    def __read_float(self, file):
        return struct.unpack(TorrojaSpectra.FMT_float, file.read(4))[0]
    
    def _load_header_1(self, file):
        file.read(4)
        #self.file_time  = self.__read_int(file)
        self.utau       = self.__read_float(file)
        self.re         = self.__read_float(file)
        self.alp        = self.__read_float(file)
        self.bet        = self.__read_float(file)
        self.mx         = self.__read_int(file)
        self.my         = self.__read_int(file)
        self.mz         = self.__read_int(file)
        self.nplan      = self.__read_int(file)
        self.nacum      = self.__read_int(file)
        self.jind       = self.__read_int(file)
        self.nvar       = self.__read_int(file)
        file.read(4)
        
        self.header_1_size = 13*4
    
    def _skip_header_1(self, file):
        file.read(self.header_1_size)
    
    def _verify_header_1(self, file):
        file.read(4)
        raise NotImplementedError
        file.read(4)
    
    def _load_header_2(self, file):
        file.read(4)
        self.jsp = [self.__read_int(file) for j in range(self.nplan)]
        self.pos_y = [self.__read_float(file) for j in range(self.nplan)]
        self.pos_y_wall = [pos_y * self.re * self.utau for pos_y in self.pos_y]
        file.read(4)
        
        self.header_2_size = 2*4 + 2*4*self.nplan
    
    def _skip_header_2(self, file):
        file.read(self.header_2_size)
    
    def _verify_header_2(self, file):
        file.read(4)
        raise NotImplementedError
        file.read(4)
    
    def _load_data_record(self, file, jind, data_type):
        nx1 = self.mx//2
        nz1 = (self.mz+1)//2
        
        buffer_size = nz1 * nx1 * 4
        
        file.read(4)
        self.spectra[data_type][jind] = np.frombuffer(file.read(buffer_size), dtype="<f4").reshape(nz1, nx1)
        file.read(4)

    
    def print_header_info(self, write_fn):
        write_fn("utau: %s", self.utau)
        write_fn("re: %s", self.re)
        write_fn("alp: %s", self.alp)
        write_fn("bet: %s", self.bet)
        write_fn("mx: %s", self.mx)
        write_fn("my: %s", self.my)
        write_fn("mz: %s", self.mz)
        write_fn("nplan: %s", self.nplan)
        write_fn("nacum: %s", self.nacum)
        write_fn("jind: %s", self.jind)
        write_fn("nvar: %s", self.nvar)
        write_fn("jsp: %s", self.jsp)
        write_fn("pos_y: %s", self.pos_y)
    
    @property
    def nu(self):
        return 1/self.re
    
    def get_phi(self, data_type, jind):
        lstar = 1/self.re / self.utau
        
        x1 = 4*np.pi
        z1 = 2*np.pi
        
        kx = np.arange(1, self.mx//2 + 0.1, 1)
        kz = np.arange(1, (self.mz+1)//2 + 0.1, 1)
        kxx, kzz = np.meshgrid(kx,kz)
        
        lambdax = 1/(kx/(2*x1))/lstar
        lambdaz = 1/(kz/(2*z1))/lstar
        
        data = self.spectra[data_type][jind]
        data = kxx*kzz*data
        
        return (lambdax,lambdaz), data
    
    def plot_spectra(self, path, file_type="svg", jinds=None, data_types=None, log_fn=None):
        if jinds is None:
            jinds = list(range(self.nplan))
        if data_types is None:
            data_types = self.data_types
        
        lstar = 1/self.re / self.utau
        
        x1 = 4*np.pi
        z1 = 2*np.pi
        
        kx = np.arange(1, self.mx//2 + 0.1, 1)
        kz = np.arange(1, (self.mz+1)//2 + 0.1, 1)
        kxx, kzz = np.meshgrid(kx,kz)
        
        lambdax = 1/(kx/(2*x1))/lstar
        lambdaz = 1/(kz/(2*z1))/lstar
        
        nrows=len(jinds)
        ncols=len(data_types)
        ax_width=6.4/2
        ax_height=4.8/2
        fig, axs = plt.subplots(nrows,ncols, figsize=(ax_width*ncols, ax_height*nrows))
        
        if log_fn is not None: log_fn("lstar=%s", lstar) #, lambdax=%s, lambdaz=%s", lstar, lambdax, lambdaz)
        
        for col, data_type in enumerate(data_types):
            for row, jind in enumerate(jinds):
                if log_fn is not None: log_fn("plotting '%s' for j=%d", data_type, jind+1)
                data = self.spectra[data_type][jind]
                if data is None: continue
                data = kxx*kzz*data
                data[0,0] = 0
                data = data/np.max(data)
                if log_fn is not None: log_fn("normalized data stats: mean %s, min %s, max %s", np.mean(data), np.min(data), np.max(data))
                
                ax = axs[row][col]
                ax.contourf(lambdax,lambdaz, data, levels=[0.1,0.5,0.9,1], cmap="Greys")
                ax.set(xscale="log", yscale="log")
            
        fig.align_labels()
        fig.tight_layout()
        fig.savefig(os.path.join(path, "Reference_spectra_Re%d.%s"%(self.Re_wall, file_type)))
        plt.close(fig)
        plt.clf()


