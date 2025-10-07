import os, glob
import numpy as np
import torch

from lib.util.profiling import SAMPLE
from lib.data.online_statistics import WelfordOnlineParallel_Torch, CovarianceOnlineParallel_Torch, MultivariateMomentsOnlineParallel_Torch, TurbulentEnergyBudgetsOnlineParallel_Torch, TemporalTwoPointCorrelation_Online_torch
from lib.util.output import ttonp, ntonp
from lib.data.torroja import TorrojaProfile
from lib.data.resample import sample_coords_to_uniform_grid
import lib.util.domain_io as domain_io

from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

def Re_cl_to_wall(Re_cl):
    return 0.116 * (Re_cl ** 0.88)

def Re_wall_to_cl(Re_wall):
    return (Re_wall / 0.116)**(1/0.88)

# Eddy turnover time
def t_to_ETT(t, u_wall, delta=1):
    return t * u_wall / delta

def ETT_to_t(ETT, u_wall, delta=1):
    return ETT * delta / u_wall

# time non-dimensionalization
def t_star(visc, u_wall):
    return visc / (u_wall ** 2)

def t_to_t_wall(t, visc, u_wall):
    return t / t_star(visc, u_wall)

def t_wall_to_t(t_wall, visc, u_wall):
    return t_wall * t_star(visc, u_wall)

def vel_to_vel_wall(vel, u_wall, order=1):
    return vel * (1/(u_wall**order))

def pos_to_pos_wall(pos, viscosity, u_wall):
    return pos * ((1/viscosity) * u_wall)

def get_closest_index(data, value):
    return np.abs(data - value).argmin()

def interpolate_ref_statistics(ref_statistics, pos_y, stat_keys=[]):
    assert isinstance(ref_statistics, (TorrojaProfile, PISOTCFProfile))
    ref_pos_y = ref_statistics.get_full_pos_y()
    assert isinstance(pos_y, np.ndarray)
    stats = []
    for key in stat_keys:
        if (not isinstance(ref_statistics, PISOTCFProfile)) and (key in ['V+','W+']):
            stats.append(np.zeros_like(pos_y))
        else:
            stats.append(np.interp(pos_y, ref_pos_y, ref_statistics.get_full_data(key)))
    return stats

# drop-in for TorrojaProfile with own PISO TCF statistics
class PISOTCFProfile:
    # re-direction for compatibility with TorrojaProfile.profiles
    class Profiles:
        def __init__(self, parent):
            self.parent = parent
        def __getitem__(self, key):
            if key not in ["U+", "V+", "W+", "u'+", "v'+", "w'+", "uv'+", "p'", "y+", "y/h",
                    "Su'+", "Sv'+", "Sw'+", "Sp'+",
                    "Fu'+", "Fv'+", "Fw'+", "Fp'+"]:
                raise KeyError("Unsupported profile: %s"%(key,))
            
            if key=="y+":
                data = self.parent.pos_y_wall
            elif key=="y/h":
                data = 1 - np.abs(self.parent.pos_y)
            else:
                data = self.parent.get_full_data(key)
            
            return data[:(len(data)+1)//2]
    
    def __init__(self, base_path, run_id, frames="ALL", avg_steps="PARSE", load_steps=False, load_moments=False, device=None, dtype=torch.float32):
        #LOG.info("Loading stats from run %s", run_id)
        self.run_id = run_id
        
        self.get_run_path(run_id, base_path)
        
        self.frames = frames
        self.start_step, self.end_step = 0, None
        if frames!="ALL":
            if avg_steps=="PARSE":
                self.get_start_end_steps_from_end_frames()
            else:
                self.start_step = -frames*avg_steps
        
        self.load_steps = load_steps or not ((self.start_step==0) and (self.end_step is None))
        self.load_run_vel_stats(load_moments=load_moments, device=device, dtype=dtype)
        self.load_run_pos_y(device=device, dtype=dtype)
        
        self.profiles = PISOTCFProfile.Profiles(self)
    
    def get_run_path(self, run_id, base_path="./test_runs", sub_dir=None):
        load_path = os.path.join(base_path, "%s_*"%(run_id,))
        if sub_dir is not None:
            load_path = os.path.join(load_path, sub_dir)
        path = glob.glob(load_path)
        if len(path)==0:
            raise IOError("No run found for path %s"%load_path)
        if len(path)>1:
            raise IOError("No unique run found for path %s, found %s"%load_path, path)
        self.run_dir = path[0]
    
    def get_start_end_steps_from_end_frames(self):
        path = os.path.join(self.run_dir, "log")
        
        sim_frames, sim_substeps = parse_adaptive_substeps(path)
        if len(sim_frames)<self.frames:
            raise ValueError("The simulation does not have that many frames.")
        end_steps = sum(sim_substeps[-self.frames:])
        avg_end_step = end_steps//self.frames
        #LOG.info("parsed log: %d frames, end steps: %d, avg: %d", self.frames, end_steps, avg_end_step)
        
        self.start_step = -end_steps
        self.end_step = None
    
    def load_run_vel_stats(self, dims=[2,4], load_moments=False, device=None, dtype=torch.float32):
        
        path = os.path.join(self.run_dir, "stats")
        
        self.vel_stats = WelfordOnlineParallel_Torch(dims, record_steps=self.load_steps)
        self.vel_cov = CovarianceOnlineParallel_Torch(dims, record_steps=self.load_steps)
        self.p_stats = WelfordOnlineParallel_Torch(dims, record_steps=self.load_steps)
        
        self.vel_stats.load(os.path.join(path, "online_stats_vel.npz"), self.start_step, self.end_step, device, dtype)
        self.vel_cov.load(os.path.join(path, "online_stats_vel_cov.npz"), self.start_step, self.end_step, device, dtype)
        self.p_stats.load(os.path.join(path, "online_stats_p.npz"), self.start_step, self.end_step, device, dtype)
        
        if load_moments:
            self.moments = MultivariateMomentsOnlineParallel_Torch.from_file(os.path.join(path, "online_moments.npz"),
                start=self.start_step, end=self.end_step, device=device, dtype=dtype)
        else:
            self.moments = None
    
    def load_run_pos_y(self, device=None, dtype=torch.float32): #, viscosity=1):
        #load domain, get wall positions from coords
        path = os.path.join(self.run_dir, "domain")
        
        domain = domain_io.load_domain(path, dtype=dtype, device=device)
        viscosity = ttonp(domain.viscosity)
        self.viscosity = viscosity
        self.pos_y = ttonp(domain.getBlock(0).getCellCoordinates()[0,1,0,:,0]) #NCDHW -> H
        
        def to_wall_pos(coords):
            return (coords + 1) * (1/viscosity)
        self.pos_y_wall = to_wall_pos(self.pos_y)
    
    def get_avg_u_wall(self):
        if self.vel_stats.n==0:
            raise RuntimeError("Requested average u_wall but no velocity statistics are recorded.")
        mean_vel_u = ntonp(self.vel_stats.mean[0,0])
        
        dudy_n = mean_vel_u[0]/(1+self.pos_y[0])
        dudy_p = mean_vel_u[-1]/(1-self.pos_y[-1])
        dudy = (dudy_n + dudy_p) * 0.5
        
        tau_wall = dudy * self.viscosity
        
        return np.sqrt(tau_wall)
    
    @property
    def u_wall(self):
        if self._u_wall is not None:
            return self._u_wall
        else:
            return self.get_avg_u_wall()
    
    def set_u_wall(self, u_wall):
        self._u_wall = u_wall
    
    def to_wall_pos(self, coords):
        #return (coords + 1) * ((1/self.viscosity) * self.u_wall)
        #assumes coords in [-1,1]
        return pos_to_pos_wall(coords + 1, self.viscosity, self.u_wall)
    
    def to_wall_vel(self, vel, order=1):
        #return vel * (1/(self.u_wall**order))
        return vel_to_vel_wall(ntonp(vel), self.u_wall, order)
        
    ### TorrojaProfile compatible
    
    def get_full_pos_y(self):
        return self.pos_y
        
        
    def get_full_data(self, key, to_wall_fn=lambda x, o: x):
        
        if key=="U+":
            return ttonp(self.vel_stats.mean[0,0])
        elif key=="V+":
            return ttonp(self.vel_stats.mean[0,1])
        elif key=="W+":
            return ttonp(self.vel_stats.mean[0,2])
        elif key=="u'+":
            return ttonp(self.vel_stats.standard_deviation[0,0])
        elif key=="v'+":
            return ttonp(self.vel_stats.standard_deviation[0,1])
        elif key=="w'+":
            return ttonp(self.vel_stats.standard_deviation[0,2])
        elif key=="uv'+":
            return - ttonp(self.vel_cov.covariance[0,0])
        elif key=="p'":
            return ttonp(self.p_stats.standard_deviation[0,0])
        elif key.startswith("S") or key.startswith("F"):
            if self.moments is None:
                raise RuntimeError("Higher order moments are not loaded.")
            order = 3 if key.startswith("S") else 4
            if key[1:]=="u'+":
                i = 0
            if key[1:]=="v'+":
                i = 1
            if key[1:]=="w'+":
                i = 2
            if key[1:]=="p'+":
                i = 3
            return ntonp(self.moments.get_moment_standardized(i, order, to_wall_fn))
        elif key=="y/h":
            return ntonp(self.get_full_pos_y())
        else:
            raise KeyError("Unsupported profile: %s"%(key,))
    
    def get_half_data(self, key):
        raise NotImplementedError


class VelocityStats:
    def __init__(self, domain, path, logger, dims=[2,4], record_online_steps=False, u_wall=None, PSD_planes=[],
                 use_moments=False, use_energy_budgets=True,
                 use_moments_3=False, use_moments_4=False, use_temporal_correlation_dims=None):
        self.pos_y = ntonp(domain.getBlock(0).getCellCoordinates()[0,1,0,:,0]) #NCDHW -> H
        #self.size_y = domain.getBlock(0).transform[0,0,:,0,4].cpu().numpy()
        self.domain_size = (torch.abs(domain.getBlock(0).vertexCoordinates[0,2,-1,0,0] - domain.getBlock(0).vertexCoordinates[0,2,0,0,0]).cpu().numpy(),
                            torch.abs(domain.getBlock(0).vertexCoordinates[0,1,0,-1,0] - domain.getBlock(0).vertexCoordinates[0,1,0,0,0]).cpu().numpy(),
                            torch.abs(domain.getBlock(0).vertexCoordinates[0,0,0,0,-1] - domain.getBlock(0).vertexCoordinates[0,0,0,0,0]).cpu().numpy()) #zyx
        self.size_y = torch.abs(domain.getBlock(0).vertexCoordinates[0,1,0,1:,0] - domain.getBlock(0).vertexCoordinates[0,1,0,:-1,0]).cpu().numpy()
        self.mean_vel_u = []
        self.vel_RMS = [] # standard deviation of velocity
        self.RE_shear_stress = [] # covariance of x and y velocity
        self.p_RMS = []
        self.step = []
        
        self.path = path
        self.log=logger
        os.makedirs(path, exist_ok=True)
        self.has_reference = False
        
        self.viscosity = ntonp(domain.viscosity)[0]
        self.set_u_wall(u_wall)
        
        
        self.plot_u_vel_max = 20
        self.plot_vel_rms_max = 3
        
        self.ref_profiles = None
        self.has_ref_profiles = False
        
        self.additional_reference_stats = []

        self.vel_stats = WelfordOnlineParallel_Torch(dims, record_steps=record_online_steps)
        self.vel_cov = CovarianceOnlineParallel_Torch(dims, record_steps=record_online_steps)
        self.p_stats = WelfordOnlineParallel_Torch(dims, record_steps=record_online_steps)
        
        self.use_moments = use_moments or use_moments_3 or use_moments_4
        self.use_moments_base = use_moments
        self.use_moments_3 = use_moments_3
        self.use_moments_4 = use_moments_4
        if self.use_moments:
            moments = []
            if self.use_moments_base:
                # u, v, w, p -> uu, vv, ww, uv, pp
                moments.extend([(2,0,0,0), (0,2,0,0), (0,0,2,0), (1,1,0,0), (0,0,0,2)])
            if self.use_moments_3:
                moments.extend([(3,0,0,0), (0,3,0,0), (0,0,3,0), (0,0,0,3)])
                moments.extend([(2,0,0,0), (0,2,0,0), (0,0,2,0), (0,0,0,2)])
            if self.use_moments_4:
                moments.extend([(4,0,0,0), (0,4,0,0), (0,0,4,0), (0,0,0,4)])
                moments.extend([(2,0,0,0), (0,2,0,0), (0,0,2,0), (0,0,0,2)])
            self.moments = MultivariateMomentsOnlineParallel_Torch(moments, avg_dims=[dim-2 for dim in dims])
        
        self.use_temporal_correlation = use_temporal_correlation_dims is not None
        if self.use_temporal_correlation:
            #self.temporal_correlation_stats = WelfordOnlineParallel_Torch(use_temporal_correlation_dims, record_steps=True)
            self.temporal_correlation_stats = TemporalTwoPointCorrelation_Online_torch(use_temporal_correlation_dims, record_steps=True)
        
        self.use_energy_budgets = use_energy_budgets
        if self.use_energy_budgets:
            cell_coordinates = self.to_wall_pos(domain.getBlock(0).getCellCoordinates())
            self.energy_budgets = TurbulentEnergyBudgetsOnlineParallel_Torch(avg_dims=[dim-2 for dim in dims], grid_coordinates=cell_coordinates)
        
        self.use_PSD = len(PSD_planes)>0
        if self.use_PSD:
            from lib.data.online_statistics import PSDOnline_Torch
            plane_size = domain.getBlock(0).velocity.size(3)
            self.PSD = PSDOnline_Torch(total_dims=5, fft_dims=dims, fft_sizes=[domain.getBlock(0).velocity.size(size) for size in dims],
                mean_dims=[0], planes=PSD_planes, planes_dim=3, planes_symmetric=True, device=domain.getDevice())
    
    @property
    def u_wall(self):
        if self._u_wall is not None:
            return self._u_wall
        else:
            return self.get_avg_u_wall()
    
    def set_u_wall(self, u_wall):
        self._u_wall = u_wall
    
    def to_wall_pos(self, coords):
        #return (coords + 1) * ((1/self.viscosity) * self.u_wall)
        #assumes coords in [-1,1]
        return pos_to_pos_wall(coords + 1, self.viscosity, self.u_wall)
    
    def to_wall_vel(self, vel, order=1):
        #return vel * (1/(self.u_wall**order))
        return vel_to_vel_wall(vel, self.u_wall, order)
    
    @property
    def Re_wall(self):
        return self.u_wall / self.viscosity
    
    def _load_reference(self, path):
        val_x = []
        val_y = []
        with open(path, newline='') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                val_x.append(float(row[0]))
                val_y.append(float(row[1]))
        return np.asarray(val_x), np.asarray(val_y)
    
    def load_references(self, path=None):
        path = path or "./data/tcf_KMM_reference/"
        self.ref_mean_vel_u = self._load_reference(os.path.join(path, "u_mean.csv"))
        self.ref_vel_RMS = [
            self._load_reference(os.path.join(path, "u_fluc.csv")),
            self._load_reference(os.path.join(path, "w_fluc.csv")), # v and w files are swapped(?)
            self._load_reference(os.path.join(path, "v_fluc.csv")),
        ]
        self.ref_RE_shear_stress = self._load_reference(os.path.join(path, "rey_stress.csv"))
        
        self.has_reference = True
    
    def set_references(self, torroja_profiles):
        assert isinstance(torroja_profiles, TorrojaProfile)
        self.ref_profiles = torroja_profiles
        self.has_ref_profiles = True
    
    def add_additional_reference_stats(self, profiles, **plot_args):
        self.additional_reference_stats.append((profiles, plot_args))
    
    def _get_avg_vel_stats(self):
        mean_vel_u = np.mean(self.mean_vel_u, axis=0)
        vel_RMS = np.mean(self.vel_RMS, axis=0)
        RE_shear_stress = np.mean(self.RE_shear_stress, axis=0)
        p_RMS = np.mean(self.p_RMS, axis=0)
        return mean_vel_u, vel_RMS, RE_shear_stress, p_RMS
    
    def _get_vel_stats(self, domain):
        vel = domain.getBlock(0).velocity[0] #NCDHW -> CDHW
        mean_vel_u = torch.mean(vel[0], [0,2]) #H
        mean_vel_v = torch.mean(vel[1], [0,2]) #H
        mean_vel_w = torch.mean(vel[2], [0,2]) #H
        mean_vel = torch.mean(vel, [1,3], keepdims=True) # C1H1
        vel_fluctuations = vel - mean_vel # CDHW
        vel_RMS = torch.sqrt(torch.mean(torch.square(vel_fluctuations), [1,3])) #CH
        RE_shear_stress = - torch.mean(vel_fluctuations[0] * vel_fluctuations[1], [0,2]) #H
        
        p = domain.getBlock(0).pressure[0,0] # DHW
        mean_p = torch.mean(p, [0,2], keepdims=True) #1H1
        p_fluctuations = p - mean_p
        p_RMS = torch.sqrt(torch.mean(torch.square(p_fluctuations), [0,2])) #H
        
        return ttonp(mean_vel_u), ttonp(mean_vel_v), ttonp(mean_vel_w), ttonp(vel_RMS), ttonp(RE_shear_stress), ttonp(p_RMS)
    
    def _plot_vel_moments_standardized(self, moments_data, order, component_idxs, labels, colors, name, y_min=None, y_max=None, file_type="pdf"):
        with SAMPLE("plot moments"):
            
            assert order in [3,4]
            order_to_key = {3:"S", 4:"F"}
            comp_idx_to_key = {0:"u'+", 1:"v'+", 2:"w'+", 3:"p'+"}
            #moments = [(self.to_wall_vel(moment, order) if (order is not None) else moment) for moment, order in zip(moments, to_wall_orders)]
            moments = [ntonp(moments_data.get_moment_standardized(i, order, self.to_wall_vel)) for i in component_idxs]
            
            nrows=1
            ncols=1
            plot_scale = 5/6.4 # 0.5
            ax_width=6.4 * plot_scale
            ax_height=4.8* plot_scale
            fig, axs = plt.subplots(nrows,ncols, figsize=(ax_width*ncols, ax_height*nrows))
            
            ax = axs
            #ax.set_title(name)
            ax.set_xlabel("$y/\\delta$")
            ax.set_ylabel("$\overline{(u_i'^+)^%d} / \overline{(u_i'^+)^2}^{%d/2}$"%(order, order))
            #ax.set_ylabel("$U^+$")
            ax.grid()
            
            for moment, comp_idx, label, color in zip(moments, component_idxs, labels, colors):
                ref_key = order_to_key[order] + comp_idx_to_key[comp_idx]
                for ref, ref_plot_args in self.additional_reference_stats:
                    ref.set_u_wall(self._u_wall) # overwrite ref normalization of self is overwritten
                    ax.plot(ref.get_full_pos_y(), ref.get_full_data(ref_key, ref.to_wall_vel), color=color, **ref_plot_args)
                ax.plot(self.pos_y, moment, label=label, color=color)
            ax.set_ylim(y_min, y_max)
            ax.legend()
            
            fig.align_labels()
            fig.tight_layout()
            fig.savefig(os.path.join(self.path, "vel_moments_%s.%s"%(name,file_type)))
            plt.close(fig)
            plt.clf()
    
    def _plot_vel_stats(self, mean_vel_u, vel_RMS, RE_shear_stress, p_RMS, name, with_reference=True, file_type="pdf", mean_vel_v=None, mean_vel_w=None):
        with SAMPLE("plot stats"):
            plt.clf()
            
            with_prof_ref = with_reference and self.has_ref_profiles
            if with_prof_ref:
                prof_ref_pos_y_key = "y/h"
                prof_ref_pos_wall_key = "y+"
                prof_ref_pos_y_full = self.ref_profiles.get_full_pos_y()
            
            with_reference = with_reference and self.has_reference
            
            nrows=2
            ncols=3
            plot_scale = 5/6.4 # 0.5
            ax_width=6.4 * plot_scale
            ax_height=4.8* plot_scale
            fig, axs = plt.subplots(nrows,ncols, figsize=(ax_width*ncols, ax_height*nrows))
            
            to_wall_pos = self.to_wall_pos
            pos_y_wall = to_wall_pos(self.pos_y) # for Re-wall=180, viscosity=1/180
            
            mean_vel_u = self.to_wall_vel(mean_vel_u)
            if mean_vel_v is not None:
                mean_vel_v = self.to_wall_vel(mean_vel_v)
            if mean_vel_w is not None:
                mean_vel_w = self.to_wall_vel(mean_vel_w)
            vel_RMS = self.to_wall_vel(vel_RMS)
            RE_shear_stress = self.to_wall_vel(RE_shear_stress, order=2)
            p_RMS = self.to_wall_vel(p_RMS)
            
            color_default = "tab:blue"
            color_reference = "k" #k=black
            color_x = "tab:red"
            color_y = "tab:green"
            color_z = "tab:blue"
            
            # u mean, linear and log scale
            ax_lin_u_mean = axs[0][0]
            ax_lin_u_mean.set_xlabel("$y/\\delta$")
            ax_lin_u_mean.set_ylabel("$U^+$")
            ax_lin_u_mean.grid()
            if with_reference:
                ax_lin_u_mean.plot(self.ref_mean_vel_u[0], self.ref_mean_vel_u[1], linestyle="x", color=color_x)
            if with_prof_ref:
                ax_lin_u_mean.plot(prof_ref_pos_y_full, self.ref_profiles.get_full_data("U+"), linestyle=":", color=color_x)
            ax_lin_u_mean.plot(self.pos_y, mean_vel_u, color=color_x)
            if mean_vel_v is not None:
                if with_prof_ref and isinstance(self.ref_profiles, PISOTCFProfile):
                    ax_lin_u_mean.plot(prof_ref_pos_y_full, self.ref_profiles.get_full_data("V+"), linestyle=":", color=color_y)
                ax_lin_u_mean.plot(self.pos_y, mean_vel_v, color=color_y)
            if mean_vel_w is not None:
                if with_prof_ref and isinstance(self.ref_profiles, PISOTCFProfile):
                    ax_lin_u_mean.plot(prof_ref_pos_y_full, self.ref_profiles.get_full_data("W+"), linestyle=":", color=color_z)
                ax_lin_u_mean.plot(self.pos_y, mean_vel_w, color=color_z)
                    
            #ax_lin_u_mean.tick_params(axis='x', labelcolor=color)
            ax_lin_u_mean.yaxis.set_major_locator(MultipleLocator(5))
            ax_lin_u_mean.yaxis.set_minor_locator(AutoMinorLocator(5))
            ax_lin_u_mean.set_ylim(0, self.plot_u_vel_max)
            
            # color = 'tab:red'
            # ax_log_u_mean = ax_lin_u_mean.twiny()
            # if with_reference:
                # ax_log_u_mean.plot(self.ref_mean_vel_u[0] + 1, self.ref_mean_vel_u[1], "xr")
            # if with_prof_ref:
                # ax_log_u_mean.plot(prof_ref_pos_y_full + 1, self.ref_profiles.get_full_data("U+"), ":r")
            # ax_log_u_mean.plot(self.pos_y + 1, mean_vel_u, color=color)
            # ax_log_u_mean.tick_params(axis='x', labelcolor=color)
            # ax_log_u_mean.set_xscale('log')
            
            #u mean at wall, log scale
            law_wall = np.arange(1, 14)
            law_log_x = np.arange(10, 180, 10)
            law_log_y = 2.5*np.log(law_log_x) + 5.5
            ax_lin_u_mean_wall = axs[1][0]
            ax_lin_u_mean_wall.set_xlabel("$y^+$")
            ax_lin_u_mean_wall.set_ylabel("$U^+$")
            ax_lin_u_mean_wall.grid()
            if with_reference:
                ax_lin_u_mean_wall.plot( to_wall_pos(self.ref_mean_vel_u[0]), self.ref_mean_vel_u[1], "xk")
            if with_prof_ref:
                ax_lin_u_mean_wall.plot(self.ref_profiles.profiles["y+"], self.ref_profiles.profiles["U+"], linestyle=":", color=color_reference)
            ax_lin_u_mean_wall.plot(law_wall, law_wall, "--k", linewidth=0.5)
            ax_lin_u_mean_wall.plot(law_log_x, law_log_y, "--k", linewidth=0.5)
            ax_lin_u_mean_wall.plot(pos_y_wall, mean_vel_u, color=color_default)
            ax_lin_u_mean_wall.yaxis.set_major_locator(MultipleLocator(5))
            ax_lin_u_mean_wall.yaxis.set_minor_locator(AutoMinorLocator(5))
            ax_lin_u_mean_wall.set_ylim(0, self.plot_u_vel_max)
            ax_lin_u_mean_wall.set_xscale('log')
            ax_lin_u_mean_wall.set_xlim(1, 180)
            
            
            # RMS velocity fluctuations
            ax_vel_RMS = axs[0][2]
            ax_vel_RMS.set_xlabel("$y/\\delta$")
            #ax_vel_RMS.set_ylabel("u'+,v'+,w'+ (r,g,b)")
            ax_vel_RMS.grid()
            if with_reference:
                ax_vel_RMS.plot(self.ref_vel_RMS[0][0], self.ref_vel_RMS[0][1], linestyle="x", color=color_x)
                ax_vel_RMS.plot(self.ref_vel_RMS[1][0], self.ref_vel_RMS[1][1], linestyle="x", color=color_y)
                ax_vel_RMS.plot(self.ref_vel_RMS[2][0], self.ref_vel_RMS[2][1], linestyle="x", color=color_z)
            if with_prof_ref:
                ax_vel_RMS.plot(prof_ref_pos_y_full, self.ref_profiles.get_full_data("u'+"), linestyle=":", color=color_x)
                ax_vel_RMS.plot(prof_ref_pos_y_full, self.ref_profiles.get_full_data("v'+"), linestyle=":", color=color_y)
                ax_vel_RMS.plot(prof_ref_pos_y_full, self.ref_profiles.get_full_data("w'+"), linestyle=":", color=color_z)
            ax_vel_RMS.plot(self.pos_y, vel_RMS[0], color=color_x, label="$\\overline{u'u'}$")
            ax_vel_RMS.plot(self.pos_y, vel_RMS[1], color=color_y, label="$\\overline{v'v'}$")
            ax_vel_RMS.plot(self.pos_y, vel_RMS[2], color=color_z, label="$\\overline{w'w'}$")
            ax_vel_RMS.yaxis.set_major_locator(MultipleLocator(0.5))
            ax_vel_RMS.yaxis.set_minor_locator(AutoMinorLocator(5))
            ax_vel_RMS.set_ylim(0, self.plot_vel_rms_max)
            ax_vel_RMS.legend()
            
            # RMS velocity fluctuations at wall
            ax_vel_RMS_wall = axs[1][2]
            ax_vel_RMS_wall.set_xlabel("$y^+$")
            #ax_vel_RMS_wall.set_ylabel("u'+,v'+,w'+ (r,g,b)")
            ax_vel_RMS_wall.grid()
            if with_reference:
                ax_vel_RMS_wall.plot(to_wall_pos(self.ref_vel_RMS[0][0]), self.ref_vel_RMS[0][1], linestyle="x", color=color_x)
                ax_vel_RMS_wall.plot(to_wall_pos(self.ref_vel_RMS[1][0]), self.ref_vel_RMS[1][1], linestyle="x", color=color_y)
                ax_vel_RMS_wall.plot(to_wall_pos(self.ref_vel_RMS[2][0]), self.ref_vel_RMS[2][1], linestyle="x", color=color_z)
            if with_prof_ref:
                ax_vel_RMS_wall.plot(self.ref_profiles.profiles["y+"], self.ref_profiles.profiles["u'+"], linestyle=":", color=color_x)
                ax_vel_RMS_wall.plot(self.ref_profiles.profiles["y+"], self.ref_profiles.profiles["v'+"], linestyle=":", color=color_y)
                ax_vel_RMS_wall.plot(self.ref_profiles.profiles["y+"], self.ref_profiles.profiles["w'+"], linestyle=":", color=color_z)
            ax_vel_RMS_wall.plot(pos_y_wall, vel_RMS[0], color=color_x, label="$\\overline{u'u'}$")
            ax_vel_RMS_wall.plot(pos_y_wall, vel_RMS[1], color=color_y, label="$\\overline{v'v'}$")
            ax_vel_RMS_wall.plot(pos_y_wall, vel_RMS[2], color=color_z, label="$\\overline{w'w'}$")
            ax_vel_RMS_wall.yaxis.set_major_locator(MultipleLocator(0.5))
            ax_vel_RMS_wall.yaxis.set_minor_locator(MultipleLocator(0.1))
            ax_vel_RMS_wall.set_ylim(0, self.plot_vel_rms_max)
            ax_vel_RMS_wall.xaxis.set_major_locator(MultipleLocator(10))
            ax_vel_RMS_wall.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax_vel_RMS_wall.set_xlim(0, 80)
            ax_vel_RMS_wall.legend()
            
            # RMS pressure fluctuations
            ax_p_RMS = axs[0][1]
            ax_p_RMS.set_xlabel("$y/\\delta$")
            ax_p_RMS.set_ylabel("$\\overline{p'p'}$")
            ax_p_RMS.grid()
            if with_prof_ref:
                ax_p_RMS.plot(prof_ref_pos_y_full, self.ref_profiles.get_full_data("p'"), linestyle=":", color=color_reference)
            ax_p_RMS.plot(self.pos_y, p_RMS, color=color_default)
            ax_p_RMS.yaxis.set_minor_locator(AutoMinorLocator(5))
            #ax_p_RMS.set_ylim(0, 3)
            
            # Reynolds shear stress
            ax_RE_shear_stress = axs[1][1]
            ax_RE_shear_stress.set_xlabel("$y/\\delta$")
            ax_RE_shear_stress.set_ylabel("$-\\overline{u'v'}$")
            ax_RE_shear_stress.grid()
            if with_reference:
                ax_RE_shear_stress.plot(self.ref_RE_shear_stress[0], self.ref_RE_shear_stress[1], linestyle="x", color=color_reference)
            if with_prof_ref:
                ax_RE_shear_stress.plot(prof_ref_pos_y_full, self.ref_profiles.get_full_data("uv'+"), linestyle=":", color=color_reference)
            ax_RE_shear_stress.plot(self.pos_y, RE_shear_stress, color=color_default)
            ax_RE_shear_stress.yaxis.set_major_locator(MultipleLocator(0.5))
            ax_RE_shear_stress.yaxis.set_minor_locator(AutoMinorLocator(5))
            ax_RE_shear_stress.set_ylim(-1, 1)
            
            
            fig.align_labels()
            fig.tight_layout()
            fig.savefig(os.path.join(self.path, "vel_stats_%s.%s"%(name,file_type)))
            plt.close(fig)
            plt.clf()
    
    def _plot_avg_vel_stats_half(self, name, with_reference=True, file_type="pdf"):
        with SAMPLE("plot stats half"):
            plt.clf()
            
            with_prof_ref = with_reference and self.has_ref_profiles
            if with_prof_ref:
                prof_ref_pos_y_key = "y/h"
                prof_ref_pos_wall_key = "y+"
                prof_ref_pos_y_full = self.ref_profiles.get_full_pos_y()
            
            with_reference = False #with_reference and self.has_reference
            
            nrows=1
            ncols=3
            plot_scale = 5/6.4 # 0.5
            ax_width=6.4 * plot_scale
            ax_height=4.8* plot_scale
            fig, axs = plt.subplots(nrows,ncols, figsize=(ax_width*ncols, ax_height*nrows), squeeze=False)
            
            vel_stats_half, vel_cov_half, p_stats_half, pos_y = self.get_merged_half_avg_vel_stats(axis=2)
            
            pos_y_wall = self.to_wall_pos(ntonp(pos_y))
            pos_y_wall_last = pos_y_wall[-1]
            pos_y_wall_mid = self.to_wall_pos(0)
            pos_y = 1 - np.abs(ntonp(pos_y))
            mean_vel_u = self.to_wall_vel(ntonp(vel_stats_half.mean[0,0]))
            vel_RMS = self.to_wall_vel(ntonp(vel_stats_half.standard_deviation[0]))
            RE_shear_stress = self.to_wall_vel(ntonp(- vel_cov_half.covariance[0,0]), order=2)
            #p_RMS = self.to_wall_vel(p_RMS)
            
            color_default = "tab:blue"
            color_reference = "k" #k=black
            color_x = "tab:red"
            color_y = "tab:green"
            color_z = "tab:blue"
            
            #u mean at wall, log scale
            law_wall = np.arange(1, 14)
            law_log_x = np.arange(10, pos_y_wall_mid, 10)
            law_log_y = 2.5*np.log(law_log_x) + 5.5
            ax_lin_u_mean_wall = axs[0][0]
            ax_lin_u_mean_wall.set_xlabel("$y^+$")
            ax_lin_u_mean_wall.set_ylabel("$U^+$")
            ax_lin_u_mean_wall.grid()
            if with_prof_ref:
                ax_lin_u_mean_wall.plot(self.ref_profiles.profiles["y+"], self.ref_profiles.profiles["U+"], linestyle=":", color=color_reference)
            for ref, ref_plot_args in self.additional_reference_stats:
                ax_lin_u_mean_wall.plot(ref.profiles["y+"], ref.profiles["U+"], color=color_default, **ref_plot_args)
            ax_lin_u_mean_wall.plot(law_wall, law_wall, "--k", linewidth=0.5)
            ax_lin_u_mean_wall.plot(law_log_x, law_log_y, "--k", linewidth=0.5)
            ax_lin_u_mean_wall.plot(pos_y_wall, mean_vel_u, color=color_default)
            ax_lin_u_mean_wall.yaxis.set_major_locator(MultipleLocator(5))
            ax_lin_u_mean_wall.yaxis.set_minor_locator(AutoMinorLocator(5))
            ax_lin_u_mean_wall.set_ylim(0, self.plot_u_vel_max)
            ax_lin_u_mean_wall.set_xscale('log')
            ax_lin_u_mean_wall.set_xlim(0.75, pos_y_wall_mid)
            
            # RMS velocity fluctuations at wall
            ax_vel_RMS_wall = axs[0][1]
            ax_vel_RMS_wall.set_xlabel("$y^+$")
            #ax_vel_RMS_wall.set_ylabel("u'+,v'+,w'+ (r,g,b)")
            ax_vel_RMS_wall.grid()
            if with_prof_ref:
                ax_vel_RMS_wall.plot(self.ref_profiles.profiles["y+"], self.ref_profiles.profiles["u'+"], linestyle=":", color=color_x)
                ax_vel_RMS_wall.plot(self.ref_profiles.profiles["y+"], self.ref_profiles.profiles["v'+"], linestyle=":", color=color_y)
                ax_vel_RMS_wall.plot(self.ref_profiles.profiles["y+"], self.ref_profiles.profiles["w'+"], linestyle=":", color=color_z)
            for ref, ref_plot_args in self.additional_reference_stats:
                ax_vel_RMS_wall.plot(ref.profiles["y+"], ref.profiles["u'+"], color=color_x, **ref_plot_args)
                ax_vel_RMS_wall.plot(ref.profiles["y+"], ref.profiles["v'+"], color=color_y, **ref_plot_args)
                ax_vel_RMS_wall.plot(ref.profiles["y+"], ref.profiles["w'+"], color=color_z, **ref_plot_args)
            ax_vel_RMS_wall.plot(pos_y_wall, vel_RMS[0], color=color_x, label="$\\overline{u'u'}$")
            ax_vel_RMS_wall.plot(pos_y_wall, vel_RMS[1], color=color_y, label="$\\overline{v'v'}$")
            ax_vel_RMS_wall.plot(pos_y_wall, vel_RMS[2], color=color_z, label="$\\overline{w'w'}$")
            ax_vel_RMS_wall.yaxis.set_major_locator(MultipleLocator(0.5))
            ax_vel_RMS_wall.yaxis.set_minor_locator(MultipleLocator(0.1))
            ax_vel_RMS_wall.set_ylim(0, self.plot_vel_rms_max)
            ax_vel_RMS_wall.set_xscale('log')
            #ax_vel_RMS_wall.xaxis.set_major_locator(MultipleLocator(20))
            #ax_vel_RMS_wall.xaxis.set_minor_locator(AutoMinorLocator(10))
            #ax_vel_RMS_wall.set_xlim(0, 100)
            ax_vel_RMS_wall.set_xlim(0.75, pos_y_wall_mid)
            ax_vel_RMS_wall.legend()
            
            # Reynolds shear stress
            ax_RE_shear_stress = axs[0][2]
            ax_RE_shear_stress.set_xlabel("$y/\\delta$")
            #ax_RE_shear_stress.set_xlabel("$y^+$")
            ax_RE_shear_stress.set_ylabel("$-\\overline{u'v'}$")
            ax_RE_shear_stress.grid()
            if with_prof_ref:
                ax_RE_shear_stress.plot(self.ref_profiles.profiles["y/h"], -self.ref_profiles.profiles["uv'+"], linestyle=":", color=color_reference)
            for ref, ref_plot_args in self.additional_reference_stats:
                ax_RE_shear_stress.plot(ref.profiles["y/h"], -ref.profiles["uv'+"], color=color_default, **ref_plot_args)
            ax_RE_shear_stress.plot(np.append(pos_y, 2-pos_y[-1]), np.append(RE_shear_stress, -RE_shear_stress[-1]), color=color_default)
            ax_RE_shear_stress.yaxis.set_major_locator(MultipleLocator(0.5))
            ax_RE_shear_stress.yaxis.set_minor_locator(AutoMinorLocator(5))
            ax_RE_shear_stress.set_ylim(0, 1)
            #ax_RE_shear_stress.set_xscale('log')
            ax_RE_shear_stress.set_xlim(0, 1)
            
            
            fig.align_labels()
            fig.tight_layout()
            fig.savefig(os.path.join(self.path, "vel_stats_half_%s.%s"%(name,file_type)))
            plt.close(fig)
            plt.clf()
    
    def _plot_temporal_correlation(self, steps_coefficients, steps_ETT, name="", file_type="pdf"):
        with SAMPLE("plot Tcorr"):
            
            # steps_sum_squares = np.asarray(steps_sum_squares) # shape: steps, N, C
            # steps_n = np.reshape(np.asarray(steps_n), (len(steps_n),1,1)) # shape: steps -> steps, 1, 1
            
            # steps_variance = steps_sum_squares / steps_n
            # base_variance = steps_variance[0]
            # steps_correlation = steps_variance / base_variance
            
            # steps_coefficients # shape: steps[NCH]
            average_stats = False # gives wrong results
            y_wall_slices = [17.2] #, self.Re_wall]
            cell_slices = [] #[0]
            ETT_offset = 0
            ETT_max = 2

            if average_stats:
                ETT_window = ETT_max
                ETT_max = steps_ETT[-1]
                #ETT_max = t_to_ETT(t_wall_to_t(314, self.viscosity, self.u_wall), self.u_wall)

                averaging_steps = int(ETT_max / ETT_window)
                window_steps = int(len(steps_ETT) / averaging_steps)

                steps_coefficients = np.asarray(steps_coefficients)
                coefficients_normalized = 0
                ETT_avg = 0
                for i in range(averaging_steps):
                    start_idx = window_steps*i
                    end_idx = window_steps*(i+1)
                    ETT_avg = ETT_avg + (steps_ETT[start_idx:end_idx] - steps_ETT[start_idx])
                    coefficients_normalized = coefficients_normalized + steps_coefficients[start_idx:end_idx] / steps_coefficients[start_idx]

                coefficients_normalized = coefficients_normalized / averaging_steps
                steps_ETT = ETT_avg / averaging_steps

            else:
                base_coefficient = steps_coefficients[0] # shape: NCH

                if ETT_max is not None:
                    ETT_start_idx = get_closest_index(steps_ETT, ETT_offset)
                    ETT_end_idx = get_closest_index(steps_ETT, ETT_offset + ETT_max)
                    steps_coefficients = steps_coefficients[ETT_start_idx:ETT_end_idx+1]
                    steps_ETT = steps_ETT[ETT_start_idx:ETT_end_idx+1]
                
                coefficients_normalized = np.asarray(steps_coefficients) / base_coefficient
            
            pos_y_wall = self.to_wall_pos(self.pos_y)
            y_slices = [get_closest_index(pos_y_wall, y_wall) for y_wall in y_wall_slices] + cell_slices
            y_slices.sort()
            
            nrows=1
            ncols=len(y_slices)
            plot_scale = 5/6.4 # 0.5
            ax_width=6.4 * plot_scale
            ax_height=4.8* plot_scale
            fig, axs = plt.subplots(nrows,ncols, figsize=(ax_width*ncols, ax_height*nrows), squeeze=False)
            
            for idx, y_slice in enumerate(y_slices):
                ax = axs[0][idx]
                ax.set_title("$y^+[%d]=%.02f$"%(y_slice, pos_y_wall[y_slice]))
                ax.set_xlabel("ETT")
                ax.set_ylabel("$R(\\text{ETT}) / R(0)$")
                ax.grid()
                ax.plot(steps_ETT, coefficients_normalized[:, 0, 0, y_slice], label="$u'u'$")
                ax.plot(steps_ETT, coefficients_normalized[:, 0, 1, y_slice], label="$v'v'$")
                ax.plot(steps_ETT, coefficients_normalized[:, 0, 2, y_slice], label="$w'w'$")
                ax.legend()
            
            # TODO
            
            fig.align_labels()
            fig.tight_layout()
            fig.savefig(os.path.join(self.path, "temporal_correlation_%s.%s"%(name,file_type)))
            plt.close(fig)
            plt.clf()
    
    def _plot_energy_budgets(self, energy_budgets, name, file_type="svg", reference_budgets=None):
        
        nrows=4
        ncols=2
        plot_scale = 5/6.4 # 0.5
        ax_width=6.4 * plot_scale
        ax_height=4.8* plot_scale
        fig, axs = plt.subplots(nrows,ncols, figsize=(ax_width*ncols, ax_height*nrows), squeeze=False)
        
        pos_y_wall_limit = 150
        
        energy_budgets.u_wall = self.u_wall
        
        y_limits = {
            (0,0): (-0.45,0.45),
            (1,1): (-0.0035,0.01),
            (2,2): (-0.075,0.075),
            (0,1): (-0.05,0.05),
        }
        
        colors = {
            "production": "tab:blue",
            "dissipation": "tab:orange",
            "transport": "tab:green",
            "diffusion": "tab:red",
            "vel p-grad": "tab:purple",
            "vel source": "tab:brown",
        }
        
        if True:
            def limits_from_budgets(budgets, i, j):
                def get_min_max(data):
                    return np.min(ntonp(data)), np.max(ntonp(data))
                min_val, max_val = get_min_max(budgets.production(i,j))
                
                def get_min_max(data, min_val, max_val):
                    return min(min_val, np.min(ntonp(data))), max(max_val, np.max(ntonp(data)))
                
                min_val, max_val = get_min_max(- budgets.dissipation(i,j), min_val, max_val)
                min_val, max_val = get_min_max(budgets.turbulent_transport(i,j), min_val, max_val)
                min_val, max_val = get_min_max(budgets.viscous_diffusion(i,j), min_val, max_val)
                min_val, max_val = get_min_max(budgets.velocity_pressure_gradient(i,j), min_val, max_val)
                
                diff = max_val - min_val
                pad = diff*0.05
                
                return (min_val-pad, max_val+pad)
            
            for key in y_limits.keys():
                y_limits[key] = limits_from_budgets(energy_budgets if reference_budgets is None else reference_budgets, *key)
        
        def plot_budgets(ax, ax_wall, i, j, title):
            ax.set_xlabel("$y/\\delta$")
            ax_wall.set_xlabel("$y^+$")
            ax.set_title(title)
            ax_wall.set_title(title)
            ax.grid()
            ax_wall.grid()
            
            def plot_data(data, pos_y, pos_y_wall, **plot_kwargs):
                ax.plot(pos_y, data, **plot_kwargs)
                ax_wall.plot(pos_y_wall, data[:len(pos_y_wall)], **plot_kwargs)
            
            # axt = ax.twinx()
            # axt_wall = ax_wall.twinx()
            
            # def plot_data_t(data, **plot_kwargs):
                # axt.plot(pos_y, data, ":k", **plot_kwargs)
                # axt_wall.plot(pos_y_wall, data, ":k", **plot_kwargs)
            
            def _plot_budgets(budgets, with_label=True, **plot_kwargs):
                if budgets.has_pos_y:
                    pos_y = budgets.get_pos_y(i,j, as_wall=False)
                    pos_y_wall = budgets.get_pos_y(i,j, as_wall=True)
                else:
                    pos_y = self.pos_y
                    pos_y_wall = self.to_wall_pos(self.pos_y)
                
                plot_args = dict(plot_kwargs)
                plot_args["color"] = colors["production"]
                if with_label: plot_args["label"] = "$P_{%d%d}$"%(i,j)
                plot_data(ntonp(budgets.production(i,j)), pos_y, pos_y_wall, **plot_args) # production
                
                plot_args = dict(plot_kwargs)
                plot_args["color"] = colors["dissipation"]
                if with_label: plot_args["label"] = "$- \\epsilon_{%d%d}$"%(i,j)
                plot_data(- ntonp(budgets.dissipation(i,j)), pos_y, pos_y_wall, **plot_args) # - dissipation # *viscosity ?
                
                plot_args = dict(plot_kwargs)
                plot_args["color"] = colors["transport"]
                if with_label: plot_args["label"] = "$T_{%d%d}$"%(i,j)
                plot_data(ntonp(budgets.turbulent_transport(i,j)), pos_y, pos_y_wall, **plot_args) # turbulent transport
                
                plot_args = dict(plot_kwargs)
                plot_args["color"] = colors["diffusion"]
                if with_label: plot_args["label"] = "$D_{%d%d}$"%(i,j)
                plot_data(ntonp(budgets.viscous_diffusion(i,j)), pos_y, pos_y_wall, **plot_args) # viscous diffusion # *viscosity ?
                
                plot_args = dict(plot_kwargs)
                plot_args["color"] = colors["vel p-grad"]
                if with_label: plot_args["label"] = "$\\Pi_{%d%d}$"%(i,j)
                plot_data(ntonp(budgets.velocity_pressure_gradient(i,j)), pos_y, pos_y_wall, **plot_args) # vel p-grad term
                
                if budgets.use_forcing:
                    plot_args = dict(plot_kwargs)
                    plot_args["color"] = colors["vel source"]
                    if with_label: plot_args["label"] = "$S_{%d%d}$"%(i,j)
                    plot_data(ttonp(budgets.velocity_forcing(i,j)), label="$S_{%d%d}$"%(i,j), color=colors["vel source"], **plot_kwargs) #
            
            if reference_budgets is not None:
                _plot_budgets(reference_budgets, with_label=False, linestyle=":")
            
            _plot_budgets(energy_budgets)
            
            ax.legend()
            ax_wall.legend()
            ax_wall.set_xlim(0, pos_y_wall_limit)
            if (i,j) in y_limits:
                ax.set_ylim(*y_limits[(i,j)])
                ax_wall.set_ylim(*y_limits[(i,j)])
        
        plot_budgets(axs[0][0], axs[0][1], 0, 0, "$\\overline{u'u'}$")
        plot_budgets(axs[1][0], axs[1][1], 1, 1, "$\\overline{v'v'}$")
        plot_budgets(axs[2][0], axs[2][1], 2, 2, "$\\overline{w'w'}$")
        plot_budgets(axs[3][0], axs[3][1], 0, 1, "$\\overline{u'v'}$")
        
        fig.align_labels()
        fig.tight_layout()
        fig.savefig(os.path.join(self.path, "energy_budgets_%s.%s"%(name,file_type)))
        plt.close(fig)
    
    def _plot_PSD(self, name, file_type="svg"):
        if not self.use_PSD:
            return
        if not len(self.PSD.fft_dims)==2:
            raise ValueError("can only plot 2D spectra")
        
        lambdas, phi = self.PSD.get_phi([self.domain_size[dim-2] for dim in self.PSD.fft_dims], self.viscosity, self.u_wall)
        planes = ttonp(self.PSD.planes)
        pos_y_wall = self.to_wall_pos(self.pos_y)
        
        nrows=self.PSD.planes.size(0)
        ncols=3
        col_names = ["$\\phi_{uu}$", "$\\phi_{vv}$", "$\\phi_{ww}$"]
        text_pad_x = 10
        text_pad_y = 10
        plot_scale = 1
        ax_width= 4 * plot_scale
        ax_height= 4 * plot_scale
        fig, axs = plt.subplots(nrows,ncols, figsize=(ax_width*ncols, ax_height*nrows)) #, sharex=True, sharey=True)
        
        for col in range(ncols):
            for row in range(nrows):
                data = ttonp(phi[col,:,row,:])
                data = data/np.max(data)
                data[0,0] = 0
                self.log.info("normalized data stats [%d,%d]: mean %s, min %s, max %s", row, col, np.mean(data), np.min(data), np.max(data))
                
                ax = axs[row][col]
                if col==0:
                    ax.set_ylabel("$\\lambda_z^+$")
                if row==(nrows-1):
                    ax.set_xlabel("$\\lambda_x^+$")
                if row==0:
                    ax.annotate(col_names[col], xy=(0.5,1), xytext=(0,text_pad_y), 
                        xycoords='axes fraction', textcoords='offset points',
                        size='large', ha='center', va='baseline')
                if col==(ncols-1):
                    ax.annotate("$y^+=%.01f$"%(pos_y_wall[planes[row]],), xy=(1,0.5), xytext=(text_pad_x, 0), rotation="vertical",
                        xycoords='axes fraction', textcoords='offset points',
                        size='large', ha='left', va='center')
                ax.contourf(*lambdas[::-1], data, levels=[0.1,0.5,0.9,1], cmap="Greys")
                ax.set(xscale="log", yscale="log")
                #ax.title.set_text("%s, $y^+=$%.03e\ncell=%d, $y/\\delta$=%.03e"%(col_names[col], pos_y_wall[planes[row]], planes[row], self.pos_y[planes[row]]))
                ax.axis("equal")
            
        fig.align_labels()
        fig.tight_layout()
        fig.savefig(os.path.join(self.path, "spectra_%s.%s"%(name, file_type)))
        plt.close(fig)
        plt.clf()
    
    def _get_dudy_wall(self, domain):
        vel = domain.getBlock(0).velocity
        mean_vel_u = torch.mean(vel[0,0], dim=(0,2)) # H
        
        dudy_n = mean_vel_u[0]/(1+self.pos_y[0])
        dudy_p = mean_vel_u[-1]/(1-self.pos_y[-1])
        
        return (dudy_n + dudy_p) * 0.5
    
    def get_avg_u_wall(self):
        if self.vel_stats.n==0:
            raise RuntimeError("Requested average u_wall but no velocity statistics are recorded.")
        mean_vel_u = ntonp(self.vel_stats.mean[0,0])
        
        dudy_n = mean_vel_u[0]/(1+self.pos_y[0])
        dudy_p = mean_vel_u[-1]/(1-self.pos_y[-1])
        dudy = (dudy_n + dudy_p) * 0.5
        
        tau_wall = dudy * self.viscosity
        
        return np.sqrt(tau_wall)
    
    def record_vel_stats(self, domain, total_step, total_time, **kwargs):
        with SAMPLE("record stats"):
            self.step.append(total_step)

            with SAMPLE("update stats"):
                self.vel_stats.update_from_data(domain.getBlock(0).velocity)
                self.vel_cov.update_from_data(domain.getBlock(0).velocity[:,:1], domain.getBlock(0).velocity[:,1:2])
                self.p_stats.update_from_data(domain.getBlock(0).pressure)
            if self.use_PSD:
                with SAMPLE("update PSD"):
                    # TODO: resample to regular for FFT? slices should already be regular
                    #PSD_vel = domain.getBlock(0).velocity
                    #if domain.hasVertexCoordinates():
                    #    if self.PSD.resampling_shape is not None:
                    #        PSD_vel = sample_coords_to_uniform_grid(PSD_vel, domain.getVertexCoordinates(), self.PSD.resampling_shape)
                    #    else:
                    #        raise ValueError("Domain has grid but resampling is not set for PSD.")
                    self.PSD.update_from_data(domain.getBlock(0).velocity)
            
            if self.use_moments:
                with SAMPLE("update moments"):
                    self.moments.update_from_data(torch.unbind(domain.getBlock(0).velocity[0], dim=0) + (domain.getBlock(0).pressure[0,0],))
            if self.use_temporal_correlation:
                self.temporal_correlation_stats.update_from_data(domain.getBlock(0).velocity, total_time)
            if self.use_energy_budgets:
                with SAMPLE("update budgets"):
                    self.energy_budgets.update_from_data(domain.getBlock(0).velocity, domain.getBlock(0).pressure)
    
    def log_wall_stats(self, domain, **kwargs):
        vel = domain.getBlock(0).velocity
        mean_vel_u = ntonp(torch.mean(vel[0,0], dim=(0,2))) # H
        
        dudy_n = mean_vel_u[0]/(1+self.pos_y[0])
        dudy_p = mean_vel_u[-1]/(1-self.pos_y[-1])
        
        visc = ntonp(domain.viscosity)[0]
        tau_wall_n = visc * dudy_n
        tau_wall_p = visc * dudy_p
        
        #self.log.info("visc: %.03e, %.03e", visc, self.viscosity)
        self.log.info("wall dudy: %.03e, %.03e", dudy_n, dudy_p)
        self.log.info("wall shear stress: %.03e, %.03e", tau_wall_n, tau_wall_p)
        self.log.info("wall shear velocity: %.03e, %.03e", np.sqrt(tau_wall_n), np.sqrt(tau_wall_p))
        self.log.info("wall Re: %.03e, %.03e", np.sqrt(tau_wall_n) / visc, np.sqrt(tau_wall_p) / visc)
        
    
    def log_vel_stats(self, domain, it, file_type="png", **kwargs):
        mean_vel_u, mean_vel_v, mean_vel_w, vel_RMS, RE_shear_stress, p_RMS = self._get_vel_stats(domain)
        self._plot_vel_stats(mean_vel_u, vel_RMS, RE_shear_stress, p_RMS, mean_vel_v=mean_vel_v, mean_vel_w=mean_vel_w,
            name="%04d"%it, with_reference=kwargs.get("with_reference", False), file_type=file_type)
        
        self.log_wall_stats(domain)
    
    def get_merged_half_avg_vel_stats(self, axis):
        dims = self.vel_stats.dims
        
        shape = self.vel_stats.mean.size(axis)
        shape_half = shape//2
        slice_1 = [slice(None)]*self.vel_stats.mean.dim()
        slice_1[axis] = slice(0, shape_half)
        def get_slice_1(tensor):
            return tensor[slice_1]
        slice_2 = [slice(None)]*self.vel_stats.mean.dim()
        slice_2[axis] = slice(shape-shape_half, shape)
        def get_slice_2(tensor):
            tensor = tensor[slice_2]
            return torch.flip(tensor, (axis,))

        vel_stats_half = WelfordOnlineParallel_Torch(dims, record_steps=False)
        vel_stats_half.update(self.vel_stats.n, get_slice_1(self.vel_stats.mean), get_slice_1(self.vel_stats.sum_squares))
        vel_stats_half.update(self.vel_stats.n, get_slice_2(self.vel_stats.mean), get_slice_2(self.vel_stats.sum_squares))

        vel_cov_half = CovarianceOnlineParallel_Torch(dims, record_steps=False)
        vel_cov_half.update(self.vel_cov.n, get_slice_1(self.vel_cov.mean_x), get_slice_1(self.vel_cov.mean_y), get_slice_1(self.vel_cov.C))
        vel_cov_half.update(self.vel_cov.n, get_slice_2(self.vel_cov.mean_x), get_slice_2(self.vel_cov.mean_y), -get_slice_2(self.vel_cov.C))

        p_stats_half = WelfordOnlineParallel_Torch(dims, record_steps=False)
        p_stats_half.update(self.p_stats.n, get_slice_1(self.p_stats.mean), get_slice_1(self.p_stats.sum_squares))
        p_stats_half.update(self.p_stats.n, get_slice_2(self.p_stats.mean), get_slice_2(self.p_stats.sum_squares))

        pos_y = self.pos_y[:shape_half]

        return vel_stats_half, vel_cov_half, p_stats_half, pos_y
    
    def plot_avg_vel_stats(self, online=True, file_type="svg", name="", _old=False, reference_budgets=None):
        if _old:
            self.log.warning("Old vel stats have wrong temporal averaging for second order statistcs!")
            #raise NotImplementedError("basic vel stats are deprecated")
            mean_vel_u, vel_RMS, RE_shear_stress, p_RMS = self._get_avg_vel_stats()
            self._plot_vel_stats(mean_vel_u, vel_RMS, RE_shear_stress, p_RMS, name="avg"+name, file_type=file_type)
        
        if online and self.vel_stats.n>0:
            mean_vel_u = self.vel_stats.mean[0,0].cpu().numpy()
            mean_vel_v = self.vel_stats.mean[0,1].cpu().numpy()
            mean_vel_w = self.vel_stats.mean[0,2].cpu().numpy()
            vel_RMS = self.vel_stats.standard_deviation[0].cpu().numpy()
            RE_shear_stress = - self.vel_cov.covariance[0,0].cpu().numpy()
            p_RMS = self.p_stats.standard_deviation[0,0].cpu().numpy()
            self._plot_vel_stats(mean_vel_u, vel_RMS, RE_shear_stress, p_RMS, mean_vel_v=mean_vel_v, mean_vel_w=mean_vel_w,
                name="avg_online"+name, file_type=file_type)
            self._plot_avg_vel_stats_half(name="avg_online"+name, file_type=file_type)
            if self.use_PSD:
                self._plot_PSD(name="PSD", file_type=file_type)
            
            if self.use_moments_base:
                mean_vel_u = self.moments.get_mean(0).cpu().numpy()
                vel_RMS = [torch.sqrt(self.moments.get_moment_normalized(self.moments._get_moment_key(i,i))).cpu().numpy() for i in range(3)]
                RE_shear_stress = - self.moments.get_moment_normalized(self.moments._get_moment_key(0,1)).cpu().numpy() #self.moments._get_moment_key(0,1)=(1,1,0,0)
                p_RMS = torch.sqrt(self.moments.get_moment_normalized(self.moments._get_moment_key(3,3))).cpu().numpy()
                
                self._plot_vel_stats(mean_vel_u, vel_RMS, RE_shear_stress, p_RMS, name="avg_moments"+name, file_type=file_type)
            
            if self.use_moments_3:
                self._plot_vel_moments_standardized(self.moments, 3, list(range(3)), #[
                    #    self.to_wall_vel(self.moments.get_moment_normalized(self.moments._get_moment_key(i,i,i)).cpu().numpy(), 3) / 
                    #    (self.to_wall_vel(self.moments.get_moment_normalized(self.moments._get_moment_key(i,i)).cpu().numpy(), 2) ** (3/2))
                    #    for i in range(4)],
                    #to_wall_orders=[None]*4,
                    #labels=["$S(u')$", "$S(v')$", "$S(w')$", "$S(p')$"],
                    #colors=["tab:red", "tab:green", "tab:blue", "k"],
                    labels=["$u$", "$v$", "$w$"],
                    colors=["tab:red", "tab:green", "tab:blue"],
                    name="Skewness"+name,
                    y_min=-1.5, y_max=1.5)
            
            if self.use_moments_4:
                self._plot_vel_moments_standardized(self.moments, 4, list(range(3)), #[
                        # self.to_wall_vel(self.moments.get_moment_normalized(self.moments._get_moment_key(i,i,i,i)).cpu().numpy(), 4) / 
                        # (self.to_wall_vel(self.moments.get_moment_normalized(self.moments._get_moment_key(i,i)).cpu().numpy(), 2) ** (4/2))
                        # for i in range(4)],
                    # to_wall_orders=[None]*4,
                    #labels=["$F(u')$", "$F(v')$", "$F(w')$", "$F(p')$"],
                    #colors=["tab:red", "tab:green", "tab:blue", "k"],
                    labels=["$u$", "$v$", "$w$"],
                    colors=["tab:red", "tab:green", "tab:blue"],
                    name="Flatness"+name,
                    y_min=0, y_max=10)
                    
            if self.use_temporal_correlation:
                self._plot_temporal_correlation(self.temporal_correlation_stats.steps_coefficients, t_to_ETT(np.asarray(self.temporal_correlation_stats.steps_time), self.u_wall),
                    name=name)
            
            if self.use_energy_budgets:
                self._plot_energy_budgets(self.energy_budgets, name="moments"+name, file_type=file_type, reference_budgets=reference_budgets)
    
    def save_vel_stats(self, basic=True, online=True, psd=True, save_steps=True):
        if basic:
            #mean_vel_u = np.asarray(self.mean_vel_u)
            #vel_RMS = np.asarray(self.vel_RMS)
            #RE_shear_stress = np.asarray(self.RE_shear_stress)
            #p_RMS = np.asarray(self.p_RMS)
            step = np.asarray(self.step)
            np.savez_compressed(os.path.join(self.path, "vel_stats.npz"),
                                #mean_vel_u=mean_vel_u, vel_RMS=vel_RMS, RE_shear_stress=RE_shear_stress, p_RMS=p_RMS,
                                step=step)
        if online:
            self.vel_stats.save(os.path.join(self.path, "online_stats_vel.npz"), save_steps=save_steps)
            self.vel_cov.save(os.path.join(self.path, "online_stats_vel_cov.npz"), save_steps=save_steps)
            self.p_stats.save(os.path.join(self.path, "online_stats_p.npz"), save_steps=save_steps)
            
            if self.use_moments:
                self.moments.save(os.path.join(self.path, "online_moments.npz"), save_steps=False)
            
            if self.use_temporal_correlation:
                self.temporal_correlation_stats.save(os.path.join(self.path, "online_stats_vel_temporal.npz"), save_steps=True)
            
            if self.use_energy_budgets:
                self.energy_budgets.save(self.path, save_steps=False)
        
        if psd and self.use_PSD:
            self.PSD.save(self.path, "PSD")
    
    def load_vel_stats(self, path, start=0, end=None, load_online=True, load_energy_budgets=False, load_psd=False, device=None, dtype=torch.float32,
            _load_old=False):
        with np.load(os.path.join(path, "vel_stats.npz")) as np_file:
            total_steps = len(np_file["step"])
            end_base = end
            if end is None:
                end = total_steps
            assert start>=-total_steps and end<=total_steps

            self.step = list(np_file["step"][start:end])
            
            if _load_old:
                self.mean_vel_u = list(np_file["mean_vel_u"][start:end])
                self.vel_RMS = list(np_file["vel_RMS"][start:end])
                self.RE_shear_stress = list(np_file["RE_shear_stress"][start:end])
                self.p_RMS = list(np_file["p_RMS"][start:end])
        
        if load_online:
            self.vel_stats.load(os.path.join(path, "online_stats_vel.npz"), start, end, device, dtype)
            self.vel_cov.load(os.path.join(path, "online_stats_vel_cov.npz"), start, end, device, dtype)
            self.p_stats.load(os.path.join(path, "online_stats_p.npz"), start, end, device, dtype)
            
        if self.use_moments:
            self.moments.load(os.path.join(path, "online_moments.npz"), start, end_base, device, dtype)
        
        if self.use_temporal_correlation:
            self.temporal_correlation_stats.load(os.path.join(path, "online_stats_vel_temporal.npz"), start, end, device, dtype)
            
        if load_energy_budgets and self.use_energy_budgets:
            self.energy_budgets.load(path, start=start, end=end_base, device=device, dtype=dtype)
        
        if load_psd and self.use_PSD:
            from lib.data.online_statistics import PSDOnline_Torch
            self.PSD = PSDOnline_Torch.from_file(path, "PSD", device, dtype)
    
    def slice_stats_by_step(self, start, end=None):
        raise NotImplementedError("use start and end parameters of load_vel_stats")
        # N.B. recorded step is not output frame!
        if end is None:
            end = len(self.step)
        self.mean_vel_u = self.mean_vel_u[start:end]
        self.vel_RMS = self.vel_RMS[start:end]
        self.RE_shear_stress = self.RE_shear_stress[start:end]
        self.p_RMS = self.p_RMS[start:end]
        self.step = self.step[start:end]
    
    def get_bulk_umean(self):
        if not self.vel_stats.record_steps:
            raise RuntimeError("no per-step vel stats available")
        bulk_vels = np.asarray([np.sum(ntonp(vel_mean[0,0,:])*self.size_y) for vel_mean in self.vel_stats.steps_mean])
        return bulk_vels
    
    def plot_bulk_umean(self, name, with_reference=True, file_type="svg"):
        bulk_vels = self.get_bulk_umean()

        nrows=1
        ncols=1
        ax_width=6.4/2
        ax_height=4.8/2
        fig, axs = plt.subplots(nrows,ncols, figsize=(ax_width*ncols, ax_height*nrows))

        ax = axs

        ax.set_xlabel("step")
        ax.set_ylabel("channel flux")
        ax.plot(bulk_vels, color="tab:blue")
        
        fig.align_labels()
        fig.tight_layout()
        fig.savefig(os.path.join(self.path, "vel_bulk_%s.%s"%(name,file_type)))
        plt.close(fig)
        plt.clf()
    
    def _get_dudy_wall_steps(self):
        if not self.vel_stats.record_steps:
            raise RuntimeError("No steps recorded.")
        
        steps_mean_u = np.asarray(self.vel_stats.steps_mean)[:,0,0,:] #SNCH -> SH
        
        dudy_n = steps_mean_u[:,0]/(1+self.pos_y[0])
        dudy_p = steps_mean_u[:,-1]/(1-self.pos_y[-1])
        
        return (dudy_n + dudy_p) * 0.5
    
    def plot_wall_stats(self, name, file_type="svg"):
        if len(self.step)<1: return
        
        step = np.asarray(self.step)
        dudy_wall = self._get_dudy_wall_steps()
        delta = 1
        visc = self.viscosity
        
        nrows=1
        ncols=1 #4
        plot_scale = 5/6.4 # 0.5
        ax_width=6.4 * plot_scale
        ax_height=4.8* plot_scale
        fig, axs = plt.subplots(nrows,ncols, figsize=(ax_width*ncols, ax_height*nrows), squeeze=False)
        
        # ax = axs[0][0]
        # ax.set_xlabel("Step")
        # ax.set_ylabel("$\\partial u / \\partial y |_w$")
        # ax.plot(step, dudy_wall)
        
        tau_wall = visc * dudy_wall
        # ax = axs[0][1]
        # ax.set_xlabel("Step")
        # ax.set_ylabel("$\\tau_w$")
        # ax.plot(step, tau_wall)
        
        u_tau = np.sqrt(tau_wall)
        # ax = axs[0][2]
        # ax.set_xlabel("Step")
        # ax.set_ylabel("$u_{\\tau}$")
        # ax.plot(step, u_tau)
        
        Re_tau = (delta / visc) * u_tau
        ax = axs[0][0] #[3]
        ax.set_xlabel("Step")
        ax.set_ylabel("$Re_{\\tau}$")
        ax.plot(step, Re_tau)
        
        fig.align_labels()
        fig.tight_layout()
        fig.savefig(os.path.join(self.path, "Re_wall_%s.%s"%(name,file_type)))
        plt.close(fig)
    
    def plot_final_stats(self, online=True, file_type="pdf", reference_budgets=None, total_time=None):
        
        u_wall = self._u_wall
        if self._u_wall is not None:
            self.log.info("Fix u_wall: %.03e", self.u_wall)
            self.plot_avg_vel_stats(name="_fix-u-wall", online=online, file_type=file_type, reference_budgets=reference_budgets)
            
            self.set_u_wall(None) # computes avg from collected statistics
        
        avg_u_wall = self.u_wall
        self.log.info("Avg u_wall: %.03e -> Re_wall %.03e.", avg_u_wall, avg_u_wall / self.viscosity)
        if total_time is not None:
            self.log.info("Avg T+: %.03e, avg ETT: %.03e.", t_to_t_wall(total_time, self.viscosity, avg_u_wall), t_to_ETT(total_time, avg_u_wall))
        self.plot_avg_vel_stats(name="_avg-u-wall", online=online, file_type=file_type, reference_budgets=reference_budgets)
        
        self.set_u_wall(u_wall) # reset u_wall
        
        if self.vel_stats.record_steps:
            self.plot_wall_stats(name="final", file_type=file_type)
            self.plot_bulk_umean(name="final", file_type=file_type)
    
    def _plot_stats_errors_half(self, errors, pos_y, ref_errors=[], file_type="pdf"):
        with SAMPLE("plot stats half"):
            plt.clf()
            
            nrows=1
            ncols=3
            plot_scale = 5/6.4 # 0.5
            ax_width=6.4 * plot_scale
            ax_height=4.8* plot_scale
            fig, axs = plt.subplots(nrows,ncols, figsize=(ax_width*ncols, ax_height*nrows), squeeze=False)
            
            pos_y_wall = self.to_wall_pos(pos_y-1)
            pos_y_wall_last = pos_y_wall[-1]
            pos_y_wall_mid = self.to_wall_pos(0)
            
            color_default = "tab:blue"
            color_reference = "k" #k=black
            color_x = "tab:red"
            color_y = "tab:green"
            color_z = "tab:blue"
            
            #u mean at wall, log scale
            ax_lin_u_mean_wall = axs[0][0]
            ax_lin_u_mean_wall.set_xlabel("$y^+$")
            ax_lin_u_mean_wall.set_ylabel("$U^+$")
            ax_lin_u_mean_wall.grid()
            for (ref, ref_plot_args), ref_error in zip(self.additional_reference_stats, ref_errors):
                ax_lin_u_mean_wall.plot(ref.profiles["y+"], ref_error["U+"][0], color=color_default, **ref_plot_args)
            ax_lin_u_mean_wall.plot(pos_y_wall, errors["U+"][0], color=color_default)
            ax_lin_u_mean_wall.set_xscale('log')
            ax_lin_u_mean_wall.set_xlim(0.75, pos_y_wall_mid)
            
            # RMS velocity fluctuations at wall
            ax_vel_RMS_wall = axs[0][1]
            ax_vel_RMS_wall.set_xlabel("$y^+$")
            #ax_vel_RMS_wall.set_ylabel("u'+,v'+,w'+ (r,g,b)")
            ax_vel_RMS_wall.grid()
            for (ref, ref_plot_args), ref_error in zip(self.additional_reference_stats, ref_errors):
                ax_vel_RMS_wall.plot(ref.profiles["y+"], ref_error["u'+"][0], color=color_x, **ref_plot_args)
                ax_vel_RMS_wall.plot(ref.profiles["y+"], ref_error["v'+"][0], color=color_y, **ref_plot_args)
                ax_vel_RMS_wall.plot(ref.profiles["y+"], ref_error["w'+"][0], color=color_z, **ref_plot_args)
            ax_vel_RMS_wall.plot(pos_y_wall, errors["u'+"][0], color=color_x, label="$\\overline{u'u'}$")
            ax_vel_RMS_wall.plot(pos_y_wall, errors["v'+"][0], color=color_y, label="$\\overline{v'v'}$")
            ax_vel_RMS_wall.plot(pos_y_wall, errors["w'+"][0], color=color_z, label="$\\overline{w'w'}$")
            ax_vel_RMS_wall.set_xscale('log')
            #ax_vel_RMS_wall.set_xlim(0, 100)
            ax_vel_RMS_wall.set_xlim(0.75, pos_y_wall_mid)
            ax_vel_RMS_wall.legend()
            
            # Reynolds shear stress
            ax_RE_shear_stress = axs[0][2]
            ax_RE_shear_stress.set_xlabel("$y/\\delta$")
            #ax_RE_shear_stress.set_xlabel("$y^+$")
            ax_RE_shear_stress.set_ylabel("$-\\overline{u'v'}$")
            ax_RE_shear_stress.grid()
            for (ref, ref_plot_args), ref_error in zip(self.additional_reference_stats, ref_errors):
                ax_RE_shear_stress.plot(ref.profiles["y/h"], ref_error["uv'+"][0], color=color_default, **ref_plot_args)
            ax_RE_shear_stress.plot(pos_y, errors["uv'+"][0], color=color_default)
            #ax_RE_shear_stress.set_xscale('log')
            ax_RE_shear_stress.set_xlim(0, 1)
            
            
            fig.align_labels()
            fig.tight_layout()
            fig.savefig(os.path.join(self.path, "vel_stats_errors_half.%s"%(file_type)))
            plt.close(fig)
            plt.clf()
    
    def log_stats_errors_half(self, plot=True):
        if not self.has_ref_profiles:
            raise RuntimeError("A reference profile is neede to calculate statistics errors.")
        
        pos_y_min = 0
        pos_y_max = 1
        size_y_total = pos_y_max - pos_y_min
        
        stat_keys = ["U+", "u'+", "v'+", "w'+", "uv'+"]
        norm_data = True
        
        def get_size_y(pos_y):
            pos_y_mid = (pos_y[1:] + pos_y[:-1]) * 0.5
            pos_y_mid = np.insert(pos_y_mid, 0, pos_y_min)
            pos_y_mid = np.append(pos_y_mid, pos_y_max)
            return pos_y_mid[1:] - pos_y_mid[:-1]
        
        def compute_error(data, pos_y, size_y, stat_key, ref_profile):
            ref_data = interpolate_ref_statistics(ref_profile, pos_y-1, stat_keys=[stat_key])[0]
            if norm_data:
                norm = 1 / np.max(np.abs(ref_profile.profiles[stat_key]))
                data = data*norm
                ref_data = ref_data*norm
            #error_cell = np.abs(data - ref_data) #absolute error
            error_cell = np.square(data - ref_data) #square error
            error_mean = np.mean(error_cell)
            error_mean_weighted = np.sum(error_cell * size_y) / size_y_total #mean weighted with cell size
            return (error_cell, error_mean, error_mean_weighted)
        
        def get_total_error(errors):
            errors_total = [0,0,0]
            for stat_key, (error_cell, error_mean, error_mean_weighted) in errors.items():
                errors_total[0] = errors_total[0] + error_cell
                errors_total[1] = errors_total[1] + error_mean
                errors_total[2] = errors_total[2] + error_mean_weighted
            return tuple(errors_total)
        
        errors = {}
        vel_stats_half, vel_cov_half, p_stats_half, pos_y = self.get_merged_half_avg_vel_stats(axis=2)
        pos_y = 1 - np.abs(ntonp(pos_y))
        size_y = get_size_y(pos_y)
        errors["U+"] = compute_error(self.to_wall_vel(ntonp(vel_stats_half.mean[0,0])), pos_y, size_y, "U+", self.ref_profiles)
        errors["u'+"] = compute_error(self.to_wall_vel(ntonp(vel_stats_half.standard_deviation[0,0])), pos_y, size_y, "u'+", self.ref_profiles)
        errors["v'+"] = compute_error(self.to_wall_vel(ntonp(vel_stats_half.standard_deviation[0,1])), pos_y, size_y, "v'+", self.ref_profiles)
        errors["w'+"] = compute_error(self.to_wall_vel(ntonp(vel_stats_half.standard_deviation[0,2])), pos_y, size_y, "w'+", self.ref_profiles)
        errors["uv'+"] = compute_error(self.to_wall_vel(ntonp(- vel_cov_half.covariance[0,0]), order=2), pos_y, size_y, "uv'+", self.ref_profiles)
        errors["TOTAL"] = get_total_error(errors)
        
        self.log.info("CNN SGS errors:")
        for stat_key, (_, error_mean, error_mean_weighted) in errors.items():
            self.log.info("\t'%s': %.03e, weighted %.03e", stat_key, error_mean, error_mean_weighted)
        
        ref_errors = []
        for ref_idx, (ref, _) in enumerate(self.additional_reference_stats):
            ref_pos_y = ref.profiles["y/h"]
            ref_size_y = get_size_y(ref_pos_y)
            ref_errors.append({})
            for stat_key in stat_keys:
                ref_stats = ref.profiles[stat_key]
                if stat_key=="uv'+": ref_stats = -ref_stats
                ref_errors[ref_idx][stat_key] = compute_error(ref_stats, ref_pos_y, ref_size_y, stat_key, self.ref_profiles)
            ref_errors[ref_idx]["TOTAL"] = get_total_error(ref_errors[ref_idx])
            
            self.log.info("Ref %d errors:", ref_idx)
            for stat_key, (_, error_mean, error_mean_weighted) in ref_errors[ref_idx].items():
                self.log.info("\t'%s': %.03e, weighted %.03e", stat_key, error_mean, error_mean_weighted)
        
        if plot:
            self._plot_stats_errors_half(errors, pos_y, ref_errors)
        
        return errors, ref_errors
