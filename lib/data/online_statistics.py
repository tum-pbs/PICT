import numpy as np
import scipy, itertools, os
import torch

from lib.util.output import ttonp, ntonp

def torch_squeeze_multidim(tensor, dims):
    # dims must be sorted ascending
    for dim in reversed(dims):
        tensor = torch.squeeze(tensor, dim=dim)
    return tensor

class WelfordOnlineParallel_Torch:
    # for variance and mean
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, dims, record_steps=False, squeeze_dims=True):
        self.n = 0
        self.mean = 0
        self.sum_squares = 0 # M2

        if not (isinstance(dims, (list, tuple)) and all(isinstance(dim, int) for dim in dims) and (len(dims)<2 or all(dims[i]<dims[i+1] for i in range(len(dims)-1)))):
            raise ValueError("'dims' must be a list of rising int.")
        self.dims = dims
        self._squeeze_dims = squeeze_dims

        self.record_steps = record_steps
        if record_steps:
            self.steps_n = []
            self.steps_mean = []
            self.steps_sum_squares = []
    
    def update(self, n_b, mean_b, sum_squares_b, record_step=True):
        if self.record_steps and record_step:
            self.steps_n.append(n_b)
            self.steps_mean.append(ttonp(mean_b))
            self.steps_sum_squares.append(ttonp(sum_squares_b))
        
        n = self.n + n_b
        delta = mean_b - self.mean
        #mean = self.mean - delta * (n_b/n)
        mean = (self.n*self.mean + n_b*mean_b)/n # should be more stable
        sum_squares = self.sum_squares + sum_squares_b + torch.square(delta)*(self.n*n_b / n)

        self.n = n
        self.mean = mean
        self.sum_squares = sum_squares

    def update_from_data(self, data, record_step=True):
        if not isinstance(data, torch.Tensor):
            raise TypeError
        n = np.prod([data.size(dim) for dim in self.dims]).tolist()
        mean = torch.mean(data, dim=self.dims, keepdim=True)
        sum_squares = torch.sum(torch.square(data - mean), dim=self.dims, keepdim=not self._squeeze_dims)
        if self._squeeze_dims:
            mean = torch_squeeze_multidim(mean, dims=self.dims)

        self.update(n, mean, sum_squares, record_step)
        #LOG.info("data.size %s, n %s, self.n %s", data.size(), n, self.n)
    
    def save(self, path, save_steps=True):
        data = {
            "n": ntonp(self.n),
            "mean": ntonp(self.mean),
            "sum_squares": ntonp(self.sum_squares)
        }
        
        if self.record_steps and save_steps:
            data["steps_n"] = np.asarray(self.steps_n)
            data["steps_mean"] = np.asarray(self.steps_mean)
            data["steps_sum_squares"] = np.asarray(self.steps_sum_squares)
        
        np.savez_compressed(path, **data)

    def load(self, path, start=0, end=None, device=None, dtype=torch.float32):
        with np.load(path) as np_file:
            self.n = torch.tensor(np_file["n"], device=device, dtype=dtype)
            self.mean = torch.tensor(np_file["mean"], device=device, dtype=dtype)
            self.sum_squares = torch.tensor(np_file["sum_squares"], device=device, dtype=dtype)
            
            if self.record_steps:
                if not "steps_n" in np_file:
                    raise IOError("no recorded steps in stats file.")
                total_steps = len(np_file["steps_n"])
                end = end or total_steps
                if not (start>=-total_steps and end<=total_steps and start<end):
                    raise ValueError("Invalid steps slicing.")
                self.steps_n = list(np_file["steps_n"][start:end])
                self.steps_mean = list(np_file["steps_mean"][start:end])
                self.steps_sum_squares = list(np_file["steps_sum_squares"][start:end])
        
        if self.record_steps and (start!=0 or end!=total_steps):
            self.n = 0
            self.mean = 0
            self.sum_squares = 0
            for step in range(len(self.steps_n)):
                self.update(torch.tensor(self.steps_n[step], device=device, dtype=dtype),
                            torch.tensor(self.steps_mean[step], device=device, dtype=dtype),
                            torch.tensor(self.steps_sum_squares[step], device=device, dtype=dtype),
                            record_step=False)

    @property
    def variance(self):
        return self.sum_squares / self.n
    
    @property
    def standard_deviation(self):
        return torch.sqrt(self.variance)

class CovarianceOnlineParallel_Torch:
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Weighted_batched_version
    # "Numerically Stable Parallel Computation of (Co-)Variance", Schubert and Gertz, 2018
    def __init__(self, dims, record_steps=False, squeeze_dims=True):
        self.n = 0
        self.mean_x = 0
        self.mean_y = 0
        self.C = 0

        if not (isinstance(dims, (list, tuple)) and all(isinstance(dim, int) for dim in dims) and (len(dims)<2 or all(dims[i]<dims[i+1] for i in range(len(dims)-1)))):
            raise ValueError("'dims' must be a list of rising int.")
        self.dims = dims
        self._squeeze_dims = squeeze_dims
        
        self.record_steps = record_steps
        if record_steps:
            self.steps_n = []
            self.steps_mean_x = []
            self.steps_mean_y = []
            self.steps_C = []

    def update(self, n_b, mean_x_b, mean_y_b, C_b, record_step=True):
        if self.record_steps and record_step:
            self.steps_n.append(n_b)
            self.steps_mean_x.append(ttonp(mean_x_b))
            self.steps_mean_y.append(ttonp(mean_y_b))
            self.steps_C.append(ttonp(C_b))
        
        n = self.n + n_b
        mean_x = (self.n*self.mean_x + n_b*mean_x_b)/n 
        mean_y = (self.n*self.mean_y + n_b*mean_y_b)/n 
        C = self.C + C_b + (self.mean_x - mean_x_b)*(self.mean_y - mean_y_b) * self.n*n_b/n

        self.n = n
        self.mean_x = mean_x
        self.mean_y = mean_y
        self.C = C


    def update_from_data(self, data_x, data_y, record_step=True):
        n = np.prod([data_x.size(dim) for dim in self.dims]).tolist()
        mean_x = torch.mean(data_x, dim=self.dims, keepdim=True)
        mean_y = torch.mean(data_y, dim=self.dims, keepdim=True)
        C = torch.sum((data_x - mean_x)*(data_y - mean_y), dim=self.dims, keepdim=not self._squeeze_dims)
        if self._squeeze_dims:
            mean_x = torch_squeeze_multidim(mean_x, dims=self.dims)
            mean_y = torch_squeeze_multidim(mean_y, dims=self.dims)

        self.update(n, mean_x, mean_y, C, record_step)

    
    def save(self, path, save_steps=True):
        data = {
            "n": ntonp(self.n),
            "mean_x": ntonp(self.mean_x),
            "mean_y": ntonp(self.mean_y),
            "C": ntonp(self.C)
        }
        
        if self.record_steps and save_steps:
            data["steps_n"] = np.asarray(self.steps_n)
            data["steps_mean_x"] = np.asarray(self.steps_mean_x)
            data["steps_mean_y"] = np.asarray(self.steps_mean_y)
            data["steps_C"] = np.asarray(self.steps_C)
        
        np.savez_compressed(path, **data)

    def load(self, path, start=0, end=None, device=None, dtype=torch.float32):
        with np.load(path) as np_file:
            self.n = torch.tensor(np_file["n"], device=device, dtype=dtype)
            self.mean_x = torch.tensor(np_file["mean_x"], device=device, dtype=dtype)
            self.mean_y = torch.tensor(np_file["mean_y"], device=device, dtype=dtype)
            self.C = torch.tensor(np_file["C"], device=device, dtype=dtype)
            
            if self.record_steps:
                if not "steps_n" in np_file:
                    raise IOError("no recorded steps in stats file.")
                total_steps = len(np_file["steps_n"])
                end = end or total_steps
                if not (start>=-total_steps and end<=total_steps and start<end):
                    raise ValueError("Invalid steps slicing.")
                self.steps_n = list(np_file["steps_n"][start:end])
                self.steps_mean_x = list(np_file["steps_mean_x"][start:end])
                self.steps_mean_y = list(np_file["steps_mean_y"][start:end])
                self.steps_C = list(np_file["steps_C"][start:end])
        
        if self.record_steps and (start!=0 or end!=total_steps):
            self.n = 0
            self.mean_x = 0
            self.mean_y = 0
            self.C = 0
            for step in range(len(self.steps_n)):
                self.update(torch.tensor(self.steps_n[step], device=device, dtype=dtype),
                            torch.tensor(self.steps_mean_x[step], device=device, dtype=dtype),
                            torch.tensor(self.steps_mean_y[step], device=device, dtype=dtype),
                            torch.tensor(self.steps_C[step], device=device, dtype=dtype),
                            record_step=False)
    
    @property
    def covariance(self):
        return self.C / self.n

class PSDOnline_Torch:
    def __init__(self, total_dims, fft_dims, fft_sizes, mean_dims, planes=None, planes_dim=0, planes_symmetric=False, device=None):
        # total_dims: number of input dimensions
        # fft_dims: the dimensions to compute the nD spectra over
        # fft_sizes: size limit for the fft result
        # mean_dims: the dimensions to average over
        # planes, planes_dim: a dimension to choose only some indices from, optional
        if not (isinstance(fft_dims, (list, tuple)) and all(isinstance(_, int) and _<total_dims for _ in fft_dims)):
            raise TypeError("'dims' must be a list of int")
        #if not len(dims)==2:
        #    raise ValueError("'dims' must have length 2")
        if planes is not None:
            if not (isinstance(planes, (list, tuple)) and all(isinstance(_, int) for _ in planes)):
                raise TypeError("'planes' must be a list of int")
            if not (len(planes)>0 and len(set(planes))==len(planes)):
                raise ValueError("at least one plane must be provided and all planes must be distinct")
        
        self.total_dims = total_dims
        self.fft_dims = fft_dims
        self.fft_sizes = fft_sizes
        self.mean_dims = mean_dims
        self.planes_dim = planes_dim
        self.planes_symmetric = planes_symmetric
        if planes is not None:
            self.planes = torch.tensor(planes, device=device, dtype=torch.int64)
            #if planes_symmetric:
            #    raise NotImplementedError
        else:
            self.planes = None
        
        self.fft_slice = [slice(None)] * total_dims
        for fft_dim, fft_size in zip(fft_dims, fft_sizes):
            self.fft_slice[fft_dim] = slice((fft_size+1)//2)
        self.fft_slice = tuple(self.fft_slice)
        
        self.n = 0
        self.fft = 0

    def update(self, fft_b, n_b):
        n = self.n + n_b
        fft = (self.fft*self.n + fft_b*n_b) / n
        
        self.n = n
        self.fft = fft
    
    def __update_from_data(self, data):
        
        data = torch.abs(torch.fft.fftn(data, dim=self.fft_dims))
        
        data = data[self.fft_slice]
        
        n = np.prod([data.size(dim) for dim in self.mean_dims])
        data = torch.mean(data, dim=self.mean_dims)
        
        self.update(data, n)
    
    def update_from_data(self, data):
        if not isinstance(data, torch.Tensor):
            raise TypeError("'data' must be a torch tensor")
        if not data.dim()==self.total_dims:
            raise ValueError
        
        if self.planes is not None:
            data_planes = data.index_select(self.planes_dim, self.planes)
        else:
            data_planes = data
        self.__update_from_data(data_planes)
        
        if self.planes_symmetric and (self.planes is not None):
            planes_sym = data.size(self.planes_dim) - self.planes - 1
            data_planes = data.index_select(self.planes_dim, planes_sym)
            self.__update_from_data(data_planes)
    
    def save(self, path, name):
        import json, os
        params = {
            "total_dims": self.total_dims,
            "fft_dims": self.fft_dims,
            "fft_sizes": self.fft_sizes,
            "mean_dims": self.mean_dims,
            "planes": ntonp(self.planes).tolist(),
            "planes_dim": self.planes_dim,
            "planes_symmetric": self.planes_symmetric,
        }
        data = {
            "n": self.n,
            "fft": ntonp(self.fft),
        }
        with open(os.path.join(path, name + ".json"), "w") as file:
            json.dump(params, file)
        np.savez_compressed(os.path.join(path, name + ".npz"), **data)
    
    @classmethod
    def from_file(cls, path, name, device=None, dtype=torch.float32):
        import json, os
        with open(os.path.join(path, name + ".json"), "r") as file:
            params = json.load(file)
        psd = cls(**params, device=device)
        with np.load(os.path.join(path, name + ".npz")) as np_file:
            psd.n = torch.tensor(np_file["n"], device=device, dtype=dtype)
            psd.fft = torch.tensor(np_file["fft"], device=device, dtype=dtype)
        return psd

    def get_phi(self, phys_sizes, nu, utau):
        if not len(phys_sizes)==len(self.fft_dims):
            raise ValueError("'phys_sizes' must match 'fft_dims'")
        lstar = nu/utau
        
        ks = [np.arange(1, size/2 +0.1, 1) for size in self.fft_sizes]
        lambdas = [1 / (k/(2*size))/lstar for k, size in zip(ks, phys_sizes)]
        
        grid_shape = []
        for dim in range(self.total_dims):
            if dim in self.mean_dims:
                continue
            elif dim in self.fft_dims:
                grid_shape.append(self.fft.size(len(grid_shape)))
            else:
                grid_shape.append(1)

        #print("grid shape", grid_shape, "fft shape", self.fft.size(), "fft sizes", self.fft_sizes, self.fft_slice)

        grid = np.prod(np.meshgrid(*ks), axis=0).reshape(grid_shape)

        grid = torch.tensor(grid, device=self.fft.device, dtype=self.fft.dtype)

        
        return lambdas, grid*self.fft

class MultivariateMomentsOnlineParallel_Torch:
    # "Numerically stable, scalable formulas for parallel and online computation of higher-order multivariate central moments with arbitrary weights"
    # Pebay at al., 2016
    # DOI 10.1007/s00180-015-0637-z
    # This is the multivariate un-weighted version
    class MultivariateMomentsData:
        def __init__(self, moments, channels):
            self.channels = channels
            self.n = 0
            self.means = [0] * self.channels
            self.moments = { moment: 0 for moment in moments}
        
        @classmethod
        def from_data(cls, moments, data_list, avg_dims):
            if not (isinstance(data_list, (list, tuple)) and all(isinstance(data, torch.Tensor) for data in data_list)):
                raise TypeError("data_list must be a list of torch.Tensor")
            channels = len(data_list)
            if channels==0:
                raise ValueError("No data provided.")
            if len(moments)==0:
                raise ValueError("No moments specified.")
            if any(len(moment)!=channels for moment in moments):
                raise ValueError("Moments must be specified with all channels.")
            
            shape = data_list[0].size()
            if any(data.size()!=shape for data in data_list):
                raise ValueError("All data tensors must have same shape. Shapes: %s"%([data.shape for data in data_list],))
            
            self = cls(moments, channels)
            
            self.n = np.prod([shape[dim] for dim in avg_dims])
            
            self.means = [torch.mean(data, dim=avg_dims, keepdim=True) for data in data_list]
            
            deviations = [data - mean for data, mean in zip(data_list, self.means)]
            
            for moment in moments:
                moment_data = 1
                for channel, exp in enumerate(moment):
                    if exp>0:
                        moment_data = moment_data * torch.pow(deviations[channel], exp)
                self.moments[moment] = torch.sum(moment_data, dim=avg_dims, keepdim=True)
            
            return self
        
        def _set_data(self, n, means, moments):
            self.n = n
            self.means = means
            self.moments = moments
        
        def set_data(self, n, means, moments):
            
            if not (isinstance(means, list) and all(isinstance(mean, torch.Tensor) for mean in means)):
                raise TypeError("means must be a list of torch.Tensor.")
            if not len(means)==self.channels:
                raise ValueError("MultivariateMomentsData has %d channels but got %d means."%(self.channels, len(means)))
            
            if not (isinstance(moments, dict) and not all(isinstance(moment_data, torch.Tensor) for moment_data in moments.items())):
                raise TypeError("moments must be a dict of torch.Tensor.")
            # overwriting the contained moments is fine
            
            self._set_data(n, means, moments)
        
        def get_mean(self, channel):
            if not channel<self.channels:
                raise IndexError("Channel %d not available."%(channel,))
            return self.means[channel]
        
        def get_moment(self, moment):
            sum_moment = sum(moment)
            if sum_moment==1:
                return 0
            if sum_moment==0:
                return self.n
            
            if not moment in self.moments:
                raise KeyError("Moment %s not available."%(moment,))
            return self.moments[moment]
        
        def get_moment_normalized(self, moment):
            norm = 1.0 / self.n
            return self.get_moment(moment) * norm
        
        def get_moment_keys(self):
            return set(self.moments.keys())
    
        def save(self, path):
            data = {
                "channels": self.channels,
                "n": self.n,
                "num_means": len(self.means),
                "num_moments": len(self.moments),
            }
            
            for idx, mean in enumerate(self.means):
                data["mean_%06d"%idx] = ntonp(mean)
            for key, moment in self.moments.items():
                data["moment_%s"%("_".join(str(k) for k in key),)] = ntonp(moment)
            
            np.savez_compressed(path, **data)
        
        @classmethod
        def from_file(cls, path, start=0, end=None, device=None, dtype=torch.float32):
            if not (start==0 and (end is None)):
                raise NotImplementedError("Sliced loading is not supported.") # needs step recording
            
            with np.load(path) as np_file:
                channels = np_file["channels"].tolist()
                n = np_file["n"].tolist()
                num_means = np_file["num_means"].tolist()
                num_moments = np_file["num_moments"].tolist()
                
                means = [torch.tensor(np_file["mean_%06d"%idx], device=device, dtype=dtype) for idx in range(num_means)]
                
                moment_keys = set()
                for key in np_file.keys():
                    if key.startswith("moment_"):
                        moment_keys.add((key, tuple(int(channel) for channel in key[7:].split("_"))))
                
                if not len(moment_keys)==num_moments:
                    raise IOError("Expected %d moments, but found %d: %s"%(num_moments, len(moment_keys), [_[1] for _ in moment_keys]))
                
                moments = {moment_key: torch.tensor(np_file[key], device=device, dtype=dtype) for key, moment_key in moment_keys}
            
            self = cls(set(moments.keys()), channels)
            self.set_data(n, means, moments)
            
            return self
        
        def load(self, path, start=0, end=None, device=None, dtype=torch.float32):
            raise NotImplementedError("Use MultivariateMomentsData.from_file().")
    
    # END class MultivariateMomentsData
    
    def __init__(self, moments, spatial_dims=3, avg_dims=[0,2]):
        self.dims = spatial_dims
        self.avg_dims = avg_dims
        self.record_steps = False
        
        self.moments = set(moments)
        
        self.max_order = max(sum(moment) for moment in moments)
        self.max_channels = max(len(moment) for moment in moments)
        
        if any(len(moment)!=self.max_channels for moment in moments):
            raise ValueError("All moments must specify all channels.")
        
        print("Requested moments:", self.moments, "(order:", self.max_order, ", channels:", self.max_channels, ")")
        
        self._add_required_moments()
        
        print("Required moments:", self.moments)
        
        self.data = None #MultivariateMomentsData(self.moments, self.max_channels)
    
    def _required_moments(self, moment):
        _, moment_minus_betas = self._get_betas(moment)
        moments = set(filter(lambda moment: sum(moment)>1, moment_minus_betas))
        return moments
    
    def _add_required_moments(self):
        num_moments = len(self.moments)
        
        moments_updated = set(self.moments)
        for moment in self.moments:
            moments_updated.update(self._required_moments(moment))
        
        self.moments = moments_updated
        if len(self.moments)>num_moments:
            self._add_required_moments()
    
    def _get_moment_key(self, *indices):
        key = [0] * self.max_channels
        for idx in indices:
            key[idx] += 1
        return tuple(key)
        
    
    def _multi_binom(self, x, y):
        return np.prod([scipy.special.binom(_x, _y) for _x, _y in zip(x, y)])
    
    def _abs_moment(self, moment):
        return sum(moment)
    
    def _get_betas(self, moment):
        x = [range(channel+1) for channel in moment]
        betas = list(itertools.product(*x))
        moment_minus_betas = [tuple(np.subtract(moment, beta)) for beta in betas]
        return betas, moment_minus_betas
    
    def _get_delta(self, means_diff, moment):
        delta = 1
        for channel, exp in enumerate(moment):
            if exp>0:
                delta = delta * means_diff[channel]**exp
        return delta
    
    def _update(self, other):
        means_diff = [self.data.get_mean(channel) - other.get_mean(channel) for channel in range(self.max_channels)]
        n_merged = self.data.n + other.n
        r_n_merged = 1.0/n_merged
        
        means_merged = [(self.data.get_mean(channel)*self.data.n + other.get_mean(channel)*other.n) * r_n_merged for channel in range(self.max_channels)]
        
        moments_merged = {}
        for moment in self.moments:
            moment_data_merged = 0
            for beta, alpha_minus_beta in zip(*self._get_betas(moment)):
                
                if sum(beta)==0:
                    # binom_factor = 1
                    # exp = 0
                    # delta = 1
                    moment_data_merged = moment_data_merged + self.data.get_moment(alpha_minus_beta) + other.get_moment(alpha_minus_beta)
                elif sum(alpha_minus_beta)==1:
                    # get_moment() = 0
                    pass
                else:
                    binom_factor = self._multi_binom(moment, beta)
                    exp = self._abs_moment(beta)
                    moment_data_merged = moment_data_merged + binom_factor * (
                            ((-other.n*r_n_merged)**exp)      * self.data.get_moment(alpha_minus_beta)
                            + ((self.data.n*r_n_merged)**exp) * other.get_moment(alpha_minus_beta)
                        ) * self._get_delta(means_diff, beta)
                    
            
            moments_merged[moment] = moment_data_merged
        
        self.data._set_data(n_merged, means_merged, moments_merged)
    
    def update_from_data(self, data_list):
        if not len(data_list)==self.max_channels:
            raise ValueError("All channels must be provided for update.")
        
        other = MultivariateMomentsOnlineParallel_Torch.MultivariateMomentsData.from_data(self.moments, data_list, self.avg_dims)
        if self.data is None:
            self.data = other
        else:
            self._update(other)
        
    def get_mean(self, channel, squeeze=True):
        mean = self.data.get_mean(channel)
        if squeeze:
            mean = torch_squeeze_multidim(mean, self.avg_dims)
        return mean
    
    def get_moment(self, moment, squeeze=True):
        moment = self.data.get_moment(moment)
        if squeeze:
            moment = torch_squeeze_multidim(moment, self.avg_dims)
        return moment
    
    def get_moment_normalized(self, moment, squeeze=True):
        moment = self.data.get_moment_normalized(moment)
        if squeeze:
            moment = torch_squeeze_multidim(moment, self.avg_dims)
        return moment
    
    def save(self, path, save_steps=True):
        if save_steps:
            raise NotImplementedError
        
        self.data.save(path)

    def load(self, path, start=0, end=None, device=None, dtype=torch.float32):
        new_data = MultivariateMomentsOnlineParallel_Torch.MultivariateMomentsData.from_file(path, start=start, end=end, device=device, dtype=dtype)
        if not (new_data.channels==self.max_channels and new_data.get_moment_keys()==self.moments):
            raise IOError("loaded moment data does not match configuration.")
        self.data = new_data

class TurbulentEnergyBudgetsOnlineParallel_Torch:
    def __init__(self, avg_dims=[0,2], grid_coordinates=None, with_forcing=False, u_wall=None):
        # grid coords needs to be already non-dimensionalized with u_wall
        # u_wall is only used when making the final statistics, so it may be changed after accumulation but before plotting
        # only for orthogonal grids
        self.dims = 3
        if not (isinstance(avg_dims, (list, tuple)) and len(set(avg_dims))==len(avg_dims) and all(0<=dim and dim<self.dims for dim in avg_dims)):
            raise ValueError("avg_dims must contain only 0,1, and/or 2.")
        self.avg_dims = avg_dims
        self.spatial_averaging = len(avg_dims)>0
        
        # for gradient calculation on refined grids
        # NCDHW
        self.grid_coordinates = grid_coordinates
        self.u_wall = u_wall
        
        # (co)variances (2nd order) and skewness (3rd order) for (u,v,w)
        variance_moments = list(filter(lambda moment: sum(moment) in [2,3], itertools.product((0,1,2,3), repeat=self.dims)))
        variance_moments = [var_moment + (0,0,0) for var_moment in variance_moments] # pad pressure gradient channels
        
        pressure_variance_moments = list(filter(lambda moment: sum(moment)==1, itertools.product((0,1), repeat=self.dims))) # all tuples of len dims with one 1
        pressure_variance_moments = list(itertools.product(pressure_variance_moments, repeat=2)) # any combination of these tuples
        pressure_variance_moments = [vel + p for vel, p in pressure_variance_moments] # concatenate tuples
        
        # channels: u, v, w, px, py, pz
        variance_moments = variance_moments + pressure_variance_moments
        del pressure_variance_moments
        
        self.use_forcing = with_forcing
        if with_forcing:
            variance_moments = [moment + (0,0,0) for moment in variance_moments] # channels: u, v, w, px, py, pz, sx, sy, sz
            
            velocity_forcing_moments = list(filter(lambda moment: sum(moment)==1, itertools.product((0,1), repeat=self.dims))) # all tuples of len dims with one 1
            velocity_forcing_moments = list(itertools.product(velocity_forcing_moments, repeat=2)) # any combination of these tuples
            velocity_forcing_moments = [vel + (0,0,0) + p for vel, p in velocity_forcing_moments] # concatenate tuples, pad unused pressure
            
            variance_moments = variance_moments + velocity_forcing_moments
            del velocity_forcing_moments
        
        self.moments_data = MultivariateMomentsOnlineParallel_Torch(variance_moments, self.dims*2, self.avg_dims)
        
        # (co)variances
        variance_moments = list(filter(lambda moment: sum(moment)==2, itertools.product((0,1,2), repeat=self.dims)))
        self.moments_data_grad = [
            # ux, vx, wx
            MultivariateMomentsOnlineParallel_Torch(variance_moments, self.dims, self.avg_dims),
            # uy, vy, wy
            MultivariateMomentsOnlineParallel_Torch(variance_moments, self.dims, self.avg_dims),
            # uz, vz, wz
            MultivariateMomentsOnlineParallel_Torch(variance_moments, self.dims, self.avg_dims),
        ]
    
    @property
    def has_pos_y(self):
        return False
    
    def _to_wall(self, data, order=1):
        if self.u_wall is None:
            return data
        return data * (1 / (self.u_wall ** order))
    
    def update_from_data(self, vel_data, p_data, s_data=None):
        # input format: NCDHW (zyx)
        # vel data: C=uvw (xyz), p data: C=1
        if not vel_data.dim()==self.dims+2:
            raise ValueError
        if not p_data.dim()==self.dims+2:
            raise ValueError
        if self.use_forcing:
            if s_data is None or s_data.dim()!=self.dims+2:
                raise ValueError
        
        
        p_grads = tuple(self._data_grad(p_data[0,0], grad_dim) for grad_dim in range(self.dims))
        data_channel_list = torch.unbind(vel_data[0], dim=0) + p_grads
        
        if self.use_forcing:
            data_channel_list = data_channel_list + torch.unbind(s_data[0], dim=0)
        
        self.moments_data.update_from_data(data_channel_list)
        
        for grad_dim in range(self.dims):
            self.moments_data_grad[grad_dim].update_from_data(torch.unbind(self._data_grad(vel_data, grad_dim)[0], dim=0))
    
    
    def _data_grad(self, data, grad_dim, borders="ZERO"):
        """
            
            grad_dim: index of the dimension to calculate the gradient over. Counted from the last dimension,
            
            borders: how to handle gradient computation at the borders 
                "ZERO": pad with zeros before computing central difference gradients
                "ONESIDED": use one-sided gradients for the border cells
                "PAD": use the central gradients of the cells next to the borders cells for the border cells
                "NONE" or None: do not compute border cell gradients, results has size-2 on the gradient dimesion
        """
        dims = data.dim()
        
        if self.grid_coordinates is not None:
            if dims==self.dims:
                pos = self.grid_coordinates[0,grad_dim]
            elif dims==(self.dims+1):
                pos = self.grid_coordinates[0,grad_dim:grad_dim+1]
            elif dims==(self.dims+2):
                pos = self.grid_coordinates[:,grad_dim:grad_dim+1]
            else:
                raise ValueError("Grid and data dimensionality does not match.")
        
        # dims = (NC)zyx, grad_dim=xyz
        grad_dim = dims-1 - grad_dim
        
        if data.size(grad_dim)==1:
            raise ValueError("Can't compute gradients for size 1 dimensions.")
        
        if self.grid_coordinates is not None:
            mean_dims = []
            for dim in range(dims):
                if data.size(dim)==1 and pos.size(dim)>1:
                    mean_dims.append(dim)
            if mean_dims:
                pos = torch.mean(pos, dim=mean_dims, keepdim=True)
        
        if not all(pos.size(-dim-1)==data.size(-dim-1) for dim in range(self.dims)):
            raise ValueError("grid and data sizes do not match: %s, %s"%(pos.size(), data.size()))
        
        if borders=="ZERO":
            pad = [0]*(dims*2)
            pad[grad_dim*2] = 1
            pad[grad_dim*2+1] = 1
            pad = tuple(reversed(pad))
            data = torch.nn.functional.pad(data, pad)
        
        slice_lower = [slice(None)]*dims
        slice_upper = [slice(None)]*dims
        
        if (self.grid_coordinates is not None) and (borders=="ZERO"):
            pos_size = pos.size(grad_dim)
            slice_lower[grad_dim] = slice(0,1)
            slice_upper[grad_dim] = slice(1,2)
            pos_lower = 2 * pos[slice_lower] - pos[slice_upper]
            
            slice_lower[grad_dim] = slice(pos_size-2, pos_size-1)
            slice_upper[grad_dim] = slice(pos_size-1, pos_size)
            pos_upper = 2 * pos[slice_upper] - pos[slice_lower]
            
            #print("pos sizes:", pos_lower.size(), pos.size(), pos_upper.size())
            
            pos = torch.cat([pos_lower, pos, pos_upper], dim=grad_dim)
        
        if not all(pos.size(-dim-1)==data.size(-dim-1) for dim in range(self.dims)):
            raise ValueError("grid and data sizes do not match: %s, %s"%(pos.size(), data.size()))
        
        
        dim_size = data.size(grad_dim)
        slice_lower[grad_dim] = slice(0, dim_size-2)
        slice_upper[grad_dim] = slice(2, dim_size)
        
        if self.grid_coordinates is not None:
            cell_distance = torch.abs(pos[slice_upper] - pos[slice_lower])
        else:
            cell_distance = 2.0
        
        data_grad = (data[slice_upper] - data[slice_lower]) / cell_distance
        
        if borders=="ONESIDED":
            slice_lower[grad_dim] = slice(0,1)
            slice_upper[grad_dim] = slice(1,2)
            data_grad_lower = data[slice_upper] - data[slice_lower]
            
            slice_lower[grad_dim] = slice(dim_size-2, dim_size-1)
            slice_upper[grad_dim] = slice(dim_size-1, dim_size)
            data_grad_upper = data[slice_upper] - data[slice_lower]
            
            data_grad = torch.cat([data_grad_lower, data_grad, data_grad_upper], dim=grad_dim)
            if self.grid_coordinates is not None:
                raise NotImplementedError
        elif borders=="PAD":
            slice_lower[grad_dim] = slice(0,1)
            slice_upper[grad_dim] = slice(dim_size-1, dim_size)
            data_grad = torch.cat([data_grad[slice_lower], data_grad, data_grad[slice_upper]], dim=grad_dim)
        elif borders is None or borders=="NONE" or borders=="ZERO":
            pass
        else:
            raise ValueError("Unkown border mode.")
        
        return data_grad
    
    def _data_grad2(self, data, grad_dim, borders="ZERO"):
        # second derivative version
        dims = data.dim()
        
        if self.grid_coordinates is not None:
            if dims==self.dims:
                pos = self.grid_coordinates[0,grad_dim]
            elif dims==(self.dims+1):
                pos = self.grid_coordinates[0,grad_dim:grad_dim+1]
            elif dims==(self.dims+2):
                pos = self.grid_coordinates[:,grad_dim:grad_dim+1]
            else:
                raise ValueError("Grid and data dimensionality does not match.")
        
        # dims = (NC)zyx, grad_dim=xyz
        grad_dim = dims-1 - grad_dim
        
        if data.size(grad_dim)==1:
            raise ValueError("Can't compute gradients for size 1 dimensions.")
        
        if self.grid_coordinates is not None:
            mean_dims = []
            for dim in range(dims):
                if data.size(dim)==1 and pos.size(dim)>1:
                    mean_dims.append(dim)
            if mean_dims:
                pos = torch.mean(pos, dim=mean_dims, keepdim=True)
        
        
        if borders=="ZERO":
            pad = [0]*(dims*2)
            pad[grad_dim*2] = 1
            pad[grad_dim*2+1] = 1
            pad = tuple(reversed(pad))
            data = torch.nn.functional.pad(data, pad)
        
        slice_lower = [slice(None)]*dims
        slice_mid   = [slice(None)]*dims
        slice_upper = [slice(None)]*dims
        
        if (self.grid_coordinates is not None) and (borders=="ZERO"):
            pos_size = pos.size(grad_dim)
            slice_lower[grad_dim] = slice(0,1)
            slice_upper[grad_dim] = slice(1,2)
            pos_lower = 2 * pos[slice_lower] - pos[slice_upper]
            
            slice_lower[grad_dim] = slice(pos_size-2, pos_size-1)
            slice_upper[grad_dim] = slice(pos_size-1, pos_size)
            pos_upper = 2 * pos[slice_upper] - pos[slice_lower]
            
            pos = torch.cat([pos_lower, pos, pos_upper], dim=grad_dim)
        
        
        dim_size = data.size(grad_dim)
        slice_lower[grad_dim] = slice(0, dim_size-2)
        slice_mid[grad_dim]   = slice(1, dim_size-1)
        slice_upper[grad_dim] = slice(2, dim_size)
        
        #data_grad = data[slice_upper] - 2*data[slice_mid] + data[slice_lower]
        data_grad_lower = data[slice_mid] - data[slice_lower]
        data_grad_upper = data[slice_upper] - data[slice_mid]
        
        if self.grid_coordinates is not None:
            data_grad_lower = data_grad_lower / torch.abs(pos[slice_mid] - pos[slice_lower])
            data_grad_upper = data_grad_lower / torch.abs(pos[slice_upper] - pos[slice_mid])
        
        data_grad = data_grad_upper - data_grad_lower
        
        if self.grid_coordinates is not None:
            data_grad = data_grad / (torch.abs(pos[slice_upper] - pos[slice_lower]) * 0.5)
        
        
        if borders=="PAD":
            slice_lower[grad_dim] = slice(0,1)
            slice_upper[grad_dim] = slice(dim_size-1, dim_size)
            data_grad = torch.cat([data_grad[slice_lower], data_grad, data_grad[slice_upper]], dim=grad_dim)
        elif borders is None or borders=="NONE" or borders=="ZERO":
            pass
        else:
            raise ValueError("Unkown border mode.")
        
        return data_grad
    
    def _get_moment_key(self, *indices, for_grad=False):
        key = [0] * (3 if for_grad else (6 if not self.use_forcing else 9))
        for idx in indices:
            key[idx] += 1
        return tuple(key)
    
    def mean(self, i, as_wall=True):
        data = self.moments_data.get_mean(i)
        if as_wall:
            data = self._to_wall(data, order=1)
        return data
    
    def mean_grad(self, i, grad_dim, as_wall=True):
        if grad_dim in self.avg_dims:
            return 0
        else:
            #self.mean() removes the first 2 dimensions
            #return self._data_grad(self.mean(i), grad_dim - 2)
            data = self.moments_data_grad[grad_dim].get_mean(i)
            if as_wall:
                data = self._to_wall(data, order=1)
            return data
    
    def covariance(self, i, j, squeeze=True, as_wall=True):
        moment = self._get_moment_key(i,j, for_grad=False)
        data = self.moments_data.get_moment_normalized(moment, squeeze=squeeze)
        if as_wall:
            data = self._to_wall(data, order=2)
        return data
    
    def covariance_grad(self, i, j, grad_dim, as_wall=True):
        moment = self._get_moment_key(i,j, for_grad=True)
        data = self.moments_data_grad[grad_dim].get_moment_normalized(moment)
        if as_wall:
            data = self._to_wall(data, order=2)
        return data
    
    def skewness(self, i, j, k, squeeze=True, as_wall=True):
        moment = self._get_moment_key(i, j, k, for_grad=False)
        data = self.moments_data.get_moment_normalized(moment, squeeze=squeeze)
        if as_wall:
            data = self._to_wall(data, order=3)
        return data
    
    def production(self, i, j, as_wall=True):
        result = 0
        
        for k in range(self.dims):
            if k not in self.avg_dims: #self.mean_grad() would be 0
                result = result - self.covariance(i, k, as_wall=as_wall) * self.mean_grad(j, k, as_wall=as_wall)
                result = result - self.covariance(j, k, as_wall=as_wall) * self.mean_grad(i, k, as_wall=as_wall)
        
        return result
    
    def dissipation(self, i, j, as_wall=True):
        result = 0
        
        for k in range(self.dims):
            #if k not in self.avg_dims:
                result = result + self.covariance_grad(i, j, k, as_wall=as_wall)
        
        result = result * 2
        return result
    
    def turbulent_transport(self, i, j, as_wall=True):
        result = 0
        
        for k in range(self.dims):
            if k not in self.avg_dims:
                result = result - self._data_grad(self.skewness(i, j, k, squeeze=False, as_wall=as_wall), k)
        
        return torch_squeeze_multidim(result, self.avg_dims)
    
    def viscous_diffusion(self, i, j, as_wall=True):
        result = 0
        
        for k in range(self.dims):
            if k not in self.avg_dims:
                result = result + self._data_grad2(self.covariance(i, j, squeeze=False, as_wall=as_wall), k)
        
        return torch_squeeze_multidim(result, self.avg_dims)
    
    def velocity_pressure_gradient(self, i, j, as_wall=True):
        
        result = - (self.covariance(i, j+3, as_wall=as_wall) + self.covariance(j, i+3, as_wall=as_wall))
        
        return result
    
    def velocity_forcing(self, i, j, as_wall=True):
        if not self.use_forcing:
            raise RuntimeError("Forcing moments are not tracked.")
        result = self.covariance(i, j+6, as_wall=as_wall) + self.covariance(j, i+6, as_wall=as_wall)
        
        return result
    
    def save(self, path, name="budgets", save_steps=True):
        if save_steps:
            raise NotImplementedError("Step recording is not supported.")
        
        self.moments_data.save(os.path.join(path, "%s_moments.npz"%(name,)), save_steps=save_steps)
        for i, moments_grad in enumerate(self.moments_data_grad):
            moments_grad.save(os.path.join(path, "%s_grad_%04d.npz"%(name, i)), save_steps=save_steps)

    def load(self, path, name="budgets", start=0, end=None, device=None, dtype=torch.float32):
        if not (start==0 and (end is None)):
            raise NotImplementedError("Sliced loading [%s:%s] is not supported."%(start, end)) # needs step recording
        
        self.moments_data.load(os.path.join(path, "%s_moments.npz"%(name,)), start=start, end=end, device=device, dtype=dtype)
        for i, moments_grad in enumerate(self.moments_data_grad):
            moments_grad.load(os.path.join(path, "%s_grad_%04d.npz"%(name, i)), start=start, end=end, device=device, dtype=dtype)




