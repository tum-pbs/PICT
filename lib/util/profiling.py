from contextlib import contextmanager
from collections import deque
import numpy as np
import time, sys, json

ENABLE_TENSORFLOW = False
if ENABLE_TENSORFLOW:
	try:
		import tensorflow as tf
	except ModuleNotFoundError:
		ENABLE_TENSORFLOW = False

DEFAULT_STATS_MODE = "WELFORD"


#https://stackoverflow.com/questions/27779677/how-to-format-elapsed-time-from-seconds-to-hours-minutes-seconds-and-milliseco
def format_time(t):
	'''t (float): time in seconds'''
	h, r = divmod(t, 3600)
	m, s = divmod(r, 60)
	return '{:02d}:{:02d}:{:06.3f}'.format(int(h), int(m), s)

def time_unit(t, m=1000.):
	'''t (float): time in seconds'''
	units = ['ns', 'us', 'ms', 's', 'm', 'h', 'd', 'y']
	x = [1e-9, 1e-6, 1e-3, 1.0, 60.0, 3600.0, 3600.0*24.0, 3600.0*24.0*365.0]
	for d, u in zip(x, units):
		if t/d < m or u==units[-1]:
			return '{:.3f} {:<2}'.format(t/d, u)

class Profiler:
	class Sample:
		def __init__(self, name, parent, group=None):
			self.name = name
			self.parent = parent
			self.group = group
			self.children = {}
			self.samples = []
			self.start = time.time()
		def add_sample(self, sample):
			self.samples.append(sample)
		def begin(self):
			self.start = time.time()
		def end(self):
			self.add_sample(time.time()-self.start)
		def get_child(self, name, group=None):
			if not name in self.children:
				self.children[name] = self.__class__(name, self, group=group)
			return self.children[name]
		@property
		def num_samples(self):
			return len(self.samples)
		def __len__(self):
			return self.num_samples
		def __getitem__(self, idx):
			return self.samples[idx]
		@property
		def min(self):
			return np.amin(self.samples)
		@property
		def max(self):
			return np.amax(self.samples)
		@property
		def mean(self):
			return np.mean(self.samples)
		@property
		def var(self):
			return np.var(self.samples)
		@property
		def std(self):
			return np.std(self.samples)
		@property
		def sum(self):
			return np.sum(self.samples)
	
	class StreamingSample(Sample):
		def __init__(self, name, parent, group=None):
			self._min = np.finfo(np.float64).max
			self._max = np.finfo(np.float64).min
			self._sum = np.float64(0)
			self._sum_sq = np.float64(0)
			self._num_samples = 0
			self._last_sample = np.float64(0)
			super().__init__(name, parent, group=group)
			del self.samples
		def add_sample(self, sample):
			sample = np.float64(sample)
			self._min = np.minimum(self._min, sample)
			self._max = np.maximum(self._max, sample)
			self._sum += sample
			self._sum_sq += sample*sample
			self._num_samples += 1
			self._last_sample = sample
		def __getitem__(self, idx):
			if idx==-1:
				return self._last_sample
			else:
				raise IndexError("StreamingSample only keeps the last sample (idx = -1).")
		@property
		def num_samples(self):
			return self._num_samples
		@property
		def min(self):
			return self._min
		@property
		def max(self):
			return self._max
		@property
		def mean(self):
			return np.divide(self._sum, self._num_samples, dtype=np.float64)
		@property
		def var(self):
			mean = self.mean
			return (np.divide(self._sum_sq, self._num_samples, dtype=np.float64)) - (mean*mean)
		@property
		def std(self):
			return np.sqrt(self.var)
		@property
		def sum(self):
			return self._sum
	
	class WelfordOnlineSample(StreamingSample):
		#https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
		def __init__(self, name, parent, group=None):
			super().__init__(name, parent, group=group)
			del self._sum_sq
			self._mean = np.float64(0)
			self._M2 = np.float64(0)
		def add_sample(self, sample):
			sample = np.float64(sample)
			self._last_sample = sample
			self._min = np.minimum(self._min, sample)
			self._max = np.maximum(self._max, sample)
			self._sum += sample
			#self._sum_sq += sample*sample
			
			self._num_samples += 1
			delta = sample - self._mean
			self._mean += np.divide(delta, self._num_samples, dtype=np.float64)
			self._M2 += delta * (sample - self._mean)
		@property
		def mean(self):
			return self._mean
		@property
		def var(self):
			return np.divide(self._M2, self._num_samples, dtype=np.float64)
	
	# SAMPLE_TYPES = {
		# "LIST":Profiler.Sample,
		# "STREAMING":Profiler.StreamingSample,
		# "WELFORD":Profiler.WelfordOnlineSample,
	# }
	
	def __init__(self, verbose=False, active=True, stats_mode=DEFAULT_STATS_MODE):
		stats_mode = stats_mode.upper()
		if stats_mode=="STREAMING":
			self._root = Profiler.StreamingSample("__root__", None)
		elif  stats_mode=="LIST":
			self._root = Profiler.Sample("__root__", None)
		elif  stats_mode=="WELFORD":
			self._root = Profiler.WelfordOnlineSample("__root__", None)
		else:
			raise ValueError("Unknown stats_mode '%s'"%stats_mode)
		self._current = self._root
		self._groups = {}
		self.verbose = verbose
		self._active = active
	
	@classmethod
	def from_file(cls, path, verbose=False, active=True):
		p = cls(verbose, active)
		with open(path, 'r') as file:
			t = json.load(path)
		p.timings = t
		return p
	
	@property
	def is_active(self):
		return self._active
	
	def current_sample_path(self):
		names = []
		c = self._current
		while c!=self._root:
			names.append(c.name)
			c = c.parent
		return "/".join(names[::-1])
	
	def _get_group(self, group):
		if group not in self._groups:
			self._groups[group] = [self._root.get_child(group, group=None), 0]
		return self._groups[group]
	
	def _begin_group(self, group):
		if group is None:
			return
		grp = self._get_group(group)
		if grp[1] <= 0:
			grp[1] = 0
			grp[0].begin()
		grp[1] += 1
	
	def _end_group(self, group):
		if group is None:
			return
		grp = self._get_group(group)
		grp[1] -= 1
		if grp[1] <= 0:
			grp[0].end()
			grp[1] = 0
	
	def _begin_sample(self, name, group=None):
		if not self._active: return
		self._current = self._current.get_child(name, group)
		self._begin_group(group)
		self._current.begin()
	
	def _end_sample(self, verbose=None):
		if not self._active: return
		
		self._current.end()
		if verbose or (self.verbose and verbose is None):
			print('\'{}\': {}'.format(self._current.name, time_unit(self._current[-1])))
		self._end_group(self._current.group)
		self._current = self._current.parent
		
	
	@contextmanager
	def sample(self, name, verbose=None, group=None):
		if self._active:
			self._begin_sample(name, group=group)
		try:
			yield # run code to measure
		finally:
			if self._active:
				self._end_sample(verbose)
	
	if ENABLE_TENSORFLOW:
		def begin_gradient_sample(self, x, verbose=None):
			@tf.custom_gradient
			def f(x):
				#print("REGISTER: end gradient sample")
				def grad(dy):
					self._end_sample(verbose)
					#print("END gradient sample")
					return tf.identity(dy)
				return tf.identity(x), grad
			return f(x)
			
		def end_gradient_sample(self, x, name):
			@tf.custom_gradient
			def f(x):
				#print("REGISTER: begin gradient sample: ", name)
				def grad(dy):
					self._begin_sample("grad:"+name)
					#print("BEGIN gradient sample: ", name)
					return tf.identity(dy)
				return tf.identity(x), grad
			return f(x)
		
#	
#	def start_sample(self, name):
#		if name in self.last
#	
#	def end_sample(self, verbose=False):
#		pass
	
	# for formatting
	def _get_max_depth(self, sample, level, level_indent):
		depth = 0
		for name in sample.children:
			depth = max(depth, level_indent*level + len(name))
			depth = max(depth, self._get_max_depth(sample.children[name], level+1, level_indent))
		return depth
	
	# for formatting
	def _get_indent(self, level_indent, max_indent):
		indent = self._get_max_depth(self._root, 0, level_indent)
		if max_indent<0:
			return indent
		return min(max_indent, indent)
	
	def _print_stats(self, sample, level, t_parent, t_root, file, level_indent, max_indent):
		t_remaining = t_parent
		for name, current in sorted(sample.children.items(), key=lambda e: e[1].sum, reverse=True):
			total_time = current.sum
			if level==0:
				t_parent = total_time
				t_root = total_time
			t_remaining -= total_time
			s = ' '*level_indent*level+'\'{}\''.format(name)
			s = ('{:<'+str(max_indent)+'}').format(s)
			file.write('{}: {:>10}, {:10d}, {:>12}, {: 10.05f}, {: 10.05f}, {:>10}, {:>10}, {:>10}\n'.format(s, time_unit(current.mean), current.num_samples, format_time(total_time), 100.*total_time/t_parent, 100.*total_time/t_root, \
				time_unit(current.std), time_unit(current.min), time_unit(current.max)))
			self._print_stats(current, level+1, total_time, t_root, file, level_indent, max_indent)
		if t_remaining>0.0 and len(sample.children)>0:
			s = ' '*level_indent*level+'{}'.format("-")
			s = ('{:<'+str(max_indent)+'}').format(s)
			file.write('{}: {:>10}, {:10d}, {:>12}, {: 10.05f}, {: 10.05f}\n'.format(s, '-', 0, format_time(t_remaining), 100.*t_remaining/t_parent, 100.*t_remaining/t_root))
	
	def stats(self, file=sys.stdout, level_indent=4, max_indent=-1):
		if self._active:
			max_indent = self._get_indent(level_indent, max_indent) +2
			file.write(('{:<'+str(max_indent)+'}: {:^10}| {:^10}| {:^12}| {:^10}| {:^10}| {:^10}| {:^10}| {:^10}|\n').format('Sample name', "average", "# samples", "total", "% parent", "% root", "std", "min", "max"))
			self._print_stats(self._root, 0, 0, 0, file, level_indent, max_indent)
		else:
			file.write('\nProfiling disabled.\n')
	
	def save(self, path):
		with open(path, 'w') as file:
			json.dump(self.timings, path)

if __name__=='__main__':
	#test
	print("--- Variance stats tests ---")
	samples_low = [4e-3, 6e-4, 1e-3, 45e-3, 8e-4, 1.605e-3]
	samples_hight = [0.2, 0.00001, 12, 15983, 2.0400862, 3e-12]
	def _test_stats(stats_mode):
		p = Profiler(stats_mode=stats_mode)
		sample = p._root.get_child("low-var")
		for s in samples_low: sample.add_sample(s)
		
		sample = p._root.get_child("high-var")
		for s in samples_hight: sample.add_sample(s)
		p.stats()
	
	for mode in ["LIST", "STREAMING", "WELFORD"]:
		print(mode)
		_test_stats(mode)
		print()
	print()
	print()
	
	print("--- Profiler tests ---")
	def _test_profiler(p):
		with p.sample('total (root)', True):
			for i in range(2):
				with p.sample('loop 4'):
					with p.sample('test 1'):
						time.sleep(0.3)
						print(p.current_sample_path())
					with p.sample('test 2'):
						time.sleep(0.7)
					time.sleep(0.12)
				#time.sleep(0.11)
			for i in range(65):
				with p.sample('loop 40'):
					time.sleep(0.01)
			time.sleep(0.1)
			print(p.current_sample_path())
		with p.sample('test (root 2)'):
			time.sleep(0.1)
	
	p_samples = Profiler(stats_mode="LIST")
	p_stream = Profiler(stats_mode="STREAMING")
	p_welford = Profiler(stats_mode="WELFORD")
	
	try:
		_test_profiler(p_samples)
		_test_profiler(p_stream)
		_test_profiler(p_welford)
	except KeyboardInterrupt:
		pass
	print()
	print("sample-list profiler:")
	p_samples.stats()
	print()
	print("streaming profiler:")
	p_stream.stats()
	print()
	print("welford profiler:")
	p_welford.stats()
	print()
	
	def debug_mode(profiler, sample_type):
		def debug_sample(sample, sample_type):
			print("checking %s... "%(sample.name))
			assert isinstance(sample, sample_type), "is: %s, expected: %s"%(type(sample).__name__, sample_type.__name__)
			print("OK")
			for name, child in sample.children.items():
				debug_sample(child, sample_type)
		debug_sample(profiler._root, sample_type)
	
	debug_mode(p_welford, Profiler.WelfordOnlineSample)
	#debug_mode(p_samples, Profiler.WelfordOnlineSample)
	
	sys.exit()
else:
	DEFAULT_PROFILER = Profiler()
	sample = DEFAULT_PROFILER.sample
	SAMPLE = sample
	stats = DEFAULT_PROFILER.stats
	STATS = stats