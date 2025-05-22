import os, sys, socket
import logging, datetime
from lib.util.profiling import DEFAULT_PROFILER

_LOG_INITIALZED = False
_LOG_PATH = ""
_LOG_ROOT = None
_LOG = None

_STDERR = sys.stderr


class StreamCapture(object):
	def __init__(self, file, stream=sys.stdout):
		self.stream = stream
		self.log = open(file, "a") 
	def write(self, msg):
		self.stream.write(msg)
		self.log.write(msg) 
	def flush(self):
		self.stream.flush()
		self.log.flush()
	def close(self):
		self.log.close()
		return self.stream
	def __del__(self):
		self.flush()
		self.close()

def setup_logging(log_path, console=True, debug=False):
	global _LOG_INITIALZED
	global _LOG_PATH
	global _LOG_ROOT
	global _LOG
	os.makedirs(log_path, exist_ok=True)
	
	sys.stderr = StreamCapture(os.path.join(log_path, 'stderr.log'), _STDERR)
	
	#setup logging
	log_format = '[%(asctime)s][%(name)s:%(levelname)s] %(message)s'
	log_formatter = logging.Formatter(log_format)
#	logging.basicConfig(level=logging.INFO,
#		format=log_format,
#		#datefmt='%Y.%m.%d-%H:%M:%S'
#		filename=os.path.join(image_path, 'logfile.log'))
	root_logger = logging.getLogger()
	root_logger.setLevel(logging.INFO)
	logfile = logging.FileHandler(os.path.join(log_path, 'logfile.log'))
	logfile.setLevel(logging.INFO)
	logfile.setFormatter(log_formatter)
	root_logger.addHandler(logfile)
	errlog = logging.FileHandler(os.path.join(log_path, 'error.log'))
	errlog.setLevel(logging.WARNING)
	errlog.setFormatter(log_formatter)
	root_logger.addHandler(errlog)
	if debug:
		debuglog = logging.FileHandler(os.path.join(log_path, 'debug.log'))
		debuglog.setLevel(logging.DEBUG)
		debuglog.setFormatter(log_formatter)
		root_logger.addHandler(debuglog)
	if console:
		console = logging.StreamHandler(sys.stdout)
		console.setLevel(logging.INFO)
		console_format = logging.Formatter('[%(name)s:%(levelname)s] %(message)s')
		console.setFormatter(console_format)
		root_logger.addHandler(console)
	log = logging.getLogger('log setup')
	log.setLevel(logging.DEBUG)
	
	logging.captureWarnings(True)
	
	if debug:
		root_logger.setLevel(logging.DEBUG)
		log.info("Debug output active")
	
	_LOG_PATH = log_path
	_LOG_INITIALZED = True
	_LOG_ROOT = root_logger
	_LOG = log
	
	log.info('--- Log Start ---')
	log.info('host: %s, pid: %d', socket.gethostname(), os.getpid())
	log.info('Python: %s', sys.version)
	log.info('Log directory: %s', log_path)
	#log.info('TensorFlow version: %s', tf.__version__)

def get_now_string():
	now = datetime.datetime.now()
	now_str = now.strftime("%y%m%d-%H%M%S")
	return now_str

def setup_run(base_dir, name="TEST", logging=True, console=True, debug=False):
	now_str = get_now_string()
	
	run_dir = os.path.join(base_dir, now_str + "_" + name)
	os.makedirs(run_dir)
	
	if logging:
		setup_logging(os.path.join(run_dir, "log"), console, debug)
		
	return run_dir

def get_logger(name):
	return logging.getLogger(name)

def close_logging():
	if _LOG_INITIALZED:
		with open(os.path.join(_LOG_PATH, 'profiling.txt'), 'w') as f:
			DEFAULT_PROFILER.stats(f)
		_LOG.info("DONE")
		logging.shutdown()
