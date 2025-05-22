
import os, signal
from lib.util.logging import setup_run, get_logger, close_logging

if __name__=="__main__":

    # choose which GPU to use, BEFORE importing torch
    from lib.util.GPU_info import get_available_GPU_id
    cudaID = None
    os.environ["CUDA_VISIBLE_DEVICES"] = cudaID or str(get_available_GPU_id(active_mem_threshold=0.8, default=None))

    from PISOtorch_simulation import StopHandler

    from tests.validations import plane_poiseuille_flow, lid_driven_cavity_2D, lid_driven_cavity_3D


    run_dir = setup_run("./test_runs", #BoundaryLayer  
        #name="validations_Lid2D-resampled_rot90sine_nonOrtho_sp" #
        #name="validations_Lid3D" #-resampled_rot60sine
        #name="validations_Lid2D-Re100_plot-grid"
        #name="validations_Lid3D-Re1000"
        name="validations_PHPF"
    )
    LOG = get_logger("VAL")
    stop_handler = StopHandler(LOG)
    stop_handler.register_signal()

    tests = ["PHPF"]
    #tests = ["Lid3D"]
    LOG.info("Running tests %s on GPU %s", tests, cudaID)
    
    if "PHPF" in tests and not stop_handler():
        try:
            sub_dir = os.path.join(run_dir, "plane_poiseuille_flow_it-10")
            os.makedirs(sub_dir)
            plane_poiseuille_flow(sub_dir, it=10, STOP_FN=stop_handler, resolutions=[8,16,32,64], max_size_diff_rel=[1,4]) #, rot_distortion_max_angle=60) #, max_size_diff_rel=[1], resolutions=[16,64])
        except:
            LOG.exception("Validation tests failed:")
    
    if "Lid2D" in tests and not stop_handler():
        try:
            sub_dir = os.path.join(run_dir, "Lid2D_Re100")
            os.makedirs(sub_dir)
            lid_driven_cavity_2D(sub_dir, Re=100, it=1, STOP_FN=stop_handler, resolutions=[32])#, rot_distortion_max_angle=90, dp=False) #   16,,24
            # sub_dir = os.path.join(run_dir, "Lid2D_Re5000")
            # os.makedirs(sub_dir)
            # lid_driven_cavity_2D(sub_dir, Re=5000, it=100, STOP_FN=stop_handler, resolutions=[8,16,32,64,128])
        except:
            LOG.exception("Validation tests failed:")
    
    if "Lid3D" in tests and not stop_handler():
        try:
            sub_dir = os.path.join(run_dir, "Lid3D_Re1000")
            os.makedirs(sub_dir)
            lid_driven_cavity_3D(sub_dir, it=50, STOP_FN=stop_handler, resolutions=[8,16,32,64])
        except:
            LOG.exception("Validation tests failed:")
    
    stop_handler.unregister_signal()
    close_logging()