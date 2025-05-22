from lib.util.logging import setup_run, get_logger, close_logging
from lib.util.profiling import SAMPLE
from lib.util.output import StringWriter

import os, signal, itertools
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch

assert torch.cuda.is_available()
cuda_device = torch.device("cuda")
cpu_device = torch.device("cpu")

def add_failed_test(dict, test_name, setup_name):
    if test_name not in dict:
        dict[test_name] = []
    dict[test_name].append(setup_name)

def run_test(tests_fn, test_params_fn, name):
    tests = tests_fn()

    failed_tests = {}

    LOG.info("Running %s tests.", name)
    for test_name, (test_fn, setups) in tests.items():
        LOG.info("\tTest: %s", test_name)
        for setup_name, domain_setup_fn in setups.items():
            LOG.info("Setup: %s", setup_name)
            try:
                test_params = test_params_fn(domain_setup_fn)
            except KeyboardInterrupt:
                raise
            except:
                LOG.exception("setup of %s failed for test %s with params %s", setup_name, test_name)
                add_failed_test(failed_tests, test_name+"_setup", setup_name)
            else:
                try:
                    test_fn(**test_params)
                except KeyboardInterrupt:
                    raise
                except:
                    LOG.exception("%s failed for %s.", test_name, setup_name)
                    add_failed_test(failed_tests, test_name, setup_name)
    if len(failed_tests)>0:
        LOG.warning("%d %s tests failed", len(failed_tests), name)
    else:
        LOG.info("%s tests done.", name)
    return failed_tests

if __name__=="__main__":
    #signal.signal(signal.SIGINT, handle_interrupt)
    
    run_dir = setup_run("./test_runs",
        #name="AdvStatic-x_r4_s1_divFree-gradTEST" #gridTransform-scaling_s1-advectGradCheck" #-aSwap-cInv-dSwap"
        #name="gradcheck_tests"
        #name="optim_tests"
        name="PISOtorch_tests-full"
    )
    
    LOG = get_logger("main")


    import tests.gradcheck
    import tests.optim
    #import tests.pisosim
    import tests.pisosteps
    import tests.domain

    tests_modules = [
        (tests.pisosteps.get_tests, tests.pisosteps.get_test_params, "pisosteps"),
        (tests.pisosteps.get_tests, tests.pisosteps.get_test_params_diff, "pisosteps-diff"),
        (tests.gradcheck.get_tests, tests.gradcheck.get_test_params, "gradcheck"),
        (tests.optim.get_tests, tests.optim.get_test_params, "optim"),
        (tests.domain.get_tests, tests.domain.get_test_params, "domain"),
    ]
    
    failed_tests = {}
    try:
        for test_mod in tests_modules:
            with SAMPLE(test_mod[-1]):
                ft = run_test(*test_mod)
            if len(ft)>0:
                failed_tests[test_mod[-1]] = ft
        #failed_tests.extend(run_step_tests())
        #failed_tests.extend(run_sim_tests())
        #failed_tests.extend(run_step_tests(diff_backend=True))
        #failed_tests.extend(run_gradcheck_tests())
        #failed_tests.extend(run_optim_tests())
    except KeyboardInterrupt:
        LOG.info("Tests interrupted.")
    
    if len(failed_tests)==0:
        LOG.info("All tests DONE.")
    else:
        s = StringWriter()
        for mod_name, tests_failed in failed_tests.items():
            s.write_line(mod_name)
            for test_name, setups_failed in tests_failed.items():
                s.write("\t")
                s.write_line(test_name)
                for setup_name in setups_failed:
                    s.write("\t\t")
                    s.write_line(setup_name)
        LOG.warning("FAILED tests:\n%s", s.get_string())
    
    close_logging()