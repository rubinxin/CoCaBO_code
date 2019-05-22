
# =============================================================================
#  CoCaBO Algorithms 
# =============================================================================
import sys
sys.path.append('../bayesopt')
sys.path.append('../ml_utils')
import argparse
import os
import testFunctions.syntheticFunctions
from methods.CoCaBO import CoCaBO
from methods.BatchCoCaBO import BatchCoCaBO


def CoCaBO_Exps(obj_func, budget, initN=24 ,trials=40, kernel_mix = 0.5, batch=None):

    saving_path = f'data/syntheticFns/{obj_func}/'

    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    if obj_func == 'func2C':
        f = testFunctions.syntheticFunctions.func2C
        categories = [3, 5]

        bounds = [{'name': 'h1', 'type': 'categorical', 'domain': (0, 1, 2)},
            {'name': 'h2', 'type': 'categorical', 'domain': (0, 1, 2, 3, 4)},
            {'name': 'x1', 'type': 'continuous', 'domain': (-1, 1)},
            {'name': 'x2', 'type': 'continuous', 'domain': (-1, 1)}]

    elif obj_func == 'func3C':
        f = testFunctions.syntheticFunctions.func3C
        categories = [3, 5, 4]

        bounds = [{'name': 'h1', 'type': 'categorical', 'domain': (0, 1, 2)},
            {'name': 'h2', 'type': 'categorical', 'domain': (0, 1, 2, 3, 4)},
            {'name': 'h3', 'type': 'categorical', 'domain': (0, 1, 2, 3)},
            {'name': 'x1', 'type': 'continuous', 'domain': (-1, 1)},
            {'name': 'x2', 'type': 'continuous', 'domain': (-1, 1)}]

    else:
        raise NotImplementedError

    # Run CoCaBO Algorithm
    if batch == 1:
        mabbo = CoCaBO(objfn=f, initN=initN, bounds=bounds,
                       acq_type='LCB', C=categories,
                       kernel_mix = kernel_mix)

    else:
        mabbo = BatchCoCaBO(objfn=f, initN=initN, bounds=bounds,
                            acq_type='LCB', C=categories,
                            kernel_mix=kernel_mix,
                            batch_size=batch)
    mabbo.runTrials(trials, budget, saving_path)




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run BayesOpt Experiments")
    parser.add_argument('-f', '--func', help='Objective function',
                        default='func2C', type=str)
    parser.add_argument('-mix', '--kernel_mix',
                        help='Mixture weight for production and summation kernel. Default = 0.0', default=0.0,
                        type=float)
    parser.add_argument('-n', '--max_itr', help='Max Optimisation iterations. Default = 100',
                        default=100, type=int)
    parser.add_argument('-tl', '--trials', help='Number of random trials. Default = 20',
                        default=20, type=int)
    parser.add_argument('-b', '--batch', help='Batch size. Default = 1',
                        default=1, type=int)

    args = parser.parse_args()
    print(f"Got arguments: \n{args}")
    obj_func = args.func
    kernel_mix = args.kernel_mix
    n_itrs = args.max_itr
    n_trials = args.trials
    batch = args.batch

    CoCaBO_Exps(obj_func=obj_func, budget=n_itrs,
                 trials=n_trials, kernel_mix = kernel_mix, batch=batch)
