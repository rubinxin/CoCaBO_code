# Code for Bayesian Optimisation over Multiple Continuous and Categorical Inputs

### Usage:
Run CoCaBO experiments: `python run_cocabo_exps.py` followed by the following flags:
  * `-f` Objective function: default=`'func2C'`
  * `-n` Max Optimisation iterations: default = `200`
  * `-tl` Number of random initialisation: default = `40`
  * `-mix` Mixture weight for categorical and continous kernel: default=`0.0`
  * `-b` Batch size (>1 for batch CoCaBO and =1 for sequential CoCaBO). Recommed to use batch CoCaBO when the number of your categories for each categorical variable > batch size.: default=`1`
 
  E.g. `python run_cocabo_exps.py -f='func3C' -n=200 -tl=40 -mix=0.0 -b=1`
  


### Dependencies:
* python 3
* numpy
* scipy
* matplotlib
* tqdm
* GPy
* pandas

