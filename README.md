# Bayesian Optimisation over Multiple Continuous and Categorical Inputs

We propose a new Bayesian optimisation approach for optimising a black-box function
with multiple continuous and categorical inputs, termed Continuous and Categorical Bayesian Optimisation (CoCaBO) and this is the Python code repository. For more details on the method, please read our paper [Bayesian Optimisation over Multiple Continuous and Categorical Inputs](https://arxiv.org/abs/1906.08878). 

### Dependencies:
* python 3
* numpy
* scipy
* matplotlib
* tqdm
* gpy
* pandas

### Usage:
Run CoCaBO experiments: `python run_cocabo_exps.py` followed by the following flags:
  * `-f` Objective function: default=`'func2C'`
  * `-n` Max Optimisation iterations: default = `200`
  * `-tl` Number of random initialisation: default = `40`
  * `-mix` Mixture weight for categorical and continous kernel: default=`0.0`
  * `-b` Batch size (>1 for batch CoCaBO and =1 for sequential CoCaBO). Recommed to use batch CoCaBO when the number of your categories for each categorical variable > batch size.: default=`1`
 
  E.g. `python run_cocabo_exps.py -f='func3C' -n=200 -tl=40 -mix=0.0 -b=1`
  
### Citation
Please cite our paper if you would like to use the code.

```
@article{ru2019bayesian,
  title={Bayesian Optimisation over Multiple Continuous and Categorical Inputs},
  author={Ru, Binxin and Alvi, Ahsan S and Nguyen, Vu and Osborne, Michael A and Roberts, Stephen J},
  journal={arXiv preprint arXiv:1906.08878},
  year={2019}
}
```



