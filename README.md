# bip-pde-vi
Codes for performing variational inference for Bayesian Inverse Problems that involve PDEs.

1. To run MCMC methods, run `python mcmc_launcher.py` with appropriate flags.
2. To run VB methods, run `python vi_launcher.py` with appropriate flags.

Note that one needs Python bindings for an FEM solves. The python must be called `poisson_1d_fem_driver` to be able to perform inference and needs to implement the following methods:
1. `set_data_gen_seed(data_seed)`
2. `set_params(kappa)`
3. `get_log_lik()`
4. `get_log_lik_grad()`
5. `get_param_coords()` (this is the coordinates of the centroids of elements for the discretisation of kappa)
6. `generate_observations(kappa)`
7. `get_solution_mean()`
8. `get_observations()`
