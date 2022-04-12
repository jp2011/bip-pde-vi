import logging
import os
import pickle
import sys
import time
from abc import abstractmethod, ABC
from pathlib import Path

import click
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tfp.math.psd_kernels

from hmc import HMCSampler
from pcn import PcnSampler

import poisson_1d_fem_driver as fem_cpp_driver


class McmcExperiment(ABC):
    experiment_folder = Path(os.getenv("STATFEMROOT")) / "del2" / "apps" / "user" / "inverseVBpoisson1D"
    def __init__(self, *, true_lengthscale=None, prior_lengthscale=None, infer_lengthscale=None, total_samples=None,
                 fem=None, experiment_name=None, inference_seed=None, data_seed=None):
        self.true_lengthscale = true_lengthscale
        self.prior_lengthscale = prior_lengthscale
        self.infer_lengthscale = infer_lengthscale
        self.total_samples = total_samples
        self.fem = fem
        self.experiment_name = experiment_name
        self.data_seed = data_seed
        self.inference_seed = inference_seed
        self.algorithm = 'specified_by_subclass'

        self.loglik_grad_evals = 0
        self.loglik_evals = 0


        # generate the observations
        tf.random.set_seed(data_seed)
        np.random.seed(data_seed)
        true_kappa_lengthscale = tf.constant(self.true_lengthscale, dtype=tf.float64)
        true_kappa_amplitude = tf.constant(1.0, dtype=tf.float64)

        kappa_prior = tfd.GaussianProcess(
            kernel=tfk.ExponentiatedQuadratic(length_scale=true_kappa_lengthscale,
                                                               amplitude=true_kappa_amplitude),
            index_points=self.fem.get_param_coords())

        self.true_kappa = kappa_prior.sample().numpy()
        fem.generate_observations(self.true_kappa)
        self.true_solution = fem.get_solution_mean()
        self.observations = fem.get_observations()

        # set the seed for inference
        tf.random.set_seed(inference_seed)
        np.random.seed(inference_seed)

        self.clean_samples = []
        self.full_trace = np.asarray([])

    def save_output(self):

        filepath_clean = McmcExperiment.experiment_folder / "output" / "data"  / f"clean-samples-{self.get_experiment_id()}.npy"
        filepath_full = McmcExperiment.experiment_folder / "output" / "data"  / f"full-trace-{self.get_experiment_id()}.npy"
        filepath_context = McmcExperiment.experiment_folder / "output" / "data" / f"context-{self.get_experiment_id()}.pickle"
        os.makedirs(os.path.dirname(filepath_clean), exist_ok=True)

        np.save(filepath_clean, np.asarray(self.clean_samples))
        np.save(filepath_full, np.asarray(self.full_trace))
        pickle.dump({
            'true_lengthscale': self.true_lengthscale,
            'prior_lengthscale': self.prior_lengthscale,
            'infer_lengthscale': self.infer_lengthscale,
            'true_kappa': self.true_kappa,
            'true_solution': self.true_solution,
            'observations': self.observations,
            'total_samples': self.total_samples,
            'experiment_name': self.experiment_name,
            'data_seed': self.data_seed,
            'inference_seed': self.inference_seed,
            'algorithm': self.algorithm,
            'time': self.elapsed_time,
            'loglik_grad_evals': self.loglik_grad_evals,
            'loglik_evals': self.loglik_evals
        }, open(filepath_context, "wb" ))

    @classmethod
    def build_experiment_id(cls, *, algorithm=None, experiment_name=None, true_lengthscale=None, prior_lengthscale=None,
                       infer_lengthscale=None, total_samples=None, data_seed=None, inference_seed=None):
        experiment_id = f"{algorithm}--{experiment_name}--true-len-{str(true_lengthscale).replace('.', '_')}--prior-len-{str(prior_lengthscale).replace('.', '_')}--inferred-len-{infer_lengthscale}--total-samples-{total_samples}--data-seed-{data_seed}--inference-seed-{inference_seed}"
        return experiment_id

    @classmethod
    def load_from_file(cls, *, algorithm=None, experiment_name=None, true_lengthscale=None, prior_lengthscale=None,
                       infer_lengthscale=None, total_samples=None, data_seed=None, inference_seed=None, include_full_trace=False):
        experiment_id = McmcExperiment.build_experiment_id(algorithm=algorithm, experiment_name=experiment_name,
                                            true_lengthscale=true_lengthscale, prior_lengthscale=prior_lengthscale,
                                            infer_lengthscale=infer_lengthscale, total_samples=total_samples,
                                            data_seed=data_seed, inference_seed=inference_seed)

        filepath_clean = McmcExperiment.experiment_folder / "output" / "data" / f"clean-samples-{experiment_id}.npy"
        filepath_context = McmcExperiment.experiment_folder / "output" / "data" / f"context-{experiment_id}.pickle"

        clean_samples = np.load(filepath_clean)
        context = pickle.load(open(filepath_context, "rb"))

        if include_full_trace:
            filepath_full = McmcExperiment.experiment_folder / "output" / "data" / f"full-trace-{experiment_id}.npy"
            full_trace = np.load(filepath_full)
            return context, clean_samples, full_trace
        else:
            return context, clean_samples

    def get_experiment_id(self):
        return McmcExperiment.build_experiment_id(algorithm=self.algorithm, experiment_name=self.experiment_name,
                                                  true_lengthscale=self.true_lengthscale,
                                                  prior_lengthscale=self.prior_lengthscale,
                                                  infer_lengthscale=self.infer_lengthscale,
                                                  total_samples=self.total_samples, data_seed=self.data_seed,
                                                  inference_seed=self.inference_seed)

    def run(self):
        start = time.time()
        self.infer()
        end = time.time()
        self.elapsed_time = end - start
        self.save_output()

    @abstractmethod
    def infer(self):
        pass

class HmcExperiment(McmcExperiment):

    def __init__(self, *, true_lengthscale=None, prior_lengthscale=None, infer_lengthscale=None, total_samples=None,
                 fem=None, experiment_name=None, data_seed=None, inference_seed=None):
        super(HmcExperiment, self).__init__(true_lengthscale=true_lengthscale, prior_lengthscale=prior_lengthscale,
                                            infer_lengthscale=infer_lengthscale, total_samples=total_samples, fem=fem,
                                            experiment_name=experiment_name, data_seed=data_seed, inference_seed=inference_seed)
        self.algorithm = 'hmc'

    def infer(self):
        prior_kernel = tfk.ExponentiatedQuadratic(length_scale=self.prior_lengthscale,
                                                  amplitude=tf.constant(1.0, dtype=tf.float64))

        prior_covmat = prior_kernel.matrix(self.fem.get_param_coords(), self.fem.get_param_coords()).numpy()
        prior_covmat = prior_covmat + 1e-6 * np.eye(self.fem.get_param_coords().shape[0])
        inv_prior_covmat = np.linalg.pinv(prior_covmat)

        n_total_iterations = self.total_samples
        n_burnin = int(0.25 * self.total_samples)
        n_calibration = int(0.5 * self.total_samples)
        def log_pdf(log_kappa):
            self.loglik_evals += 1
            self.fem.compute_likelihood(log_kappa)
            prior_lpdf = -0.5 * np.dot(log_kappa, np.dot(inv_prior_covmat, log_kappa))
            return self.fem.get_log_lik() + prior_lpdf

        def nabla_log_pdf(log_kappa):
            self.loglik_grad_evals += 1
            self.fem.set_params(log_kappa)
            nabla_prior_lpdf = -1 * np.dot(inv_prior_covmat, log_kappa)
            return np.multiply(self.fem.get_log_lik_grad(), np.exp(log_kappa)) + nabla_prior_lpdf

        init_estimand = np.random.normal(0, 1, self.fem.get_param_coords().shape[0])
        mass_matrix_diag = 1e3 * np.ones(self.fem.get_param_coords().shape[0])
        sampler = HMCSampler(func_lpdf=log_pdf,
                             func_nabla_lpdf=nabla_log_pdf,
                             func_plot=None,
                             init_estimand=init_estimand,
                             init_M_diag=mass_matrix_diag,
                             init_L=30,
                             init_epsilon=0.03,
                             n_burnin=n_burnin,
                             n_calib=n_calibration,
                             S=n_total_iterations,
                             n_info_interval=max(2, int(0.3 * n_total_iterations)),
                             thinning=1,
                             snapshot_path=McmcExperiment.experiment_folder / "output" / "data" / "snapshots",
                             unique_estimation_id=self.get_experiment_id(),
                             adaptive=False, save_samples=True, record_full_trace=True)
        sampler.sample()
        self.clean_samples = sampler.samples
        self.full_trace = sampler.full_trace


class PcnExperiment(McmcExperiment):

    def __init__(self, *, true_lengthscale=None, prior_lengthscale=None, infer_lengthscale=None, total_samples=None,
                 fem=None, experiment_name=None, data_seed=None, inference_seed=None):
        super(PcnExperiment, self).__init__(true_lengthscale=true_lengthscale, prior_lengthscale=prior_lengthscale,
                                            infer_lengthscale=infer_lengthscale, total_samples=total_samples, fem=fem,
                                            experiment_name=experiment_name, data_seed=data_seed, inference_seed=inference_seed)
        self.algorithm = 'pcn'

    def infer(self):
        fem = self.fem
        prior_lengthscale = self.prior_lengthscale
        class PcnPrior():
            def __init__(self):
                prior_kernel = tfk.ExponentiatedQuadratic(length_scale=prior_lengthscale,
                                                          amplitude=tf.constant(1.0, dtype=tf.float64))
                self.gp = tfd.GaussianProcess(kernel=prior_kernel, index_points=fem.get_param_coords())

            def sample(self):
                return self.gp.sample().numpy()

            def mean(self):
                return self.gp.mean().numpy()

        def likelihood(kappa_param):
            self.loglik_evals += 1
            self.fem.compute_likelihood(kappa_param)
            return self.fem.get_log_lik()

        n_total_iterations = self.total_samples
        n_burnin = int(0.6 * self.total_samples)
        pcn_sampler = PcnSampler(init_estimand=np.random.normal(0, 1, self.fem.get_param_coords().shape[0]),
                                 n_burnin=n_burnin,
                                 S=n_total_iterations,
                                 n_info_interval=int(0.1 * n_total_iterations), thinning=1,
                                 unique_estimation_id=self.get_experiment_id(),
                                 save_samples=True, record_full_trace=True,
                                 snapshot_path=McmcExperiment.experiment_folder / "output" / "data" / "snapshots",
                                 prior=PcnPrior(), likelihood=likelihood)
        pcn_sampler.sample()
        self.clean_samples = pcn_sampler.samples
        self.full_trace = pcn_sampler.full_trace


@click.command()
@click.option('--experiment_name', type=str, default="POISSON-1D-2021-06-15")
@click.option('--algorithm', type=click.Choice(['hmc', 'pcn'], case_sensitive=False), default='pcn')
@click.option('--true_lengthscale', type=np.float64, default=0.2)
@click.option('--prior_lengthscale', type=np.float64, default=0.2)
@click.option('--infer_lengthscale', is_flag=True)
@click.option('--total_iters', type=int, default=10_000)
@click.option('--data_seed', type=int, default=42)
@click.option('--inference_seed', type=int, default=42)
@click.option('--verbose', is_flag=True)
def main(experiment_name, algorithm, true_lengthscale, prior_lengthscale, infer_lengthscale, total_iters, data_seed, inference_seed, verbose):

    fem = fem_cpp_driver.FemSolverInterface()
    fem.set_data_gen_seed(data_seed)

    if algorithm == 'pcn':
        mcmc_runner = PcnExperiment(true_lengthscale=true_lengthscale, prior_lengthscale=prior_lengthscale,
                                    infer_lengthscale=infer_lengthscale, total_samples=total_iters, fem=fem,
                                    experiment_name=experiment_name, data_seed=data_seed, inference_seed=inference_seed)
    else:
        mcmc_runner = HmcExperiment(true_lengthscale=true_lengthscale, prior_lengthscale=prior_lengthscale,
                                    infer_lengthscale=infer_lengthscale, total_samples=total_iters, fem=fem,
                                    experiment_name=experiment_name, data_seed=data_seed, inference_seed=inference_seed)

    log_fmt = '[%(levelname)s] [%(asctime)s] [%(name)s] %(message)s'
    datefmt = '%H:%M:%S'
    if verbose:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=log_fmt)
    else:
        log_folder = McmcExperiment.experiment_folder / 'logs'
        os.makedirs(log_folder, exist_ok=True)
        logging.basicConfig(filename= log_folder / f"log-{mcmc_runner.get_experiment_id()}.log",
                            filemode='a',
                            format=log_fmt,
                            datefmt=datefmt,
                            level=logging.DEBUG)
    mcmc_runner.run()

if __name__ == "__main__":
    main()
