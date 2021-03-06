import os
from pathlib import Path
import logging

import numpy as np


class MomentumDistribution:
    def __init__(self, mass_matrix_diag):
        self.mass_matrix_diag = mass_matrix_diag
        self.mass_matrix_inv_diag = 1 / mass_matrix_diag

    def sample(self):
        n = self.mass_matrix_diag.shape[0]
        standard_normal_samples = np.random.normal(size=n)
        mass_matrix_diag_sqrt = np.sqrt(self.mass_matrix_diag)
        return np.multiply(standard_normal_samples, mass_matrix_diag_sqrt)

    def log_pdf(self, phi):
        return -0.5 * np.dot(phi, np.multiply(self.mass_matrix_inv_diag, phi))

    def mass_inverse_mvm(self, phi):
        result = np.multiply(self.mass_matrix_inv_diag, phi)
        return result

class HMCSampler:

    def __init__(self, *, func_lpdf=None, func_nabla_lpdf=None, func_plot=None,
                 init_estimand=None, init_M_diag=None, init_L=100, init_epsilon=0.01,
                 n_burnin=20_000, n_calib=40_000, S=80_000, n_info_interval=500, thinning=5,
                 unique_estimation_id=0,
                 save_samples=False, adaptive=False, record_full_trace=False,
                 snapshot_path=None):

        self.uid = unique_estimation_id
        self.thinning = thinning

        self.func_lpdf = func_lpdf
        self.func_nabla_lpdf = func_nabla_lpdf
        self.func_plot = func_plot

        self.estimand = init_estimand
        self.L = init_L
        self.epsilon = init_epsilon
        self.adaptive = adaptive
        self.save_samples = save_samples

        self.S = S
        self.n_info_interval = n_info_interval
        self.n_calib = n_calib
        self.n_burnin = n_burnin
        self.adjust_interval_calib = 100
        self.adjust_interval_burnin = 100

        self.s = 0
        self.samples = []
        self.total_samples = 0
        self.accepted_samples = 0
        self.samples = []
        self.likelihood_samples = []

        self.record_full_trace = record_full_trace
        if self.record_full_trace:
            self.full_trace = np.zeros((S, init_estimand.shape[0]))
        self.snapshot_path = snapshot_path

        self.momentum = MomentumDistribution(init_M_diag)
        self.logger = logging.getLogger(__name__)

    def reset_counters(self):
        self.samples = []
        self.total_samples = 0
        self.accepted_samples = 0
        self.samples = []
        self.likelihood_samples = []

    def snapshot(self):
        file_name_clean = f"hmc-samples--{self.uid}--{self.s}"
        file_path_clean = Path(self.snapshot_path) / file_name_clean
        directory = os.path.dirname(file_path_clean)
        if not os.path.exists(directory):
            os.makedirs(directory)
        hmc_samples_array = np.asarray(self.samples)
        np.save(file_path_clean, hmc_samples_array[::self.thinning ,:])

        if self.record_full_trace:
            file_name_full = f"hmc-full_trace--{self.uid}"
            file_path_full = Path(self.snapshot_path) / file_name_full
            np.save(file_path_full, np.asarray(self.full_trace))

    def adjust_leapfrogging(self):
        if self.total_samples > 0:
            acceptance = self.accepted_samples / self.total_samples

            leapfrog_prod = 1

            if acceptance > 0.80 or acceptance < 0.50:
                change_sign = np.sign(acceptance - 0.65)
                self.epsilon = self.epsilon + change_sign * 0.2 * self.epsilon

                # This can cause problems for certain scenarios
                if self.adaptive:
                    self.L = int(leapfrog_prod / self.epsilon) + 1

                self.reset_counters()

                self.logger.info \
                    (f"Iteration {self.s}, AR: {acceptance:0.4f}, adjusted to L={self.L}, ??={self.epsilon:0.4f}")

    def adjust_momentum(self):
        if self.total_samples > 0:
            hmc_samples = np.asarray(self.samples)
            cov_estim = np.cov(hmc_samples.T)
            self.momentum = MomentumDistribution(1 / np.diag(cov_estim))
            # this is a safety mechanism so that it does not propose unrealistic
            # values after calibration of the mass matrix. epsilon will adjust itself
            # in subsequent iterations
            self.epsilon = 0.1 * self.epsilon
            self.reset_counters()

    def sample_one(self):
        #### BURN-IN ####
        if self.s == self.n_burnin:
            self.reset_counters()

        #### REPORTING ####
        if self.total_samples > 0 and (self.s + 1) % self.n_info_interval == 0:
            self.logger.info(f"Iteration {self.s}, AR: {(self.accepted_samples / self.total_samples):0.3f}")
            if self.save_samples:
                self.snapshot()

        #### ADJUSTMENTS ####
        # adjustment L-epsilon
        if self.s < self.n_burnin and (self.s + 1) % self.adjust_interval_burnin == 0:
            self.adjust_leapfrogging()

        if self.s < self.n_calib + self.n_burnin and (self.s + 1) % self.adjust_interval_calib == 0:
            self.adjust_leapfrogging()

        # calibration of the mass matrix
        if self.s == self.n_calib - 1:
            self.adjust_momentum()

        #### SAMPLING CORE ####
        phi = self.momentum.sample()
        phi_star = phi.copy()
        estimand = self.estimand
        estimand_star = estimand.copy()

        try:
            for leapfrog_iter in range(self.L):
                if leapfrog_iter == 0:
                    phi_star = phi_star + 0.5 * self.epsilon * self.func_nabla_lpdf(estimand_star)

                estimand_star = estimand_star + self.epsilon * self.momentum.mass_inverse_mvm(phi_star)

                if leapfrog_iter != self.L - 1:
                    phi_star = phi_star + 1 * self.epsilon * self.func_nabla_lpdf(estimand_star)
                else:
                    phi_star = phi_star + 0.5 * self.epsilon * self.func_nabla_lpdf(estimand_star)

            if np.any(np.isinf(phi_star)) or np.any(np.isnan(phi_star)):
                self.logger.info("??* is NaN or Inf")

            if not np.any(np.isinf(phi_star)) and not np.any(np.isnan(phi_star)):
                likelihood = self.func_lpdf(estimand)
                likelihood_star = self.func_lpdf(estimand_star)

                mom_lpdf = self.momentum.log_pdf(phi)
                mom_star_lpdf = self.momentum.log_pdf(phi_star)

                ratio = np.exp(likelihood_star + mom_star_lpdf - likelihood - mom_lpdf)
                self.logger.info(
                    f"Iteration {self.s},\tLL*: {likelihood_star:.2f},\t??*: {mom_star_lpdf:.2f},\tLL: {likelihood:.2f},\t??: {mom_lpdf:.2f}\tR:{ratio:.2f}")
                if ratio > np.random.uniform(0, 1):
                    estimand = estimand_star
                    self.accepted_samples += 1

        except np.linalg.LinAlgError as error:
            self.logger.warning(f"Failed to sample: {error}")
        except Exception as e:
            self.logger.warning(f"Failed to sample: {e}")

        if self.record_full_trace:
            self.full_trace[self.s, :] = estimand.copy()

        self.total_samples += 1
        self.estimand = estimand
        self.samples.append(estimand)
        self.likelihood_samples.append(self.func_lpdf(estimand))
        self.s += 1

    def sample(self):
        while self.s < self.S:
            self.sample_one()

        if self.save_samples:
            self.snapshot()
