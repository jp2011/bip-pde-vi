import os
from pathlib import Path
import logging

import numpy as np


class PcnSampler:

    def __init__(self, *, init_estimand=None, n_burnin=20_000, S=80_000, n_info_interval=500, thinning=1,
                 unique_estimation_id=0, save_samples=False, record_full_trace=False, snapshot_path=None, prior=None,
                 likelihood=None):

        self.prior = prior
        self.likelihood = likelihood

        self.uid = unique_estimation_id
        self.thinning = thinning

        self.estimand = init_estimand
        self.save_samples = save_samples

        self.S = S
        self.n_info_interval = n_info_interval
        self.n_burnin = n_burnin
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

        self.logger = logging.getLogger(__name__)

    def reset_counters(self):
        self.samples = []
        self.total_samples = 0
        self.accepted_samples = 0
        self.likelihood_samples = []

    def snapshot(self):
        file_name_clean = f"pcn-samples--{self.uid}--{self.s}"
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

    def adjust_beta(self):
        # calibrate the beta
        ar = self.accepted_samples / self.total_samples
        if ar < 0.18:
            self.beta = max(0, self.beta - 0.15 * self.beta)
            self.reset_counters()
            self.logger.info(f"Iteration {self.s + 1}: AR={ar}. Adjusting the beta: decreasing to {self.beta}")

        elif ar > 0.28:
            self.beta = min(self.beta + 0.05 * self.beta, 1.0)
            self.reset_counters()
            self.logger.info(f"Iteration {self.s + 1}: AR={ar}. Adjusting the beta: increasing to {self.beta}")

    def sample(self, initial_beta=0.5):
        self.s = 0
        self.beta = initial_beta

        self.estimand = self.prior.sample()
        log_lik_current = self.likelihood(self.estimand)

        for i in range(self.S):
            self.s = i

            #### BURN-IN ####
            if self.s == self.n_burnin:
                self.reset_counters()

            #### REPORTING ####
            if self.total_samples > 0 and (self.s + 1) % self.n_info_interval == 0:
                self.logger.info(f"Iteration {self.s}, AR: {(self.accepted_samples / self.total_samples):0.3f}")
                if self.save_samples:
                    self.snapshot()

            #### ADJUSTMENTS ####
            # adjustment beta
            if self.s < self.n_burnin and (self.s + 1) % self.adjust_interval_burnin == 0:
                self.adjust_beta()

            #### SAMPLING CORE ####
            # Propose new sample
            prior_sample = self.prior.sample()
            estimand_star = self.prior.mean() + np.sqrt(1 - self.beta ** 2) * (self.estimand - self.prior.mean()) + self.beta * (prior_sample - self.prior.mean())

            # Compute acceptance probability
            log_lik_new = self.likelihood(estimand_star)

            # Accept-reject step
            a = min(1, np.exp((-1) * log_lik_current - (-1) * log_lik_new))
            if a > np.random.uniform():
                self.estimand = estimand_star
                log_lik_current = log_lik_new
                self.accepted_samples += 1

            self.total_samples += 1
            self.samples.append(self.estimand)
            self.full_trace[i, :] = self.estimand.copy()
        if self.save_samples:
            self.snapshot()