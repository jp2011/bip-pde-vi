import logging
import os
import sys
import pickle
import time
from abc import abstractmethod, ABC
from pathlib import Path
import click
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy import sparse

tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tfp.math.psd_kernels

import poisson_1d_fem_driver as fem_cpp_driver


class ViExperiment(ABC):
    experiment_folder = Path(os.getenv("STATFEMROOT")) / "del2" / "apps" / "user" / "inverseVBpoisson1D"
    def __init__(self, *, true_lengthscale=None, prior_lengthscale=None, infer_lengthscale=None, total_samples=None,
                 fem=None, experiment_name=None, data_seed=None, inference_seed=None, svi_samples=3, extra_param=None):
        self.true_lengthscale = true_lengthscale
        self.prior_lengthscale = prior_lengthscale
        self.infer_lengthscale = infer_lengthscale
        self.total_samples = total_samples
        self.fem = fem
        self.experiment_name = experiment_name
        self.data_seed = data_seed
        self.inference_seed = inference_seed
        self.svi_samples = svi_samples
        self.extra_param = extra_param
        self.algorithm = 'specified_by_subclass'

        self.window_size = 5_000

        def log_of_data(kappas_tf):
            lls_and_grads = []
            for kappa_tf in kappas_tf:
                fem.set_params(kappa_tf.numpy())
                loglik = fem.get_log_lik()
                loglik_grad = np.multiply(fem.get_log_lik_grad(), np.exp(kappa_tf.numpy()))
                lls_and_grads.append(np.concatenate([[loglik], loglik_grad], axis=0))
            lls_and_grads = np.stack(lls_and_grads, axis=0)
            return lls_and_grads

        @tf.custom_gradient
        def tf_loglik(kappa_tf):
            ll_kappa_grad = tf.py_function(log_of_data, [kappa_tf], tf.float64)
            ll = ll_kappa_grad[:, 0]
            kappa_grad = ll_kappa_grad[:, 1:]

            def grad(dy):
                return tf.expand_dims(dy, axis=1) * kappa_grad

            return ll, grad
        self.tf_loglik = tf_loglik

        # generate the observations
        tf.random.set_seed(self.data_seed)
        np.random.seed(self.data_seed)
        true_kappa_lengthscale = tf.constant(self.true_lengthscale, dtype=tf.float64)
        true_kappa_amplitude = tf.constant(1.0, dtype=tf.float64)

        kappa_prior = tfd.GaussianProcess(kernel=tfk.ExponentiatedQuadratic(length_scale=true_kappa_lengthscale,
                                                                            amplitude=true_kappa_amplitude),
                                          index_points=self.fem.get_param_coords())

        self.true_kappa = kappa_prior.sample().numpy()
        fem.generate_observations(self.true_kappa)
        self.true_solution = fem.get_solution_mean()
        self.observations = fem.get_observations()

        tf.random.set_seed(self.inference_seed)
        np.random.seed(self.inference_seed)


        self.loss_trace = None
        self.mean = None
        self.cov = None


    def save_output(self):

        filepath_output = ViExperiment.experiment_folder / "output" / "data" / f"output-{self.get_experiment_id()}.pickle"
        directory = os.path.dirname(filepath_output)
        if not os.path.exists(directory):
            os.makedirs(directory)

        pickle.dump({
            'true_lengthscale': self.true_lengthscale,
            'prior_lengthscale': self.prior_lengthscale,
            'infer_lengthscale': self.infer_lengthscale,
            'true_kappa': self.true_kappa,
            'true_solution': self.true_solution,
            'observations': self.observations,
            'total_samples': self.total_samples,
            'experiment_name': self.experiment_name,
            'inference_seed': self.inference_seed,
            'data_seed': self.data_seed,
            'algorithm': self.algorithm,
            'time': self.elapsed_time,
            'mean' : self.mean,
            'cov' : self.cov,
            'extra_param': self.extra_param,
            'loss_trace': self.loss_trace,
        }, open(filepath_output, "wb" ))

    @classmethod
    def build_experiment_id(cls, *, algorithm=None, experiment_name=None, true_lengthscale=None, prior_lengthscale=None,
                       infer_lengthscale=None, total_samples=None, svi_samples=None, data_seed=None, inference_seed=None, extra_param=None):
        experiment_id = f"{algorithm}--{experiment_name}--true-len-{str(true_lengthscale).replace('.', '_')}--prior-len-{str(prior_lengthscale).replace('.', '_')}--inferred-len-{infer_lengthscale}--total-samples-{total_samples}--svi-samples-{svi_samples}--data-seed-{data_seed}--inference-seed-{inference_seed}--extra-param-{extra_param}"
        return experiment_id

    @classmethod
    def load_from_file(cls, *,algorithm=None, experiment_name=None, true_lengthscale=None, prior_lengthscale=None,
                       infer_lengthscale=None, total_samples=None, svi_samples=None, data_seed=None, inference_seed=None, extra_param=None):
        experiment_id = ViExperiment.build_experiment_id(algorithm=algorithm, experiment_name=experiment_name,
                                                         true_lengthscale=true_lengthscale,
                                                         prior_lengthscale=prior_lengthscale,
                                                         infer_lengthscale=infer_lengthscale, total_samples=total_samples,
                                                         svi_samples=svi_samples, data_seed=data_seed,
                                                         inference_seed=inference_seed, extra_param=extra_param)

        filepath_output = ViExperiment.experiment_folder / "output" / "data" / f"output-{experiment_id}.pickle"
        output = pickle.load(open(filepath_output, "rb"))
        return output

    def get_experiment_id(self):
        return ViExperiment.build_experiment_id(algorithm=self.algorithm, experiment_name=self.experiment_name,
                                                true_lengthscale=self.true_lengthscale,
                                                prior_lengthscale=self.prior_lengthscale,
                                                infer_lengthscale=self.infer_lengthscale,
                                                total_samples=self.total_samples, svi_samples=self.svi_samples,
                                                data_seed=self.data_seed, inference_seed=self.inference_seed,
                                                extra_param=self.extra_param)
    def run(self):
        start = time.time()
        self.infer()
        end = time.time()
        self.elapsed_time = end - start
        self.save_output()

    @abstractmethod
    def infer(self):
        pass

class MeanFieldVarBayes(ViExperiment):

    def __init__(self, *, true_lengthscale=None, prior_lengthscale=None, infer_lengthscale=None, total_samples=None,
                 fem=None, experiment_name=None, data_seed=None, inference_seed=None, svi_samples=None, extra_param=None):
        super(MeanFieldVarBayes, self).__init__(true_lengthscale=true_lengthscale, prior_lengthscale=prior_lengthscale,
                                                infer_lengthscale=infer_lengthscale, total_samples=total_samples,
                                                fem=fem, experiment_name=experiment_name, data_seed=data_seed,
                                                inference_seed=inference_seed, svi_samples=svi_samples,
                                                extra_param=extra_param)
        self.algorithm = 'mfvb'

    def infer(self):
        inference_points = self.fem.get_param_coords()
        num_inference_points = inference_points.shape[0]

        opt_elbo = tf.Variable(np.finfo(np.float64).min, dtype=tf.float64, trainable=False)

        t_mean = tf.Variable(1e-1 * np.random.randn(num_inference_points))
        opt_mean = tf.Variable(1e-1 * np.random.randn(num_inference_points), trainable=False)

        t_sigma_flat = tfp.util.TransformedVariable(
            1e0 * tf.random.uniform([num_inference_points], dtype=tf.float64), bijector=tfb.Exp())
        opt_sigma = tf.Variable(1e0 * tf.random.uniform([num_inference_points], dtype=tf.float64), trainable=False)

        prior_loglen = tf.constant(np.log(self.prior_lengthscale), dtype=tf.float64)
        prior_logamp = tf.constant(np.log(1), dtype=tf.float64)

        kappa_prior = tfd.GaussianProcess(kernel=tfk.ExponentiatedQuadratic(length_scale=tf.exp(prior_loglen),
                                                                            amplitude=tf.exp(prior_logamp)),
                                          index_points=inference_points)

        def target_log_prob_fn(t_mean, t_sigma_flat):
            q_k = tfd.MultivariateNormalDiag(loc=t_mean, scale_diag=t_sigma_flat)


            # kld = tfd.kl_divergence(q_k, kappa_prior)
            q_k_samples = q_k.sample(self.svi_samples)

            # log q_kappa
            log_q_kappa = tf.reduce_mean(q_k.log_prob(q_k_samples))

            # log p_kappa
            log_p_kappa = tf.reduce_mean(kappa_prior.log_prob(q_k_samples))
            kld = log_q_kappa - log_p_kappa

            data_lik = tf.reduce_mean(self.tf_loglik(q_k_samples))

            elbo = data_lik - kld
            tf.print(tf.timestamp(), " ---- ELBO: ", elbo, ", Data likelihood:", data_lik)

            def update_opt_values():
                opt_elbo.assign(elbo)
                opt_mean.assign(t_mean)
                opt_sigma.assign(t_sigma_flat)
                return None

            tf.cond(tf.greater(elbo, opt_elbo), lambda: update_opt_values(), lambda: None)
            return elbo

        def trace_func(traceable_quantities):
            return {'loss': traceable_quantities.loss}

        learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.1,
                                                                          decay_steps=1000,
                                                                          decay_rate=0.96)

        vi_convergence_criterion = tfp.optimizer.convergence_criteria.LossNotDecreasing(rtol=1e-14,
                                                                                        window_size=self.window_size,
                                                                                        min_num_steps=150_000)

        opt_trace = tfp.math.minimize(loss_fn=lambda: -target_log_prob_fn(t_mean, t_sigma_flat),
                                      optimizer=tf.optimizers.Adam(learning_rate=learning_rate_fn),
                                      num_steps=self.total_samples, trace_fn=trace_func, return_full_length_trace=False,
                                      convergence_criterion=vi_convergence_criterion)

        cov_mat = tf.linalg.diag(opt_sigma)
        self.loss_trace = opt_trace['loss'].numpy()
        self.mean = opt_mean.numpy()
        self.cov = cov_mat.numpy()


class FullCovVarBayes(ViExperiment):

    def __init__(self, *, true_lengthscale=None, prior_lengthscale=None, infer_lengthscale=None, total_samples=None,
                 fem=None, experiment_name=None, data_seed=None, inference_seed=None, svi_samples=None, extra_param=None):
        super(FullCovVarBayes, self).__init__(true_lengthscale=true_lengthscale, prior_lengthscale=prior_lengthscale,
                                              infer_lengthscale=infer_lengthscale, total_samples=total_samples, fem=fem,
                                              experiment_name=experiment_name, data_seed=data_seed,
                                              inference_seed=inference_seed, svi_samples=svi_samples,
                                              extra_param=extra_param)
        self.algorithm = 'fcvb'
        self.tril_to_unconstrained = tfb.Chain([
            tfb.Invert(tfb.FillTriangular(validate_args=False)),
            # take the log of the diagonals
            tfb.TransformDiagonal(tfb.Log(validate_args=False)),
        ])

    def infer(self, init_mean=None, init_cov_mat=None):
        inference_points = self.fem.get_param_coords()
        num_inference_points = inference_points.shape[0]

        opt_elbo = tf.Variable(np.finfo(np.float64).min, dtype=tf.float64, trainable=False)

        if np.all(init_mean != None):
            t_mean = tf.Variable(init_mean, dtype=tf.float64)
        else:
            t_mean = tf.Variable(1e-1 * np.random.randn(num_inference_points), dtype=tf.float64)

        if np.all(init_cov_mat != None):
            L = np.linalg.cholesky(init_cov_mat)
            print("Using the following Cholesky factor.")
            print(L)
            t_sigma_flat = self.tril_to_unconstrained(L)
            print("Logged and flattened:")
            print(t_sigma_flat.numpy())
        else:
            t_sigma_flat = tf.Variable(1e-2 * np.random.normal(0, 1, int(num_inference_points * (num_inference_points + 1) / 2)), dtype=tf.float64)

        opt_mean = tf.Variable(t_mean, trainable=False)
        opt_sigma = tf.Variable(t_sigma_flat, trainable=False)

        prior_loglen = tf.constant(np.log(self.prior_lengthscale), dtype=tf.float64)
        prior_logamp = tf.constant(np.log(1), dtype=tf.float64)

        kappa_prior = tfd.GaussianProcess(kernel=tfk.ExponentiatedQuadratic(length_scale=tf.exp(prior_loglen),
                                                                            amplitude=tf.exp(prior_logamp)),
                                          index_points=inference_points)

        def target_log_prob_fn(t_mean, t_sigma_flat):
            q_k = tfd.MultivariateNormalTriL(loc=t_mean, scale_tril=self.tril_to_unconstrained.inverse(t_sigma_flat))
            kld = tfd.kl_divergence(q_k, kappa_prior)
            q_k_samples = q_k.sample(self.svi_samples)

            data_lik = tf.reduce_mean(self.tf_loglik(q_k_samples))

            elbo = data_lik - kld
            tf.print(tf.timestamp(), " ---- ELBO: ", elbo, ", Data likelihood:", data_lik)

            def update_opt_values():
                opt_elbo.assign(elbo)
                opt_mean.assign(t_mean)
                opt_sigma.assign(t_sigma_flat)
                return None
            tf.cond(tf.greater(elbo, opt_elbo), lambda: update_opt_values(), lambda: None)
            return elbo

        def trace_func(traceable_quantities):
            return {'loss': traceable_quantities.loss}

        learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.01,
                                                                          decay_steps=5000,
                                                                          decay_rate=0.98)

        vi_convergence_criterion = tfp.optimizer.convergence_criteria.LossNotDecreasing(rtol=1e-14,
                                                                                        window_size=self.window_size,
                                                                                        min_num_steps=500_000)

        opt_trace = tfp.math.minimize(loss_fn=lambda: -target_log_prob_fn(t_mean, t_sigma_flat),
                                      optimizer=tf.optimizers.Adam(learning_rate=learning_rate_fn),
                                      num_steps=self.total_samples, trace_fn=trace_func, return_full_length_trace=False,
                                      convergence_criterion=vi_convergence_criterion)

        vi_fc_cov_l_factor = self.tril_to_unconstrained.inverse(opt_sigma)
        cov_mat = vi_fc_cov_l_factor @ tf.transpose(vi_fc_cov_l_factor)
        self.loss_trace = opt_trace['loss'].numpy()
        self.mean = opt_mean.numpy()
        self.cov = cov_mat.numpy()


class PrecisionVarBayes(ViExperiment):

    def __init__(self, *, true_lengthscale=None, prior_lengthscale=None, infer_lengthscale=None, total_samples=None,
                 fem=None, experiment_name=None, data_seed=None, inference_seed=None, svi_samples=None, extra_param=None):
        super(PrecisionVarBayes, self).__init__(true_lengthscale=true_lengthscale, prior_lengthscale=prior_lengthscale,
                                                infer_lengthscale=infer_lengthscale, total_samples=total_samples,
                                                fem=fem, experiment_name=experiment_name, data_seed=data_seed,
                                                inference_seed=inference_seed, svi_samples=svi_samples,
                                                extra_param=extra_param)
        
        num_hops = 2
        adjacency_fname = ViExperiment.experiment_folder / f'bar_{num_hops}ring.dat'

        precision_indices = []
        adj_file = open(adjacency_fname, 'r')
        for i, line in enumerate(adj_file):
            adjacent_elems = line.split(', ')
            for j in adjacent_elems:
                precision_indices.append([i, int(j)])
        adj_file.close()
        precision_indices = np.asarray(precision_indices)

        # reorder the graph to reduce bandwidth
        Q_original = sparse.csr_matrix(
            (np.ones(precision_indices.shape[0]), (precision_indices[:, 0], precision_indices[:, 1])))
        self.new_to_original_indices = sparse.csgraph.reverse_cuthill_mckee(Q_original, symmetric_mode=True)
        Q_reordered = Q_original[self.new_to_original_indices, :][:, self.new_to_original_indices]

        original_coords = fem.get_param_coords()
        map_from_new_to_orig = np.stack((np.arange(original_coords.shape[0]), self.new_to_original_indices), axis=1)
        self.original_to_new_indices = map_from_new_to_orig[map_from_new_to_orig[:, 1].argsort()][:, 0]

        # Create a template for the parametrisation and extract non-zero indices
        forced_bandwidth = 0 if extra_param is None else extra_param
        self.param_bandwidth = max(PrecisionVarBayes.get_csr_mat_bandwidth(Q_reordered), forced_bandwidth)
        precision_L_template = tf.linalg.band_part(
            tf.ones((fem.get_param_coords().shape[0], fem.get_param_coords().shape[0])), self.param_bandwidth, 0)
        precision_L_template_sparse = tf.sparse.reorder(tf.sparse.from_dense(precision_L_template))


        self.precision_L_shape = precision_L_template_sparse.dense_shape
        self.precision_L_indices = precision_L_template_sparse.indices

        self.algorithm = 'pmvb'

    @staticmethod
    def get_csr_mat_bandwidth(csr_mat):
        coo = csr_mat.tocoo()
        return np.max(np.abs(coo.row - coo.col))

    def infer(self):

        def log_of_data(kappas_tf):
            lls_and_grads = []
            for kappa_tf in kappas_tf:
                kappa_np = kappa_tf.numpy()
                reordered_params = kappa_np[self.original_to_new_indices]  # from new to old
                self.fem.set_params(reordered_params)
                loglik = self.fem.get_log_lik()
                loglik_grad = np.multiply(self.fem.get_log_lik_grad(), np.exp(reordered_params))
                loglik_grad_reordered = loglik_grad[self.new_to_original_indices]
                lls_and_grads.append(np.concatenate([[loglik], loglik_grad_reordered], axis=0))
            lls_and_grads = np.stack(lls_and_grads, axis=0)
            return lls_and_grads

        @tf.custom_gradient
        def tf_loglik(kappa_tf):
            ll_kappa_grad = tf.py_function(log_of_data, [kappa_tf], tf.float64)
            ll = ll_kappa_grad[:, 0]
            kappa_grad = ll_kappa_grad[:, 1:]

            def grad(dy):
                return tf.expand_dims(dy, axis=1) * kappa_grad
            return ll, grad
        
        @tf.function
        def kld(mean_q, precL_q, mean_p, covL_p):
            def squared_frobenius_norm(x):
                return tf.reduce_sum(tf.square(x), axis=[-2, -1])

            covL_p_op = tf.linalg.LinearOperatorLowerTriangular(covL_p)
            precL_q_adjoint_inverse = tf.linalg.LinearOperatorAdjoint(tf.linalg.LinearOperatorLowerTriangular(precL_q).inverse()).to_dense()

            b_inv_a = covL_p_op.solve(precL_q_adjoint_inverse)
            precL_q_log_abs_det = 2 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(precision_L)))
            d = tf.cast(mean_q.shape[0], tf.double)

            kld = 0.5 * (2 * covL_p_op.log_abs_determinant() + 2 * precL_q_log_abs_det - d + squared_frobenius_norm(b_inv_a) + squared_frobenius_norm(covL_p_op.solve((mean_p - mean_q)[..., tf.newaxis])))
            return kld

        inference_points = self.fem.get_param_coords()[self.new_to_original_indices]
        num_inference_points = inference_points.shape[0]
        
        # init procedure
        prior_kernel = tfk.ExponentiatedQuadratic(length_scale=0.01, amplitude=1.0)
        cov_mat = prior_kernel.matrix(inference_points, inference_points).numpy() + np.diag(1e-5*np.ones(len(inference_points)))
        prec_L_full = np.linalg.cholesky(np.linalg.inv(cov_mat))
        init_values = np.zeros(len(self.precision_L_indices))
        for i, mat_xy in enumerate(self.precision_L_indices):
            if mat_xy[0] == mat_xy[1]:
                init_values[i] = np.log(prec_L_full[mat_xy[0], mat_xy[1]])
            else:
                init_values[i] = prec_L_full[mat_xy[0], mat_xy[1]]
        
        opt_elbo = tf.Variable(np.finfo(np.float64).min, dtype=tf.float64, trainable=False)

        t_mean = tf.Variable(1e-4 * np.random.randn(num_inference_points))
        opt_mean = tf.Variable(1e-3 * np.random.randn(num_inference_points), trainable=False)

        t_precision_L_flat = tf.Variable(init_values, dtype=tf.double)
        opt_precision_L_flat = tf.Variable(t_precision_L_flat, dtype=tf.double, trainable=False)

        prior_loglen = tf.constant(np.log(self.prior_lengthscale), dtype=tf.float64)
        prior_logamp = tf.constant(np.log(1), dtype=tf.float64)

        # tfd_std_normal = tfd.Normal(loc=tf.zeros(num_inference_points, dtype=tf.double), scale=tf.ones(num_inference_points, dtype=tf.double))
        kappa_prior = tfd.GaussianProcess(kernel=tfk.ExponentiatedQuadratic(length_scale=tf.exp(prior_loglen),
                                                                            amplitude=tf.exp(prior_logamp)),
                                          index_points=inference_points)

        def target_log_prob_fn(t_mean, t_precision_L_flat):
            scatter = tf.scatter_nd(self.precision_L_indices, t_precision_L_flat, self.precision_L_shape)
            precision_L = tfb.TransformDiagonal(tfb.Exp()).forward(scatter)

                ##### MORE MANUAL WAY
                # precision_L_operator = tf.linalg.LinearOperatorLowerTriangular(tf.stack([precision_L] * self.svi_samples, axis=0))
                # precision_L_T_operator = tf.linalg.LinearOperatorAdjoint(precision_L_operator)
                #
                # epsilon_samples = tfd_std_normal.sample(svi_samples)
                # q_k_centred_samples = precision_L_T_operator.solvevec(epsilon_samples)
                # q_k_samples = tf.expand_dims(t_mean, axis=0) + q_k_centred_samples
                #
                # ### OPTION 1 MONTE CARLO WAY
                # # log q_kappa
                # L_diags = tf.linalg.diag_part(precision_L)
                # epsilon_dot_product = tf.linalg.matmul(tf.expand_dims(epsilon_samples, axis=1), tf.expand_dims(epsilon_samples, axis=2))
                # log_q_kappa = svi_samples * tf.reduce_sum(tf.math.log(L_diags)) - 0.5 * tf.reduce_sum(epsilon_dot_product)

                # # log p_kappa
                # log_p_kappa - tf.reduce_mean(kappa_prior.log_prob(q_k_samples))
                # kl_q_p = log_p_kappa + log_q_kappa

                # ### OPTION 2: CLOSED-FORM WAY
                # kl_q_p = kld(t_mean, precision_L, kappa_prior.mean(),
                #              kappa_prior.get_marginal_distribution().scale.to_dense())

            covariance_L = tfb.CholeskyToInvCholesky().forward(precision_L)
            q_k = tfd.MultivariateNormalTriL(loc=t_mean, scale_tril=covariance_L)
            kl_q_p = tfp.distributions.kl_divergence(q_k, kappa_prior)

            # data likelihood
            q_k_samples = q_k.sample(self.svi_samples)
            data_lik = tf.reduce_mean(tf_loglik(q_k_samples))
            elbo = data_lik - kl_q_p

            tf.print(tf.timestamp(), " ---- ELBO: ", elbo, ", Data likelihood:", data_lik)

            def update_opt_values():
                opt_elbo.assign(elbo)
                opt_mean.assign(t_mean)
                opt_precision_L_flat.assign(t_precision_L_flat)
                return None

            tf.cond(tf.greater(elbo, opt_elbo), lambda: update_opt_values(), lambda: None)
            return elbo

        def trace_func(traceable_quantities):
            return {'loss': traceable_quantities.loss}

        learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.01,
                                                                          decay_steps=5000,
                                                                          decay_rate=0.98)

        vi_convergence_criterion = tfp.optimizer.convergence_criteria.LossNotDecreasing(rtol=1e-14,
                                                                                        window_size=self.window_size,
                                                                                        min_num_steps=500_000)

        opt_trace = tfp.math.minimize(loss_fn=lambda: -target_log_prob_fn(t_mean, t_precision_L_flat),
                                      optimizer=tf.optimizers.Adam(learning_rate=learning_rate_fn),
                                      num_steps=self.total_samples, trace_fn=trace_func, return_full_length_trace=False,
                                      convergence_criterion=vi_convergence_criterion)

        scatter = tf.scatter_nd(self.precision_L_indices, opt_precision_L_flat, self.precision_L_shape)
        precision_L = tfb.TransformDiagonal(tfb.Exp()).forward(scatter)
        covariance_L = tfb.CholeskyToInvCholesky().forward(precision_L)
        cov_mat = (covariance_L @ tf.transpose(covariance_L)).numpy()[self.original_to_new_indices, :][:, self.original_to_new_indices]

        self.loss_trace = opt_trace['loss'].numpy()
        self.mean = opt_mean.numpy()[self.original_to_new_indices]
        self.cov = cov_mat


class FullCovInitMfVarBayes(ViExperiment):

    def __init__(self, *, true_lengthscale=None, prior_lengthscale=None, infer_lengthscale=None, total_samples=None,
                 fem=None, experiment_name=None, data_seed=None, inference_seed=None, svi_samples=None, extra_param=None):
        super(FullCovInitMfVarBayes, self).__init__(true_lengthscale=true_lengthscale, prior_lengthscale=prior_lengthscale,
                                              infer_lengthscale=infer_lengthscale, total_samples=total_samples, fem=fem,
                                              experiment_name=experiment_name, data_seed=data_seed,
                                              inference_seed=inference_seed, svi_samples=svi_samples,
                                              extra_param=extra_param)
        self.algorithm = 'fcvb_init'

        self.mfvb_runner = MeanFieldVarBayes(true_lengthscale=true_lengthscale, prior_lengthscale=prior_lengthscale,
                                             infer_lengthscale=infer_lengthscale, total_samples=total_samples, fem=fem,
                                             experiment_name=experiment_name, data_seed=data_seed, inference_seed=inference_seed,
                                             svi_samples=svi_samples, extra_param=extra_param)

        self.fcvb_runner = FullCovVarBayes(true_lengthscale=true_lengthscale, prior_lengthscale=prior_lengthscale,
                                           infer_lengthscale=infer_lengthscale, total_samples=total_samples, fem=fem,
                                           experiment_name=experiment_name, data_seed=data_seed, inference_seed=inference_seed,
                                           svi_samples=svi_samples, extra_param=extra_param)

    def infer(self):
        self.mfvb_runner.infer()
        mfvb_mean = self.mfvb_runner.mean

        self.fcvb_runner.infer(init_mean=mfvb_mean)

        self.loss_trace = self.fcvb_runner.loss_trace
        self.mean = self.fcvb_runner.mean
        self.cov = self.fcvb_runner.cov

@click.command()
@click.option('--experiment_name', type=str, default="POISSON-1D-2021-06-15")
@click.option('--algorithm', type=click.Choice(['mfvb', 'fcvb', 'pmvb', 'fcvb_init'],
                                               case_sensitive=False), default='pmvb')
@click.option('--true_lengthscale', type=np.float64, default=0.2)
@click.option('--prior_lengthscale', type=np.float64, default=0.2)
@click.option('--infer_lengthscale', is_flag=True)
@click.option('--total_iters', type=int, default=10_000)
@click.option('--svi_samples', type=int, default=3)
@click.option('--data_seed', type=int, default=42)
@click.option('--inference_seed', type=int, default=42)
@click.option('--extra_param', type=int, default=None)
@click.option('--verbose', is_flag=True)
def main(experiment_name, algorithm, true_lengthscale, prior_lengthscale, infer_lengthscale, total_iters, svi_samples,
         data_seed, inference_seed, extra_param, verbose):

    fem = fem_cpp_driver.FemSolverInterface()
    fem.set_data_gen_seed(data_seed)

    if algorithm == 'fcvb':
        ViRunnerClass = FullCovVarBayes
    elif algorithm == 'pmvb':
        ViRunnerClass = PrecisionVarBayes
    elif algorithm == 'fcvb_init':
        ViRunnerClass = FullCovInitMfVarBayes
    else:
        ViRunnerClass = MeanFieldVarBayes
    vi_runner = ViRunnerClass(true_lengthscale=true_lengthscale, prior_lengthscale=prior_lengthscale,
                              infer_lengthscale=infer_lengthscale, total_samples=total_iters, fem=fem,
                              experiment_name=experiment_name, data_seed=data_seed, inference_seed=inference_seed,
                              svi_samples=svi_samples, extra_param=extra_param)

    log_fmt = '[%(levelname)s] [%(asctime)s] [%(name)s] %(message)s'
    datefmt = '%H:%M:%S'
    if verbose:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=log_fmt)
    else:
        log_folder = ViExperiment.experiment_folder / 'logs'
        os.makedirs(log_folder, exist_ok=True)
        logging.basicConfig(filename=log_folder / f"log-{vi_runner.get_experiment_id()}.log",
                            filemode='a',
                            format=log_fmt,
                            datefmt=datefmt,
                            level=logging.DEBUG)
    vi_runner.run()


if __name__ == "__main__":
    main()
