import numpy as np
import logging


class ApproximateZINBFit:
    """Approximate zero-inflated negative binomial (ZINB) distribution fitting based on
    the first two moments and number of zero counts.
    """
    
    EPS = 1e-8
    
    def __init__(self,
                 min_nb_phi = 1e-3,
                 max_nb_phi = 1.0,
                 max_zinb_p_zero = 0.999,
                 min_zinb_p_zero = 0.001,
                 max_iters = 10_000,
                 lr = 0.25,
                 p_zero_l1_reg = 0.05,
                 outlier_stringency = np.inf,
                 atol = 1e-2,
                 verbose = False):
        self.min_nb_phi = min_nb_phi
        self.max_nb_phi = max_nb_phi
        self.max_zinb_p_zero = max_zinb_p_zero
        self.min_zinb_p_zero = min_zinb_p_zero
        self.max_iters = max_iters
        self.lr = lr
        self.p_zero_l1_reg = p_zero_l1_reg
        self.outlier_stringency = outlier_stringency
        self.atol = atol
        self.verbose = verbose
        self._logger = logging.getLogger()
        
    def __call__(self, data):
        assert isinstance(data, np.ndarray)
        assert len(data) > 0
        assert data.ndim == 1
        
        # remove outliers
        data_nnz = data[data > 0]
        med_nnz = np.median(data_nnz)
        scale_nnz = np.mean(np.abs(data_nnz - med_nnz))
        outlier_threshold = med_nnz + self.outlier_stringency * scale_nnz
        data = data[data <= outlier_threshold]
        num_data_points = len(data)
        
        # ransac estimation of mean and variance
        empirical_mean = np.mean(data)
        empirical_var = np.var(data)
        empirical_p_zero = (data == 0).sum() / num_data_points
        empirical_second_moment = empirical_var + empirical_mean ** 2
        
        # initial guess
        p_zero = np.clip(empirical_p_zero, self.min_zinb_p_zero, self.max_zinb_p_zero)
        nb_mu = empirical_mean
        nb_phi =  np.clip((empirical_var - empirical_mean) / (empirical_mean ** 2 + self.EPS),
                          self.min_nb_phi, self.max_nb_phi)
        
        if self.verbose:
            self._logger.warning(f"Initial guess: (p_zero: {p_zero:.3f}, mu: {nb_mu:.3f}, phi: {nb_phi:.3f})")
    
        # relaxation iterations
        converged = False
        for i in range(self.max_iters):
            new_nb_mu = empirical_mean / (1 - p_zero)
            new_nb_var = empirical_second_moment / (1 - p_zero)
            new_nb_phi = np.clip((new_nb_var - nb_mu) / (nb_mu ** 2 + self.EPS) - 1,
                                 self.min_nb_phi, self.max_nb_phi)

            nb_alpha = 1 / nb_phi
            nb_p_zero = np.power((nb_alpha / (nb_mu + nb_alpha)), nb_alpha)
            new_p_zero = np.clip(empirical_p_zero - (1 - p_zero) * nb_p_zero + self.p_zero_l1_reg,
                                 self.min_zinb_p_zero, self.max_zinb_p_zero)

            nb_mu_atol_satisfied = np.abs(new_nb_mu - nb_mu) < 0.1 * self.atol
            nb_phi_atol_satisfied = np.abs(new_nb_phi - nb_phi) < 0.1 * self.atol
            p_zero_atol_satisfied = np.abs(new_p_zero - p_zero) < 0.1 * self.atol

            nb_mu = nb_mu + self.lr * (new_nb_mu - nb_mu)
            nb_phi = nb_phi + self.lr * (new_nb_phi - nb_phi)
            p_zero = p_zero + self.lr * (new_p_zero - p_zero)

            if self.verbose:
                self._logger.warning(f"Iteration: {i}, p_zero: {p_zero:.3f}, mu: {nb_mu:.3f}, phi: {nb_phi:.3f}")
            
            if all((nb_mu_atol_satisfied, nb_phi_atol_satisfied, p_zero_atol_satisfied)):
                converged = True
                break
                
        if not converged:
            self._logger.warning("The ZINB fit was not successful -- please increase max_iter and/or " \
                                 + "decrease the relaxation parameter.")

        return {'mu': nb_mu,
                'phi': nb_phi,
                'p_zero': p_zero,
                'iters': i,
                'converged': converged}


def int_ndarray_mode(arr: np.ndarray, axis: int):
    return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=axis, arr=arr)
