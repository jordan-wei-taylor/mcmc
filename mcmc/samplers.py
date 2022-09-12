from   mcmc.utils       import check_verbose, Verbose, softmax
from   collections import defaultdict
from   scipy       import stats

import numpy       as np


class MCMCSampler():

    def __init__(self, log_pstar, covariance, data, **kwargs):
        self.log_pstar  = log_pstar
        self.covariance = covariance
        self.data       = data
        self.kwargs     = kwargs

    def transition(self, theta):
        return stats.multivariate_normal(theta, self.covariance).rvs()

    def get_samples(self, burn_period = 0.2):
        if isinstance(burn_period, float):
            burn_period = int(burn_period * len(self.samples) + 1)
        return self.samples[burn_period:]
        

class MetropolisHastingsSampler(MCMCSampler):

    def __init__(self, log_pstar, covariance, data, **kwargs):
        super().__init__(log_pstar, covariance, data, **kwargs)

    def fit(self, n_samples, theta0, verbose = 'auto', random_state = None):

        if random_state is not None:
            np.random.seed(random_state)

        N                  = n_samples + 1 # add 1 to include theta0
        m                  = len(theta0)   # no. of parameters
        
        assert n_samples > 0
        assert isinstance(theta0, np.ndarray) and (theta0.ndim == 1)

        verbose            = check_verbose(N, verbose)

        # samples
        self.samples       = np.empty((N, m))
        self.samples[0]    = theta0

        # record of all log_pstar evaluations
        self.log_pstars    = np.empty(N)
        self.log_pstars[0] = self.log_pstar(theta0, self.data, **self.kwargs)

        # acceptance indicator for all new samples
        self.acceptance    = np.zeros(N - 1, dtype = bool)

        if verbose:
            message = Verbose(N, verbose)
            message.print(0, self.log_pstars[0])

        for i in range(1, N):

            # sample a new theta
            theta = self.transition(self.samples[i - 1])

            # compute log pstar of new theta
            logp  = self.log_pstar(theta, self.data, **self.kwargs)

            # accept with p_new / p_old probability
            if np.log(np.random.uniform()) < (logp - self.log_pstars[i - 1]):
                self.samples[i]        = theta
                self.log_pstars[i]     = logp
                self.acceptance[i - 1] = True

            # reject and add the previous sample
            else:
                self.samples[i]        = self.samples[i - 1]
                self.log_pstars[i]     = self.log_pstars[i - 1]
            
            if verbose:
                message.print(i, self.log_pstars[i], '' if i % verbose else '\n')

        return self


class GibbsSampler(MCMCSampler):

    def __init__(self, log_pstar, deviation, data, **kwargs):
        super().__init__(log_pstar, deviation, data, **kwargs)

        self.__dict__['deviation'] = self.__dict__.pop('covariance')

    def transition(self, value, j):
        return stats.norm(value, self.deviation[j]).rvs()

    def fit(self, n_samples, theta0, verbose = 'auto', random_state = None):

        if random_state is not None:
            np.random.seed(random_state)

        N                  = n_samples + 1 # add 1 to include theta0
        m                  = len(theta0)   # no. of parameters

        assert n_samples > 0
        assert isinstance(theta0, np.ndarray) and (theta0.ndim == 1)

        verbose = check_verbose(N, verbose)

        # samples
        self.samples       = np.empty((N, m))
        self.samples[0]    = theta0

        # record of all log_pstar evaluations
        self.log_pstars    = np.empty(N)
        self.log_pstars[0] = self.log_pstar(theta0, self.data, **self.kwargs)

        # acceptance rate indicator for all new samples (per parameter)
        self.acceptance    = np.ones((N - 1, m), dtype = bool)

        if verbose:
            message = Verbose(N, verbose)
            message.print(0, self.log_pstars[0])

        for i in range(1, N):

            # theta and logp baseline
            theta_baseline = self.samples[i - 1]
            logp_baseline  = self.log_pstars[i - 1]

            # loop through each parameter in theta
            for j in range(m):
                
                # copy most recent theta baseline
                theta    = theta_baseline.copy()

                # sample the j-th element
                theta[j] = self.transition(theta[j], j)
                
                # compute log_pstar for theta
                logp     = self.log_pstar(theta, self.data, **self.kwargs)

                # accept with p_new / p_old probability
                if np.log(np.random.uniform()) < (logp - logp_baseline):
                    theta_baseline = theta
                    logp_baseline  = logp
                
                # reject the new sample
                else:
                    self.acceptance[i - 1,j] = False
                
            # append new sample and log_pstar values
            self.samples[i]    = theta_baseline
            self.log_pstars[i] = logp_baseline
            
            if verbose:
                message.print(i, self.log_pstars[i], '' if i % verbose else '\n')

        return self


class AdaptiveGibbsSampler(MCMCSampler):

    def __init__(self, log_pstar, deviation, data, rate = 0.8, **kwargs):

        assert isinstance(rate, float) and 0 < rate < 1

        super().__init__(log_pstar, deviation, data, **kwargs)

        self.__dict__['deviation'] = self.__dict__.pop('covariance')
        self.rate = rate

    def transition(self, value, j):
        return stats.norm(value, self.deviation[j]).rvs()

    def fit(self, n_samples, theta0, verbose = 'auto', random_state = None):

        if random_state is not None:
            np.random.seed(random_state)

        N                  = n_samples + 1 # add 1 to include theta0
        m                  = len(theta0)   # no. of parameters

        assert n_samples > 0
        assert isinstance(theta0, np.ndarray) and (theta0.ndim == 1)

        # samples
        self.samples       = np.empty((N, m))
        self.samples[0]    = theta0

        # record of all log_pstar evaluations
        self.log_pstars    = np.empty(N)
        self.log_pstars[0] = self.log_pstar(theta0, self.data, **self.kwargs)

        # acceptance rate indicator for all new samples
        self.acceptance    = {}

        # log_alpha for the dirichlet prior
        log_alpha          = np.zeros(m)

        counter            = np.zeros(m)

        if verbose:
            verbose = check_verbose(N, verbose)
            message = Verbose(N, verbose)
            message.print(0, self.log_pstars[0])

        for i in range(1, N):
            
            # theta and logp baseline
            theta_baseline = self.samples[i - 1]
            logp_baseline  = self.log_pstars[i - 1]

            # initialise the acceptance rate for the i-th sample to be a dictionary with list values
            self.acceptance[i] = defaultdict(list)

            for _ in range(m):
                
                # compute alpha
                alpha    = softmax(log_alpha) + 1e-8             # numerical stability as alpha needs to be greater than 0

                # sample the j-th parameter to update
                j        = stats.dirichlet(alpha).rvs().argmax() # draws m numbers but select the one with the highest value

                counter += 1
                counter[j] = 0

                # copy most recent theta baseline
                theta    = theta_baseline.copy()

                # sample the j-th element
                theta[j] = self.transition(theta[j], j)
                
                # compute log_pstar for theta
                logp     = self.log_pstar(theta, self.data, **self.kwargs)

                delta          = logp - logp_baseline

                # accept with probability p_new / p_old
                if delta or np.log(np.random.uniform()) < delta:
                    
                    # increment the j-th element of log_alpha by the improvement in log_pstar value
                    log_alpha[j]  += delta

                    theta_baseline = theta
                    logp_baseline  = logp
                    self.acceptance[i][j].append(True)
                    
                # reject the new sample
                else:
                    self.acceptance[i][j].append(False)

                if delta > 0:
                    log_alpha += delta ** counter / counter.sum()

                # pull all values towards 0 (this prevents exploding values)
                # encourages indices that have not been picked
                log_alpha *= self.rate
                
            self.samples[i]    = theta_baseline
            self.log_pstars[i] = logp_baseline
            
            if verbose:
                message.print(i, self.log_pstars[i], '' if i % verbose else '\n')
                
        return self