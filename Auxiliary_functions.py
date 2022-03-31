"""
Auxiliary functions for CVI
See https://github.com/lteacy/vdpgm_stack/blob/master/vdpgm_srv/matlab/vdpgm.m
for the original matlab implementations.
"""
import numpy as np
from scipy.special import gammaln, digamma


def E_log_p_of_z_given_other_z(alpha, phi):
    # Returns E[log p(z_n | z_{-n})]_q_{-n} as a TxN matrix

    T, N = phi.shape

    # Compute E[Nz_{-n}], a TxN matrix
    E_Nt = np.sum(phi, axis=1)
    E_Nt_minus_n = np.tile(np.array([E_Nt]).transpose(), (1, N)) - phi    
    # Compute V[Nz_{-n}], a TxN matrix
    aux = phi*(1-phi)
    V_Nt = np.sum(aux, axis=1)
    V_Nt_minus_n = np.tile(np.array([V_Nt]).transpose(), (1, N)) - aux 
    
    # Auxiliary cumulative sums of E[Nz_{-n}]
    E_Nt_minus_n_cumsum_geq = np.flipud(np.cumsum(np.flipud(E_Nt_minus_n), 0))
    E_Nt_minus_n_cumsum = E_Nt_minus_n_cumsum_geq - E_Nt_minus_n
    # Auxiliary cumulative sums of V[Nz_{-n}]
    phi_cumsum_geq = np.flipud(np.cumsum(np.flipud(phi), 0))
    aux = phi_cumsum_geq*(1 - phi_cumsum_geq)
    V_Nt_cumsum_geq = np.tile(np.array([np.sum(aux, axis=1)]).transpose(), (1, N)) - aux
    
    phi_cumsum = phi_cumsum_geq - phi
    aux = phi_cumsum*(1 - phi_cumsum)
    V_Nt_cumsum = np.tile(np.array([np.sum(aux, axis=1)]).transpose(), (1, N)) - aux
    
    
    first_term = np.log(1 + E_Nt_minus_n) - .5*V_Nt_minus_n/(1 + E_Nt_minus_n)**2\
                - np.log(1 + alpha + E_Nt_minus_n_cumsum_geq)\
                + .5*V_Nt_cumsum_geq/(1 + alpha + E_Nt_minus_n_cumsum_geq)**2
    first_term[-1] = 0

    aux = np.log(alpha + E_Nt_minus_n_cumsum) - np.log(1 + alpha + E_Nt_minus_n_cumsum_geq)\
        - .5*V_Nt_cumsum/(alpha + E_Nt_minus_n_cumsum)**2\
        + .5*V_Nt_cumsum_geq/(1 + alpha + E_Nt_minus_n_cumsum_geq)**2
        
    second_term = np.cumsum(aux, 0) - aux

    return first_term + second_term


def ELBO_term_for_pz(alpha, phi):
    
    # Auxiliary sums and cumulative sums of E[N_z]
    E_Nt = np.sum(phi, axis=1)
    E_Nt_geq_i = np.cumsum(E_Nt, 0)
    E_Nt_g_i = E_Nt_geq_i - E_Nt

    # Auxiliary sums and cumulative sums of V[N_z]
    V_Nt = np.sum(phi*(1-phi), axis=1)
    phi_geq_i = np.cumsum(phi, 0)
    V_Nt_geq_i = np.sum(phi_geq_i*(1-phi_geq_i), axis=1)
    phi_g_i = phi_geq_i - phi
    V_Nt_g_i = np.sum(phi_g_i*(1-phi_g_i), axis=1)
    
    aux = gammaln(1 + E_Nt) -.5*digamma(1 + E_Nt)*V_Nt + gammaln(alpha + E_Nt_g_i)\
        - .5*digamma(1 + E_Nt_g_i)*V_Nt_g_i - gammaln(1 + alpha + E_Nt_geq_i)\
        + .5*digamma(1 + alpha + E_Nt_geq_i)*V_Nt_geq_i

    E_log_p_of_z = np.sum(aux[:-1])
    
    return E_log_p_of_z - np.sum(E_log_p_of_z_given_other_z(alpha, phi)*phi)




def Update_MHPP_parameters(a_tao, b_tao, a_0, b_0, mean_mu, cov_mu, prior_mean, prior_cov, delta):

    T = a_tao.shape[0]
    M = mean_mu.shape[1]
    # Compute gamma parameters
    # KL(q|p_0)
    aux_1 = (a_tao - 1.)*digamma(a_tao) - gammaln(a_tao) + np.log(b_tao) - a_tao + a_tao/b_tao
    # KL(q|q_{t-1})
    aux_2 = (a_tao - a_0)*digamma(a_tao) - gammaln(a_tao) + a_0*(np.log(b_tao) - np.log(b_0))\
          + gammaln(a_0) - a_tao + b_0*a_tao/b_tao
    gamma = aux_1 - aux_2 + delta

    for t in range(T):
        aux_inv = np.linalg.pinv(prior_cov[t])
        aux_mean = prior_mean[t] - mean_mu[t]
        sign_q, logdet_q = np.linalg.slogdet(cov_mu[t])       
        sign_p, logdet_p = np.linalg.slogdet(prior_cov[t])
        # Compute mean parameters
        # KL(q|p_0)        
        aux_1[t] = .5*(np.sum(cov_mu[t]) - M + mean_mu[t].dot(mean_mu[t]) - sign_q*logdet_q)
        # KL(q|q_{t-1})
        aux_2[t] = .5*(aux_mean.T.dot(aux_inv.dot(aux_mean)) + np.trace(np.matmul(aux_inv, cov_mu[t])) - M\
                 + sign_p*logdet_p - sign_q*logdet_q)
           
    mean = aux_1 - aux_2 + delta

    omegas = np.zeros((2,T))
    omegas[0] = gamma
    omegas[1] = mean
    return omegas

def Update_HPP_parameters(a_tao, b_tao, a_0, b_0, mean_mu, cov_mu, prior_mean, prior_cov, delta):

    T = a_tao.shape[0]
    M = mean_mu.shape[1]
    # Compute gamma parameters
    # KL(q|p_0)
    aux_1 = (a_tao - 1.)*digamma(a_tao) - gammaln(a_tao) + np.log(b_tao) - a_tao + a_tao/b_tao
    # KL(q|q_{t-1})
    aux_2 = (a_tao - a_0)*digamma(a_tao) - gammaln(a_tao) + a_0*(np.log(b_tao) - np.log(b_0))\
          + gammaln(a_0) - a_tao + b_0*a_tao/b_tao
    gamma = aux_1 - aux_2

    for t in range(T):
        aux_inv = np.linalg.pinv(prior_cov[t])
        aux_mean = prior_mean[t] - mean_mu[t]
        sign_q, logdet_q = np.linalg.slogdet(cov_mu[t])       
        sign_p, logdet_p = np.linalg.slogdet(prior_cov[t])
        # Compute mean parameters
        # KL(q|p_0)        
        aux_1[t] = .5*(np.sum(cov_mu[t]) - M + mean_mu[t].dot(mean_mu[t]) - sign_q*logdet_q)
        # KL(q|q_{t-1})
        aux_2[t] = .5*(aux_mean.T.dot(aux_inv.dot(aux_mean)) + np.trace(np.matmul(aux_inv, cov_mu[t])) - M\
                 + sign_p*logdet_p - sign_q*logdet_q)
           
    mean = aux_1 - aux_2
    
    omega = np.tile(np.sum(mean + gamma) + delta, (2, T))
    return omega



def lowerbound(X, T, N, M, alpha, prior_mean, prior_cov, a_0, b_0,\
               mean_mu, cov_mu, a_tao, b_tao, phi, likx, N_t, omega, delta, batch, E_q_rho):
    """
    Computes the ELBO of the log-likelihood function 
    """
    elbo = 0

    # tau terms
    # KL(q|p_0)
    aux_1 = (a_tao-1.)*digamma(a_tao) - gammaln(a_tao) + np.log(b_tao) - a_tao + a_tao/b_tao
    # KL(q|q_{t-1})
    aux_2 = (a_tao-a_0)*digamma(a_tao) - gammaln(a_tao) + a_0*(np.log(b_tao) - np.log(b_0))\
          + gammaln(a_0) - a_tao + b_0*a_tao/b_tao

    elbo -= np.sum(E_q_rho[0]*aux_2 + (1 - E_q_rho[0])*aux_1)

    # mu terms
    for t in range(T):
        aux_inv = np.linalg.pinv(prior_cov[t])
        aux_mean = prior_mean[t] - mean_mu[t]
        sign_q, logdet_q = np.linalg.slogdet(cov_mu[t])       
        sign_p, logdet_p = np.linalg.slogdet(prior_cov[t])
        # Compute mean parameters
        # KL(q|p_0)        
        aux_1[t] = .5*(np.sum(cov_mu[t]) - M + mean_mu[t].dot(mean_mu[t]) - sign_q*logdet_q)
        # KL(q|q_{t-1})
        aux_2[t] = .5*(aux_mean.T.dot(aux_inv.dot(aux_mean)) + np.trace(np.matmul(aux_inv, cov_mu[t])) - M\
                  + sign_p*logdet_p - sign_q*logdet_q)

    elbo -= np.sum(E_q_rho[1]*aux_2 + (1 - E_q_rho[1])*aux_1)

    #z
    elbo += ELBO_term_for_pz(alpha, phi)
    if batch > 0:
        # rho
        # KL[q(rho)||p(rho)]
        aux_1 = delta/omega*(omega/(np.exp(omega) - 1) - 1) + 1 - omega/(np.exp(omega) - 1)
        aux_2 = np.log((1. - np.exp(-1.*delta))/(delta/omega - delta/omega*np.exp(-1.*omega)))

        elbo -= np.sum(aux_1 + aux_2)

    #x
    # Eq[log p(X)
    lpx = np.sum(phi*likx)
    elbo += lpx
    
    return elbo


def compute_test_perplexity(y, phi, mean_mu, cov_mu, a_tao, b_tao, alpha):
    N, M = y.shape
    T = phi.shape[0]
    
    suff = np.zeros((T, N))
    for t in range(T):
        suff[t] = np.sum((y - mean_mu[t])**2, axis=1) + np.trace(cov_mu[t])
    liky = np.zeros(suff.shape)
    for t in range(T):
        liky[t, :] = .5*M*(digamma(a_tao[t]) - np.log(b_tao[t]) - np.log(2*np.pi))
        tao_t = a_tao[t]/b_tao[t]
        liky[t, :] -= .5*tao_t*suff[t]

    aux = E_log_p_of_z_given_other_z(alpha, np.c_[phi, np.zeros(T)])[:, -1]
    likz = np.tile(aux, (N, 1)).T
    s = likz + liky
    # Compute phi_y
    phi_y = log_normalize(s, axis=0)

    # Return log(perplexity)
    lik = np.sum(phi_y*liky) - np.sum(np.log(phi_y)*phi_y)
    return -lik/N


def log_normalize(v, axis=0):
    """
    Normalized probabilities from unnormalized log-probabilites
    """
    v = np.rollaxis(v, axis)
    v = v.copy()
    v -= v.max(axis=0)
    out = logsumexp(v)
    v = np.exp(v - out)
    v += np.finfo(np.float32).eps
    v /= np.sum(v, axis=0)
    return np.swapaxes(v, 0, axis)

def logsumexp(arr, axis=0):  
    """
    Computes the sum of arr assuming arr is in the log domain.
    Returns log(sum(exp(arr))) while minimizing the possibility of
    over/underflow
    """
    arr = np.rollaxis(arr, axis)
    # Use the max to normalize, as with the log this is what accumulates
    # the less errors
    vmax = arr.max(axis=0)
    out = np.log(np.sum(np.exp(arr - vmax), axis=0))
    out += vmax
    return out