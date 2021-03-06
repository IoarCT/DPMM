import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
import numpy as np
from HPP_CVI_DPMM import HPP_DPMM
from MHPP_CVI_DPMM import MHPP_DPMM
from CVI_for_DPMM import CV_DPMM
from Auxiliary_functions import compute_test_likelihood
from Synth_drift_generator import drift_generator_2D
import time

# Use numbers 1 to n for the nonempty clusters so that there is no confusion
# with the colors
def regularize_for_print(array):
    for t in range(len(np.unique(array))):
        array[array == np.unique(array)[t]] = t
    return array
# Aux. function to print ellipses
def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

colors = sns.color_palette("Paired", n_colors=100)
n_batches=30

X, test_data =  drift_generator_2D(n_samples=1000, n_test_samples=200,\
                                   n_batches=n_batches, n_stationary_batches=15, random_seed=999)

start = time.time()


# Execute three algorithms
params, clusters, N_t, iteration, log_lik, cluster_centers, cluster_covs = CV_DPMM(X,
                                                                              alpha=2.,
                                                                              thresh=1.e-3,
                                                                              max_iter=100,
                                                                              T=10)


params_1, clusters, N_t, iteration, log_lik, cluster_centers, cluster_covs = HPP_DPMM(X,
                                                                              alpha=2.,
                                                                              thresh=1.e-3,
                                                                              max_iter=100,
                                                                              T=10)


params_2, clusters, N_t, iteration, log_lik, cluster_centers, cluster_covs = MHPP_DPMM(X,
                                                                              alpha=2.,
                                                                              thresh=1.e-3,
                                                                              max_iter=100,
                                                                              T=10)

# Compute and plot perplexity

perp = np.zeros((3, n_batches))
for i in range(n_batches):
    perp[0][i] = compute_test_likelihood(test_data[i], params[i][4], params[i][0], params[i][1], params[i][2], params[i][3], 2)
    perp[1][i] = compute_test_likelihood(test_data[i], params_1[i][4], params_1[i][0], params_1[i][1], params_1[i][2], params_1[i][3], 2)
    perp[2][i] = compute_test_likelihood(test_data[i], params_2[i][4], params_2[i][0], params_2[i][1], params_2[i][2], params_2[i][3], 2)


fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
ax1.plot(perp[0], 'rx-', linewidth=4, label='SVB-DPM')
ax1.plot(perp[1], 'bx--', linewidth=4, label='HPP-DPM')
ax1.plot(perp[2], 'gx-', linewidth=4, label='MHPP-DPM')
plt.locator_params(axis="x", integer=True, tight=True)
ax1.legend(frameon=False, fontsize=17)
plt.xlabel('Batch n??', fontsize=18)
plt.ylabel('test-likelihood/N', fontsize=18)
plt.show()

fig.savefig('synth.pdf', format='pdf', bbox_inches='tight')

end = time.time()
time_ = end-start


# Scatterplot of the clusters found

# for i in range(n_steps):
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
#     clusters[i] = regularize_for_print(clusters[i])
#     n_clust = len(np.unique(clusters[i]))
#     for n in range(n_clust):
#         data = X[i, clusters[i] == n]
#         ax1.scatter(data[:, 0], data[:, 1], s=10, color=colors[n])
#         ax1.scatter(cluster_centers[i][n, 0], cluster_centers[i][n, 1], color='black')
#         # Print cov. ellipses with 2 std.
#         cov = cluster_covs[i][n]
#         vals, vecs = eigsorted(cov)
#         theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
#         w, h = 4 * np.sqrt(vals)
#         ell = Ellipse(xy=(cluster_centers[i][n, 0], cluster_centers[i][n, 1]),
#                   width=w, height=h,
#                   angle=theta, color='black')
#         ell.set_facecolor('none')
#         ax1.add_artist(ell)
#     ax1.set_xlim([-2,2])
#     ax1.set_ylim([-2,2])
#     fig.suptitle("Batched Variational DPMM Clustering")
#     ax2.plot(log_lik[i])
#     ax2.set(xlabel="n?? of iterations", ylabel="log likelihood")

#     plt.show()
print('Algorithm running time:',np.round(time_,2),'sec')






