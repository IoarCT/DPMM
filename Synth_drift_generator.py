from sklearn.datasets import make_blobs
import numpy as np


# Create batches with user controlled concept drift
def drift_generator_2D(n_samples, n_test_samples, n_batches, n_stationary_batches, random_seed=999):

    X = np.zeros((n_batches, n_samples, 2))
    test_data = np.zeros((n_batches, n_test_samples, 2))

    np.random.seed(random_seed)
    
    center_var = np.random.uniform(-.4, .4, (n_batches//n_stationary_batches, 4, 2))
    std_var = np.random.uniform(-.15, .15, (n_batches//n_stationary_batches, 4))

    center_drift = np.zeros((n_batches, 4, 2))
    std_drift = np.zeros((n_batches, 4))
    
    for i in range(n_batches):
        center_drift[i] = center_var[i//n_stationary_batches]
        std_drift[i] = std_var[i//n_stationary_batches]

    for i in range(n_batches):
        X[i] = make_blobs(n_samples=n_samples,
                          cluster_std=[.15, .15, .15, .15] + std_drift[i],
                          n_features=2,
                          centers=[(1.5, 1.5),
                                   (-1, 1),
                                   (-.5, -1),
                                   (1, -.5)] + center_drift[i])[0]

        test_data[i] = make_blobs(n_samples=n_test_samples,
                      cluster_std=[.15, .15, .15, .15] + std_drift[i],
                      n_features=2,
                      centers=[(1.5, 1.5),
                                (-1, 1),
                                (-.5, -1),
                                (1, -.5)] + center_drift[i])[0]
    
    return X, test_data
        
        
        
        
