"""
EECS 445 - Introduction to Machine Learning
Winter 2025 - HW4 - q1_run_animate.py
Script to visualize a GMM.
"""

from q1_gmm import gmm
from q1_run import get_data
import numpy as np

np.random.seed(445)
X, test = get_data()
mu, p, z, si2, BIC = gmm(X[:,1:3], 3, num_iter=30, plot=True)
# Here, we train on only the first two columns of our data for visualization
# purposes. In run.py, you should use all columns of X
assert np.all(np.isclose(p, test, rtol=0.01)), 'Expected p = [[0.24954319], [0.39035819], [0.36009862]], got p = ' + str(p)
print('Correct EM parameters!')
assert BIC > 1524.01 and BIC < 1524.03, 'Expected BIC = 1524.02, got BIC = ' + str(BIC)
print('Correct BIC value!')