# solutions.py
"""Volume 3: Gaussian Mixture Models. Solutions File."""

import numpy as np
from scipy import stats as st
from scipy.special import logsumexp
from scipy.optimize import linear_sum_assignment
from scipy.stats import multivariate_normal as mn
from matplotlib import pyplot as plt
import time
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture as SKGMM
from sklearn.cluster import KMeans


class GMM:
    # Problem 1
    def __init__(self, n_components, weights=None, means=None, covars=None):
        """
        Initializes a GMM.
        
        The parameters weights, means, and covars are optional. If fit() is called,
        they will be automatically initialized from the data.
        
        If specified, the parameters should have the following shapes, where d is
        the dimension of the GMM:
            weights: (n_components,)
            means: (n_components, d)
            covars: (n_components, d, d)
        """
        # Set attributes
        self.n_components = n_components
        self.weights = weights
        self.means = means
        self.covars = covars
        if means is not None:
            self.d = np.shape(means)[1]
    
    # Problem 2
    def component_logpdf(self, k, z):
        """
        Returns the logarithm of the component pdf. This is used in several computations
        in other functions.
        
        Parameters:
            k (int) - the index of the component
            z ((d,) or (..., d) ndarray) - the point or points at which to compute the pdf
        Returns:
            (float or ndarray) - the value of the log pdf of the component at 
        """

        return np.log(self.weights[k]) + mn.logpdf(z, self.means[k, :], self.covars[k, :, :])
    
    # Problem 2
    def pdf(self, z):
        """
        Returns the probability density of the GMM at the given point or points.
        
        Parameters:
            z ((d,) or (..., d) ndarray) - the point or points at which to compute the pdf
        Returns:
            (float or ndarray) - the value of the GMM pdf at z
        """

        return np.sum([self.weights[k]*mn.pdf(z, self.means[k], self.covars[k]) for k in range(len(self.weights))])
    
    # Problem 3
    def draw(self, n):
        """
        Draws n points from the GMM.
        
        Parameters:
            n (int) - the number of points to draw
        Returns:
            ((n,d) ndarray) - the drawn points, where d is the dimension of the GMM.
        """
        # Get components
        xs = np.random.choice(self.n_components, n, p=self.weights, replace=True)

        zs = []
        # Go through components
        for x in xs:
            mean = self.means[x]
            covar = self.covars[x]
            # Use component means and variances to draw z
            z = np.random.multivariate_normal(mean, cov=covar)
            zs.append(z)

        return np.array(zs)

    # Problem 4
    def _compute_e_step(self, Z):
        """
        Computes the values of q_i^t(k) for the given data and current parameters.
        
        Parameters:
            Z ((n, d) ndarray): the data that is being used for training; d is the
                    dimension of the data.
        Returns:
            ((n_components, n) ndarray): an array of the computed q_i^t(k) values, such
                    that result[k,i] = q_i^t(k).
        """
        l_s = np.array([self.component_logpdf(k, Z) for k in range(self.n_components)])
        L = np.amax(l_s, axis=0)
        # Sum over components in denominator to get right answers but lower accuracy and slower runtime.
        qitk = np.exp(l_s-L) / (np.sum([np.exp(l_s[kp, :]-L) for kp in range(self.n_components)], axis=0))

        return qitk


    # Problem 5
    def _compute_m_step(self, Z):
        """
        Takes a step of the expectation maximization (EM) algorithm. Return
        the updated parameters.
        
        Parameters:
            Z (n,d) ndarray): the data that is being used for training; d is the
                    dimension of the data.
        Returns:
            ((n_components,) ndarray): the updated component weights
            ((n_components,d) ndarray): the updated component means
            ((n_components,d,d) ndarray): the updated component covariance matrices
        """
        n = Z.shape[0]
        q = self._compute_e_step(Z)

        # Find new weights
        wk_new = 1/n * np.sum(q, axis=1)
        # Find new means
        mu_new = q@Z/np.sum(q, axis=1).reshape(-1, 1)

        centered = np.expand_dims(Z, 0) - np.expand_dims(mu_new, 1)
        # Here's the magic einsum
        cov_new = np.einsum("Kn, Knd, KnD -> KdD", q, centered, centered) / np.sum(q, axis=1).reshape(-1, 1, 1)

        return wk_new, mu_new, cov_new
        
    # Problem 6
    def fit(self, Z, tol=1e-3, maxiter=200):
        """
        Fits the model by applying the Expectation Maximization algorithm until the
        parameters appear to converge.
        
        Parameters:
            Z ((n,d) ndarray): the data to use for training; d is the
                dimension of the data.
            tol (float): the tolderance to check for convergence
            maxiter (int): the maximum number of iterations allowed
        Returns:
            self
        """
        n, d = np.shape(Z)

        # If attributes are none then set
        if self.means is None:
            indices = np.random.randint(0, n, size=self.n_components)
            self.means = np.array([Z[i] for i in indices])
            self.d = np.shape(self.means)[1]

        if self.weights is None:
            self.weights = np.ones(self.n_components)/self.n_components

        if self.covars is None:
            vars = []
            for col in range(d):
                var = np.var(Z[:, col])
                vars.append(var)
            covars = np.diag(vars)
            self.covars = np.array([covars for k in range(self.n_components)])

        old_weights, old_means, old_covars = self.weights, self.means, self.covars

        # Iterate
        for i in range(maxiter):

            new_weights, new_means, new_covars = self._compute_m_step(Z)
            change = (np.max(np.abs(new_weights - old_weights)) + np.max(np.abs(new_means - old_means))
                      + np.max(np.abs(new_covars - old_covars)))

            # Check convergence
            if change < tol:
                break

            # Reset attributes
            self.weights, self.means, self.covars = new_weights, new_means, new_covars
            old_weights, old_means, old_covars = new_weights, new_means, new_covars
            
        return self
        
    # Problem 8
    def predict(self, Z):
        """
        Predicts the labels of data points using the trained component parameters.
        
        Parameters:
            Z ((m,d) ndarray): the data to label; d is the dimension of the data.
        Returns:
            ((m,) ndarray): the predicted labels of the data
        """

        return np.argmax([np.exp(self.component_logpdf(k, Z)) for k in range(self.n_components)], axis=0)
        
    def fit_predict(self, Z, tol=1e-3, maxiter=200):
        """
        Fits the model and predicts cluster labels.
        
        Parameters:
            Z ((m,d) ndarray): the data to use for training; d is the
                dimension of the data.
            tol (float): the tolderance to check for convergence
            maxiter (int): the maximum number of iterations allowed
        Returns:
            ((m,) ndarray): the predicted labels of the data
        """
        return self.fit(Z, tol, maxiter).predict(Z)


# Problem 3
def problem3():
    """
    Draw a sample of 10,000 points from the GMM defined in the lab pdf. Plot a heatmap
    of the pdf of the GMM (using plt.pcolormesh) and a hexbin plot of the drawn points.
    How do the plots compare?
    """
    n = 10000

    x = np.linspace(-5, 5, 100)
    y = np.linspace(-8, 8, 100)
    X, Y = np.meshgrid(x, y)

    # Draw from pdf
    Z = np.array([[gmm.pdf([X[i, j], Y[i, j]]) for j in range(100)] for i in range(100)])
    draws = np.array(gmm.draw(n))

    # Plot it
    plt.subplot(121)
    plt.pcolormesh(X, Y, Z, shading='auto')
    plt.title("PDF")
    plt.subplot(122)
    plt.hexbin(draws[:, 0], draws[:, 1], gridsize=(25, 25))
    plt.xlim((-5,5))
    plt.ylim((-8,8))
    plt.title("Draws")
    plt.show()

# Problem 7
def problem7(filename='problem7.npy'):
    """
    The file problem7.npy contains a collection of data drawn from a GMM.
    Train a GMM on this data with n_components=3. Plot the pdf of your
    trained GMM, as well as a hexbin plot of the data.
    """
    # Load data
    data = np.load(filename)
    gmm = GMM(3)
    gmm.fit(data)

    n = 10000

    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x, y)

    # Draw
    Z = np.array([[gmm.pdf([X[i, j], Y[i, j]]) for j in range(100)] for i in range(100)])
    draws = np.array(gmm.draw(n))

    # Plot it
    plt.subplot(121)
    plt.pcolormesh(X, Y, Z, shading='auto')
    plt.title("PDF")
    plt.subplot(122)
    plt.hexbin(draws[:, 0], draws[:, 1], gridsize=(25, 25))
    plt.xlim((-4, 4))
    plt.ylim((-4, 4))
    plt.title("Draws")
    plt.show()

# Problem 8
def get_accuracy(pred_y, true_y):
    """
    Helper function to calculate the actually clustering accuracy, accounting for
    the possibility that labels are permuted.
    
    This computes the confusion matrix and uses scipy's implementation of the Hungarian
    Algorithm (linear_sum_assignment) to find the best combination, which is generally
    much faster than directly checking the permutations.
    """
    # Compute confusion matrix
    cm = confusion_matrix(pred_y, true_y)
    # Find the arrangement that maximizes the score
    r_ind, c_ind = linear_sum_assignment(cm, maximize=True)
    return np.sum(cm[r_ind, c_ind]) / np.sum(cm)
    
def problem8(filename='classification.npz'):
    """
    The file classification.npz contains a set of 3-dimensional data points "X" and 
    their labels "y". Use your class with n_components=4 to cluster the data.
    Plot the points with the predicted and actual labels, and compute and return
    your model's accuracy. Be sure to check for permuted labels.
    
    Returns:
        (float) - the GMM's accuracy on the dataset
    """

    data = np.load(filename)
    X = data["X"]
    y = data['y']
    gmm = GMM(4)

    # Make predictions
    predictions = gmm.fit_predict(X)

    # Make the Masks
    zeros_train = y == 0
    ones_train = y == 1
    twos_train = y == 2
    threes_train = y == 3

    zeros_predict = predictions == 0
    ones_predict = predictions == 1
    twos_predict = predictions == 2
    threes_predict = predictions == 3

    # graph the labels of the original data
    plt.subplot(221)
    plt.scatter(X[:, 0][zeros_train], y[zeros_train], color='red', s=.5, alpha=.1)
    plt.scatter(X[:, 0][ones_train], y[ones_train], color='blue', s=.5,alpha=.1)
    plt.scatter(X[:, 0][twos_train], y[twos_train], color='green',s=.5, alpha=.1)
    plt.scatter(X[:, 0][threes_train], y[threes_train], color='orange', s=.5,alpha=.1)

    plt.title('Actual Data')

    plt.subplot(222)
    # graph the labels of the predicted data
    plt.scatter(X[:, 0][zeros_predict], y[zeros_predict], color='red', s=.5,alpha=.1)
    plt.scatter(X[:, 0][ones_predict], y[ones_predict], color='blue', s=.5,alpha=.1)
    plt.scatter(X[:, 0][twos_predict], y[twos_predict], color='green',s=.5, alpha=.1)
    plt.scatter(X[:, 0][threes_predict], y[threes_predict], color='orange', s=.5,alpha=.1)
    plt.title('Predictions')

    # plt.subplot(223)
    # # graph the labels of the predicted data
    # plt.scatter(X[:, 0][zeros_train], X[:, 1][zeros_train], color='red', s=.75, alpha=.1)
    # plt.scatter(X[:, 0][ones_train], X[:, 1][ones_train], color='blue', s=.75, alpha=.1)
    # plt.scatter(X[:, 0][twos_train], X[:, 1][twos_train], color='green', s=.75, alpha=.1)
    # plt.scatter(X[:, 0][threes_train], X[:, 1][threes_train], color='orange', s=.75, alpha=.1)
    # plt.title('Actual')
    #
    # plt.subplot(224)
    # # graph the labels of the predicted data
    # plt.scatter(X[:, 0][zeros_predict], X[:, 1][zeros_predict], color='red', s=.75, alpha=.1)
    # plt.scatter(X[:, 0][ones_predict], X[:, 1][ones_predict], color='blue', s=.75, alpha=.1)
    # plt.scatter(X[:, 0][twos_predict], X[:, 1][twos_predict], color='green', s=.75, alpha=.1)
    # plt.scatter(X[:, 0][threes_predict], X[:, 1][threes_predict], color='orange', s=.75, alpha=.1)
    # plt.title('Predictions')

    ax = plt.subplot(223, projection='3d')
    # graph the labels of the predicted data
    ax.scatter(X[:, 0][zeros_train], X[:, 1][zeros_train], X[:, 2][zeros_train], color='red', s=.75, alpha=.1)
    ax.scatter(X[:, 0][ones_train], X[:, 1][ones_train], X[:, 2][ones_train], color='blue', s=.75,alpha=.1)
    ax.scatter(X[:, 0][twos_train], X[:, 1][twos_train], X[:, 2][twos_train], color='green',s=.75, alpha=.1)
    ax.scatter(X[:, 0][threes_train], X[:, 1][threes_train],  X[:, 2][threes_train], color='orange',s=.75, alpha=.1)

    plt.title('Actual')

    ax = plt.subplot(224, projection='3d')
    # graph the labels of the predicted data
    ax.scatter(X[:, 0][zeros_predict], X[:, 1][zeros_predict], X[:, 2][zeros_predict], color='red', s=.75,alpha=.1)
    ax.scatter(X[:, 0][ones_predict], X[:, 1][ones_predict], X[:, 2][ones_predict], color='blue', s=.75,alpha=.1)
    ax.scatter(X[:, 0][twos_predict], X[:, 1][twos_predict], X[:, 2][twos_predict], color='green', s=.75,alpha=.1)
    ax.scatter(X[:, 0][threes_predict], X[:, 1][threes_predict], X[:, 2][threes_predict], color='orange',  s=.75,alpha=.1)
    plt.title('Predictions')

    plt.tight_layout()

    plt.show()

    return get_accuracy(predictions, y)

# Problem 9
def problem9(filename='classification.npz'):
    """
    Again using classification.npz, compare your class, sklearn's GMM implementation, 
    and sklearn's K-means implementation for speed of training and for accuracy of 
    the resulting clusters. Print your results. Be sure to check for permuted labels.
    """
    data = np.load(filename)
    X = data["X"]
    y = data['y']

    # SKLearn's GMM
    skgmm = SKGMM(4, max_iter=200)
    start = time.time()
    predictions = skgmm.fit_predict(X)
    end = time.time()
    print("Accuracy for SkLearn's GMM: ", get_accuracy(predictions, y))
    print("Time for SkLearn's GMM ", end-start)
    print()

    # Sklearn's KMeans
    kmeans = KMeans(4, max_iter=200, tol=1e-3)
    start = time.time()
    predictions = kmeans.fit_predict(X)
    end = time.time()
    print("Accuracy for SkLearn's KMeans: ", get_accuracy(predictions, y))
    print("Time for SkLearn's Kmeans ", end - start)
    print()

    # My GMM
    gmm = GMM(4)
    start = time.time()
    predictions = gmm.fit_predict(X)
    end = time.time()
    print("Accuracy for my GMM: ", get_accuracy(predictions, y))
    print("Time for my GMM ", end - start)



if __name__ == "__main__":
    # gmm = GMM(n_components=2, weights=np.array([.6,.4]), means=np.array([[-.5,-4],[.5,.5]]), covars=np.array([[[1,0],[0,1]],[[.25,-1],[-1,8]]]))
    # print(gmm.pdf(np.array([1, -3.5])))
    # print(gmm.component_logpdf(0, np.array([1, -3.5])))
    # print(gmm.component_logpdf(1, np.array([1, -3.5])))
    # gmm.draw(10000)
    # problem3()
    # data = np.array([[.5, 1], [1, .5], [-2, .7]])
    # print(gmm._compute_e_step(data))
    # print(gmm._compute_m_step(data))
    # problem7()
    problem8()
    # problem9()
