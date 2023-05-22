import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error as mse


class NMFRecommender:
    def __init__(self, random_state=15, rank=2, maxiter=200, tol=1e-3):
        """
        Save the parameter values as attributes.
        """
        self.random_state = random_state
        self.tol = tol
        self.maxiter = maxiter
        self.rank = rank

    def initialize_matrices(self, m, n):
        """
        Initialize the W and H matrices.
        
        Parameters:
            m (int): the number of rows
            n (int): the number of columns
        Returns:
            W ((m,k) array)
            H ((k,n) array)
        """
        np.random.seed(self.random_state)
        self.W = np.random.random((m, self.rank))
        self.H = np.random.random((self.rank, n))

        return self.W, self.H

    def _compute_loss(self, V, W, H):
        """
        Compute the loss of the algorithm according to the
        Frobenius norm.

        Parameters:
            V ((m,n) array): the array to decompose
            W ((m,k) array)
            H ((k,n) array)
        """
        return np.linalg.norm(V - W @ H, 'fro')

    def _update_matrices(self, V, W, H):
        """
        The multiplicative update step to update W and H
        Return the new W and H (in that order).
        
        Parameters:
            V ((m,n) array): the array to decompose
            W ((m,k) array)
            H ((k,n) array)
        Returns:
            New W ((m,k) array)
            New H ((k,n) array)
        """
        # Equations from book
        H1 = H * (W.T @ V) / (W.T @ W @ H)
        W1 = W * (V @ H1.T) / (W @ H1 @ H1.T)

        return W1, H1

    def fit(self, V):
        """
        Fits W and H weight matrices according to the multiplicative 
        update algorithm. Save W and H as attributes and return them.
        
        Parameters:
            V ((m,n) array): the array to decompose
        Returns:
            W ((m,k) array)
            H ((k,n) array)
        """
        W, H = self.initialize_matrices(np.shape(V)[0], np.shape(V)[1])

        # Update and compute loss
        for _ in range(self.maxiter):
            W1, H1 = self._update_matrices(V, W, H)
            W, H = W1, H1

            if self._compute_loss(V, W1, H1) < self.tol:
                break

        self.W = W
        self.H = H

        return W, H


    def reconstruct(self):
        """
        Reconstruct and return the decomposed V matrix for comparison against 
        the original V matrix. Use the W and H saved as attrubutes.
        
        Returns:
            V ((m,n) array): the reconstruced version of the original data
        """
        V = self.W @ self.H
        return V


def prob4(rank=2):
    """
    Run NMF recommender on the grocery store example.
    
    Returns:
        W ((m,k) array)
        H ((k,n) array)
        The number of people with higher component 2 than component 1 scores
    """
    V = np.array([[0,1,0,1,2,2],
                  [2,3,1,1,2,2],
                  [1,1,1,0,1,1],
                  [0,2,3,4,1,1],
                  [0,0,0,0,1,0]])

    nmf = NMFRecommender(rank=rank)
    W, H = nmf.fit(V)

    # Go through people and count
    peeps = 0
    for person in range(np.shape(H)[1]):
        if H[:, person][1] > H[:, person][0]:
            peeps += 1

    return W, H, peeps

# print(prob4())


def prob5(filename='artist_user.csv'):
    """
    Read in the file `artist_user.csv` as a Pandas dataframe. Find the optimal
    value to use as the rank as described in the lab pdf. Return the rank and the reconstructed matrix V.
    
    Returns:
        rank (int): the optimal rank
        V ((m,n) array): the reconstructed version of the data
    """
    # Load file and set benchmark
    df = pd.read_csv("artist_user.csv", index_col=0)
    bench = np.linalg.norm(df, 'fro') * .0001

    rank = 3
    # Iterate until convergence
    while True:
        nmf = NMF(n_components=rank, init='random', random_state=0)
        W = nmf.fit_transform(df)
        H = nmf.components_
        V = W @ H

        if np.sqrt(mse(df, V)) < bench:
            break

        rank += 1

    return rank, V


def discover_weekly(user_id, V):
    """
    Create the recommended weekly 30 list for a given user.
    
    Parameters:
        userid (int): which user to do the process for
        V ((m,n) array): the reconstructed array
    """
    # Read in the csv files
    df = pd.read_csv("artist_user.csv", index_col=0)
    df.rename(columns=pd.read_csv('artists.csv').astype(str).set_index('id').to_dict()['name'], inplace=True)
    user_all = df.loc[user_id, :].reset_index()
    user_all.rename(columns={'index': 'artist', user_id: 'listens'}, inplace=True)

    # Get Recommendations
    user_all['weights'] = V[user_id - 2]
    user_all = user_all[user_all['listens'] == 0]
    user_all.sort_values(by='weights', ascending=False, inplace=True)

    return user_all['artist'].tolist()[:30]