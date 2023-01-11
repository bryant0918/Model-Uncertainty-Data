import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import ClassifierMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import itertools


class NaiveBayesFilter(ClassifierMixin):
    '''
    A Naive Bayes Classifier that sorts messages in to spam or ham.
    '''

    def __init__(self):
        self.data = None
        self.ham_count = None
        self.spam_count = None
        self.words = None

        return

    def tokenize(self, message, words=None):
        if words is None:
            return message.split()
        else:
            return [w if w in words else "<UNK>" for w in message.split()]

    def fit(self, X, y):
        '''
        Create a table that will allow the filter to evaluate P(H), P(S)
        and P(w|C)

        Parameters:
            X (pd.Series): training data
            y (pd.Series): training labels
        '''
        # Get dictionary of words
        words_label = [(self.tokenize(X.iloc[i]), y.iloc[i]) for i in range(len(X))]
        self.words = set(itertools.chain.from_iterable([tup[0] for tup in words_label]))
        word_dict = {word:{"spam": 0, "ham": 0} for word in self.words}
        word_dict['<UNK>'] = {"spam": 1, "ham": 1}

        # Start counting
        self.ham_count = 0
        self.spam_count = 0
        for words, label in words_label:
            if label == "spam":
                self.spam_count += 1
            elif label == "ham":
                self.ham_count += 1
            for word in words:
                word_dict[word][label] += 1

        # Create dataframe
        self.data = pd.DataFrame(word_dict)


    def predict_proba(self, X):
        '''
        Find P(C=k|x) for each x in X and for each class k by computing
        P(C=k)P(x|C=k)

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,2): Probability each message is ham, spam
                0 column is ham
                1 column is spam
        '''
        # Find the probabilities
        prob_spam = self.spam_count / (self.spam_count + self.ham_count)
        prob_ham = self.ham_count / (self.ham_count + self.spam_count)

        # Multiply by conditional probabilities
        spam_probs = [prob_spam*np.product([self.data.loc['spam'][xi]/self.data.loc['spam'].sum(axis=0) for xi in self.tokenize(X.iloc[i], self.words)]) for i in range(len(X))]
        ham_probs = [prob_ham*np.product([self.data.loc['ham'][xi]/self.data.loc['ham'].sum(axis=0) for xi in self.tokenize(X.iloc[i], self.words)]) for i in range(len(X))]

        return np.array(np.column_stack((ham_probs, spam_probs)))


    def predict(self, X):
        '''
        Use self.predict_proba to assign labels to X,
        the label will be a string that is either 'spam' or 'ham'

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,): label for each message
        '''
        # Get probabilities and take max
        probabilities = self.predict_proba(X)
        binary_labels = np.array([np.argmax(probabilities[i]) for i in range(len(probabilities))])

        return np.array(['ham' if idx == 0 else "spam" for idx in binary_labels])

    def predict_log_proba(self, X):
        '''
        Find ln(P(C=k|x)) for each x in X and for each class k

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,2): Probability each message is ham, spam
                0 column is ham
                1 column is spam
        '''

        # Find the probabilities
        prob_spam = np.log(self.spam_count / (self.spam_count + self.ham_count))
        prob_ham = np.log(self.ham_count / (self.ham_count + self.spam_count))

        # Add to conditional probabilities
        spam_probs = [prob_spam + np.sum([np.log((self.data.loc['spam'][xi] + 1) / (self.data.loc['spam'].sum(axis=0) + 2)) for xi in
                                              self.tokenize(X.iloc[i], self.words)]) for i in range(len(X))]
        ham_probs = [prob_ham + np.sum([np.log((self.data.loc['ham'][xi] + 1) / (self.data.loc['ham'].sum(axis=0) + 2)) for xi in
                                            self.tokenize(X.iloc[i], self.words)]) for i in range(len(X))]

        return np.array(np.column_stack((ham_probs, spam_probs)))

    def predict_log(self, X):
        '''
        Use self.predict_log_proba to assign labels to X,
        the label will be a string that is either 'spam' or 'ham'

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,): label for each message
        '''
        # Get probabilities and take max
        probabilities = self.predict_log_proba(X)
        binary_labels = np.array([np.argmax(probabilities[i]) for i in range(len(probabilities))])

        return np.array(['ham' if idx == 0 else "spam" for idx in binary_labels])


class PoissonBayesFilter(ClassifierMixin):
    '''
    A Naive Bayes Classifier that sorts messages in to spam or ham.
    This classifier assumes that words are distributed like
    Poisson random variables
    '''

    def __init__(self):
        return

    def tokenize(self, message, words=None):
        if words is None:
            return message.split()
        else:
            return [w if w in words else "<UNK>" for w in message.split()]

    def fit(self, X, y):
        '''
        Uses bayesian inference to find the poisson rate for each word
        found in the training set. For this we will use the formulation
        of l = rt since we have variable message lengths.

        This method creates a tool that will allow the filter to
        evaluate P(H), P(S), and P(w|C)


        Parameters:
            X (pd.Series): training data
            y (pd.Series): training labels

        Returns:
            self: this is an optional method to train
        '''
        # Get list of messages and labels
        words_label = [(self.tokenize(X.iloc[i]), y.iloc[i]) for i in range(len(X))]
        # Get a set of unique words
        self.words = set(itertools.chain.from_iterable([tup[0] for tup in words_label]))
        # Create dictionaries to count occurances and add unknown token
        word_dict = {word: {"spam": 0, "ham": 0} for word in self.words}
        word_dict['<UNK>'] = {"spam": 1, "ham": 1}

        # Start counting
        self.ham_count = 0
        self.spam_count = 0
        for words, label in words_label:
            if label == "spam":
                self.spam_count += 1
            elif label == "ham":
                self.ham_count += 1
            for word in words:
                word_dict[word][label] += 1

        # Create data frame
        self.data = pd.DataFrame(word_dict)

        # redefine dataframe
        self.data.loc['ham'] = (self.data.loc['ham']) / (self.spam_count + self.ham_count)
        self.data.loc['spam'] = (self.data.loc['spam']) / (self.spam_count + self.ham_count)

        # Get spam and ham rates
        self.ham_rates = self.data.loc['ham'] / self.data.loc['ham'].sum(axis=0)
        self.spam_rates = self.data.loc['spam'] / self.data.loc['spam'].sum(axis=0)

        # self.data.loc['ham'] = (self.data.loc['ham'] + 1) / (self.spam_count + self.ham_count + 2)
        # self.data.loc['spam'] = (self.data.loc['spam'] + 1) / (self.spam_count + self.ham_count + 2)

        # self.ham_rates = self.data.loc['ham'] / self.data.loc['ham'].sum(axis=0)
        # self.spam_rates = self.data.loc['spam'] / self.data.loc['spam'].sum(axis=0)

    def predict_log_proba(self, X):
        '''
        Find ln(P(C=k|x)) for each x in X and for each class

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,2): Log probability each message is ham or spam
                column 0 is ham, column 1 is spam
        '''

        # Find the probabilities of spam and ham
        prob_spam = np.log(self.spam_count / (self.spam_count + self.ham_count))
        prob_ham = np.log(self.ham_count / (self.ham_count + self.spam_count))

        # Add the conditional probabilities
        ham_probs = [
            prob_ham + np.sum(stats.poisson.logpmf(np.unique(self.tokenize(X.iloc[i],self.words), return_counts=True)[1],
                                                              self.data.loc['ham'][np.unique(self.tokenize(X.iloc[i],self.words), return_counts=True)[0]] *
                                                              len(np.unique(self.tokenize(X.iloc[i],self.words), return_counts=True)[0]))) for i in range(len(X))]

        # Add the conditional probabilities
        spam_probs = [prob_spam + np.sum(stats.poisson.logpmf(np.unique(self.tokenize(X.iloc[i],self.words), return_counts=True)[1],
                                                              self.data.loc['spam'][np.unique(self.tokenize(X.iloc[i],self.words), return_counts=True)[0]] *
                                                              len(np.unique(self.tokenize(X.iloc[i],self.words), return_counts=True)[0]))) for i in range(len(X))]


        return np.array(np.column_stack((ham_probs, spam_probs)))

    def predict(self, X):
        '''
        Use self.predict_log_proba to assign labels to X

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,): label for each message
        '''
        # Get probabilities and take the max
        probabilities = self.predict_log_proba(X)
        binary_labels = np.array([np.argmax(probabilities[i]) for i in range(len(probabilities))])

        return np.array(['ham' if idx == 0 else "spam" for idx in binary_labels])


def sklearn_method(X_train, y_train, X_test):
    '''
    Use sklearn's methods to transform X_train and X_test, create a
    naïve Bayes filter, and classify the provided test set.

    Parameters:
        X_train (pandas.Series): messages to train on
        y_train (pandas.Series): labels for X_train
        X_test  (pandas.Series): messages to classify

    Returns:
        (ndarray): classification of X_test
    '''

    # Transform data
    vectorizer = CountVectorizer()
    train_counts = vectorizer.fit_transform(X_train)

    # Train model
    clf = MultinomialNB()
    clf = clf.fit(train_counts, y_train)

    # Get prediction labels
    test_counts = vectorizer.transform(X_test)
    labels = clf.predict(test_counts)

    return labels


if __name__ == "__main__":
    df = pd.read_csv("sms_spam_collection.csv")

    X = df['Message']
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7)

    # NBF = NaiveBayesFilter()
    # NBF.fit(X_train, y_train)
    # # print(X[:1])
    # probs = NBF.predict_proba(X_test)
    # labels = NBF.predict_log(X_test)
    # print(accuracy_score(y_test, labels))

    # NBF = NaiveBayesFilter()
    # NBF.fit(X[:300], y[:300])
    # labels = NBF.predict(X[530:535])
    # print(labels)
    # log_labels = NBF.predict_log(X[530:535])
    # print(log_labels)

    # PB = PoissonBayesFilter()
    # PB.fit(X[:300], y[:300])
    #
    # print(PB.ham_rates['in'])
    # print(PB.spam_rates['in'])

    PB = PoissonBayesFilter()
    PB.fit(X_train, y_train)
    labels = PB.predict(X_test)
    print(accuracy_score(y_test, labels))

    # labels = sklearn_method(X_train, y_train, X_test)
    # print(accuracy_score(y_test, labels))
