from typing import TypeVar
from collections import Counter
import itertools
import numpy as np
from scipy import linalg


def sigmoid(z):
    # sigmoid function
    return 1.0/(1.0+np.exp(-z))


def pack_idx_with_frequency(words, word2idx):
    "yields positional encoding of words with their relative frequency in given input"
    freqs = Counter(words)
    for word in words:
        word_idx = word2idx[word]
        freq = freqs[word]
        yield word_idx, freq


def target_label_pairs(input, word2idx, V, C):
    """
    yields pair of target and context vectors.
    Context vector is a bag-of-words around target.
    Target is one-hot encoded
    ### Input:
    - input: list of words
    - word2idx: mapping of word to index
    - V: size of vocabulary
    - C: size of context window (lookahead and lookbehind)
    """
    for slice in itertools.cycle(window(input, C * 2 + 1)):
        center = slice[C]
        context = slice[:C] + slice[C + 1:]

        target = np.zeros(V)
        target[word2idx[center]] = 1

        input_bow = np.zeros(V)
        for idx, freq in pack_idx_with_frequency(context, word2idx):
            input_bow[idx] = freq/len(context)
        yield input_bow, target


T = TypeVar('T')

def window(items: list[T], window_size: int):
    """
    yields values in a sliding window of a size `window_size`

    ### Example:
    `items = [1,2,3,4,5,6]`

    `window_size = 4`

    yields:
    - [1,2,3,4]
    - [2,3,4,5]
    - [3,4,5,6]
    """
    for i in range(len(items) - window_size + 1):
        yield items[i:i + window_size]


def batches(data, word2idx, V, C, batch_size):
    batch_x = []
    batch_y = []
    for x, y in target_label_pairs(data, word2idx, V, C):
        while len(batch_x) < batch_size:
            batch_x.append(x)
            batch_y.append(y)
        yield np.array(batch_x).T, np.array(batch_y).T
        batch_x = []
        batch_y = []


def compute_pca(data, n_components=2):
    """
    Input: 
        data: of dimension (m,n) where each row corresponds to a word vector
        n_components: Number of components you want to keep.
    Output: 
        X_reduced: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """

    m, n = data.shape

    ### START CODE HERE ###
    # mean center the data
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric,
    # the performance gain is substantial
    evals, evecs = linalg.eigh(R)
    # sort eigenvalue in decreasing order
    # this returns the corresponding indices of evals and evecs
    idx = np.argsort(evals)[::-1]

    evecs = evecs[:, idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :n_components]
    ### END CODE HERE ###
    return np.dot(evecs.T, data.T).T


def positional_encoding(items):
    """
    Returns mapping of indicies to the value (also known as an array), and value to the index
    """
    idx2word = list(items)
    word2idx = { value: idx for idx, value in enumerate(idx2word) }
    return word2idx, idx2word
