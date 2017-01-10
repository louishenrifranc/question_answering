import numpy as np
import theano.tensor as T
from theano import scan, function, shared, In
from theano.ifelse import ifelse
import os
import sys; sys.setrecursionlimit(50000)
from termcolor import cprint
import math
from six.moves import cPickle
import matplotlib.pyplot as plt

from utils.nlp_utils import load_embeddings, sentence2index, zero_pad
from utils.theano_utils import get_embeddings, get_weigths_and_bias
from utils.theano_utils import embedding_lookup, softmax


class LSTM(object):
    """
    Long short-term memory.
    """

    def __init__(self, vocabulary, n_hid=100):
        """
        Parameters
        ----------
        vocabulary : set of str
            Vocabulary to use
        n_hid : int (default : 100)
            Dimension of the intermediary states
        """

        cprint("Building model...", "magenta", attrs=["bold"])

        self._n_vocab = len(vocabulary) + 2

        self._word2index, self._index2word = self._get_word_dictionaries(vocabulary)

        self._n_emb = n_emb
        self._n_hid = n_hid

        self._params = []

        self._context = T.itensor3("context")
        self._question = T.imatrix("question")
        self._correct_answer = T.ivector("correct_answer")

    def 


    def _encode_context(self, context):
        """
        Encodes the context.

        Parameters
        ----------
        context : Tensor-3 of size batch_size x n_context x context_length
            Context indices
        
        Returns
        -------
        Tensor-2 of size batch_size x n_hid
            Encoded context
        """
