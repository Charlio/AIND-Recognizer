import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        best_hmm_model = None
        lowest_BIC = float('inf')
        n = len(self.lengths)
        logN = np.log(n)
        for i in range(self.min_n_components, self.max_n_components+1):
            try:
                hmm_model = self.base_model(i) 
                logL = hmm_model.score(self.X, self.lengths)
                p = n**2 + 2*i*n - 1
                bic = -2*logL + p*logN
                if bic < lowest_BIC:
                    lowest_BIC = bic
                    best_hmm_model = hmm_model
            except:
                continue
        return best_hmm_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        best_hmm_model = None
        highest_DIC = float('-inf')
        for i in range(self.min_n_components, self.max_n_components+1):
            try:
                hmm_model = self.base_model(i)
                other_scores = []
                for word in self.words.keys():
                    if word != self.this_word:
                        X, lengths = self.all_word_Xlengths[word]
                        other_scores.append(hmm_model.score(X, lengths))
                dic = hmm_model.score(self.X, self.lengths) - np.mean(other_scores)
                if dic > highest_DIC:
                    highest_DIC = dic
                    best_hmm_model = hmm_model
            except:
                continue
        return best_hmm_model or self.base_model(self.n_constant)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        highest_score = float('-inf')
        best_hmm_model = None
        for i in range(self.min_n_components, self.max_n_components+1):
            try:
                split_method = KFold()
                logLs = []
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)
                    hmm_model = self.base_model(i)
                    test_X, test_lengths = combine_sequences(cv_test_idx, self.sequences)
                    logL = hmm_model.score(test_X, test_lengths)
                    logLs.append(logL)
                avg_score = np.mean(logLs)
                if avg_score > highest_score:
                    highest_score = avg_score
                    best_hmm_model = hmm_model
            except:
                continue
        return best_hmm_model or self.base_model(self.n_constant)        
        