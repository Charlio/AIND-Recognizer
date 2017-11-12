import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    words = test_set.get_all_Xlengths()
    for word_id in range(len(test_set.get_all_sequences())):
        prob_dict = dict()
        highest_score = float('-inf')
        guess = None
        X, lengths = words[word_id]
        for word, hmm_model in models.items():
            try:
                curr_score = hmm_model.score(X, lengths)
                prob_dict[word] = curr_score
                if curr_score > highest_score:
                    highest_score = curr_score
                    guess = word
            except:
                prob_dict[word] = highest_score
        probabilities.append(prob_dict)
        guesses.append(guess)
    return probabilities, guesses
