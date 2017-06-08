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
    # Bhinesh Patel - 6/6/2017
    # raise NotImplementedError
    
    for i, (X, lengths) in test_set.get_all_Xlengths().items():
        # Record scores in a dict
        scores = {}
        # for each word-model pair get score. Code snippet from forum posting.
        for word, model in models.items():
            try:
                # score word with the current_model
                scores[word] = model.score(X, lengths)
            except:
                # unable to score word with model so give a low value. This idea was from a forum posting.
                scores[word] = float("-inf")
                pass

        probabilities.append(scores)
        # guess the word based on the highest score. Get highest score and return the index, which is the word.
        guess_word = max(scores, key= scores.get)
        guesses.append(guess_word)

    return probabilities, guesses


