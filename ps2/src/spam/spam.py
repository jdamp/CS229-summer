import collections

import numpy as np
import util
import svm
import pickle


def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    words = message.split()
    normalized_words = [word.lower() for word in words]
    return normalized_words
    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """
    word_counts = collections.defaultdict(lambda: 0)
    # *** START CODE HERE ***
    for message in messages:
        for word in get_words(message):
            word_counts[word] += 1
    # Filter out words appearing less than five times
    common_words = [word for word in word_counts if word_counts[word] >= 5]
    return dict(zip(common_words, range(len(common_words))))
    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message. 
    Each row in the resulting array should correspond to each message 
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    # *** START CODE HERE ***
    n_messages = len(messages)
    n_words = len(word_dictionary)
    rep_arr = np.zeros((n_messages, n_words))
    for imsg, message in enumerate(messages):
        words = get_words(message)
        for word in words:
            try:
                rep_arr[imsg, word_dictionary[word]] += 1
            except KeyError:  # word not in dictionary, just ignore
                continue
    return rep_arr
    # *** END CODE HERE ***


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # *** START CODE HERE ***
    # Number of training examples
    num_examples = matrix.shape[0]
    # Number of words in the dictionary (features)
    num_words = matrix.shape[1]

    # Select columns of M belonging to labels = 1 and labels = 0
    matrix_ones = matrix[np.argwhere(labels == 1).reshape(-1)]
    matrix_zeros = matrix[np.argwhere(labels == 0).reshape(-1)]

    # Maximum likelihood estimate of phi_k|y=1
    k_vec = np.arange(0, np.max(matrix) + 1).reshape(1, 1, -1)
    # M_(i,j,k): word j in document i has the count k
    # Count number of times that feature j has value k in a certain class
    # => Feature word/j appears k times in documents that are spam(y=1) or not (y=0)
    m1 = (matrix_ones[:, :, np.newaxis] == k_vec)
    m0 = (matrix_zeros[:, :, np.newaxis] == k_vec)

    # Slightly different normailization than in the script to make sure that sum_k phi_1k = sum_k phi_0k = 1 for each
    # feature, i.e. the probabilities of the multinomial model are properly normalized
    phi_1 = (1 + np.sum(m1, axis=0)) / (np.sum(m1) + num_words)
    phi_0 = (1 + np.sum(m0, axis=0)) / (np.sum(m0) + num_words)
    # phi_y:
    phi_y = np.mean(labels)

    model = {"logphi_1": np.log(phi_1), 'logphi_0': np.log(phi_0), 'logphi_y1': np.log(phi_y),
             'logphi_y0': np.log(1-phi_y)}
    return model
    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    # phi.shape: values k)
    # matrix.shape: (documents n, features d)
    observed = matrix.astype(int)
    logprobs0 = np.array([model["logphi_0"][d, entry] for row in observed for d, entry in enumerate(row)]).reshape(matrix.shape)
    logprobs1 = np.array([model["logphi_1"][d, entry] for row in observed for d, entry in enumerate(row)]).reshape(matrix.shape)

    # Posterior probability
    # P(y=1|x, phi) = p(y=1|phi) * Prod_1^(d) p(x_d| y=1, phi_1) / (p(y=0)
    #=> log(P(y=1|x, phi)) = log(p(y=1|phi)) + Sum_1^(d) log(p(x_d| y=1, phi_1)) - log(p(x))

    p1 = np.sum(logprobs1, axis=1) + model['logphi_y1']
    p0 = np.sum(logprobs0, axis=1) + model['logphi_y0']
    return p1 > p0

    # *** END CODE HERE ***


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in part-c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of the top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    word_arr = np.array(list(dictionary.values()))
    # Calculate the model as follows:
    # log (P(token i at least once in mail| mail is SPAM) / P(token i at least once in mail| mail not SPAM) )
    logdiff = model["logphi_1"][:, 1:].sum(axis=1) - model["logphi_0"][:, 1:].sum(axis=1)
    top_5 = (-logdiff).argsort()[:5]
    # We need an inverse mapping of indices to words:
    inverse_dict = {val: key for key, val in dictionary.items()}
    return [inverse_dict[index] for index in top_5]


def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spma or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider

    Returns:
        The best radius which maximizes SVM accuracy.
    """
    # *** START CODE HERE ***
    # *** END CODE HERE ***


def main():
    train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('spam_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('spam_test.tsv')

    dictionary = create_dictionary(train_messages)

    print('Size of dictionary: ', len(dictionary))

    util.write_json('spam_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('spam_sample_train_matrix', train_matrix[:100,:])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('spam_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('spam_top_indicative_words', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util.write_json('spam_optimal_radius', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))


if __name__ == "__main__":
    main()
