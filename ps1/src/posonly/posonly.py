import matplotlib.pyplot as plt
import numpy as np
import util
import sys

sys.path.append('../linearclass')

### NOTE : You need to complete p01b_logreg implementation first!

from logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, save_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on t-labels,
        2. on y-labels,
        3. on y-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        save_path: Path to save predictions.
    """
    output_path_true = save_path.replace(WILDCARD, 'true')
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_adjusted = save_path.replace(WILDCARD, 'adjusted')

    # Load data for all problems
    X_train, t_train = util.load_dataset(train_path, add_intercept=True, label_col='t')
    _, y_train = util.load_dataset(train_path, add_intercept=True, label_col='y')
    X_test, t_test = util.load_dataset(test_path, add_intercept=True, label_col='t')

    X_val, y_val = util.load_dataset(valid_path, add_intercept=True, label_col='y')

    # Part (a): Train and test on true labels

    classifier = LogisticRegression()
    classifier.fit(X_train, t_train)
    predictions = classifier.predict(X_test)
    # Make sure to save predicted probabilities to output_path_true using np.savetxt()
    np.savetxt(output_path_true, predictions)
    util.plot(X_test, t_test, classifier.theta, "part_a.pdf")


    # Part (b): Train on y-labels and test on true labels
    classifier_b = LogisticRegression()
    classifier_b.fit(X_train, y_train)
    predictions = classifier_b.predict(X_test)
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    np.savetxt(output_path_naive, predictions)
    util.plot(X_test, t_test, classifier_b.theta, "part_b.pdf")

    # Part (f): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to output_path_adjusted
    h_x = classifier_b.predict(X_val[y_val == 1])
    alpha = np.sum(h_x)/np.sum(y_val)

    scaled_predictions = 1/alpha * predictions
    np.savetxt(output_path_adjusted, scaled_predictions)
    util.plot(X_test, t_test, classifier_b.theta, correction=alpha, save_path="part_f.pdf")


if __name__ == '__main__':
    main(train_path='train.csv',
        valid_path='valid.csv',
        test_path='test.csv',
        save_path='posonly_X_pred.txt')
