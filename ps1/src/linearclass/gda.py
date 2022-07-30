import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=False)

    # Train a GDA classifier
    # Plot decision boundary on validation set
    # Use np.savetxt to save outputs from validation set to save_path
    classifier = GDA()
    classifier.fit(x_train, y_train)

    util.plot(x_valid, y_valid, classifier.theta, f"{save_path}.pdf")
    predictions = classifier.predict(x_valid)
    np.savetxt(f"{save_path}.txt", predictions)


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """

        # Find phi, mu_0, mu_1, and sigma
        # Write theta in terms of the parameters
        n = y.shape[0]
        y = np.expand_dims(y, axis=1)
        y_inv = np.where(y == 1, 0, 1)

        phi = np.sum(y) / n
        mu_0 = np.dot(y_inv.T, x) / np.sum(y_inv)
        mu_1 = np.dot(y.T, x) / np.sum(y)
        mu = np.where(y == 1, mu_1, mu_0)
        sigma = 1/n * np.dot((x-mu).T, (x-mu))
        sigma_inv = np.linalg.inv(sigma)

        theta_0 = -np.log((1-phi)/phi) + 0.5 * (mu_0@sigma_inv@mu_0.T - (mu_1@sigma_inv@mu_1.T))
        theta = -sigma_inv@(mu_0-mu_1).T
        self.theta = np.concatenate((theta_0, theta))


    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        print(self.theta[1:].T.shape, x.shape)
        return 1/(1 + np.exp(x@self.theta[1:]) + self.theta[0])


if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2')
