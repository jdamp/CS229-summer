import numpy as np
import util


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logistic_cost(theta, x, y, eps):
    h_theta = sigmoid(x @ theta)
    m = x.shape[0]
    return (-1/m) * (y.T @ np.log(h_theta + eps) + (1 - y).T @ np.log(1 - h_theta + eps))


def logistic_gradient(theta, x, y):
    h_theta = sigmoid(x @ theta)
    m = x.shape[0]
    return 1/m * x.T @ (y - h_theta)


def logistic_hesse(theta, x):
    m = x.shape[0]# Make sure to save predicted probabilities to output_path_true using np.savetxt()
    h_theta = sigmoid(x @ theta)
    D = np.diag(h_theta * (1 - h_theta))
    return 1/m * x.T @ D @ x


def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    # Train a logistic regression classifier
    classifier = LogisticRegression()
    classifier.fit(x_train, y_train)
    print(x_train.shape, y_train.shape, classifier.theta)
    # Plot decision boundary on top of validation set
    util.plot(x_valid, y_valid, classifier.theta, f"{save_path}.pdf")

    # Use np.savetxt to save predictions on eval set to save_path
    # *** END CODE HERE ***
    predictions = classifier.predict(x_train)
    np.savetxt(f"{save_path}.txt", predictions)


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(
        self, step_size=0.01, max_iter=10000, eps=1e-5, theta_0=None, verbose=True
    ):
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
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        if self.theta is None:
            self.theta = np.zeros(x.shape[1])

        for i in range(self.max_iter):
            theta_prev = np.copy(self.theta)
            # Newton-Raphson method
            H_inv = np.linalg.inv(logistic_hesse(self.theta, x))
            self.theta += self.step_size * H_inv @ logistic_gradient(self.theta, x, y)
            if self.verbose:
                loss = logistic_cost(self.theta, x, y, self.eps)
                print(f"Loss after iteration {i}: {loss}")

            if np.sum(np.abs(self.theta - theta_prev)) < self.eps:
                break
            i += 1

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        pred = sigmoid(x @ self.theta)
        return pred


if __name__ == "__main__":
    main(
        train_path="ds1_train.csv",
        valid_path="ds1_valid.csv",
        save_path="logreg_pred_1",
    )

    main(
        train_path="ds2_train.csv",
        valid_path="ds2_valid.csv",
        save_path="logreg_pred_2",
    )
