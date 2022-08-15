# Important note: you do not have to modify this file for your homework.

import util
import numpy as np
import matplotlib.pyplot as plt


def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    count, _ = X.shape

    probs = 1. / (1 + np.exp(-X.dot(theta)))
    grad = (Y - probs).dot(X)

    return grad


def logistic_regression(X, Y, return_grad=True):
    """Train a logistic regression model."""
    theta = np.zeros(X.shape[1])
    learning_rate = 0.1
    grads = []
    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta + learning_rate * grad
        if return_grad and i % 100 == 0:
            grads.append(np.linalg.norm(grad))
        if i % 10000 == 0:
            print('Finished %d iterations' % i)
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break
        if i > 1e6 and return_grad:
            break
    if return_grad:
        return grads
    else:
        return


def plot_grads(grads, dataset):
    fig, ax = plt.subplots()
    ax.plot(grads)
    ax.semilogy()
    ax.set(xlabel="Iterations / 10000", ylabel=r"$||\nabla\theta||^2$",
           title=f"Gradient magnitude for dataset {dataset})")
    plt.savefig(f"gradients_{dataset}.pdf")

def main():
    print('==== Training model on data set A ====')
    Xa, Ya = util.load_csv('ds1_a.csv', add_intercept=True)
    fig, ax = plt.subplots()
    util.plot_points(Xa[:, 1:], Ya, ax)
    plt.savefig("ds1_b.pdf")
    grads_a = logistic_regression(Xa, Ya)
    plot_grads(grads_a, "a")

    print('\n==== Training model on data set B ====')
    Xb, Yb = util.load_csv('ds1_b.csv', add_intercept=True)
    fig, ax = plt.subplots()

    util.plot_points(Xb[:, 1:], Yb, ax)
    plt.savefig("ds1_a.pdf")
    grads_b = logistic_regression(Xb, Yb)
    plot_grads(grads_b, "b")

if __name__ == '__main__':
    main()
