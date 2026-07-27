import pandas as pd
import numpy as np

def sigmoid(z):
    sig = 1/(1 + np.exp(-z))
    return sig
def cost(x, y, w, b):
    cost = 0
    m = x.shape[0]
    err = sigmoid(np.dot(x,w)+b)
    cost = -np.sum(y * np.log(err) + (1 - y) * np.log(1 - err)) / m
    return cost
def grad (x, y, w, b):
    m = x.shape[0]
    err = sigmoid(np.dot(x,w)+b) - y
    dj_dw = (np.dot(x.T,err)) / m
    dj_db = np.sum(err)/m
    return dj_dw, dj_db
def grad_descent(x, y, w, b, alpha, nb_iter):
    J_history = []
    for i in range (nb_iter):
        dj_dw, dj_db = grad(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        J_history.append(cost(x, y, w, b))
        if (i+1) % 1000 == 0:
            print(f"Iteration {i+1}, cost {J_history[-1]:2f}")
    return w,b,J_history
def main():
    df = pd.read_csv("students.csv")
    x_raw = df[['hours_studied', 'sleep_hours', 'practice_tests']].values
    mu = x_raw.mean(axis=0)
    sigma = x_raw.std(axis=0)
    x = (x_raw - mu) / sigma
    y = df['passed'].values
    alpha = 1e-2
    nb = 10000
    m = x.shape[1]
    w_in = np.zeros(m)
    b_in = 0
    w, b, J = grad_descent(x, y, w_in, b_in, alpha, nb)
    print(f"w = {w}, b = {b}")
    x_pred = np.zeros(m)
    x_pred[0] = float(input('How many hours studied? '))
    x_pred[1] = float(input('How many hours slept? '))
    x_pred[2] = float(input('How many practice tests? '))
    x_pred = (x_pred - mu) / sigma
    y_pred = sigmoid(np.dot(x_pred, w) + b)
    pass_percent = y_pred * 100
    fail_percent = (1 - y_pred) * 100
    print(f"Chance to pass: {pass_percent:.2f}%, chance to fail: {fail_percent:.2f}%")


if __name__ == "__main__":
    main()
