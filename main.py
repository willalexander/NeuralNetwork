import numpy as np 
from copy import deepcopy
import matplotlib.pyplot as plt

N = 4
learning_rate = 0.0001

X = np.array([
    [2, 6],
    [4, 1],
    [3, 8],
    [7, 5]
])

Y = np.array([
    [8],
    [5],
    [11],
    [12]
])

X_aug = np.append(np.array([[1], [1], [1], [1]]), X, axis=1)


def activation(x):
    return np.maximum(x, 0)

def activation_1(x):
    return 0 if x <= 0 else 1

def cost(x, y):
    return pow(x - y, 2)

def compute_output(X, alpha, beta):
    N = X.shape[0]
    Yhat = np.zeros(N)
    for i in range(N):
        hidden_layer = np.matmul(alpha, X[i])
        hidden_layer_act = activation(hidden_layer)
        hidden_layer_aug = np.append(1, hidden_layer_act)
        Yhat[i] = np.matmul(np.transpose(beta), hidden_layer_aug)
    return Yhat

def compute_loss(Y, Yhat):
    #print('\n\n')
    #print(Y)
    Yhat2 = Yhat.reshape(1, 4).transpose()
    #print(Yhat2)
    #print(Y - Yhat2)
    #print(np.power(Y - Yhat2, 2.0))
    #print(np.sum(np.power(Y - Yhat2, 2.0)))
    return np.sum(np.power(Y - Yhat2, 2.0))



def compute_grad(alpha, beta, X, Y, Yhat):
    b_grad = np.zeros(5)
    for i in range(N):
        b_grad_i = np.zeros(5)
        for j in range(5):
            if j == 1:
                alphaj_Xi = 1.0
            else:
                alphaj_Xi = np.matmul(alpha[j - 1], X[i])
            b_grad_i[j] = 2 * (Yhat[i] - Y[i]) * activation(alphaj_Xi)
        b_grad += b_grad_i

    a_grad = np.zeros((4, 3))
    for i in range(N):
        a_grad_i = np.zeros((4, 3))
        for j in range(4):
            for k in range(3):
                a_grad_i[j][k] = 2 * (Yhat[i] - Y[i]) * beta[j] * activation_1(np.matmul(alpha[j], X[i])) * X[i][k]
        a_grad += a_grad_i

    return a_grad, b_grad

alpha = np.array([
    [-0.2373,  0.4209,  0.0018],
    [-0.0219, -0.0093, -0.3173],
    [0.2384, -0.0101,  0.3405],
    [-0.0094,  0.2039, -0.3998]
])

beta = np.array([
    [0.1246],
    [-0.1108],
    [0.3166],
    [-0.494],
    [0.3031]
])

for epoch in range(1000):
    print("\nEpoch {}".format(epoch))

    Y_hat = compute_output(X_aug, alpha, beta)
    #print("Yhat: {}".format(Y_hat))

    loss = compute_loss(Y, Y_hat)
    print("loss: {}".format(loss))

    a_grad, b_grad = compute_grad(alpha, beta, X_aug, Y, Y_hat)

    #print("a_grad: {} b_grad: {}".format(a_grad, b_grad))

    alpha -= learning_rate * a_grad
    beta -= learning_rate * b_grad.reshape(1,5).transpose()

    #print(Y_hat)


test_set = np.array([
    [7, 4],
    [2, 8],
    [1, 4],
    [8, 3],
])

test_set_aug = np.append(np.array([[1], [1], [1], [1]]), test_set, axis=1)
Yhat = compute_output(test_set_aug, alpha, beta)

'''
for i in range(4):
    print("{} + {} = {} (pred) {} (actual)".format(
        test_set[i][0],
        test_set[i][1],
        int(Yhat[i]),
        (test_set[i][0] + test_set[i][1])))
'''




#print(0.5 - np.random.random(size=12))

#print(X)
#print(X2)
#print(Y)