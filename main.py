import numpy as np
from statistics import NormalDist
import math

# Learning Rate
learning_rate = 0.001

# Number of hidden units
K = 4

# Number of output units
O = 10

# Predictor variables
X = np.array([
    [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
])

# Response variables
Y = np.array([
    [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
])

# Number of observations
N = X.shape[0]

X_aug = np.append(np.ones((N, 1)), X, axis=1)


def activation(x):
    return np.maximum(x, 0)

def activation_1(x):
    return 0 if x <= 0 else 1

def varclamp(var):
    return np.maximum(var, 0.000001)

def varclamp_1(var):
    return 0 if var <= 0.000001 else 1



def cost(x, y):
    return pow(x - y, 2)

def compute_output(X_aug, alpha, beta):
    Yhat = np.zeros((N, O))
    for i in range(N):
        hidden_layer = np.matmul(alpha, X_aug[i])
        hidden_layer_act = activation(hidden_layer)
        hidden_layer_aug = np.append(1, hidden_layer_act)
        Yhat[i] = np.matmul(beta, hidden_layer_aug)
        Yhat[i][9] = varclamp(Yhat[i][9])
    return Yhat

def multivariate_gaussian_likelihood(X, mean, variance):
    result = 1
    for i in range(len(X)):
        result *= NormalDist(mu=mean[i], sigma=np.sqrt(variance)).pdf(X[i])
    return result

def compute_loss(Y, Yhat):
    result = 0
    for i in range(N):
        result += multivariate_gaussian_likelihood(Y[i], Yhat[i][0:O-1], Yhat[i][O-1])
    return -1 * result

def compute_grad(alpha, beta, X, Y, Yhat):
    i = 0
    H0 = np.matmul(alpha, X_aug[i])
    H = activation(H0)
    H_aug = np.append(1, H)

    variance = Yhat[i][9]
    input_length = X_aug.shape[1]

    b_grad = np.zeros((O, K + 1))
    for l in range(9):
        b_grad[l] = -1 * (1 / variance) * (Y[i][l] - Yhat[i][l]) * H_aug
    b_grad[9] = 1 * (9/2) * (1 / variance) * (Y[i][l] - Yhat[i][l]) * H_aug * varclamp_1(variance)

    a_grad = np.zeros((K, input_length))
    for j in range(K):
        for k in range(input_length):
            mean_component = 0
            for p in range(input_length - 1):
                mean_component += -1 * (1 / variance) * (Y[i][p] - Yhat[i][p]) * beta[p][j] * activation_1(H0[j]) * X_aug[i][k]
            a_grad[j][k] = 1 * (9 / 2) * (1 / variance) * beta[9, j] * activation_1(H0[j]) * X_aug[i][k] * varclamp_1(variance) + mean_component
    return a_grad, b_grad

alpha = np.random.random((K, X_aug.shape[1])) - 0.5
beta = np.random.random((O, K + 1)) - 0.5

np.set_printoptions(linewidth=2000)


print(Y[0])

print("\n")

for epoch in range(1000):
    Yhat = compute_output(X_aug, alpha, beta)
    Yhat[0][9] = 1.0
    loss = compute_loss(Y, Yhat)
    ideal_loss = compute_loss(Y, np.array([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0]]))

    
    print(Yhat[0][0:9])
    print(Yhat[0][9])
    print("Loss: {}. Ideal loss: {}".format(loss, ideal_loss))

    

    #Yhat = np.array([
    #    [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.00001]
    #])

    #loss = compute_loss(Y, Yhat)
    #print('Epoch #{}. Loss: {}'.format(epoch, loss))
    #if math.isnan(loss):
    #    break

    a_grad, b_grad = compute_grad(alpha, beta, X, Y, Yhat)
    alpha -= learning_rate * a_grad
    beta -= learning_rate * b_grad


    #Yhat = compute_output(X_aug, alpha, beta)
    #print(Yhat[0][0:9])

