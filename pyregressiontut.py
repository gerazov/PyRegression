#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear and two-layer regression in Scikit-learn, Theano and PyTorch.
Based on various tutorials including:

[1] "Linear Regression Example" by Jaques Grobler
http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py

[2] "A Real Example: Logistic Regression" in Theano Tutorial
http://deeplearning.net/software/theano/tutorial/examples.html

[3] "Learning PyTorch with Examples" by Justin Johnson, 
http://pytorch.org/tutorials/beginner/pytorch_with_examples.html

[4] PyTorch-Tutorial : 301_regression by Morvan Zhou  (cool plotting take from here)
https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/301_regression.py

@authors:
    Branislav Gerazov Jan 2018

Copyleft 2018 by Branislav Gerazov.

See the file LICENSE for the licence associated with this software.
"""

import numpy as np
import matplotlib.pyplot as plt

#%% toy data
def f(x):
    return x**3 + 2*x**2

x_data = np.linspace(-4, 2, 200)  # astype(float32)
y_data = f(x_data) + 5*np.random.randn(x_data.size)
x_data = x_data[:, None]  # make 2D
y_data  = y_data[:, None]

#%% plot data
plt.figure()
plt.scatter(x_data, y_data, alpha=.5)
plt.plot(x_data, f(x_data), lw=5, alpha=.8, label='ground truth')
plt.grid()
plt.legend()

#% ============================================================================
#%% Scikit-learn
#% =============================================================================
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

#%% sklearn linear regression - actually a wrapper for least squares
skreg = LinearRegression()
skreg.fit(x_data, y_data)

#%% predict
predict = skreg.predict(x_data)
mse = np.mean((predict - y_data)**2)
print('scikit-learn linear mse:', mse)

#%% plot
plt.plot(x_data, predict, lw=5, alpha=.8, label='sklearn linear')
plt.legend()

#%% sklearn mlp regression (2 layers)
n_hidden = 10  # neurons in hidden layer
learn = .5  # learning rate
epochs = 50
l2_reg = 0.001  # L2 regularization
mlpreg = MLPRegressor(hidden_layer_sizes=(n_hidden), activation='logistic', solver='adam', alpha=l2_reg, learning_rate_init=learn, max_iter=epochs, shuffle=False, verbose=True)
mlpreg.fit(x_data, y_data.ravel())

#%% predict
predict = mlpreg.predict(x_data)
mse = np.mean((predict - y_data.ravel())**2)
print('scikit-learn mlp mse:', mse)

#%% plot
plt.plot(x_data, predict, lw=5, alpha=.8, label='sklearn mlp')
plt.legend()

#%% add title and save
plt.title('Scikit-learn regression results')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('sklearn-regression.png', dpi='figure')

#% =============================================================================
#%% Theano
#% =============================================================================
import theano as th
import theano.tensor as T

#%% cast data to float32 so it can go on the GPU
x_data = x_data.astype('float32')
y_data = y_data.ravel().astype('float32')

#%% replot data
plt.figure()
plt.scatter(x_data, y_data, alpha=.5)
plt.plot(x_data, f(x_data), lw=5, alpha=.8, label='ground truth')
plt.grid()
plt.legend()

#%% init symbolic placeholders
x = T.matrix('x')
y = T.vector('y')

#%% init model parameters
# floatX = float32 for possibility putting data in the GPU
wy = th.shared(np.random.randn(1).astype(th.config.floatX), name='wy')  
by = th.shared(np.asarray(0, dtype=th.config.floatX), name='by')

#%% init graph
prediction = T.dot(x, wy) + by
reg_l2 = 0.001
L2 = reg_l2 * (wy ** 2).sum()
loss = T.mean((prediction - y)**2) + L2
grad_wy, grad_by = T.grad(loss, [wy, by])

#%% compile graph
learn = .001
train = th.function(inputs=[x, y], 
                    outputs=[prediction, loss],
                    updates=((wy, wy - learn * grad_wy), 
                             (by, by - learn * grad_by)))

predict = th.function(inputs=[x], outputs=prediction)

#%% visualize compiled graph
#th.printing.pydotprint(train, 'theanologreg-train.png')
th.printing.pydotprint(predict, 'theanologreg-predict.png')

#%% train
epochs = 200
for t in range(epochs):
    pred, err = train(x_data, y_data)
    print(t, err)


#%% predict
prediction = predict(x_data)
mse = np.mean((prediction - y_data)**2)
print(mse)
print('theano linear mse:', mse)

#%% plot prediction
plt.plot(x_data, prediction, lw=5, alpha=.8, label='theano linear')
plt.legend()

#%% Theano 2 layer mlp
wh = th.shared(np.random.randn(1, 10).astype(th.config.floatX), name='wh')
wy = th.shared(np.random.randn(10).astype(th.config.floatX), name='wy')
bh = th.shared(np.zeros(10, dtype=th.config.floatX), name='bh')
by = th.shared(np.asarray(0, dtype=th.config.floatX), name='by')

h = T.dot(x, wh) + bh
a = 1 / (1 + T.exp(-h))
prediction = T.dot(a, wy) + by

reg_l2 = 0.001
L2 = reg_l2 * ((wh ** 2).sum() + (wy ** 2).sum())
loss = T.mean((prediction - y)**2) + L2
params = [wh, bh, wy, by]
grads = T.grad(loss, params)

#%% compile graph
learn = .5
updates = [(par, par - learn * grad) for par, grad in zip(params, grads)]
train = th.function(inputs=[x, y], 
                    outputs=[prediction, loss],
                    updates=updates)

predict = th.function(inputs=[x], outputs=prediction)

#%% train (maybe need to run several times)
epochs = 50
for t in range(epochs):
    pred, err = train(x_data, y_data)
    print(t, err)

#%% predict and plot
prediction = predict(x_data)
mse = np.mean((prediction - y_data)**2)
print(mse)
print('theano mlp mse:', mse)

plt.plot(x_data, prediction, lw=5, alpha=.8, label='theano mlp')
plt.legend()

#%% add title and save
plt.title('Theano regression results')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('theano-regression.png', dpi='figure')

#% =============================================================================
#%% PyTorch
#% =============================================================================
import torch
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F

#%% load data into torch Tensors and then Variables
dtype = torch.FloatTensor
    
x = torch.from_numpy(x_data).type(dtype)
y = torch.from_numpy(y_data).type(dtype)
y = torch.unsqueeze(y, dim=1)  # make 2D

x, y = Variable(x), Variable(y)

#%% init model parameters
wy = Variable(torch.randn(1, 1), requires_grad=True)
by = Variable(torch.zeros(1), requires_grad=True)

#%% train linear model
learn = .05
l2_reg = 0.001

plt.figure()
plt.ion()  # interactive plotting 

for t in range(10):
    # run forward
    prediction = x.mm(wy) + by  # can also be set as F.linear(x, wy, by)
    L2 = l2_reg * (wy**2).sum()
    loss = torch.mean((prediction - y)**2) + L2     
    
    # run backward to compute gradients
    loss.backward()  
    
    # apply learning
    wy.data = wy.data - learn * wy.grad.data
    by.data = by.data - learn * by.grad.data
    
    # null gradients because they accumulate
    wy.grad.data.zero_()
    by.grad.data.zero_()

    # fancy plotting from [4]
    print(t, loss.data[0])
    # plot and show learning process
    plt.cla()
    plt.scatter(x_data, y_data, alpha=.5)
    plt.plot(x_data, f(x_data), lw=5, alpha=.8, label='ground truth')
    plt.grid()
    plt.plot(x_data, prediction.data.numpy(), lw=5, alpha=0.8, label='pytorch linear')
    plt.legend()
    plt.title('Loss = {:2f}'.format(loss.data[0]))
    plt.pause(0.1)

plt.ioff()

#%% init 2-layer model parameters
wh = Variable(torch.randn(1, 10), requires_grad=True) 
bh = Variable(torch.zeros(10), requires_grad=True)
wy = Variable(torch.randn(10, 1), requires_grad=True)
by = Variable(torch.zeros(1), requires_grad=True)
params = [wh, bh, wy, by]

#%% train mlp
plt.ion()   # interactive plotting
learn = .05
l2_reg = 0.001
for t in range(500):
    h = x.mm(wh) + bh
    a = 1 / (1 + torch.exp(-h))  # can use F.logsigmoid(h)
    prediction_mlp = a.mm(wy) + by
    L2 = l2_reg * ((wh**2).sum() + (wy**2).sum())
    loss = torch.mean((prediction_mlp - y)**2)  + L2
    loss.backward() 
    
    for param in params:
        param.data = param.data - learn * param.grad.data
        param.grad.data.zero_()

    if t % 10 == 0:
        print(t, loss.data[0])
        # plot and show learning process
        plt.cla()
        plt.scatter(x_data, y_data, alpha=.5)
        plt.plot(x_data, f(x_data), lw=5, alpha=.8, label='ground truth')
        plt.grid()
        plt.plot(x_data, prediction.data.numpy(), lw=5, alpha=0.8, label='pytorch linear')
        plt.plot(x_data, prediction_mlp.data.numpy(), lw=5, alpha=0.8, label='pytorch mlp')
        plt.legend()
        plt.title('Loss = {:2f}'.format(loss.data[0]))
        plt.pause(0.1)

plt.ioff()

#%% add title and save
plt.title('PyTorch regression results')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('pytorch-regression.png', dpi='figure')

# =============================================================================
#%% full PyTorch bells and whistles
# =============================================================================

# to put your data on the GPU
use_cuda = torch.cuda.is_available()
if use_cuda:
    dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU
else:
    dtype = torch.FloatTensor
    
x = torch.from_numpy(x_data).type(dtype)
y = torch.from_numpy(y_data).type(dtype)
y = torch.unsqueeze(y, dim=1)  # make 2D

x, y = Variable(x), Variable(y)

#%% init model class (you can use equations here too)
class torch_regressor(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(torch_regressor, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)  # output layer

    def forward(self, x):
        x = F.sigmoid(self.hidden(x))  # activation function for hidden layer
        x = self.predict(x)  # linear output
        return x

#%% init model    
reg = torch_regressor(n_feature=1, n_hidden=10, n_output=1)  # define the network
print(reg)  # net architecture
if use_cuda:
    reg.cuda()  # must be before init of optimizer

optimizer = torch.optim.Adam(reg.parameters(), lr=learn)
loss_func = torch.nn.MSELoss()

#%% train 
plt.figure()
plt.ion()   # interactive plotting
for t in range(500):
    # forward pass
    prediction_adam = reg(x)
    loss = loss_func(prediction_adam, y)

    # backward pass and learning
    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

    if t % 10 == 0:
        print(t, loss.data[0])
        # plot and show learning process
        plt.cla()
        plt.scatter(x_data, y_data, alpha=.5)
        plt.plot(x_data, f(x_data), lw=5, alpha=.8, label='ground truth')
        plt.grid()
        plt.plot(x_data, prediction.data.numpy(), lw=5, alpha=0.8, label='pytorch linear')
        plt.plot(x_data, prediction_mlp.data.numpy(), lw=5, alpha=0.8, label='pytorch mlp')
        if use_cuda:
            plt.plot(x_data, prediction_adam.cpu().data.numpy(), lw=5, alpha=0.8, label='pytorch adam')
        plt.legend()
        plt.title('Loss = {:2f}'.format(loss.data[0]))
        plt.pause(0.1)

plt.ioff()

#%% add title and save
plt.title('PyTorch regression results')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('pytorch-regression-adam.png', dpi='figure')


# =============================================================================
#%% bonus PyTorch for GPU matrix computations
# =============================================================================
import torch
from datetime import datetime

#dtype = torch.FloatTensor
dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

M, N = 10000, 10000

# Create random inputs 
x = torch.rand((M,N)).type(dtype) 
y = torch.rand(x.size()).type(dtype)   

#%% calculate z = x * y five times and time it
start_time = datetime.now()  # start stopwatch
for i in range(5):
    print('\r{}/5'.format(i), end='')
    z = x.mm(y)

# print time    
time = datetime.now() - start_time
print('\nfinished in {}'.format(time))

# I got 25.366270 for CPU and 0.001644 for GPU -> 15500x faster : )