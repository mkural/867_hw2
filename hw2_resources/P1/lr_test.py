from functools import partial 
import matplotlib
matplotlib.use('Agg')
from numpy import *
from plotBoundary import *
from part1 import *
from part2 import *
import pylab as pl
# import your LR training code

# parameters
name = '3'
print('======Training======')
# load data from csv files
train = loadtxt('data/data'+name+'_train.csv')

Z = train[:,0:2]
Y = train[:,2:3]
X = ones([Z.shape[0],Z.shape[1]+1])
X[:,1:]=Z


# Carry out training.
reg_lambda = 0.001
quadratic_w = logistic_regression(reg_lambda,X,Y)
clf = linear_logistic_regression(reg_lambda,Z,Y)
linear_w,linear_w0 = clf.coef_,clf.intercept_
### TODO ###
predictLR = partial(lr_prediction,quadratic_w)
#predictLR = partial(linear_lr_prediction,clf)
print(predictLR(np.array([1,2,3])))
# Define the predictLR(x) function, which uses trained parameters
### TODO ###

# plot training results
pl.figure(1)
plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Train')
pl.savefig('p' +name + '_train_decision_boundary.png',dpi=150)
#print('Training Score',linear_lr_score(linear_w,linear_w0,X,Y))
print('Training Accuracy', clf.score(Z,Y))
print('Training Score',linear_lr_score(quadratic_w[1:],quadratic_w[:1],X,Y))

print('======Validation======')
# load data from csv files
validate = loadtxt('data/data'+name+'_validate.csv')
Z = validate[:,0:2]
Y = validate[:,2:3]
X = ones([Z.shape[0],Z.shape[1]+1])
X[:,1:]=Z

# plot validation results
pl.figure(2)
plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Validate')
pl.savefig('p' + name + '_validate_decision_boundary.png',dpi=150)
#print('Validation Score',linear_lr_score(linear_w,linear_w0,X,Y))
print('Validation Score',linear_lr_score(quadratic_w[1:],quadratic_w[:1],X,Y))
print('Validation Accuracy', clf.score(Z,Y))