from functools import partial
import matplotlib
matplotlib.use('Agg')
from numpy import *
from plotBoundary import *
from part3 import *
import pylab as pl


def linear_kernel(s,t):
	return s.dot(t)
def rbf_kernel(sigma,s,t):
	return exp(-(s-t).dot(s-t)/(2*sigma*sigma))

def train_and_validate(name,kernel,C,eps=1E-4):
	name = '1'
	print('======Training======')
	train = loadtxt('data/data'+name+'_train.csv')
	X = train[:, 0:2].copy()
	Y = train[:, 2:3].copy()
	alpha,nontrivial,bias = svm_slack(kernel,X,Y,C,eps)
	predictSVM = partial(svm_prediction,kernel,X,Y,alpha,nontrivial,bias)
	count = 0
	for i in range(X.shape[0]):
		if(Y[i]*predictSVM(X[i,:])>0):
			count+=1
	print("Training accuracy",count/X.shape[0])
	


	#UNCOMMENT FOR FIGURE
	#pl.figure(1)
	#plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Train')
	#pl.savefig('train_svm' + name + '.png',dpi=150)




	print('======Validation======')
	# load data from csv files
	validate = loadtxt('data/data'+name+'_validate.csv')
	X = validate[:, 0:2].copy()
	Y = validate[:, 2:3].copy()
	count = 0
	for i in range(X.shape[0]):
		if(Y[i]*predictSVM(X[i,:])>0):
			count+=1
	print("Validation accuracy",count/X.shape[0])


	#UNCOMMENT FOR FIGURE
	#pl.figure(2)
	#plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Validate')
	#pl.savefig('validate_svm' + name +'.png',dpi=150)

	return
train_and_validate('4',partial(rbf_kernel,1.0),0.1,1E-4)
for name in ['1','2','3','4']:
	break
	for kernel in [linear_kernel,partial(rbf_kernel,1.0)]:
		for C in [0.01,0.1,1,10,100]:
			print(name,kernel,str(C))
			train_and_validate(name,kernel,C,1E-4)
