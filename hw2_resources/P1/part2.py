import numpy as np
from sklearn.linear_model import LogisticRegression
def linear_logistic_regression(reg_lambda,X,y):#X,y is the data/response, noting X does not begin with a column of 1s
	clf = LogisticRegression(penalty='l1',C=1E7,fit_intercept=True, intercept_scaling=1000)
	clf.fit(X,y)
	#print('C:',1/reg_lambda)
	v = np.array(clf.coef_)
	#print('Coefficient of each feature:', clf.coef_)
	#print('Intercept',clf.intercept_)
	#print('Normalized: ',v/np.linalg.norm(v))
	#print('Training accuracy:', clf.score(X,y))
	#print('')
	return clf
	#sc = StandardScaler?
def sigmoid(z):
	return 1/(1+np.exp(-z))
def linear_lr_prediction(clf,x):
	#print("predicting probability for x",x)
	w = clf.coef_
	w0 = clf.intercept_
	return sigmoid(np.append(w0,w).dot(x))
def linear_lr_score(w,w0,X,y):
	score = 0
	for k in range(X.shape[0]):
		score+=np.log(1+np.exp(-y[k]*np.append(w0,w).dot(X[k,:])))
	return score/X.shape[0]
