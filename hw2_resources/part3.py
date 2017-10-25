import numpy as np
import cvxopt as cv
from cvxopt import solvers
import pylab as pl

def svm_slack(kernel,X,y,C,eps):
	N = X.shape[0]
	nontrivial = []
	support = []

	PP = np.zeros([N,N])
	for i in range(N):
		for j in range(N):
			PP[i,j] = kernel(X[i],X[j])*y[i]*y[j]
	P = cv.matrix(PP)
	qq = -1*np.ones(N)	
	q = cv.matrix(qq)
	hh = np.zeros(2*N)
	for i in range(N,2*N):
		hh[i] = C
	h = cv.matrix(hh)
	GG = np.zeros([2*N,N])
	for i in range(N):
		GG[i,i]=-1
		GG[i+N,i]=1
	G = cv.matrix(GG)
	AA = np.zeros([1,N])
	for i in range(N):
		AA[0,i] = y[i]
	A = cv.matrix(AA)
	bb = np.zeros(1)
	b = cv.matrix(bb)
	solvers.options['show_progress']=False
	sol = solvers.qp(P,q,G,h,A,b)
	#ignore all the crap above

	alpha = np.array(sol['x'])
	for i in range(alpha.shape[0]):
		if(alpha[i]<eps):
			alpha[i]=0.0
		elif(alpha[i]>1-eps):
			nontrivial.append(i)
			alpha[i]=1.0
			#uncomment to print vectors over the margin
			#print("Over the margin! at", X[i,1:],y[i])
		else:
			nontrivial.append(i)
			support.append(i)
			#uncomment to print support vectors
			#print("Support Vector at ",X[i,1:],y[i])
	bias = 0
	wnorm = 0
	for i in nontrivial:
		for j in nontrivial:
			wnorm+=PP[i,j]*alpha[i]*alpha[j]
	margin = 1/math.sqrt(wnorm)
	print("Margin",margin)
	for n in support:
		bias+=y[n]
		for m in nontrivial:
			bias-=alpha[m]*y[m]*kernel(X[m],X[n])
	bias/=len(support)
	return alpha,nontrivial,bias
def svm_prediction(kernel, X,y,alpha,nontrivial,bias,x):

	val = 0
	for i in nontrivial:
		val+=alpha[i]*y[i]*kernel(x,X[i,:])
	return bias+val

