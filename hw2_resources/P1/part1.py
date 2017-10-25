from functools import partial
import math
import numpy as np
def logistic_regression(reg_lambda,X,y):#X,y are the coordinates in each row (including a first 1 for x)) 
	lambda_loss= partial(lr_point_loss,reg_lambda)
	lambda_gradient = partial(lr_point_gradient,reg_lambda)
	guess_w =np.zeros(X.shape[1])
	found_w=stochastic_descent(reg_lambda,lambda_loss,lambda_gradient,guess_w,X,y,0.01,1E-6)
	return found_w


def lr_point_loss(reg_lambda,w,x,y):
	return np.log(1+np.exp(-y*w.dot(x)))+reg_lambda*w[1:].dot(w[1:])

def lr_point_gradient(reg_lambda,w,x,y):
	z = np.zeros(w.shape[0])
	z[1:]=w[1:]
	return -y*(1-sigmoid(y*w.dot(x)))*x+2*reg_lambda*z


def lr_prediction(w,z):
	return sigmoid(w.dot(z))

def sigmoid(z):
	return 1/(1+math.exp(-z))


def stochastic_descent(reg_lambda,objective_f,gradient_f, guess, data, response, learning_rate=1E-1, threshold=1E-6):
	n_iter=1
	current_input = guess
	#current_value = objective_f(guess)
	gradient = 0
	avg_grad_norm=0
	avg_val=0
	#inputs = [current_input]
	#values = [current_value]
	index_array = [i for i in range(data.shape[0])]
	while(n_iter==1 or (avg_grad_norm/data.shape[0]>threshold and step_size>1E-6*learning_rate and n_iter<=1E3)):
		step_size = learning_rate*pow(n_iter,-0.7)
		if(n_iter%20==0): print("epoch ",n_iter,"gradient norm ",avg_grad_norm/data.shape[0])
		np.random.shuffle(index_array)
		avg_grad_norm=0
		avg_val = 0
		for k in index_array:

			last_input=current_input
			#last_value = current_value
			gradient = lr_point_gradient(reg_lambda,current_input,data[k],response[k])
			avg_grad_norm+=np.linalg.norm(gradient)
			avg_val+=objective_f(current_input,data[k],response[k])
			current_input = last_input - step_size*gradient
			#current_value = objective_f(guess)
		n_iter+=1
	return current_input

