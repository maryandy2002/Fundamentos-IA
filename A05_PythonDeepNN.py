#==================Red Neuronal Profunda=================
# María Andrea Ramírez López
# Fundamentos de Inteligencia Artificial
# Matemática Algorítmica
# 26 de marzo del 2025
#========================================================
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def init_params():
    W=[]
    b=[]
    mm=np.array([784,10,15,12])
    nn=np.array([10,15,12,10])
    capas=len(mm)
    print("Red de perceptrones")
    for i in range(capas):
        print(mm[i]," x ",nn[i])
        W.append(np.random.rand(nn[i],mm[i])-0.5)
        b.append(np.random.rand(nn[i],1)-0.5)
    return W,b

def ReLU(Z):
    return np.maximum(Z,0)

def softmax(Z):
    A=np.exp(Z)/sum(np.exp(Z))
    return A

def forward_prop(W,b,X):
    A=[]
    Z=[]
    AA=X
    A.append(AA)
    for i in range(len(W)):
        ZZ=W[i].dot(AA)+b[i]
        if i<len(W)-1:
            AA=ReLU(ZZ)
        if i==len(W)-1:
            AA=softmax(ZZ)
        A.append(AA)
        Z.append(ZZ)
    return Z,A

def ReLU_deriv(Z):
    return Z>0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size,Y.max()+1))
    one_hot_Y[np.arange(Y.size),Y]=1
    one_hot_Y=one_hot_Y.T
    return one_hot_Y

def backward_prop(Z,A,W,X,Y):
    dW=[]
    db=[]
    m=m_train
    n=len(W)-1
    one_hot_Y=one_hot(Y)
    dZ=A[n+1]-one_hot_Y
    dWW=1/m*dZ.dot(A[n].T)
    dbb=np.expand_dims(1/m*np.sum(dZ,axis=1),axis=1)
    db.append(dbb)
    dW.append(dWW)
    if n>1:
        for i in range(n-1,0,-1):
            dZ=W[i+1].T.dot(dZ)*ReLU_deriv(Z[i])
            dWW=1/m*dZ.dot(A[i].T)
            dbb=np.expand_dims(1/m*np.sum(dZ,axis=1),axis=1)
            db.append(dbb)
            dW.append(dWW)
    dZ=W[1].T.dot(dZ)*ReLU_deriv(Z[0])
    dWW=1/m*dZ.dot(X.T)
    dbb=np.expand_dims(1/m*np.sum(dZ,axis=1),axis=1)
    db.append(dbb)
    dW.append(dWW)
    db.reverse()
    dW.reverse()
    return dW,db

def update_params(W,b,dW,db,alpha):
    for i in range(len(W)):
        W[i]=W[i]-alpha*dW[i]
        b[i]=b[i]-alpha*db[i]
    return W,b

def get_predictions(A2):
    return np.argmax(A2,0)

def get_accuracy(predictions,Y):
    print(predictions,Y)
    return np.sum(predictions==Y)/Y.size

def gradient_descent(X,Y,alpha,iterations):
    W,b=init_params()
    for epoc in range(iterations):
        Z,A=forward_prop(W,b,X)
        dW,db=backward_prop(Z,A,W,X,Y)
        W,b=update_params(W,b,dW,db,alpha)
        if epoc%10==0:
            print("Iteracion: ",epoc)
            predictions=get_predictions(A[len(A)-1])
            print(get_accuracy(predictions,Y))
    return W,b

def make_predictions(X,W,b):
    Z,A=forward_prop(W,b,X)
    predictions=get_predictions(A[len(A)-1])
    return predictions

def test_prediction(index,W,b):
    current_image=X_train[:,index,None]
    prediction=make_predictions(X_train[:,index,None],W,b)
    label=Y_train[index]
    print("Prediction: ",prediction)
    print("Label: ",label)
    current_image=current_image.reshape((28,28))*255
    plt.gray()
    plt.imshow(current_image,interpolation='nearest')
    plt.show()
    
if __name__=="__main__":
    data=pd.read_csv('train.csv')
    data=np.array(data)
    m,n=data.shape
    print("Numero de imagenes = ",m)
    print("Numero de pixeles = ",n)
    np.random.shuffle(data)
        
    data_dev=data[0:1000].T
    Y_dev=data_dev[0]
    X_dev=data_dev[1:n]
    X_dev=X_dev/255
    data_train=data[1000:m].T
    Y_train=data_train[0]
    X_train=data_train[1:n]
    X_train=X_train/255
    _,m_train=X_train.shape
        
    W,b=gradient_descent(X_train,Y_train,0.15,1000)
        
    for i in range(20):
        test_prediction(i,W,b)
        