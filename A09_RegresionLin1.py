#===========Regresion Lineal Simple en Pytorch===========
# María Andrea Ramírez López
# Fundamentos de Inteligencia Artificial
# Matemática Algorítmica
# 8 de abril del 2025
#========================================================

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 0) Preparar datos
X_numpy,y_numpy=datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=4)

#enviar a tensor de precision sencilla
X=torch.from_numpy(X_numpy.astype(np.float32))
y=torch.from_numpy(y_numpy.astype(np.float32))
y=y.view(y.shape[0],1) #torch necesita salida del transpuesto
n_samples,n_features=X.shape

# 1) Modelo
# Modelo lineal f=wx+b
input_size=n_features
output_size=1
model=nn.Linear(input_size,output_size)

# 2) Error y optimizador
learning_rate=0.01
criterion=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate) 

# 3) Ciclo de aprendizaje
num_epochs=200
for epoch in range(num_epochs):
    #Evaluacion y error
    y_predicted=model(X)
    loss = criterion(y_predicted,y)
    #Gradiente y mejora de coeficientes
    loss.backward()
    optimizer.step()
    #resetear gradiente
    optimizer.zero_grad()
    if(epoch+1)%10==0:
        print(f'epoch: {epoch+1},loss={loss.item():.4f}')

#Grafica
predicted=model(X).detach().numpy()
plt.plot(X_numpy,y_numpy,'ro')
plt.plot(X_numpy,predicted,'b')
plt.show()