#======================Torch Básico=====================
# María Andrea Ramírez López
# Fundamentos de Inteligencia Artificial
# Matemática Algorítmica
# 1 de abril del 2025
#========================================================

import torch

#escalar vacio (trae asura)
x=torch.empty(1)
print(x)

#vector en r3
x=torch.empty(3)
print(x)

#tensor en r2xr3
x=torch.empty(2,3)
print(x)   

#tensor en r2xr2xr3
x=torch.empty(2,2,3)
print(x)

#tensor de numeros aleatorios r5xr3
x=torch.rand(5,3)
print(x)

#tensor de r5xr3 lleno con ceros
x=torch.zeros(5,3)
print(x)

#checar tamaño
print(x.size)

#checar tipo de datos
print(x.type)

#especificando tipo de dato
x=torch.zeros(5,3,dtype=torch.float16)
print(x)
print(x.dtype)

#construir vector con datos
x=torch.tensor([5.5,3])
print(x.size())

#vector optimizable (variables del gradiente)
x=torch.tensor([5.5,3],requires_grad=True)

#suma de tensores
y=torch.rand(2,2)
x=torch.rand(2,2)
z=x+y
z=torch.add(x,y)
print(z)
y.add_(x)
print(y)

#resta de tensores
z=x-y
z=torch.sub(x,y)
print(z)

#multiplicacion
z=x*y
z=torch.mul(x,y)
print(z)

#division
z=x/y
z=torch.div(x,y)
print(z)

#rebanadas
x=torch.rand(5,3)
print(x)
print(x[:,0]) #todos los renglones, columna 0
print(x[1,:]) #renglon 1, todas las columnas
print(x[1,1]) #elemento [1,1]

#valor del elemento en (1,1)
print(x[1,1].item())

#cambiar forma con torch.view()
x=torch.rand(4,4)
y=x.view(16)
z=x.view(-1,8) #-1: se infiere de las otras dimensiones
print(x.size(),y.size(),z.size())

#convertir un tensor en arreglo y viceversa
a=torch.ones(5)
b=a.numpy()
print(b)
print(type(b))

#le suma 1 a todas las entradas
a.add_(1)
print(a)
print(b)

#de numpy a torch
import numpy as np
a=np.ones(5)
b=torch.from_numpy(a)
print(a)
print(b)

#Le suma 1 a todas las entradas de a 
a+=1
print(a)
print(b)

#De CPU a GPU (si hay cuda)
if torch.cuda.is_available():
    device=torch.device("cuda")
    print("Tengo GPU"+str(device))
    y_d=torch.ones_like(x,device=device)
    x_d=x.to(device)
    z_d=x_d+y_d
    
    #de vuelta al CPU
    z=z_d.to("cpu")
    z=z.numpy()
    print(z)