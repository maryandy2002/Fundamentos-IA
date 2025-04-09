#===========Diferenciacion automatica Autograd==========
# María Andrea Ramírez López
# Fundamentos de Inteligencia Artificial
# Matemática Algorítmica
# 1 de abril del 2025
#========================================================
import torch
#requires_grad=true genera funciones gradiente
#para las operaciones que se hacen con ese tensor

x=torch.rand(3,requires_grad=True)
y=x+2

#y=y(x) tiene un grad_fn asociado
print(x)
print(y)
print(y.grad_fn)

#z=z(y)=z(y(x))
z=y*y*3 #multiplica las entradas del tensor
print(z)
z=z.mean() #calcula el promedio de las entradas
print(z)

#calculo del gradiente con retropropagacion
z.backward()
print(x.grad) #dz/dx

#torch.autograd se basa en la regla de la cadena
x=torch.rand(3,requires_grad=True)
y=x*2
for i in range(10):
    y=y*2
print(y)
print(y.shape)

#evaluar "gradiente" dy/dx en v
v=torch.tensor([0.1,1.0,0.0001],dtype=torch.float32)
y.backward(v)
print(x.grad)

#decirle a un tensor que deje de generar gradientes
a=torch.randn(2,2)
print(a.requires_grad)
b=((a*3)/(a-1))
print(b.grad_fn)

#con gradiente
a.requires_grad_(True)
print(a.requires_grad)
b=(a*a).sum()
print(b.grad_fn)
x=torch.randn(3,requires_grad=True)
print(a.requires_grad)
a=torch.randn(2,2,requires_grad=True)
print(a.requires_grad)

#sin gradiente
b=a.detach()
print(b.requires_grad)

#con envoltura que le quita el gradiente
a=torch.randn(2,2,requires_grad=True)
print(a.requires_grad)
with torch.no_grad():
    print((x**2).requires_grad)

#backward() acumula el gradiente en .grad
#.zero_() limpia el gradiente antes de comenzar
weights=torch.ones(4,requires_grad=True)
print(weights)
#Epoch: paso e optimización
for epoch in range(3):
    #ejemplito
    model_output=(weights*3).sum()
    model_output.backward()
    print(weights.grad)
    #optimización:encontrar nuevos coeficeintes
    with torch.no_grad():
        weights-=0.1*weights.grad
    #reinicializa el gradiente a cero (importante)
    weights.grad.zero_()
model_output=(weights*3).sum()
print(weights)
print(model_output)

