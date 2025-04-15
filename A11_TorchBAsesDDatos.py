#==============Manejo de Datos en Pytorch==============


# María Andrea Ramírez López
# Fundamentos de Inteligencia Artificial
# Matemática Algorítmica
# 15 de abril del 2025
#======================================================

#Modulos necesarios
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math


#Bigdata debe dividirse en pequeños grupos de datos

#Ciclo de entrenamiento
#for epoch in range(num_epochs):
#   ciclo sobre todos los grupos de datos
#   for i in range(total_batches)

#epoch = una evaluacion a rotroprograpagacion para todo los datos de entrenamiento
#total_batches = numero total de subconjuntos de datos
#batch_size = numero de datos de entrenamiento en cada subconjunto
#number of iterations = numero de iteraciones sobre todos los datos de entrenamiento
#e.g : 100 samples, batch_size=20 -> 100/20=5 iterations for 1 epoch

#DataLoader puede dividir los datos en grupos

#Implementacion de datos tipica
# implement __init__, __getitem__, and __len__

#Hijos de Dataset
class WineDataset(Dataset):
    
    def __init__(self):
        #Inicializar, bajar datos, etc.
        #lectura con numpy o pandas
        #----------------
        #tipicos datos separados por coma
        #delimiter= simbolo delimitador
        #skiprows = lineas de encabezado
        xy = np.loadtxt('wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]
        
        #primera columna es etiqueta de clase y el resto son caracteristicas
        self.x_data=torch.from_numpy(xy[:,1:]) #grupos del 1 en adelante
        self.y_data=torch.from_numpy(xy[:,[0]]) #grupos 0
        
    #permitir indexacion para obtener el dato i de dataser[i]
    #metodo getter
    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]
    
    #len(dataset) es el tamaño de la base de datos
    def __len__(self):
        return self.n_samples
    
#instanciar base de datos
dataset=WineDataset()

#leer caracteristicas del primer dato
first_data = dataset[0]
features, labels =first_data
print(features,labels)

#Cargar toda la base con DataLoader
#reborujar los datos (shuffle): bueno para el entrenamiento
#num_workers: carga rápida utilizando multiples procesos
#Si comete un erro en la carga, poner num_workers=0
train_loader=DataLoader(dataset=dataset, #base de datos
                        batch_size=4,    #cuatro grupos
                        shuffle=True,    #reborujados
                        num_workers=2)   #dos procesos

#convertir en iterador y observar un dato al azar
dataiter = iter(train_loader)
data = next(dataiter)
features,labels=data

#ciclo de aprendizaje vacio
num_epochs=2
total_samples=len(dataset)
n_iterations = math.ceil(total_samples/4)
print(total_samples, n_iterations)
for epoch in range(num_epochs):
    for i, (inputs,labels) in enumerate(train_loader):
        #178 lineas, batch_size=4, n_iters=178/4=44.5 -> 45iteraciones
        #corre tu proceso de aprendizaje
        #Diagnostico
        if (i+1)%5 == 0:
            print(f'Epoch: {epoch+1}/{num_epochs}, Step {i+1}/{n_iterations}| Inputs {inputs.shape} | Labels {labels.shape}')
            
#algunas bases de datos existen en torchvision.datasets
#e.g MINST, Fashion-MNSIT, CIFAR10, COCO
train_dataset=torchvision.datasets.MNIST(root='./data',train=True,transform=torchvision.transforms.ToTensor(),download=True)
train_loader=DataLoader(dataset=train_dataset,batch_size=3,shuffle=True)

#look at one random sample
dataiter=iter(train_loader)
data = next(dataiter)
inputs, targets=data
print(inputs.shape,targets.shape)
        