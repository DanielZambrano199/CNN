# coding=utf-8
import torch.nn as nn
from torchvision import models
import torch
#Configuración de la capa de Frontal
conv1_1 = 64
conv1_2 = 64
m_pool_1 = 'M'
conv2_1 = 128
conv2_2 = 128
m_pool_2 = 'M'
conv3_1 = 256
conv3_2 = 256
conv3_3 = 256
m_pool_3 = 'M'
conv4_1 = 512
conv4_2 = 512
conv4_3 = 512
t_kernel_front = 3

#Configuración de la capa de Fondo
convA_1 = 512
convA_2 = 512
convA_3 = 512
convA_4 = 256
convA_5 = 128
convA_6 = 64
canal_entrada = 512
ConvDila = True

#Configuración de la capa de Salida
t_kernel_cs = 1 #Tamaño del Kernel en la capa de salida
entrada_CS = 64
salida_CS = 1

t_kernel_mp = 2 #Tamaño del Kernel en el maxpooling
stride_mp=2

class CNN(nn.Module):
    def __init__(self, cargar_pesos=False):
        super(CNN, self).__init__()
        self.seen = 0

        self.conf_frontal = [conv1_1, conv1_2, m_pool_1, conv2_1, conv2_2, m_pool_2, conv3_1,
                              conv3_2, conv3_3, m_pool_3, conv4_1, conv4_2, conv4_3]
        
        self.conf_fondo  = [convA_1, convA_2, convA_3, convA_4, convA_5, convA_6]
        
        self.frontal = crear_capa(self.conf_frontal)
        
        self.fondo = crear_capa(self.conf_fondo,Ent_Canales = canal_entrada,dilation = ConvDila)
        
        self.capa_salida = nn.Conv2d(entrada_CS, salida_CS, kernel_size=t_kernel_cs)
        
        if not cargar_pesos:
            Vgg16mod = models.vgg16(pretrained = True)
            self._init_pesos()
            for i in range(len(self.frontal.state_dict().items())):
                list(self.frontal.state_dict().items())[i][1].data[:] = list(Vgg16mod.state_dict().items())[i][1].data[:]
                
    def forward(self,x):
        x = self.frontal(x)
        x = self.fondo(x)
        x = self.capa_salida(x)
        return x
    
    def _init_pesos(self):
        for j in self.modules():
            if isinstance(j, nn.Conv2d):
                nn.init.normal_(j.weight, std=0.01)
                if j.bias is not None:
                    nn.init.constant_(j.bias, 0)
            elif isinstance(j, nn.BatchNorm2d):
                nn.init.constant_(j.weight, 1)
                nn.init.constant_(j.bias, 0)
                
                
def crear_capa(config, Ent_Canales = 3,lote=False,dilation = False):
    if dilation:
        dilatacion = 2
        pad = 2
    else:
        dilatacion = 1
        pad = 1
    capas = []
    for y in config:
        if y == 'M':
            capas += [nn.MaxPool2d(kernel_size=t_kernel_mp, stride=stride_mp)]
        else:
            convolucion2d = nn.Conv2d(Ent_Canales, y, kernel_size=t_kernel_front, padding=pad, dilation = dilatacion)
            if lote:
                capas += [convolucion2d, nn.BatchNorm2d(y), nn.ReLU(inplace=True)]
            else:
                capas += [convolucion2d, nn.ReLU(inplace=True)]
            Ent_Canales = y
    return nn.Sequential(*capas)

if __name__ == "__main__":
    cnn = CNN()         
