import torch.nn as nn
from torchvision import models

class RNC(nn.Module): #Clase padre
    def __init__(self, cargar_peso=False): #Atributos
        super(RNC, self).__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512] #Configuración de front-end
        self.backend_feat  = [512, 512, 512,256,128,64] #Configuración de Back-end
        self.frontend = crear_capas(self.frontend_feat) #Crea capas
        self.backend = crear_capas(self.backend_feat,in_channels = 512,dilatacion = True)#Crea capas
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1) #Canal de Entrada, Canal de Salida, Tamaño de Kernel
        if not cargar_peso:
            mod = models.vgg16(pretrained = True) #Modelo VGG-16 
            for i in range(len(self.frontend.state_dict().items())):
                list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]          
def crear_capas(cfg, canal_ent = 3,lote=False,dilatacion = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    capas = []
    for v in cfg: #MaxPooling
        if v == 'M':
            capas += [nn.MaxPool2d(kernel_size=2, stride=2)] 
        else:
            conv2d = nn.Conv2d(canal_ent, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if lote:
                capas += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                capas += [conv2d, nn.ReLU(inplace=True)]
            canal_ent = v
    return nn.Sequential(*capas)                