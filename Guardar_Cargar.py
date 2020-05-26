import h5py
import torch
#import shutil

def Guardar_red(archnomb, red):
    with h5py.File(archnomb, 'w') as h5f:
        for k, v in red.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())
def Cargar_red(archnomb, red):
    with h5py.File(archnomb, 'r') as h5f:
        for k, v in red.state_dict().items():        
            param = torch.from_numpy(np.asarray(h5f[k]))         
            v.copy_(param)
            
#def Guardar_PuntoDeControl(estado, modeloOptimo, ident_tarea, archivo='punto_de_control.pth.tar'):
#    torch.save(estado, ident_tarea+archivo)
#    if modeloOptimo:
#        shutil.copyfile(ident_tarea+archivo, ident_tarea+'modelo.pth.tar')            