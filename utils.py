import torch
import medmnist
from medmnist import INFO

def set_medmnist_init(data_flag:str):
    #postavljanje odgovarajucih flegova za upotrebu BLOOD MNIST skupa podataka
    data_flag = data_flag
    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    #ukoliko je vrednost za broj kanala jedan, znaci da se radi sa crno-belim slikama
    #print(f"Info: {info} \nTask:{task}\nNumber of channels:{n_channels}\nNumber of classes:{n_classes}")
    DataClass = getattr(medmnist, info['python_class'])
    return [data_flag, info, task, DataClass]

def denormalize_image(image):
    MEAN = torch.tensor([0.5, 0.5, 0.5])
    STD = torch.tensor([0.5, 0.5, 0.5])
    denormalized_image = (image*torch.tensor(STD)) + torch.tensor(MEAN)
    return denormalized_image

def check_cuda():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    return device

