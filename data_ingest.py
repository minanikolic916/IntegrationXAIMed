from utils import set_medmnist_init
import random
import torchvision.transforms as transforms
import torch.utils.data as data


#zelimo lokalno da sacuvamo podatke
download = True
BATCH_SIZE = 64


def transform_data():
    #posebne transformacije se koriste za treniranje, a posebne za testiranje
    data_transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(45),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    data_transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    return data_transform_train, data_transform_test

def load_data(data_flag:str):
    #informacije o klasi podataka koju koristimo
    data_flag, info, task, DataClass = set_medmnist_init(data_flag)

    #namestanje transformacija
    data_transform_train, data_transform_test = transform_data()

    #ucitavanje podataka
    train_dataset = DataClass(split='train', transform=data_transform_train, download=download, size = 224)
    test_dataset = DataClass(split='test', transform=data_transform_test, download=download, size = 224)

    #pregled ucitanih podataka za skup za treniranje i testiranje
    print(train_dataset)
    print(test_dataset)
    print(f"Number number of instances for training: {len(train_dataset)}\nNumber of instances for test:{len(test_dataset)}")

    return train_dataset, test_dataset

def set_data_loaders(data_flag:str):
    #enkapsulacija podataka u DataLoader
    train_dataset, test_dataset = load_data(data_flag)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False, pin_memory=True)
    return train_loader, test_loader

def load_random_image(test_loader):
    img = next(iter(test_loader))[0][0]
    return img
    
