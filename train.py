import torch
import torch.nn as nn
import torch.optim as optim
import time
from data_ingest import set_data_loaders
from utils import set_medmnist_init
from utils import check_cuda
from tqdm import tqdm 

#podesavanja parametara koji ce da se koriste prilikom procesa treniranja
def set_training_params(model, num_epochs, lr):
  num_epochs = num_epochs
  lr = lr
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
  return [num_epochs, lr, criterion, optimizer]


#definisanje funkcije za treniranje modela
def model_train(train_loader, model, num_epochs, lr):
  num_epochs, lr, criterion, optimizer = set_training_params(model, num_epochs, lr)
  device = check_cuda()
  
  since = time.time()
  for epoch in range(num_epochs):

      model.train()
      running_loss = 0.0
      for inputs, targets in tqdm(train_loader):
          optimizer.zero_grad()
          if device == "cuda":
            outputs = model(inputs.cuda())
            targets = targets.cuda()
          else:
            outputs = model(inputs)
            
          targets = targets.squeeze().long()
          loss = criterion(outputs, targets)
              
          loss.backward()
          optimizer.step()

          running_loss += loss.item()
          
      epoch_loss = running_loss/len(train_loader)    
      print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}')
  
  time_elapsed = time.time()-since
  print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))