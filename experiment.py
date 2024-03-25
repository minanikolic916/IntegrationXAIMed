from data_ingest import set_data_loaders, set_medmnist_init
from model import model_resnet_trans_learn, model_resnet18_trans_learn, print_model_size, save_model
from train import model_train
from evaluate import model_eval, visualize_eval_results

DATASET_NAME = "bloodmnist"
NUM_EPOCHS = 5
LR = 0.0001

#ucitavanje odgovarajuceg seta i postavljanje data loader-a
data_flag, info, task, DataClass = set_medmnist_init(DATASET_NAME)
NUM_CLASSES = len(info['label'])
CLASS_NAMES = info['label']
train_loader, test_loader = set_data_loaders(data_flag)

#odabir modela i treniranje 
model_resnet18 = model_resnet18_trans_learn('no_quantization')
model_resnet50 = model_resnet_trans_learn('no_quantization')
print_model_size(model_resnet50)
model_train(train_loader, model_resnet50, NUM_EPOCHS, LR)

#cuvanje modela
save_model(model_resnet50, "./models/resnet50-fine-tuned-v3.pth")

#evaluacija modela sa vizuelizacijom
#model_eval(test_loader, model_resnet18, NUM_CLASSES, average_type= None)
#model_eval(test_loader, model_resnet18, NUM_CLASSES, average_type= 'macro')

visualize_eval_results(test_loader, model_resnet50, NUM_CLASSES, CLASS_NAMES, avg_type = None)
visualize_eval_results(test_loader, model_resnet50, NUM_CLASSES, CLASS_NAMES, avg_type = 'macro')





   