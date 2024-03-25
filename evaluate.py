from data_ingest import set_data_loaders, set_medmnist_init
from utils import check_cuda
#odabir konkretnih metrika za multiklasnu klasifikaciju
from torchmetrics import Precision, Recall, Accuracy
import torch
import pandas as pd

def set_metrics(num_classes, average_type):
  mca_metric = Accuracy(
    task = "multiclass", num_classes= num_classes, average= average_type
    )
  precision_metric = Precision(
      task = "multiclass", num_classes = num_classes, average = average_type
    )
  recall_metric = Recall(
      task = "multiclass", num_classes = num_classes, average = average_type
    )
  return [mca_metric, precision_metric, recall_metric]


#definisanje funkcije za evaluaciju rezultata
def model_eval(test_loader, model, num_classes, average_type):
  device = check_cuda()
  #pribavljamo metrike
  mca_metric, precision_metric, recall_metric = set_metrics(num_classes, average_type)
  #inputi i model, kao i metrike moraju svi da budu na device-u
  model.to(device)
  mca_metric.to(device)
  precision_metric.to(device)
  recall_metric.to(device)
  model.eval()

  with torch.inference_mode():
      for inputs, targets in test_loader:
          inputs, targets = inputs.to(device), targets.to(device)
          targets = targets.squeeze(1)
          test_pred = model(inputs)
          #_, preds = loss_fn(test_pred, targets)
          _, preds = torch.max(test_pred, 1)
          mca_metric(preds, targets)
          precision_metric(preds, targets)
          recall_metric(preds, targets)

  accuracy = mca_metric.compute()
  precision = precision_metric.compute()
  recall = recall_metric.compute()
  return [accuracy.cpu(), precision.cpu(), recall.cpu()]

def visualize_eval_results(test_loader, model, num_classes, class_names, avg_type):
  accuracy, precision, recall = model_eval(test_loader, model, num_classes= num_classes, average_type = avg_type)
  if avg_type is None:
    data = {
            'Class names': class_names.values(),
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall':recall }
    eval_df = pd.DataFrame(data)
    eval_df.set_index('Class names', inplace=True)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # za pretty print
      print(eval_df)
  else:
    print(f"Accuracy: {accuracy:.5f} | Precision: {precision:.5f} | Recall: {recall:.5f}\n")
