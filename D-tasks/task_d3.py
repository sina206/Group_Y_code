from backbone import new_backbone
from dataset import CIFAR100Dataset
from torch.utils.data import DataLoader,random_split
import torch.nn as nn
import torch,os
from torchvision import transforms
import matplotlib.pyplot as plt
import random 
import numpy as np
from torchvision.transforms import v2
import torch.nn.functional as F
from task_d1 import set_seed

class PreprocessBackbone(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        ##Here we apply the normalisations 
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = v2.functional.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return self.backbone(x)

class CoarseAndFineClassifer(nn.Module):
  def __init__(self):
        super().__init__()
        self.backbone = PreprocessBackbone(new_backbone())
        self.classifier_coarse = nn.Sequential( 
          nn.Linear(576, 1024), 
          nn.ReLU(), 
          nn.Linear(1024, 1024), 
          nn.ReLU(),
          nn.Linear(1024, 20))

        self.classifier_fine = nn.Sequential(
            nn.Linear(576, 512),
            nn.ReLU(),
            nn.Linear(512, 100)
        )

  def forward(self, x):
         
    emb = self.backbone(x)
    return self.classifier_fine(emb), self.classifier_coarse(emb)
  


def prepare_test():
    # TODO: Create an instance of your model here. Your model must take in input a tensor of shape
    #  (B, 3, 32, 32), where B >= 2, and output two tensors: the first of shape (B, 100), with the second of shape
    #  (B, 20). B is the batch size and 100/20 is the number of fine/coarse classes.
    #  The output is the prediction of your classifier, providing two scores for both fine and coarse classes,
    #  for each image in input

    model = CoarseAndFineClassifer()  # TODO change this to your model

    # do not edit from here downwards
    weights_path = 'models/d3.pth'
    print(f'Loading weights from {weights_path}')

    map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=map_location))

    return model

def train_multi_head_classifier(train_loader, model, loss_function_coarse, optimiser, device,loss_function_fine):
    model.train()
    correct=0
    total=0
    total_loss = 0

    for image,fine_label,coarse_label in train_loader:
        image,fine_label, coarse_label = image.to(device), fine_label.to(device),coarse_label.to(device)

        ##Here we run input x through the model to get raw class scores
        pred_fine,pred_coarse = model(image)

        ##Here we compute the loss between the raw class scores and the ground truth
        loss_coarse = loss_function_coarse(pred_coarse, coarse_label)
        loss_fine = loss_function_fine(pred_fine, fine_label)
        loss = 0.5*loss_coarse + 0.5*loss_fine

        #Backprop
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        total_loss += loss.item()

        correct += (
            (pred_coarse.argmax(dim=1) == coarse_label) &
            (pred_fine.argmax(dim=1) == fine_label)
        ).sum().item()       
        total += coarse_label.size(0)


    train_acc= correct/total 
    total_loss= total_loss/len(train_loader)
    print("loss",total_loss)
    print("accuracy",train_acc)

    return total_loss,train_acc

def validate_multi_head_classifier(test_loader, model,loss_function_coarse, device,best_acc,save_path,loss_function_fine):
    model.eval()
    correct_fine = 0
    correct_coarse =0
    overall_correct = 0
    total = 0
    total_loss=0 

    ##Here we loop over the test data
    with torch.no_grad():
        for image, fine_label, coarse_label in test_loader:

            image,fine_label,coarse_label = image.to(device), fine_label.to(device),coarse_label.to(device)

            ##Here we run input x through the model to get raw class scores
            pred_fine, pred_coarse = model(image)
            pred_fine_label   = pred_fine.argmax(dim=1)
            pred_coarse_label = pred_coarse.argmax(dim=1)

            loss_coarse = loss_function_coarse(pred_coarse, coarse_label)
            loss_fine = loss_function_fine(pred_fine, fine_label)
            loss = 0.5*loss_coarse + 0.5*loss_fine
            total_loss += loss.item()

            ##Here we compute the accuracy of the model
            correct_coarse += (pred_coarse_label == coarse_label).sum().item()
            correct_fine += (pred_fine_label == fine_label).sum().item()
            overall_correct += (
                (pred_coarse_label == coarse_label) &
                 (pred_fine_label == fine_label)
                 ).sum().item()

            total += coarse_label.size(0)

    val_accuracy = overall_correct / total
    val_loss = total_loss / len(test_loader)

    print( "accuracy coarse:", (correct_coarse/total))
    print( "accuracy fine:", (correct_fine/total))
    print( "accuracy both:", val_accuracy)

    if val_accuracy > best_acc:
      print(val_accuracy ,best_acc)
      best_acc= val_accuracy
      torch.save(model.state_dict(), save_path)
      print(f"New Best Model Saved",best_acc)
       
    return val_loss,val_accuracy,best_acc

def test_multi_head_classifier(test_loader, model, device):
    model.eval()
    correct_fine = 0
    correct_coarse =0
    overall_correct = 0
    total = 0


    ##Here we loop over the test data
    with torch.no_grad():
        for image, fine_label, coarse_label in test_loader:

            image,fine_label,coarse_label = image.to(device), fine_label.to(device),coarse_label.to(device)

            ##Here we run input x through the model to get raw class scores
            pred_fine, pred_coarse = model(image)
            pred_fine_label   = pred_fine.argmax(dim=1)
            pred_coarse_label = pred_coarse.argmax(dim=1)

            ##Here we compute the accuracy of the model
            correct_coarse += (pred_coarse_label == coarse_label).sum().item()
            correct_fine += (pred_fine_label == fine_label).sum().item()
            overall_correct += (
                (pred_coarse_label == coarse_label) &
                (pred_fine_label == fine_label)
            ).sum().item()
            total += coarse_label.size(0)

    val_accuracy = overall_correct / total

    print( "accuracy coarse:", (correct_coarse/total))
    print( "accuracy fine:", (correct_fine/total))
    print( "accuracy both:", val_accuracy)

   
       
    return val_accuracy

def save_model():
    ##First we need to load the data from the train and test pickles
  train_transform = v2.Compose([
      v2.RandomChoice([
          v2.RandomCrop(32, padding=4),
          v2.RandomHorizontalFlip(p=1.0),
          v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
          v2.RandomErasing(p=1.0, scale=(0.02, 0.2), value=0),
      ])
  ])

  train = CIFAR100Dataset("./data/train.pkl",transform=train_transform)
  test  = CIFAR100Dataset("./data/test.pkl")
    
  
  train_size = int(0.8*len(train))
  validation_size= int(0.2*len(train))
  
  #Fixing seed 
  fixed_seeds =[89]
  accuracies_val =[] 

  for seed in fixed_seeds:
    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    ##Then we need to create the dataloaders
    train_data, validation_data = random_split(train, [train_size, validation_size],generator =generator )
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader   = DataLoader(validation_data, batch_size=32, shuffle=False)
    test_loader  = DataLoader(test, batch_size=32, shuffle=False)

    # print(train.images.shape, len(train.coarse_labels))

    ##This is so we can use the GPU if we have one
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ##Here we create the model and load its parameters onto the GPU if it has one
    model = CoarseAndFineClassifer().to(device)

    #Set hyperparameters
    learning_rate = 0.0007
    weight_decay= 3e-06
    loss_function_coarse = nn.CrossEntropyLoss(label_smoothing=0.2)
    loss_function_fine = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate,weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=30)

    #Saving best model 
    best_acc = 0 
    save_path = './models/d3.pth'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  


    counter = 0
    for epoch in range(30):
      print("Epoch",epoch)
      train_loss,train_acc=train_multi_head_classifier(train_loader, model, loss_function_coarse, optimiser, device,loss_function_fine)
      print("here")
      val_loss,val_acc,best_acc =validate_multi_head_classifier(val_loader, model, loss_function_coarse, device,best_acc,save_path,loss_function_fine)
      print("done")
      scheduler.step()

    print("Best accuracy on validation set",best_acc)
    accuracies_val.append(best_acc)

    print("Accuracy on test set")
    test_multi_head_classifier(test_loader, model, device)
 
  mean = np.mean(accuracies_val)
  std = np.std(accuracies_val)
  print("Across 3 Seeds: ", mean, std)

   

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = prepare_test().to(device)
    test  = CIFAR100Dataset("./data/test.pkl")
    test_loader  = DataLoader(test, batch_size=32, shuffle=False)
    test_multi_head_classifier(test_loader, model, device)

   
  