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


def set_seed(seed=0):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

class PreprocessBackbone(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        ##Here we apply the normalisations 
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = v2.functional.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return self.backbone(x)

class FineClassifier(nn.Module):
      def __init__(self):
        super().__init__()
        self.backbone = PreprocessBackbone(new_backbone())
        self.classifier = nn.Sequential( 
          nn.Linear(576, 512), 
          nn.ReLU(),
          nn.Linear(512, 100))

      def forward(self, x):
        ##Here we run input x through the backbone to get the embedding
        emb = self.backbone(x) # (B, 576)

        ##Here we run the embedding through the classifier to get the raw class scores
        return self.classifier(emb)  # (B, 20) (B, 100)


def prepare_test():
    # TODO: Create an instance of your model here. Your model must take in input a tensor of shape
    #  (B, 3, 32, 32), where B >= 2, and output a tensor of shape (B, 20), where B is the batch size
    #  and 20 is the number of classes. The output is the prediction of your classifier, providing a score for each
    #  class, for each image in input

    model = FineClassifier()

    # do not edit from here downwards
    weights_path = 'models/d1.pth'
    print(f'Loading weights from {weights_path}')
    map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=map_location))

    return model

def train_fine_classifier(train_loader, model, loss_function, optimiser, device):
    model.train()
    correct=0
    total=0
    total_loss= 0

    for image,fine_label,_ in train_loader:
    
        image, fine_label = image.to(device), fine_label.to(device)

        ##Here we run input x through the model to get raw class scores
        pred = model(image)

        ##Here we compute the loss between the raw class scores and the ground truth
        loss = loss_function(pred, fine_label)

        #Backprop
        loss.backward()
        total_loss+= loss.item()
        optimiser.step()
        optimiser.zero_grad()

        pred = model(image).argmax(dim=1)
        correct += (pred == fine_label).sum().item()
        total += fine_label.size(0)


    train_acc= correct/total 
    total_loss= total_loss/len(train_loader)
    print("loss",total_loss)
    print("accuracy",train_acc)

    return total_loss,train_acc



def validate_fine_classifier(test_loader, model, loss_function, device,best_acc,save_path):
    model.eval()
    correct = 0
    total = 0
    total_loss=0


    ##Here we loop over the test data
    with torch.no_grad():
        for image, fine_label, _ in test_loader:

            image,fine_label = image.to(device), fine_label.to(device)

            ##Here we run input x through the model to get raw class scores
            pred = model(image)
            loss = loss_function(pred, fine_label)
            total_loss+= loss.item()


            ##Here we compute the accuracy of the model
            correct += (pred.argmax(dim=1) == fine_label).sum().item()
            total += fine_label.size(0)

    val_accuracy = correct / total
    print( "accuracy:", val_accuracy)

    val_loss = total_loss / len(test_loader)
    print( "loss:", val_loss)

    if val_accuracy > best_acc:
      print(val_accuracy ,best_acc)
      best_acc= val_accuracy
      torch.save(model.state_dict(), save_path)
      print(f"New Best Model Saved",best_acc)

    return val_loss,val_accuracy,best_acc


def test_fine_classifier(test_loader, model, device):
    model.eval()
    correct = 0
    total = 0


    all_preds, all_labels = [], []
    with torch.no_grad():
        for image, fine_label, _ in test_loader:
            image, fine_label = image.to(device), fine_label.to(device)
            pred = model(image).argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(fine_label.cpu().numpy())

            ##Here we compute the accuracy of the model
            correct += (pred == fine_label).sum().item()
            total += fine_label.size(0)

    accuracy = correct / total
    print( "accuracy:", accuracy)

  
    return accuracy 


def save_model():
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

      train_data, validation_data = random_split(train, [train_size, validation_size],generator =generator )
      train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
      val_loader   = DataLoader(validation_data, batch_size=32, shuffle=False)
      test_loader  = DataLoader(test, batch_size=32, shuffle=False)

      ##This is so we can use the GPU if we have one
      device = 'cuda' if torch.cuda.is_available() else 'cpu'

      ##Here we create the model and load its parameters onto the GPU if it has one
      model = FineClassifier().to(device)

      #Set hyperparameters
      learning_rate = 3e-4
      weight_decay= 3e-5
      loss_function = nn.CrossEntropyLoss(label_smoothing=0.05)
      optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate,weight_decay=weight_decay)

      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=30)
      #Saving best model 
      best_acc = 0 
      save_path = './models/d1.pth'
      os.makedirs(os.path.dirname(save_path), exist_ok=True) 

      # Main model building loop
      for epoch in range(30):
          print("Epoch", epoch)
          train_loss, train_acc = train_fine_classifier(train_loader, model, loss_function, optimiser, device)
          print("here")
          val_loss, val_acc, best_acc = validate_fine_classifier(val_loader, model, loss_function, device, best_acc, save_path)
          print("done")
          scheduler.step(val_acc)

      print("Best accuracy on validation set", best_acc)
      accuracies_val.append(best_acc)

      print("Accuracy on test set")
      model = FineClassifier().to(device)
      model.load_state_dict(torch.load(save_path))

      model.eval()
      all_preds, all_labels = [], []
      with torch.no_grad():
          for image, _,fine_label in test_loader:
              image, fine_label = image.to(device), fine_label.to(device)
              pred = model(image).argmax(dim=1)
              all_preds.extend(pred.cpu().numpy())
              all_labels.extend(fine_label.cpu().numpy())

      test_fine_classifier(test_loader, model, device)



    mean = np.mean(accuracies_val)
    std = np.std(accuracies_val)
    print("Across 3 Seeds: ", mean, std)



if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = prepare_test().to(device)
    test  = CIFAR100Dataset("./data/test.pkl")
    test_loader  = DataLoader(test, batch_size=32, shuffle=False)
    test_fine_classifier(test_loader, model, device)



    