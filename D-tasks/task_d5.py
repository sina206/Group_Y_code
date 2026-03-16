import torch
import os
import copy
import torch.nn.functional as F

from dataset import CIFAR100Dataset
from torch.utils.data import DataLoader, Subset,random_split
import torch.nn as nn
from backbone import new_backbone
from torchvision.transforms import v2
import random
import numpy as np

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
            nn.Dropout(p=0.3),
            nn.Linear(512, 80)
        )

    def forward(self, x):
        ##Here we run input x through the backbone to get the embedding
        emb = self.backbone(x)  # (B, 576)

        ##Here we run the embedding through the classifier to get the raw class scores
        return self.classifier(emb)  # (B, 100)

def prepare_test():
    model = FineClassifier()

    # do not edit from here downwards
    weights_path = 'models/d5.pth'
    print(f'Loading weights from {weights_path}')
    map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=map_location))

    return model.backbone

def get_embeddings(backbone, loader, device):
    backbone.eval()
    all_embs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels, _ in loader:
            inputs = inputs.to(device)
            features = backbone(inputs) 
            all_embs.append(features.cpu())
            all_labels.append(labels)

    return torch.cat(all_embs), torch.cat(all_labels)

def knn_classifier(test_embs, support_embs, support_labels, k=5):
    
    # Calculate Cosine Similarity
    # normalise vectors to unit length
    test_norm = torch.nn.functional.normalize(test_embs, p=2, dim=1)
    support_norm = torch.nn.functional.normalize(support_embs, p=2, dim=1)
    
    # dot product to compute cosine similarity
    similarity = torch.mm(test_norm, support_norm.t())
    
    # convert similarity to distance
    dists = 1 - similarity

    knn_dists, knn_indices = torch.topk(dists, k=k, largest=False, dim=1)

    neighbor_labels = support_labels[knn_indices]

    # weighted voting for k-NN:
    preds = []
    epsilon = 1e-8 

    # invert distance (so the smaller the distance the larger the weight)
    w = 1.0 / (knn_dists + epsilon)
    
    for i in range(neighbor_labels.size(0)):
        # sum weight for each class
        class_scores = {}
        for j in range(k):
            label = neighbor_labels[i, j].item()
            score = w[i, j].item()
            class_scores[label] = class_scores.get(label, 0) + score
        # take class with highest score
        preds.append(max(class_scores, key=class_scores.get))

    return torch.tensor(preds)

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

        pred_labels = pred.argmax(dim=1)
        correct += (pred_labels == fine_label).sum().item()
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

def test_fine_classifier(test_loader, model, loss_function, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for image, fine_label, _ in test_loader:
            image, fine_label = image.to(device), fine_label.to(device)
            pred = model(image).argmax(dim=1)

            correct += (pred == fine_label).sum().item()
            total += fine_label.size(0)

    accuracy = correct / total
    print( "accuracy:", accuracy)

    return accuracy 

def train_and_save_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_transform = v2.Compose([
        v2.RandomChoice([
            v2.RandomCrop(32, padding=4),
            v2.RandomHorizontalFlip(p=1.0),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            v2.RandomErasing(p=1.0, scale=(0.02, 0.2), value=0),
        ])
    ])

    base_train_data = CIFAR100Dataset("data/zero_shot/train.pkl", transform=train_transform)
    base_val_data = CIFAR100Dataset("data/zero_shot/train.pkl") 
    test_data  = CIFAR100Dataset("data/zero_shot/test.pkl")

    train_size = int(0.8 * len(base_train_data))
    indices = torch.randperm(len(base_train_data)).tolist()
    
    train_dataset = Subset(base_train_data, indices[:train_size])
    val_dataset = Subset(base_val_data, indices[train_size:])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_data, batch_size=32, shuffle=False)

    model = FineClassifier().to(device)
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=3e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=30)

    best_acc = 0.0
    save_path = 'models/d5.pth'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(30):
        print(f"\nEpoch {epoch}")
        train_loss, train_acc = train_fine_classifier(train_loader, model, loss_function, optimiser, device)
        val_loss, val_acc, best_acc = validate_fine_classifier(val_loader, model, loss_function, device, best_acc, save_path)
        scheduler.step()

    print("Best accuracy on validation set:", best_acc)
    
    print("Testing best saved model:")
    model.load_state_dict(torch.load(save_path, weights_only=True))
    test_fine_classifier(test_loader, model, loss_function, device)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #1 Prepare model by loading weights
    backbone = prepare_test().to(device)

    #2 Load eval data
    test_loader = DataLoader(CIFAR100Dataset("data/zero_shot/test.pkl"), batch_size=32, shuffle=False)
    
    support_sets = {
        "Support 1": DataLoader(CIFAR100Dataset("data/zero_shot/support_1.pkl"), batch_size=100, shuffle=False),
        "Support 5": DataLoader(CIFAR100Dataset("data/zero_shot/support_5.pkl"), batch_size=100, shuffle=False),
        "Support 10": DataLoader(CIFAR100Dataset("data/zero_shot/support_10.pkl"), batch_size=100, shuffle=False)
    }

    #3 run knn classifier
    print("Extracting test embeddings:")
    test_embs, test_labels = get_embeddings(backbone, test_loader, device)
    
    k_values = [1, 3, 5, 7, 9, 11, 20]
    
    for name, loader in support_sets.items():
        print(f"\nEvaluating {name}:")
        support_embs, support_labels = get_embeddings(backbone, loader, device)
        
        for k in k_values:
            preds = knn_classifier(test_embs, support_embs, support_labels, k=k)
            correct = (preds == test_labels.cpu()).sum().item()
            acc = correct / len(test_labels)
            print(f"k-NN Accuracy (k={k}): {acc * 100:.2f}%")
