import task_d1
from task_d1 import FineClassifier
from backbone import new_backbone
from dataset import CIFAR100Dataset
from torch.utils.data import DataLoader,random_split
import torch.nn as nn
import torch,os
import matplotlib.pyplot as plt
from task_d1 import set_seed 
import torch.nn.functional as F

class KNN:
  def __init__(self,train_embeddings,train_labels,k=3):
      self.k = k 
      self.train_embeddings = train_embeddings 
      self.train_labels = train_labels 
  
  def predict(self,images,labels,distance_metric='Euclidean',voting_strat= "Majority", tau=0.45, epsilon =  1e-08):

    if distance_metric=='Euclidean':
      #L1 normalisation
      norm_images = F.normalize(images, dim=1)
      norm_train_embeddings = F.normalize(self.train_embeddings, dim=1)

      distances = torch.cdist(norm_images, norm_train_embeddings,p=2)
     
    if distance_metric == 'Manhattan_L1':
      #L2 normalisation
      norm_train_embeddings = F.normalize(self.train_embeddings, dim=1)
      norm_images= F.normalize(images, dim=1)

      distances = torch.cdist(norm_images, norm_train_embeddings,p=1)
    
    if distance_metric == 'Manhattan_L2':
      #L2 normalisation
      norm_train_embeddings = F.normalize(self.train_embeddings, p=2, dim=1)
      norm_images= F.normalize(images, p=2, dim=1)

      distances = torch.cdist(norm_images, norm_train_embeddings,p=1)
    
    
    if distance_metric == 'Cosine':
      #L1 Normalisation
      norm_images = F.normalize(images, dim=1)
      norm_train_embeddings = F.normalize(self.train_embeddings, dim=1)

      distances = 1-(norm_images @ norm_train_embeddings.T)
    
    if distance_metric == 'Dot_Product':
      distances = 1-(images @ train_embeddings.T)



    if voting_strat== "Majority": # Baseline
      smallest_dist_indices = torch.argsort(distances, descending=False)[:,:self.k]
      predicted_labels = self.train_labels[smallest_dist_indices]
      predictions = torch.mode(predicted_labels, dim=1).values
    
    if voting_strat == "Inverse_Distance_Weighted":
      smallest_dist_indices = torch.argsort(distances, descending=False)[:,:self.k]
      predicted_labels = self.train_labels[smallest_dist_indices]
      device=distances.device 
     

      coarse_class_scores = torch.zeros(predicted_labels.size(0), 20, device=predicted_labels.device)

      neighbor_distances = distances.gather(1, smallest_dist_indices)
      weights = 1.0 / (neighbor_distances + epsilon)
      coarse_class_scores.scatter_add_(dim=1,index=predicted_labels,src=weights)
      predictions = torch.argmax(coarse_class_scores, dim=1)
    
    if voting_strat == "Rank_Weighted":
      smallest_dist_indices = torch.argsort(distances, descending=False)[:,:self.k]
      predicted_labels = self.train_labels[smallest_dist_indices]
      
      ranks = torch.arange(1, self.k + 1, device=predicted_labels.device, dtype=torch.float)
      weights = 1.0 / ranks
      votes = torch.zeros(predicted_labels.size(0), 20, device=predicted_labels.device)

      for i in range(predicted_labels.size(0)):        
          for j in range(self.k):        
              predicted_class = predicted_labels[i, j]
              votes[i, predicted_class] += weights[j]
      

      predictions = votes.argmax(dim=1)

    if voting_strat == "Softmax_Voting":
      smallest_dist_indices = torch.argsort(distances, descending=False)[:,:self.k]
      predicted_labels = self.train_labels[smallest_dist_indices]

      scores = -distances / tau                   
      weights = F.softmax(scores, dim=1)          

      probs = torch.zeros(predicted_labels.size(0), 20, device=distances.device)

      probs.scatter_add_(dim=1,index=predicted_labels,src=weights)

      predictions = probs.argmax(dim=1)      
    
    accuracy = (predictions == labels).sum().item()/ predictions.size(0)
    return accuracy 


def extract_embeddings(model,data,device):
  model.eval()
  all_embeddings = []
  all_labels = []

  with torch.no_grad():
      for images,fine_labels,coarse_labels in data:
          images = images.to(device)
          emb = model(images)
          all_embeddings.append(emb)
          all_labels.append(coarse_labels)

  embeddings = torch.cat(all_embeddings).to(device)
  labels = torch.cat(all_labels).to(device)

  return embeddings,labels 




def prepare_test():
    # TODO: Load the model from task D1 and return its **backbone**. The backbone model will be fed a batch of images,
    #  i.e. a tensor of shape (B, 3, 32, 32), where B >= 2, and must return a tensor of shape (B, 576), i.e.
    #  the embedding extracted for the input images. Hint: if the backbone is stored inside your model with the
    #  name "backbone", you can simply leave the code below as is. Otherwise, please adjust.

    model = task_d1.prepare_test()
    return model.backbone


def run_experiments():
    set_seed(12)
    generator = torch.Generator().manual_seed(12)


    save_path = './models/d1.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train = CIFAR100Dataset("./data/train.pkl")
    test  = CIFAR100Dataset("./data/test.pkl")

    train_size = int(0.8*len(train))
    validation_size= int(0.2*len(train))

    train_data, validation_data = random_split(train, [train_size, validation_size],generator =generator )
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader   = DataLoader(validation_data, batch_size=32, shuffle=False)
    test_loader  = DataLoader(test, batch_size=32, shuffle=False)


    model = FineClassifier().to(device)
    model.load_state_dict(torch.load(save_path))

    # Extract embeddings from D1 
    train_embeddings,train_labels = extract_embeddings(model,train_loader,device)
    val_embeddings,val_labels = extract_embeddings(model,val_loader,device)
    test_embeddings,test_labels = extract_embeddings(model,test_loader,device)


    k_values = range(1,50)



    distance_metrics =['Euclidean','Manhattan_L1','Manhattan_L2','Cosine']
    results_dict={}

    for metric in distance_metrics:
        train_values = []
        val_values =[]
        test_values =[]


        # Run classifer for multiple k 
        for k in k_values:
            print("-----------",metric,"-------------")
            print("Value of k",k)
            classifier = KNN(train_embeddings,train_labels,k)

            train_accuracy = classifier.predict(train_embeddings,train_labels,distance_metric=metric)
            val_accuracy = classifier.predict(val_embeddings,val_labels,distance_metric=metric)
            test_accuracy = classifier.predict(test_embeddings,test_labels,distance_metric=metric)

            print("Train Accuracy", train_accuracy)
            print("Val Accuracy", val_accuracy)
            print("Test Accuracy", test_accuracy)

            val_values.append(val_accuracy)
            train_values.append(train_accuracy)
            test_values.append(test_accuracy)

            best_accuracy = max(val_values)
            best_accuracy_index= val_values.index(best_accuracy)
            best_k = k_values[best_accuracy_index]
            print("Best Value of k ",best_k, "Val Accuracy ",best_accuracy )
            results_dict[metric]= val_values



    # Plot k value vs accuracy graph 
    plt.figure()
    # plt.plot(k_values, train_values, marker='o', label="Train Accuracy")

    for metric in distance_metrics:
        label_x= metric+ " Val Top-1 Accuracy"
        plt.plot(k_values, results_dict[metric], label=label_x)

        plt.xlabel("k value")
        plt.ylabel("Validation Top-1 Accuracy")
        plt.title("Validation Top-1 Accuracy vs k for Different Distance Metrics")
        plt.legend()
        plt.grid(True)

        plt.savefig("d7_accuracy_graph.png", dpi=300, bbox_inches="tight")  
        plt.show()

    voting_strategies=["Majority","Inverse_Distance_Weighted","Rank_Weighted","Softmax_Voting"]
    results_dict={}

    for strat in voting_strategies:
        train_values = []
        val_values =[]
        test_values =[]


        # Run classifer for multiple k 
        for k in k_values:
            print("-----------",strat,"-------------")
            print("Value of k",k)
            classifier = KNN(train_embeddings,train_labels,k)

            train_accuracy = classifier.predict(train_embeddings,train_labels,distance_metric='Cosine',voting_strat=strat)
            val_accuracy = classifier.predict(val_embeddings,val_labels,distance_metric='Cosine',voting_strat=strat)
            test_accuracy = classifier.predict(test_embeddings,test_labels,distance_metric='Cosine',voting_strat=strat)

            print("Train Accuracy", train_accuracy)
            print("Val Accuracy", val_accuracy)
            print("Test Accuracy", test_accuracy)

            val_values.append(val_accuracy)
            train_values.append(train_accuracy)
            test_values.append(test_accuracy)

        best_accuracy = max(val_values)
        best_accuracy_index= val_values.index(best_accuracy)
        best_k = k_values[best_accuracy_index]
        print("Best Value of k ",best_k, "Val Accuracy ",best_accuracy )
        results_dict[strat]= val_values


        
        
    plt.figure()
    # plt.plot(k_values, train_values, marker='o', label="Train Accuracy")
    best_acc=-1
    final_k_index=None 

    for strat in voting_strategies:
        label_x= strat+ " Val Top-1 Accuracy"
        plt.plot(k_values, results_dict[strat], label=label_x)
        if max(results_dict[strat])>best_acc:
            best_acc=max(results_dict[strat])
            final_k_index= (results_dict[strat]).index(best_acc)
            
    final_k = k_values[final_k_index]




    plt.xlabel("k value")
    plt.ylabel("Validation Top-1 Accuracy")
    plt.title("Validation Top-1 Accuracy vs k for Different Voting Strategies")
    plt.legend()
    plt.grid(True)

    plt.savefig("d7_voting_graph.png", dpi=300, bbox_inches="tight")  
    plt.show()


    print("-------","Test Accuracy","-------")
    best_classifier = KNN(train_embeddings,train_labels,final_k)
    test_accuracy = best_classifier.predict(test_embeddings,test_labels,distance_metric='Cosine',voting_strat='Softmax_Voting')
    print(final_k,test_accuracy)


if __name__ == '__main__':
    set_seed(89) #Same seed as D1 to ensure same train-val split 
    generator = torch.Generator().manual_seed(89)


    save_path = './models/d1.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train = CIFAR100Dataset("./data/train.pkl")
    test  = CIFAR100Dataset("./data/test.pkl")

    train_size = int(0.8*len(train))
    validation_size= int(0.2*len(train))

    train_data, validation_data = random_split(train, [train_size, validation_size],generator =generator )
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader   = DataLoader(validation_data, batch_size=32, shuffle=False)
    test_loader  = DataLoader(test, batch_size=32, shuffle=False)


    model = prepare_test().to(device)

    # Extract embeddings from D1 
    train_embeddings,train_labels = extract_embeddings(model,train_loader,device)
    val_embeddings,val_labels = extract_embeddings(model,val_loader,device)
    test_embeddings,test_labels = extract_embeddings(model,test_loader,device)


    k_values = range(1,50)
    print("-------","Test Accuracy","-------")
    best_classifier = KNN(train_embeddings,train_labels,7)
    test_accuracy = best_classifier.predict(test_embeddings,test_labels,distance_metric='Cosine',voting_strat='Majority')
    print(test_accuracy)




  