import torch
import torch.nn as nn
import numpy as np
from backbone import new_backbone
import torch.nn.functional as F
from dataset import CIFAR100Dataset
from torch.utils.data import DataLoader
from task_d4_batch_sampler import OnlineBatchSampler
import os
from tqdm import tqdm
from typing import Dict

### Main hyperparameter that we definitely want to experiment with
MARGIN = 0.2
USE_FINE_LABELS = True  # change to False for coarse labels
SAMPLING_STRAT = "batch_all"
DIST_METRIC = "l2"
NUM_EPOCHS = 1

### For testing different hyperparams - adjust based on number of epochs
EVALUATE_DURING_TRAINING = True
EVALUATE_EVERY_N_EPOCHS = 5
SAVE_EVERY_N_EPOCHS = 10

### Consider experimenting with these too
NUM_CLASSES_PER_BATCH = 10  # for batch sampler
NUM_SAMPLES_PER_CLASS = 4  # for batch sampler
L2_NORM_EMBEDDINGS = True

### Non hyperparameter constants
RECALL_K_VALUES = [5, 10, 50, 100]  # Do not change
DEBUG_BATCHES = False


def run_evaluation(model, eval_dataset, device):
    print("-------Starting evaluation-------")
    model.eval()

    evaluation_dataloader = DataLoader(eval_dataset, batch_size=128, shuffle=False)

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for images, fine_lbls, coarse_lbls in tqdm(
            evaluation_dataloader,
            desc="Generating embeddings for evaluation",
            leave=False,
        ):
            images = images.to(device)
            labels = fine_lbls if USE_FINE_LABELS else coarse_lbls

            emb = model(images)

            all_embeddings.append(emb)
            all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0).to(device)

    if L2_NORM_EMBEDDINGS:
        all_embeddings = F.normalize(all_embeddings, p=2)

    global_dist_matrix = calc_dist_matrix(all_embeddings, all_embeddings, DIST_METRIC)

    final_evaluation = evaluate_embeddings(global_dist_matrix, all_labels)

    print("\nEvaluation Results:")

    for k, v in final_evaluation.items():
        print(f"{k}: {v}")


# TODO Save your best models and store them at './models/d4_m={margin}_fine.pth' or ./models/d4_m={margin}_coarse.pth,
#  depending on whether you trained the model with triplets formed with the fine or coarse labels.
#  {margin} is the margin value that you used to train the model. You must upload at least two models, one for the
#  fine-grained version and one for the coarse-grained version, specifying the margin value. You can upload multiple
#  models trained with different margin values


class TripletEmbeddingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = new_backbone()

    def forward(self, x):
        # x: (B, 3, 32, 32)
        emb = self.backbone(x)  # (B, 576)
        return emb


def prepare_test(margin, fine_labels):
    # TODO: Create an instance of your model here. Your model must take in input a tensor of shape
    #  (B, 3, 32, 32), where B >= 2, and output a tensor of shape (B, 576), where B is the batch size and 576 is the
    #  embedding dimension. Make sure that the correct model is loaded depending on the margin and fine_labels parameters
    #  where `margin` is a float and `fine_labels` is a boolean that if True/False will load the model trained with triplets
    #  formed with the fine/coarse labels.

    model = TripletEmbeddingNetwork()
    model.eval()

    # do not edit from here downwards
    s = "fine" if fine_labels else "coarse"
    weights_path = f"models/d4_m={margin}_{s}.pth"

    print(f"Loading weights from {weights_path}")
    map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=map_location))

    return model


def calc_dist_matrix(x, y, method="l2"):
    """computes all pairwise euclidean/cosine distances between rows of tensors x and y and returns distance matrix"""
    if method == "l2":
        return torch.cdist(x, y, p=2)
    elif method == "cosine":
        x = F.normalize(x, dim=-1)  # x / |x|
        y = F.normalize(y, dim=-1)  # y / |y|
        return 1.0 - x @ y.T


def triplet_sampling(sampling_strat, dist_matrix, margin, labels):

    active_loss_flag = False

    loss = 0

    anchor_labels = labels

    # create masks to identify positive and negative samples
    positive_mask = anchor_labels == anchor_labels.unsqueeze(1)
    # mark all diagonals as False to prevent comparing anchor to itself
    positive_mask.fill_diagonal_(False)
    negative_mask = anchor_labels != anchor_labels.unsqueeze(1)
    triplet_mask = positive_mask.unsqueeze(2) & negative_mask.unsqueeze(1)

    if sampling_strat == "batch_all":
        # calculate differences between all triplets
        diffs = dist_matrix.unsqueeze(2) - dist_matrix.unsqueeze(1)

        # calculate loss for valid triplets
        loss = torch.relu(diffs + margin)
        loss = loss[triplet_mask]

    elif sampling_strat == "batch_hard":
        # extract hard positives and negatives from distance matrix
        pos_matrix = torch.where(positive_mask, dist_matrix, float("-inf"))
        neg_matrix = torch.where(negative_mask, dist_matrix, float("inf"))
        hard_positives = torch.max(pos_matrix, dim=1).values
        hard_negatives = torch.min(neg_matrix, dim=1).values

        # calculate differences between all triplets
        diffs = hard_positives - hard_negatives

        # calculate loss for valid triplets
        loss = torch.relu(diffs + margin)

    elif sampling_strat == "semi_hard":
        # extract hard positives and negatives from distance matrix
        pos_matrix = torch.where(positive_mask, dist_matrix, float("-inf"))
        neg_matrix = torch.where(negative_mask, dist_matrix, float("inf"))
        hard_positives = torch.max(pos_matrix, dim=1).values
        hard_negatives = torch.min(neg_matrix, dim=1).values

        # extract lower and upper bounds for semi-hard negatives
        upper_bound = hard_positives.unsqueeze(1) + margin
        lower_bound = hard_positives.unsqueeze(1)

        # extract semi-hard negatives
        semi_hard_mask = (neg_matrix < upper_bound) & (neg_matrix > lower_bound)
        semi_hard_dists = torch.where(semi_hard_mask, neg_matrix, torch.tensor(float("inf")))
        semi_hard_negatives = torch.min(semi_hard_dists, dim=1).values

        # if no semi-negatives -> use hard negatives
        check_semi_hard = semi_hard_negatives != float("inf")
        chosen_neg = torch.where(check_semi_hard, semi_hard_negatives, hard_negatives)

        # compute loss
        loss = torch.relu(hard_positives - chosen_neg + margin)

    else:
        print("invalid sampling strategy")

    # to only use active losses in mean calculation i.e., loss > 0
    # flag must be set to true
    if active_loss_flag:
        active_mask = loss > 0
        if active_mask.any():
            active_losses = loss[active_mask]
            loss = active_losses.mean()
        else:
            loss = loss.sum() * 0.0
    else:
        loss = loss.mean()

    return loss


def train_embedding_model(
    model,
    optimiser,
    dataloader,
    eval_dataset,
    device,
    margin,
    fine_labels,
    num_epochs,
    normalise_embeddings=True,
    sampling_strat="batch_all",
    dist_metric="l2",
):
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            images, batch_fine, batch_coarse = batch

            imgs = images.to(device)
            labels = batch_fine if fine_labels else batch_coarse
            labels = labels.to(device)

            optimiser.zero_grad()

            emb = model(imgs)

            if normalise_embeddings:
                emb = F.normalize(emb, p=2)

            dist_matrix = calc_dist_matrix(emb, emb, method=dist_metric)

            loss = triplet_sampling(sampling_strat, dist_matrix, margin, labels)

            loss.backward()
            optimiser.step()

            total_loss += loss.item()

        print("Epoch [", epoch + 1, "/", num_epochs, "] Loss: ", total_loss)

        if EVALUATE_DURING_TRAINING and ((epoch + 1) % SAVE_EVERY_N_EPOCHS == 0):
            print(f"Saving model at epoch {epoch+1}")
            label_type = "fine" if fine_labels else "coarse"
            save_path = f"./models/d4_m={margin}_{label_type}_epoch{epoch+1}.pth"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)

        if EVALUATE_DURING_TRAINING and ((epoch + 1) % EVALUATE_EVERY_N_EPOCHS == 0):
            run_evaluation(model, eval_dataset, device)

    label_type = "fine" if fine_labels else "coarse"
    save_path = f"./models/d4_m={margin}_{label_type}.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print("Model saved at:", save_path)


def evaluate_embeddings(dist_matrix: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """
    Evaluate our triplet loss embeddings using global recall@k and anchor-positive/negative distances

    Args:
        dist_matrix: (N,N) matrix of pairwise distances between embeddings for the ENTIRE dataset - not just a batch
        labels: (N,) labels for the ENTIRE dataset
    Returns:
        A dictionary with the following metrics as keys and float values:
            recall_at_5
            recall_at_10
            recall_at_50
            recall_at_100
            mean_d_ap
            mean_d_an
    """
    num_samples = dist_matrix.shape[0]

    ### Calculate recall@k
    recall_matrix = np.zeros((num_samples, len(RECALL_K_VALUES)))

    unique_labels, counts = torch.unique(labels, return_counts=True)
    class_counts = {int(label): int(count) for label, count in zip(unique_labels, counts)}

    # Ignore distances of images to themselves (diagonal of dist_matrix) by setting to infinity
    for i in range(num_samples):
        dist_matrix[i, i] = float("inf")

    for img in tqdm(range(num_samples), desc="Calculating recall@k", leave=False):
        # Get sorted indices of distances for all other images
        sorted_indices = torch.argsort(dist_matrix[img])
        for idx, k in enumerate(RECALL_K_VALUES):
            # Check the top-k closest images
            top_k = sorted_indices[:k]
            ### Implements the equation in the spec
            # Count how many of the top-k have the correct label
            correct_count = sum(int(labels[j]) == int(labels[img]) for j in top_k)
            # Divide by the number of samples with this label (subtract 1 to exclude the image itself)
            recall_matrix[img][idx] = correct_count / (class_counts[int(labels[img])] - 1)

    # Calculate avg recall@k across all images
    evaluation_metrics = {
        f"recall_at_{k}": float(np.mean(recall_matrix[:, idx])) for idx, k in enumerate(RECALL_K_VALUES)
    }

    ### Calculate mean distance to anchor positive and anchor negative
    label_col = labels.unsqueeze(1)
    label_row = labels.unsqueeze(0)
    positive_mask = label_col == label_row
    identical_mask = torch.eye(num_samples, dtype=torch.bool, device=labels.device)
    positive_mask = positive_mask & ~identical_mask
    negative_mask = ~positive_mask & ~identical_mask

    anchor_positive_distances = dist_matrix[positive_mask]
    anchor_negative_distances = dist_matrix[negative_mask]
    mean_d_ap = torch.mean(anchor_positive_distances).item()
    mean_d_an = torch.mean(anchor_negative_distances).item()
    evaluation_metrics["mean_d_ap"] = mean_d_ap
    evaluation_metrics["mean_d_an"] = mean_d_an

    return evaluation_metrics


if __name__ == "__main__":

    def main():
        print(
            f"Margin: {MARGIN}, \nFine-labels: {USE_FINE_LABELS}, \nL2-norm-embeddings: {L2_NORM_EMBEDDINGS}, \nSampling strat: {SAMPLING_STRAT}, \nDist metric: {DIST_METRIC}\n"
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dataset = CIFAR100Dataset("./data/train.pkl")
        eval_dataset = CIFAR100Dataset("./data/test.pkl")
        sampler = OnlineBatchSampler(dataset, NUM_CLASSES_PER_BATCH, NUM_SAMPLES_PER_CLASS, USE_FINE_LABELS)
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            # collate_fn=lambda x: x,  # return list of dataset items without stacking
        )

        if DEBUG_BATCHES:
            print("Type of dataloader:", type(dataloader))
            print("Total batches:", len(dataloader))

            # Inspect first batch
            first_batch = next(iter(dataloader))
            print("\n--- First batch ---")
            print("Batch type:", type(first_batch))
            print("Batch length:", len(first_batch))
            print("Types of batch elements:", [type(x) for x in first_batch[:5]])
            print("Shapes of first 5 images:", [x[0].shape for x in first_batch[:5]])
            print("First 10 labels:", [x[1] for x in first_batch[:10]])

        model = TripletEmbeddingNetwork()
        optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4)

        train_embedding_model(
            model,
            optimiser,
            dataloader,
            eval_dataset,
            device,
            MARGIN,
            USE_FINE_LABELS,
            num_epochs=NUM_EPOCHS,
            normalise_embeddings=L2_NORM_EMBEDDINGS,
            sampling_strat=SAMPLING_STRAT,
        )

        run_evaluation(model, eval_dataset, device)
