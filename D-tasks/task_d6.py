import torch
import numpy as np
import pickle
from task_d4 import TripletEmbeddingNetwork
from task_d2 import prepare_test as _load_d2_model
from dataset import CIFAR100Dataset
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import torch.nn.functional as F
import matplotlib.pyplot as plt


BATCH_HARD = "batch_hard"
BATCH_ALL = "batch_all"
SEMI_HARD = "semi_hard"

MARGIN_0_2 = 0.2
MARGIN_0_5 = 0.5
MARGIN_1 = 1.0

# Load the 'string' class names - the CIFAR100Dataset provides raw integers as class labels
with open("data/meta.pkl", "rb") as f:
    meta_dict = pickle.load(f, encoding="bytes")
coarse_class_names = [name.decode("utf-8") for name in meta_dict[b"coarse_label_names"]]


def generate_model_name(task_name, margin=None, sampling_strat=None, use_d2_classifier=False):
    if task_name == "d4" and not (margin and sampling_strat):
        raise ValueError("D4 models require margin and sampling strategy.")
    if task_name == "d2" and use_d2_classifier:
        return "d2_with_classifier"
    if task_name == "d2" and not use_d2_classifier:
        return "d2_without_classifier"
    if task_name == "d4":
        return f"d4_m={margin}_ss={sampling_strat}"
    raise ValueError(f"D6 only uses models from D2 and D4 {task_name}")


def _load_d4_model(margin, sampling_strat):
    model = TripletEmbeddingNetwork()
    model.eval()
    weights_path = f"_d6_models/d4_m={margin}_ss={sampling_strat}.pth"

    print(f"Loading weights from {weights_path}")
    map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=map_location))

    return model


def load_model(task_name, margin=None, sampling_strat=None):
    if task_name not in {"d2", "d4"}:
        raise ValueError(f"D6 only uses models from D2 and D4 {task_name}")
    if task_name == "d4" and not (margin and sampling_strat):
        raise ValueError("D4 models require margin and sampling strategy.")
    if task_name == "d2":
        model = _load_d2_model()
    elif task_name == "d4":
        model = _load_d4_model(margin, sampling_strat)
    model.eval()
    return model


def generate_embedding_cache(model, dataloader, model_name, normalize_embeddings, use_d2_classifier=False):
    print("Generating embeddings for model:", model_name)
    embeddings = []
    labels = []
    if model_name == "d2_without_classifier":
        model = model.backbone
    with torch.no_grad():
        for batch in dataloader:
            images, _, batch_coarse_labels = batch
            batch_embeddings = model(images)
            if normalize_embeddings:
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            embeddings.append(batch_embeddings)
            labels.append(batch_coarse_labels)

    embeddings = torch.cat(embeddings).numpy()
    labels = torch.cat(labels).numpy()
    np.savez(f"_d6_embeddings/{model_name}.npz", embeddings=embeddings, labels=labels)
    print(f"Saved embeddings and labels to {model_name}.npz")
    return


def load_embeddings_from_cache(model_name):
    data = np.load(f"_d6_embeddings/{model_name}.npz")
    return data["embeddings"], data["labels"]


def d4_plotter(margin, caption, sampling_strat, generate_cache=False, perplexity=30):
    model = load_model("d4", margin=margin, sampling_strat=sampling_strat)
    model_name = generate_model_name("d4", margin=margin, sampling_strat=sampling_strat)
    test_dataset = CIFAR100Dataset("./data/test.pkl")
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    if generate_cache:
        generate_embedding_cache(model, test_dataloader, model_name, normalize_embeddings=True)
    embeddings, labels = load_embeddings_from_cache(model_name)
    print(f"Loaded embeddings shape: {embeddings.shape}, labels shape: {labels.shape}")

    tsne = TSNE(init="pca", random_state=67, perplexity=perplexity)
    embeddings_tsne = tsne.fit_transform(embeddings)
    print("t-SNE completed.")

    cmap = plt.get_cmap("tab20")
    plt.figure(figsize=(12, 10))
    for i in range(20):
        plt.scatter(
            embeddings_tsne[labels == i, 0],
            embeddings_tsne[labels == i, 1],
            label=coarse_class_names[i],
            color=cmap(i),
            alpha=0.6,
            s=15,
        )

    plt.legend(title="Coarse Classes", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title(caption)
    plt.tight_layout()
    plt.savefig(f"_d6_figures/{model_name}_tsne.png", dpi=300, bbox_inches="tight")
    print(f"Saved plot")
    # plt.show()


def d2_plotter(caption, generate_cache=False, perplexity=30, use_classifier=False):
    model = load_model("d2")
    model_name = generate_model_name("d2", use_d2_classifier=use_classifier)
    test_dataset = CIFAR100Dataset("./data/test.pkl")
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    if generate_cache:
        generate_embedding_cache(
            model, test_dataloader, model_name, normalize_embeddings=False, use_d2_classifier=use_classifier
        )
    embeddings, labels = load_embeddings_from_cache(model_name)
    print(f"Loaded embeddings shape: {embeddings.shape}, labels shape: {labels.shape}")

    tsne = TSNE(init="pca", random_state=67, perplexity=perplexity)
    embeddings_tsne = tsne.fit_transform(embeddings)
    print("t-SNE completed.")

    cmap = plt.get_cmap("tab20")
    plt.figure(figsize=(12, 10))
    for i in range(20):
        plt.scatter(
            embeddings_tsne[labels == i, 0],
            embeddings_tsne[labels == i, 1],
            label=coarse_class_names[i],
            color=cmap(i),
            alpha=0.6,
            s=15,
        )

    plt.legend(title="Coarse Classes", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title(caption)
    plt.tight_layout()
    plt.savefig(f"_d6_figures/{model_name}_tsne.png", dpi=300, bbox_inches="tight")
    print(f"Saved plot")
    # plt.show()


if __name__ == "__main__":
    cache = False
    p = 50

    # caption = "D2 Model Without Classifier"
    # d2_plotter(caption, generate_cache=cache, use_classifier=False, perplexity=p)
    # caption = "D2 Model With Classifier"
    # d2_plotter(caption, generate_cache=cache, use_classifier=True, perplexity=p)

    caption = "D4 Model: Margin=0.2, Sampling Method=semi-hard"
    d4_plotter(caption=caption, margin=MARGIN_0_2, sampling_strat=SEMI_HARD, generate_cache=cache, perplexity=p)
    caption = "D4 Model: Margin=0.5, Sampling Method=semi-hard"
    d4_plotter(caption=caption, margin=MARGIN_0_5, sampling_strat=SEMI_HARD, generate_cache=cache, perplexity=p)
    caption = "D4 Model: Margin=1.0, Sampling Method=semi-hard"
    d4_plotter(caption=caption, margin=MARGIN_1, sampling_strat=SEMI_HARD, generate_cache=cache, perplexity=p)

    caption = "D4 Model: Margin=0.5, Sampling Method=batch-hard"
    d4_plotter(caption=caption, margin=MARGIN_0_5, sampling_strat=BATCH_HARD, generate_cache=cache, perplexity=p)
    caption = "D4 Model: Margin=0.5, Sampling Method=batch-all"
    d4_plotter(caption=caption, margin=MARGIN_0_5, sampling_strat=BATCH_ALL, generate_cache=cache, perplexity=p)
    caption = "D4 Model: Margin=0.5, Sampling Method=semi-hard"
    d4_plotter(caption=caption, margin=MARGIN_0_5, sampling_strat=SEMI_HARD, generate_cache=cache, perplexity=p)

    caption = "D4 Model: Margin=1.0, Sampling Method=batch-hard"
    d4_plotter(caption=caption, margin=MARGIN_1, sampling_strat=BATCH_HARD, generate_cache=cache, perplexity=p)
    caption = "D4 Model: Margin=1.0, Sampling Method=batch-all"
    d4_plotter(caption=caption, margin=MARGIN_1, sampling_strat=BATCH_ALL, generate_cache=cache, perplexity=p)
    caption = "D4 Model: Margin=1.0, Sampling Method=semi-hard"
    d4_plotter(caption=caption, margin=MARGIN_1, sampling_strat=SEMI_HARD, generate_cache=cache, perplexity=p)

    caption = "D4 Model: Margin=0.2, Sampling Method=batch-all"
    d4_plotter(caption=caption, margin=MARGIN_0_2, sampling_strat=BATCH_ALL, generate_cache=cache, perplexity=p)
    caption = "D4 Model: Margin=0.2, Sampling Method=batch-hard"
    d4_plotter(caption=caption, margin=MARGIN_0_2, sampling_strat=BATCH_HARD, generate_cache=cache, perplexity=p)
