# TODO: Implement the dataset class extending torch.utils.data.Dataset
import pickle
import torch

class CIFAR100Dataset(torch.utils.data.Dataset):
    def __init__(self, pickle_path, transform=None):
        # Load the pickled split (keys are bytes). Contains data, fine_labels, coarse_labels.
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f, encoding='bytes')

        # CIFAR100 in this coursework is stored flattened as (N, 3072) = (N, 3*32*32) 3 channels and 32*32 for image size.
        # Reshape once to channel-first tensors: (N, 3, 32, 32).
        imgs = data[b'data'].reshape(-1, 3, 32, 32)

        # Convert to float32 and scale to [0,1] since the backbone expects normalized floats.
        self.images = torch.tensor(imgs, dtype=torch.float32) / 255.0

        # Keep both label granularities; D1 uses fine, D2 uses coarse, others may need both.
        self.fine_labels = data[b'fine_labels']
        self.coarse_labels = data[b'coarse_labels']

        # Optional transform hook (e.g., augmentation or normalization tweaks).
        self.transform = transform

    def __len__(self):
        # Required by Dataset: number of samples in this split.
        return len(self.coarse_labels)

    def __getitem__(self, idx):
        # Fetch one sample (channel-first, already float32 in [0,1]).
        img = self.images[idx]
        fine = self.fine_labels[idx]
        coarse = self.coarse_labels[idx]

        # Apply any user-provided transforms (e.g., random crop/flip). 
        # Note: if using torchvision transforms that expect (H,W,C) or PIL, adapt accordingly.
        if self.transform:
            img = self.transform(img)

        # Return image + both labels so loaders for D1–D4 can reuse the same dataset.
        return img, fine, coarse
