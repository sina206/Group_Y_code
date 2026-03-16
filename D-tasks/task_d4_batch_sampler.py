from dataset import CIFAR100Dataset
from collections import defaultdict
from typing import Dict, List
import torch


class OnlineBatchSampler:
    """
    Custom batch sampler essential for online triplet sampling.
    Ensures that we have at least `num_samples_per_class` samples for each of the `num_classes` in every batch.
    """

    def __init__(
        self,
        dataset: CIFAR100Dataset,
        num_classes: int,
        num_samples_per_class: int,
        use_fine_labels: bool,
    ):
        self.dataset = dataset
        self.num_classes = num_classes
        self.num_samples_per_class = num_samples_per_class

        self.total_batch_size = num_classes * num_samples_per_class
        self.class_labels = (
            self.dataset.fine_labels if use_fine_labels else self.dataset.coarse_labels
        )

        # key: class label, value: list of indices belonging to that class from the dataset
        self.class_to_indices: Dict[int, List[int]] = self._get_class_to_indices_map()
        self.dataset_size = len(self.dataset)

    def __iter__(self):
        for _ in range(len(self)):
            batch_indices = []

            # Shuffle class labels and select the first `num_classes` classes for the batch
            all_classes_shuffled = torch.randperm(len(self.class_to_indices))
            selected_classes = all_classes_shuffled[: self.num_classes]

            for class_label in selected_classes:
                class_label = int(class_label.item())
                # Shuffle indices of samples belonging to the selected class and select `num_samples_per_class` samples
                indices = self.class_to_indices[class_label]
                all_indices_shuffled = torch.randperm(len(indices))
                selected_indices = all_indices_shuffled[: self.num_samples_per_class]

                # Add selected sample indices to the batch
                for idx in selected_indices:
                    idx = idx.item()
                    batch_indices.append(indices[idx])

            # Randomize the order of samples in the batch
            batch_indices = torch.tensor(batch_indices)
            shuffled_indices = torch.randperm(len(batch_indices))
            randomized_batch = batch_indices[shuffled_indices].tolist()
            yield randomized_batch

    def __len__(self):
        return self.dataset_size // self.total_batch_size

    def _get_class_to_indices_map(self) -> Dict[int, List[int]]:
        """
        Creates a mapping from class labels to the list of dataset indices belonging to that class
        """
        class_to_indices = defaultdict(list)
        for idx, class_label in enumerate(self.class_labels):
            class_to_indices[class_label].append(idx)
        return class_to_indices
