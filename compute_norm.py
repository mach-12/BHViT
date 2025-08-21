from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
from tqdm import tqdm
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def compute_mean_std(data_path):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    mean = 0.0
    std = 0.0
    nb_samples = 0

    for data, _ in tqdm(loader):
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    print("Mean:", mean)
    print("Std:", std)


if __name__ == "__main__":
    file_path = input("Enter Path to Dataset: ")
    compute_mean_std(file_path)
