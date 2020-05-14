import numpy as np
import torch
from tqdm.auto import tqdm
from matplotlib import pyplot as plt


def plot_confusion_matrix(model, test_loader, path):
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    confusion_matrix = np.zeros([200, 200], int)

    with torch.no_grad():
        for item in tqdm(test_loader):
            images, labels = item.images.to(DEVICE), item.labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            for i, l in enumerate(labels):
                confusion_matrix[l.item(), predicted[i].item()] += 1


    plt.figure(figsize=(12, 12))
    plt.imshow(confusion_matrix, cmap="gist_heat", interpolation="nearest")
    plt.savefig(path)