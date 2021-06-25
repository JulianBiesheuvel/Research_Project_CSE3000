import matplotlib.pyplot as plt
import numpy as np
import torchvision
from config import *
from custom_dataset import CustomMNISTDataset
from torchvision import datasets
from torch.backends import cudnn
from torchvision.transforms import transforms
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import *


def main(type_experiment, bwidth):
    # Load the Datasets and apply normalisation
    torch.backends.cudnn.enabled = False
    torch.manual_seed(1)

    train = datasets.MNIST(root='../data', train=True, transform=Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))])
                           , download=True)

    test = datasets.MNIST(root='../data', train=False, transform=Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))])
                          , download=True)

    # Create the Custom Datasets
    train_dataset = CustomMNISTDataset(train.data, transform=transforms.ToTensor, apply_transform=False,
                                       experiment=type_experiment)
    test_dataset = CustomMNISTDataset(test.data, transform=transforms.ToTensor, apply_transform=True,
                                      experiment=type_experiment)


    makeImage(test_dataset, type_experiment)
    # fig = plt.figure()
    #
    # values = []
    #
    # for data in enumerate(train_dataset):
    #     values.append(int(data[1][1].item()))
    #
    # bins = compute_histogram_bins(values, desired_bin_size=bwidth)
    #
    # n, bins, _ = plt.hist(x=values, bins=bins, alpha=0.5, color='steelblue', density=False, label="Data")
    #
    # typet = 'random_rot'
    # typedata = 'train_'
    #
    # plt.legend()
    # plt.xlabel(f'{type_experiment} value')
    # plt.ylabel('Frequency')
    # plt.title(f'Distribution of the {type_experiment} value on the test dataset - {typet}')
    # max_freq = n.max()
    # # Set a clean upper y-axis limit.
    # plt.ylim(ymax=np.ceil(max_freq / 10) * 10 if max_freq % 10 else max_freq + 10)
    #
    # median_value = np.median(values)
    #
    # plt.axvline(median_value, color='k', linestyle='dashed', linewidth=1)
    #
    # min_ylim, max_ylim = plt.ylim()
    # plt.text(median_value * 1.1, max_ylim * 0.9, 'Median: {:.2f}'.format(median_value))
    #
    # plt.show()
    #
    # # File name of the histogram
    # fnamesvg = '../vis3/distribution_' + type_experiment + typedata + typet + '.svg'
    #
    # fig.savefig(fnamesvg, transparent=True, format="svg")
    # plt.cla()
    # plt.clf()
    # fig.clear()


# Code from: https://stackoverflow.com/a/52323731/14998112
def compute_histogram_bins(data, desired_bin_size):
    min_val = np.min(data)
    max_val = np.max(data)
    min_boundary = -1.0 * (min_val % desired_bin_size - min_val)
    max_boundary = max_val - max_val % desired_bin_size + desired_bin_size
    n_bins = int((max_boundary - min_boundary) / desired_bin_size) + 1
    bins = np.linspace(min_boundary, max_boundary, n_bins)
    return bins

# Make grid of the images of either the training or test dataset with metric val
def makeImage(dataset, type):
    # Create the Data Loaders
    # train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE_TEST, shuffle=True)

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    fig = plt.figure()

    # Plot images from the train Dataloader with their labels
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i].squeeze(0), cmap='gray', interpolation='none')
        plt.title(f"{type} : {example_targets[i]:.0f}")
        plt.xticks([])
        plt.yticks([])

    # Either show the images or save them to the disk
    # plt.show()
    # fname = '../vis/180_degree_rotation_' + type + '.svg'
    plt.savefig('negative.png', transparent=True)
    #
    # These lines is to clear the data from the previous canvas
    # plt.cla()
    # plt.clf()
    # fig.clear()


if __name__ == '__main__':
    # experiments = [
    #     ['mean',1],
    #     ['variance', 300],
    #     ['std',2],
    #     ['median',1]
    # ]
    #
    # for exp in experiments:
    #     main(exp[0], exp[1])

    main('mean', 1)





