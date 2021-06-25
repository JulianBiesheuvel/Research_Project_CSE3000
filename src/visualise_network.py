import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
from datetime import date
from numpy.random import randint
from config import *
from sklearn.metrics import mean_absolute_error
from network import Net
from custom_dataset import CustomMNISTDataset
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torch.backends import cudnn
from torchvision.transforms import transforms
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import *


# https://debuggercafe.com/visualizing-filters-and-feature-maps-in-convolutional-neural-networks-using-pytorch/
# https://androidkt.com/how-to-visualize-feature-maps-in-convolutional-neural-networks-using-pytorch/

def main(type_ex):
    test = datasets.MNIST(root='../data', train=False, transform=Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))])
                          , download=True)

    test_dataset = CustomMNISTDataset(test.data, transform=transforms.ToTensor, is_test=True, experiment=type_ex)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE_TEST, shuffle=False)

    # load the model
    model = Net()
    model.load_state_dict(torch.load("../models_negatives/model_" + type_ex + ".pth"))
    model.eval()
    print(model)
    model_weights = []
    conv_layers = []

    model_children = list(model.children())
    # counter to keep count of the conv layers
    counter = 0
    # append all the conv layers and their respective weights to the list
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter += 1
                        model_weights.append(child.weight)
                        conv_layers.append(child)
    # print(f"Total convolutional layers: {counter}")

    # take a look at the conv layers and the respective weights
    # for weight, conv in zip(model_weights, conv_layers):
    #     print(f"WEIGHT: {weight} \nSHAPE: {weight.shape}")
    #     print(f"CONV: {conv} ====> SHAPE: {weight.shape}")

    # visualize the first conv layer filters
    plt.figure(figsize=(20, 17))
    for i, filter in enumerate(model_weights[0]):
        plt.subplot(8, 8, i + 1)  # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
        plt.imshow(filter[0, :, :].detach(), cmap='gray')
        plt.axis('off')
        plt.savefig('../network_visuals/models_negatives/filters_layer_0_model_' + type_ex + '.png')
    # plt.show()

    # visualize the first conv layer filters
    plt.figure(figsize=(20, 17))
    for i, filter in enumerate(model_weights[1]):
        plt.subplot(8, 8, i + 1)  # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
        plt.imshow(filter[0, :, :].detach(), cmap='gray')
        plt.axis('off')
        plt.savefig('../network_visuals/models_negatives/filters_layer_1_model_' + type_ex + '.png')

    plt.cla()
    plt.clf()

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    plt.imshow(example_data[1].squeeze(0), cmap='gray', interpolation='none')
    # plt.show()
    plt.savefig("../network_visuals/models_negatives/" + type_ex + "_digit.png")

    img = (example_data[1]).unsqueeze(0).float()

    # pass the image through all the layers
    results = [conv_layers[0](img)]
    for i in range(1, len(conv_layers)):
        # pass the result from the last layer to the next layer
        results.append(conv_layers[i](results[-1]))
    # make a copy of the `results`
    outputs = results

    # visualize 64 features from each layer
    # (although there are more feature maps in the upper layers)
    for num_layer in range(len(outputs)):
        plt.figure(figsize=(30, 30))
        layer_viz = outputs[num_layer][0, :, :, :]
        layer_viz = layer_viz.data
        print(layer_viz.size())
        for i, filter in enumerate(layer_viz):
            if i == 64:  # we will visualize only 8x8 blocks from each layer
                break
            plt.subplot(8, 8, i + 1)
            plt.imshow(filter, cmap='gray')
            plt.axis("off")
        print(f"Saving layer {num_layer} feature maps...")
        plt.savefig(f"../network_visuals/models_negatives/layer_{num_layer}_" + type_ex + ".png")
        # plt.show()
        plt.close()


if __name__ == '__main__':

    types = [
        'mean',
        'median',
        'std',
        'variance'
    ]

    for er in types:
        main(er)