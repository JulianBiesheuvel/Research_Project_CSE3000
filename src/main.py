import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
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


# Code from: https://nextjournal.com/gkoehler/pytorch-mnist
def main(random_seed, type_experiment, metric_val):
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    train = datasets.MNIST(root='../data', train=True, transform=Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))])
                           , download=True)

    test = datasets.MNIST(root='../data', train=False, transform=Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))])
                          , download=True)

    train_dataset = CustomMNISTDataset(train.data, transform=transforms.ToTensor, apply_transform=False,
                                       experiment=type_experiment)
    test_dataset = CustomMNISTDataset(test.data, transform=transforms.ToTensor, apply_transform=True,
                                      experiment=type_experiment)
    train_dataset, validation_dataset = random_split(train_dataset, [40000, 20000])

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, drop_last=True)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=BATCH_SIZE_TEST, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE_TEST, shuffle=True)

    # Examples of the train dataset
    # plot_examples(type_experiment, train_loader)
    # Examples of the test dataset
    # plot_examples(type_experiment, test_loader)

    # breakpoint()

    network = Net()

    optimizer = optim.Adam(network.parameters(), lr=LEARNING_RATE)
    criterion = nn.L1Loss()

    train_losses = []
    train_counter = []
    test_losses = []
    validation_losses = []

    network.eval()

    # Determine the baseline on the training set
    # We do this by taking the average value of the train dataset and calculate
    # the MAE of the average value and the train dataset

    values_test_dataset = [data[1][1].item() for data in enumerate(test_dataset)]

    print('Before training: ' + computeBaseline(values_test_dataset, metric_val) + ', of the testing set')

    # Make a random guess with the test set on untrained network
    testing(network, test_losses, test_loader, False, type_experiment, None, random_seed, metric_val)
    # testing(network, validation_losses, validation_loader, False, type_experiment, None, random_seed, metric_val)

    n_epochs_stop = 5
    epochs_no_improve = 0
    early_stop = False
    min_val_loss = np.Inf

    for epoch in range(1, EPOCHS + 1):

        if early_stop:
            print(f'Stopped! at epoch {epoch - 1}')
            break

        network.train()

        for batch_idx, (data, labels) in enumerate(train_loader):
            # forward feed
            pred = network(data.float())

            # calculate the loss
            loss = criterion(pred, labels.unsqueeze(1).float())

            # backward propagation: calculate gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # clear out the gradients from the last step loss.backward()
            optimizer.zero_grad()

            if batch_idx % 10 == 0:
                # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.2f}'.format(
                #     epoch, batch_idx * len(data), len(train_dataset),
                #            100. * batch_idx / len(train_loader.dataset), loss.item()))

                # Here we save the loss of the training, we need this to compare wiht the validation loss


                train_losses.append(loss.item())
                train_counter.append(
                    (batch_idx * 64) + ((epoch - 1) * len(train_dataset)))
                torch.save(network.state_dict(), '../models5/model_' + type_experiment + '.pth')


        testing(network, validation_losses, validation_loader, False, type_experiment, None, random_seed, metric_val)

        last_validation_value = validation_losses[len(validation_losses) - 1]

        # Here we start checking for the early loss
        # Code from: https://www.kaggle.com/akhileshrai/tutorial-early-stopping-vanilla-rnn-pytorch
        if last_validation_value < min_val_loss:
            # This case the validation loss still decreases
            # print('komt bij de if')
            epochs_no_improve = 0
            min_val_loss = last_validation_value
        else:
            # This case the last validation loss did not decrease, compared to training
            # But we do not immediately stop
            # print('komt bij de else')
            epochs_no_improve += 1

        # print(f'epoch: {epoch} and epochs_no_impr: {epochs_no_improve}')
        if epoch > 5 and epochs_no_improve == n_epochs_stop:
            print('Early Stopping!')
            early_stop = True

    idd = randint(0, 10000)

    validation_counter = [j * len(train_dataset) for j in range(len(validation_losses))]

    test_counter = [0, len(train_dataset) * len(validation_losses)]

    testing(network, test_losses, test_loader, True, type_experiment, idd, random_seed, metric_val)

    if random_seed == 0:
         plot_losses(idd, type_experiment, random_seed, train_counter, train_losses, validation_counter, validation_losses, test_counter, test_losses)

    del network, optimizer, criterion, train_losses, train_counter, validation_losses, test_dataset, \
        validation_counter, test_counter, train_loader, test_loader, validation_loader

    print('Finished!')

    return test_losses[1]


# Compute the baseline for each metric
# We want the baseline to compare in the future how well our model did
# Preferably our model of course performs better than the baseline
def computeBaseline(dataset, mean_val):
    # Create a set of the size of the training data
    y_pred = [mean_val for _ in range(len(dataset))]

    # MAE = mea(ground-truth, average_value_metric)
    mae = mean_absolute_error(dataset, y_pred)

    return f'The baseline is: {mae}'


def plot_examples(expp, loader):
    examples = enumerate(loader)
    batch_idx, (example_data, example_targets) = next(examples)

    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i].squeeze(0), cmap='gray', interpolation='none')
        plt.title(f"{expp} : {example_targets[i]:.1f}")
        plt.xticks([])
        plt.yticks([])
    plt.show()

    # plt.savefig("../plots/losses_" + expp + "_90_rotation_3.png")
    # for i in range(6):
    #     plt.subplot(2, 3, i + 1)
    #     plt.tight_layout()
    #     plt.imshow(rotate(example_data[i], 180, interpolation=InterpolationMode.BILINEAR,fill=[0, ]).squeeze(0), cmap='gray', interpolation='none')
    #     plt.title(f"{expp} : {torch.std(rotate(example_data[i], 180, interpolation=InterpolationMode.BILINEAR, fill=[0, ]).squeeze(0).float()):.1f}")
    #     plt.xticks([])
    #     plt.yticks([])
    # plt.show()
    # plt.savefig("../plots/losses_" + expp + "_90_rotation_4.png")


def plot_losses(id, experiment, loop, train_counter, train_losses, validation_counter, validation_losses, test_counter,
                test_losses):
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.plot(train_counter, train_losses, color='#B96CD8', )
    ax.plot(validation_counter, validation_losses, color='#ffa629')
    ax.scatter(test_counter, test_losses, color='#2741e8')

    # minposs = np.argmin(validation_losses)
    # 
    # val_line = validation_counter[len(validation_counter) - 1]
    # 
    # if minposs != validation_losses[len(validation_losses) - 1]:
    #     val_line = validation_counter[minposs]
    # 
    # plt.axvline(val_line, linestyle='--', color='r')

    # Add legend, title and axis labels
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    lgd = ax.legend(['Train Loss', 'Validation Loss', 'Test Loss'], loc='upper center', bbox_to_anchor=(0.5, -0.15),
                    fancybox=True, shadow=True, ncol=5)
    ax.ticklabel_format(axis='x', style='sci', scilimits=(1, 4))
    ax.set_title('Train-, validation, test- loss %s value, 90 degrees' % experiment)
    ax.set_xlabel('number of training examples seen')
    ax.set_ylabel('Mean Absolute Error Loss')

    fname = "../plots/mult5/losses_" + experiment + "_90_degree_rotated_" + str(id) + ".svg"

    writeToFile(experiment, fname)

    plt.savefig(fname, transparent=True, format="svg")

    plt.cla()
    plt.clf()
    ax.cla()

    fig.clear()


def testing(network, losses, loader, test, type_exp, id, random_seed, metric_val):
    network.eval()

    val_predicted = []
    val_actual_means = []

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(loader):
            # forward feed
            pred = network(data.float())

            # calculate the loss
            val_predicted.append(pred.numpy())
            val_actual_means.append(labels.numpy())

    flat_predicted = val_predicted[0].squeeze()

    flat_actual_means = val_actual_means[0].squeeze()

    if test:
        print('After training the performance is: ' + str(mean_absolute_error(flat_actual_means, flat_predicted)) + ', of the predicted values by the network and ground-truth values.')

    mae = mean_absolute_error(flat_actual_means, flat_predicted)

    if test and random_seed == 0:
        fig = plt.figure()
        ax = fig.add_subplot()

        # This line plots the average value of each of the metrics in the scatter plot,
        # if this want to be used, the mean value should be parameterized for this method
        # and the line should be added to the legend
        plt.axhline(y=metric_val, color='steelblue', linestyle='--')

        ax.scatter(range(0, 100), flat_predicted[:100], color='red')
        ax.scatter(range(0, 100), flat_actual_means[:100], color='orange')
        ax.set_xlabel("Number of Examples")
        ax.set_ylabel(f"{type_exp} value")

        # Add legend, title and axis labels
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        lgd = ax.legend(['baseline', 'Predicted labels', 'Ground Truth'], loc='upper center',
                        bbox_to_anchor=(0.5, -0.15),
                        fancybox=True, shadow=True, ncol=5)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(1, 4))
        ax.set_title('The difference between %s value - 90 degrees' % type_exp)

        fname = "../plots/mult5/scatter_" + type_exp + "_90_degrees_rotation_" + str(id) + ".svg"

        plt.savefig(fname, transparent=True, format="svg")

        plt.cla()
        plt.clf()

        fig.clear()

    losses.append(mae)

    print('\nTest set: Avg. loss: {:.2f}\n'.format(mae))


# Code taken from https://thispointer.com/how-to-append-text-or-lines-to-a-file-in-python/
def writeToFile(experiment, fname):
    file_name = '../plots/info5'

    today = date.today()

    lines_to_append = [
        today.strftime("%B %d, %Y"),
        'Dataset:' + str(experiment) + '_validation',
        '- learning_rate : ' + str(LEARNING_RATE),
        '- batch_size_train : ' + str(BATCH_SIZE_TRAIN),
        '- batch_size_test : ' + str(BATCH_SIZE_TEST),
        '- batch_size_val : ' + str(BATCH_SIZE_VAl),
        '- random_seed : ' + str(RANDOM_SEED),
        '- num_epoch : ' + str(EPOCHS),
        '- train_dataset_size : ' + str(DATASET_SIZE_TRAIN),
        '- test_val_dataset_size : ' + str(DATASET_SIZE_TEST_VAL),
        '- model :',
        '- fname_training_loss_file : ' + fname,
        '\n'
    ]

    with open(file_name, "a+") as file_object:
        appendEOL = False

        file_object.seek(0)

        data = file_object.read(100)
        if len(data) > 0:
            appendEOL = True

        for line in lines_to_append:

            if appendEOL:
                file_object.write("\n")
            else:
                appendEOL = True

            file_object.write(line)


if __name__ == '__main__':

    # experiments = [
    #     ['mean', 32.82],
    #     ['median', 0],
    #     ['std', 76.38]
    # ]

    # experiments = [
    #     'mean',
    #     'std',
    #     'median'
    # ]

    # all_runs = []
    #
    # for i in range(10):
    #     run = []
    #     for exp in experiments:
    #         run.append(main(i, exp))
    #
    # print(all_runs)

    means = []
    std = []
    median = []

    for i in range(10):
        means.append(main(i, 'mean', 32))

    print(means)

    for j in range(10):
        std.append(main(j, 'std', 77))

    print(std)

    for k in range(10):
        median.append(main(k, 'median', 0))

    print(median)
    #
    # variance = []
    #
    # for l in range(10):
    #     variance.append(main(l, 'variance'))
    #
    # print(variance)
