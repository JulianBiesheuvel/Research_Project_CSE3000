import matplotlib.pyplot as plt
import csv
from torchvision.transforms.functional import *


def main(type_experiment, bwidth):

    fig = plt.figure()

    values = []

    with open('../vis3/distribution_' + type_experiment + '_train_hist_randrot.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            values.append(int(row[0]))

    bins = compute_histogram_bins(values, desired_bin_size=bwidth)

    n, bins, _ = plt.hist(x=values, bins=bins, alpha=0.5, color='steelblue', density=False, label="Data")

    plt.legend()
    plt.xlabel(f'Angle')
    plt.ylabel('Frequency')
    plt.title(f'Angle distribution of the {type_experiment} value on the train dataset')
    max_freq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(max_freq / 10) * 10 if max_freq % 10 else max_freq + 10)
    average_value = sum(values)/len(values)

    plt.axvline(average_value, color='k', linestyle='dashed', linewidth=1)

    min_ylim, max_ylim = plt.ylim()
    plt.text(average_value * 1.1, max_ylim * 0.9, 'Mean: {:.2f}'.format(average_value))

    plt.show()

    # File name of the histogram
    fnamesvg = '../vis3/distribution_' + type_experiment + '_train_hist_randrot_freq.svg'
    fnamepng = '../vis2/distribution_' + type_experiment + '_train_hist_randrot_greq.png'

    fig.savefig(fnamesvg, transparent=True, format="svg")
    plt.cla()
    plt.clf()
    fig.clear()



# Code from: https://stackoverflow.com/a/52323731/14998112
def compute_histogram_bins(data, desired_bin_size):
    min_val = np.min(data)
    max_val = np.max(data)
    min_boundary = -1.0 * (min_val % desired_bin_size - min_val)
    max_boundary = max_val - max_val % desired_bin_size + desired_bin_size
    n_bins = int((max_boundary - min_boundary) / desired_bin_size) + 1
    bins = np.linspace(min_boundary, max_boundary, n_bins)
    return bins


if __name__ == '__main__':
    experiments = [
        ['mean',3],
        ['variance', 3],
        ['std',3],
        ['median',3]
    ]

    for exp in experiments:
        main(exp[0], 4)

