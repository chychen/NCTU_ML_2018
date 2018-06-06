from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from data_loader import load_mnist

colors = ['b', 'c', 'y', 'm', 'r']


def lda_plot(inputs, labels, with_legend=True):
    # LDA
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda_data = lda.fit_transform(inputs, labels)
    # Visualization
    fig, ax = plt.subplots()
    scatter_obj = []
    for i in range(5):
        indecies = (labels == (i+1))
        scatter_obj.append(ax.scatter(
            lda_data[indecies, 0], lda_data[indecies, 1], c=colors[i], s=8.0, alpha=0.5))
    if with_legend:
        ax.legend(tuple(scatter_obj),
                ('1', '2', '3', '4', '5'),
                scatterpoints=1,
                loc='lower left',
                fontsize=8)
    plt.show(block=True)

def main():
    data = load_mnist()
    lda_plot(data['X_train'], data['T_train'])


if __name__ == '__main__':
    main()
