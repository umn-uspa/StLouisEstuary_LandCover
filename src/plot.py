#!/usr/bin/env python
'''
Code plotting in Jupyter Notebooks

Author:Olena Boiko

'''

import os
import rasterio
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from base64 import b16encode

def plot_learning_history(history):
    '''
    Plot the training and validation accuracy and loss at each epoch.

    Parameters
    ----------
    history: keras History
        Model fitting history after training

    Returns
    -------
    None (Displays the figure)
    '''
    #plot the training and validation IoU and loss at each epoch
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    acc = history.history['iou_score']
    val_acc = history.history['val_iou_score']
    fig, axs = plt.subplots(1, 2, figsize=(10,3), tight_layout=True)
    axs[0].plot(epochs, loss, label='Training loss')
    axs[0].plot(epochs, val_loss, label='Validation loss')
    axs[0].set_title('Training and validation loss', weight="bold")
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[1].plot(epochs, acc, label='Training IoU')
    axs[1].plot(epochs, val_acc, label='Validation IoU')
    axs[1].set_title('Training and validation IoU', weight="bold")
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('IoU')
    axs[1].legend()
    plt.show()

def plot_example_maps(rgb_array, classified_array, palette, display_labels):
    '''
    Plots the for sample aerial image and classified map side by side for a small area.

    Parameters
    ----------
    rgb_array: array
        Array with 3 bands - red, green, and blue
    classified_array: array
        Array with with 1 band showing classified results
    palette: dict
        Color palette to use
    display_labels: list
        Class names used for the legend

    Returns
    -------
    None (Displays the figure)
    '''
    # transform color palette to a list of strings and pre-build a legend
    clrs = [(b'#' + b16encode(bytes(palette[l]))).decode() for l in palette.keys()]
    legend = [ mpatches.Patch(facecolor=clrs[i]) for i in range(len(display_labels))]
    # apply color palette to classified array
    classified_display = np.array([
        list(map(palette.__getitem__, row)) for row in classified_array])
    # make fig suplots and visualize arrays
    fig, axs = plt.subplots(1, 2, figsize=(12,4), tight_layout=True)
    axs[0].imshow(rgb_array)
    axs[1].imshow(classified_display)
    axs[0].axis("off")
    axs[1].axis("off")
    axs[1].legend(legend, display_labels, ncol=1, loc="upper left",
                  bbox_to_anchor=(1, 1))
    plt.show()

def plot_confusion_matrix(true, pred, display_labels):
    '''
    Plot the confusion matrix with some formatting.

    Parameters
    ----------
    true: array
        Ground truth (correct) value
    pred: array
        Estimated values as returned by the classifier
    display_labels: list
        Class names used to label axes of the plot

    Returns
    -------
    None (Displays the figure)
    '''
    # uses sklearn confusion matrix with some formatting
    fig, ax = plt.subplots()
    cm = confusion_matrix(true, pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap="GnBu", ax=ax)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    plt.show()