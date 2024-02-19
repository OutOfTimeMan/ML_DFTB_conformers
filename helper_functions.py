import torch
import matplotlib.pyplot as plt
import dgl
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
from descriptastorus.descriptors import rdNormalizedDescriptors, rdDescriptors


def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.
    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.
    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / y_pred.shape[0]) * 100
    return acc


def plot_loss_curves(results, learning_rate, batch_size, figsize):
    loss = results["train_loss"]
    test_loss = results["test_loss"]
    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]
    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=figsize)

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.ylabel("Loss, kcal/mol", fontsize=16)
    plt.xlabel("Epoch", fontsize=16)

    # Set tick params
    plt.gca().tick_params(axis='both', which='both', labelsize=14)
    
    accuracy = [x*100 for x in accuracy]
    test_accuracy = [x*100 for x in test_accuracy]

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="Train accuracy")
    plt.plot(epochs, test_accuracy, label="Test accuracy")
    plt.ylabel("Accuracy, %", fontsize=16)
    plt.xlabel("Epoch", fontsize=16)
    plt.legend()

    # Set tick params
    plt.gca().tick_params(axis='both', which='both', labelsize=14)

def collate_molgraphs(data):
    """Batching a list of datapoints for dataloader.
    Parameters
    ----------
    data : list of 4-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES, a DGLGraph, all-task labels and a binary
        mask indicating the existence of labels.
    Returns
    -------
    smiles : list
        List of smiles
    bg : DGLGraph
        The batched DGLGraph.
    labels : Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    masks : Tensor of dtype float32 and shape (B, T)
        Batched datapoint binary mask, indicating the
        existence of labels.
    """
    smiles, graphs, labels  = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    return smiles, bg, labels

def collate_molgraphs_conf(data):
    smiles, graphs, conf, labels  = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    conf = torch.stack(conf, dim=0)
    labels = torch.stack(labels, dim=0)

    return smiles, bg, conf, labels

def is_valid_class(lst):
    lst = list(map(lambda x: x > 0.754, lst))
    result = [float(item) for item in lst]
    return result

def is_valid_class_metric_max(lst, boundary):
    lst = list(map(lambda x: x > boundary, lst))
    result = [float(item) for item in lst]
    return result

def plot_conf_matrix(test, predicted):
    cf_matrix=confusion_matrix(test, predicted)
    ax= plt.subplot()
    sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True,fmt='.2%', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')

def plot_roc_auc(test, predicted):
    fpr, tpr, _ = roc_curve(test, predicted)
    auc = roc_auc_score(test, predicted)
    plt.plot(fpr,tpr, label="AUC = "+str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()

def smiles_to_normalized_features(smiles):
    generator = rdNormalizedDescriptors.RDKit2DNormalized()
    results = generator.process(smiles)
    processed, features = results[0], results[1:]
    if processed is None:
        pass
    return features
