from data import TracksDataset, collate_fn
import torch
from torch.utils.data import DataLoader
import torch.functional as fn
import numpy as np
import matplotlib.pyplot as plt

class LossRecorder:
    
    def __init__(self, names='train'):
        
        self.records = None
        self.size = len(names) if not isinstance(names, str) else 1
        self.free()
        self.names = names if len(names) == self.size else [names] * self.size
        
    def add(self, scalar, to=0):
        
        self.records[to].append(scalar)
        
    def free(self):
    
        self.records = [[] for i in range(self.size)]
        
    def show(self, y_lim=(0, 1), size=(16, 16)):
        
        _, axs = plt.subplots(self.size, 1, figsize=(size))
        if self.size == 1:
            axs = [axs]
        
        for i, (record, name) in enumerate(zip(self.records, self.names)):

            axs[i].set_title(f"{name} loss")
            axs[i].set_xlabel("#iteration")
            axs[i].set_ylabel("loss")
            axs[i].set_ylim(y_lim)
            axs[i].plot(record, 'b')
            axs[i].grid(True)

def normalize(vector: np.array):
    
    return vector / np.linalg.norm(vector)

def create_representation(labels, label_representations):
    
    embeddings = list(map(label_representations.get, labels))
    representation = np.sum(embeddings, axis=0)
    representation = normalize(np.array(representation))
    
    return representation

def create_dataloader(batch_size, shuffle, indexes, size, predicted=None, labels=None):
    
    dataset = TracksDataset(
        path_to_embeddings,
        indexes,
        predicted=predicted,
        labels=labels
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda x: collate_fn(x, size, get_predicted=predicted is not None, get_labels=labels is not None)
    )
    
    return dataloader

def change_mask(mask):
    
    mask_changed = mask.clone()
    mask_changed.masked_fill_(mask == 0.0, True)
    mask_changed.masked_fill_(torch.isinf(mask), False)
    
    return mask_changed.bool()

def predict_modified(model, dataloader):
    
    logits_concated = []
    
    for inputs, mask in dataloader:

        logits = model.forward(inputs)
        logits = logits.squeeze(-1)
        logits = logits.detach().cpu()
        logits = torch.repeat_interleave(logits, number_of_leaves // logits.shape[1], dim=-1)
        logits_concated.extend(logits)
        
    logits_concated = torch.stack(logits_concated)
    probas = fn.sigmoid(logits_concated)
    
    return probas

def predict(model, dataloader, k, get_probas=False):
    
    logits_concated = []
    ancestors_concated = []
    
    for inputs, mask in dataloader:

        logits = model.forward(inputs, None)
        logits = logits.squeeze(-1)
        logits = logits.detach().cpu()
        mask = mask.detach().cpu()
        logits = torch.where(change_mask(mask), logits, float('-inf'))
        _, tops = torch.topk(logits, k=k, dim=-1)
        ancestors_concated.extend(tops.tolist())
        
        if get_probas:
        
            logits = torch.repeat_interleave(logits, number_of_leaves // logits.shape[1], dim=-1)
            logits_concated.extend(logits)
    
    if get_probas:    
        
        logits_concated = torch.stack(logits_concated)
        probas = fn.sigmoid(logits_concated)
        return probas, ancestors_concated
    
    return ancestors_concated
