from torch.utils.data import Dataset
import torch
from math import ceil

class TracksDataset(Dataset):
    
    def __init__(self, path_to_data, idxs, predicted=None, labels=None, transform=None):
        
        self.path_to_data = path_to_data
        self.indexes = idxs
        self.labels = labels
        self.predicted = predicted
        
        if self.predicted is not None:
            self.predicted = [lower_by_level(x) for x in predicted]
                
    def __len__(self):
        
        return len(self.indexes)
    
    def __getitem__(self, i):
        
        embeddings = torch.from_numpy(np.load(f'{self.path_to_data}/{self.indexes[i]}.npy')).float()
        yield embeddings
        
        if self.predicted is not None:
            yield self.predicted[i]
            
        if self.labels is not None:
            yield self.labels[i]
        
def lower_by_level(indexes):
    
    return sum([list(range(c * level_growth, (c + 1) * level_growth)) for c in indexes], [])
        
def change_list_of_indexes(indexes, rule):
    
    return [list(set(map(rule, x))) for x in indexes]

def collate_fn(batch, level_size, get_predicted=True, get_labels=True):

    if get_predicted and get_labels:
        inputs, predicted, labels = zip(*batch)
    elif get_predicted:
        inputs, predicted = zip(*batch)
    elif get_labels:
        inputs, labels = zip(*batch)

    batch_size = len(inputs)
    max_len = max([tsr.size(0) for tsr in inputs])
    inputs_repeated = [tsr.repeat(ceil(max_len / tsr.size(0)), 1)[:max_len] for tsr in inputs]
    inputs = torch.stack(inputs_repeated)
    yield inputs.to(device).detach()
    
    if get_predicted:
        
        masked_predicted = torch.full((batch_size, level_size), float('-inf'))
        masked_predicted = masked_predicted.scatter(1, torch.tensor(predicted), 0.0)
        yield masked_predicted.to(device).detach()
    
    if get_labels:
        
        masked_labels = torch.zeros((batch_size, level_size))
        masked_labels[sum([[i] * len(x) for i, x in enumerate(labels)], []), sum(labels, [])] = 1.0
        yield masked_labels.to(device).detach()

def translate_labels_to_clusters(labels, clusters):
    
    label_to_cluster = dict(sum([[(label, i) for label in cluster] for i, cluster in enumerate(clusters)], []))
    labels = change_list_of_indexes(labels, label_to_cluster.get)
    
    return labels

def change_mask(mask):
    
    mask_changed = mask.clone()
    mask_changed.masked_fill_(mask == 0.0, True)
    mask_changed.masked_fill_(torch.isinf(mask), False)
    
    return mask_changed.bool()