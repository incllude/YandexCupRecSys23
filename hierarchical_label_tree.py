import numpy as np
from utils import create_representation, normalize

def init_mu(representations):
    
    L = len(representations)
    mu_neg, mu_pos = np.random.choice(np.arange(L), 2, replace=False)
    mu_neg, mu_pos = representations[mu_pos], representations[mu_neg]
    
    return mu_neg, mu_pos

def optimize_mu(neg_cluster, pos_cluster, label_representations):
    
    mu_neg = create_representation(neg_cluster, label_representations)
    mu_pos = create_representation(pos_cluster, label_representations)

    return mu_neg, mu_pos

def calc_loss(neg_cluster, pos_cluster, mu_neg, mu_pos, label_representations):
    
    loss = 0.0
    
    for label in neg_cluster:
        loss += mu_neg @ label_representations[label]
        
    for label in pos_cluster:
        loss += mu_pos @ label_representations[label]
        
    return loss

def optimize_clusters(labels, representations, mu_neg, mu_pos):
    
    L = len(labels)
    ranks_and_scores = sorted(zip(labels, list(map(lambda x: (mu_pos - mu_neg) @ x, representations))), key=lambda x: x[1])
    ranked_labels = list(map(lambda x: x[0], ranks_and_scores))
    clusters = np.sign(np.arange(1, L+1) - (L+1)/2)
    if L % 2 == 1:
        clusters[(L+1)/2] = 1 if ranks_and_scores[(L+1)/2][1] >= 0 else -1
    
    neg_cluster, pos_cluster = [], []
    for label, cluster in zip(ranked_labels, clusters):
        
        if cluster == -1:
            neg_cluster.append(label)
        else:
            pos_cluster.append(label)
    
    return neg_cluster, pos_cluster

def split_on_clusters(labels, label_representations: Dict[int, np.array], clusterization):
    if len(labels) == 1:
        return [labels], [label_representations[labels[0]]]
    
    splitter = clusterization(n_clusters=2, n_init=10)

    representations = list(map(label_representations.get, labels))
    splitter.fit(representations)
    mu_neg, mu_pos = np.array(splitter.cluster_centers_)
    
    neg_cluster, pos_cluster = optimize_clusters(labels, representations, mu_neg, mu_pos)
    
    return (neg_cluster, pos_cluster), (mu_neg, mu_pos)

def build_hlt(labels, label_representation, clusterization):
    
    hlt = [[list(labels)]]
    hlt_representations = [np.zeros((0, embedding_size))]
    leaves_level = False

    while not leaves_level:

        level = []
        level_representations = []
        for cluster in hlt[-1]:

            clusters, representations = split_on_clusters(cluster, label_embeddings, clusterization)
            level.extend(clusters)
            level_representations.extend(representations)
        
        print(len(level))
        hlt.append(level)
        hlt_representations.append(np.array(level_representations))
        if len(level) == len(labels):
            leaves_level = True
            
    return hlt, hlt_representations

def squeeze_tree(tree):
    
    squeezed = []
    
    for d, level in enumerate(tree):
        
        if d % 2 == 0:
            squeezed.append(level)
            
    return squeezed