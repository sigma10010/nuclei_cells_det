import random
import numpy as np
import torch


def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)

def pair_embeddings(embeddings, labels):
    embeddings1 = []
    embeddings2 = []
    targets = []
    
    labels = labels.squeeze()
    labels_set = set(labels.cpu().numpy())
    dic_label_indices = label_to_indices(labels)
    
    for index, embedding in enumerate(embeddings):
        embedding1 = embedding.unsqueeze(0)
        embeddings1.append(embedding1)
        
        label1 = int(labels[index])
        target = np.random.randint(0, 2)
        targets.append(torch.Tensor([target]).unsqueeze(0).to(dtype=torch.float32))
        
        if target == 1:
            siamese_index = index
            while siamese_index == index:
                siamese_index = np.random.choice(dic_label_indices[label1])
        else:
            siamese_label = np.random.choice(list(labels_set - set([label1])))
            siamese_index = np.random.choice(dic_label_indices[siamese_label])
        embedding2 = embeddings[siamese_index].unsqueeze(0)
        embeddings2.append(embedding2)
        
    return cat(embeddings1), cat(embeddings2), cat(targets)

def triplet_embeddings(embeddings, labels):
    """Arguments:
        embeddings (Tensor [N*dim_embed])
        labels (Tensor [N*1])
        outputs:
        
    """
    embeddings1 = []
    embeddings2 = []
    embeddings3 = []
    
    labels = labels.squeeze()
    labels_set = set(labels.cpu().numpy())
    dic_label_indices = label_to_indices(labels)
    
    for index, embedding in enumerate(embeddings):
        embedding1 = embedding.unsqueeze(0)
        embeddings1.append(embedding1)
        
        label1 = int(labels[index])
        positive_index = index
        while positive_index == index:
            positive_index = np.random.choice(dic_label_indices[label1])
        embedding2 = embeddings[positive_index].unsqueeze(0)
        embeddings2.append(embedding2)
        
        negative_label = np.random.choice(list(labels_set - set([label1])))
        negative_index = np.random.choice(dic_label_indices[negative_label])   
        embedding3 = embeddings[negative_index].unsqueeze(0)
        embeddings3.append(embedding3)
    
    
    return cat(embeddings1), cat(embeddings2), cat(embeddings3)

def label_to_indices(labels):
    """Arguments:
        labels (Tensor [N,])
        
    """
    labels = labels.squeeze().cpu().numpy()
    labels_set = set(labels)
    return {label: np.where(labels == label)[0] for label in labels_set}