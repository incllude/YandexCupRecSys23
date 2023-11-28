import torch.nn as nn
import torch

class AttentionXMLModified(nn.Module):
    
    def __init__(self, clusters_embeddings, inner_size, layer_size, ffnn_size, copy_model=None):
        super(AttentionXMLModified, self).__init__()
        
        head_size = clusters_embeddings.size(0)
        self.layer_size = range(layer_size)
        self.ffnn_size = ffnn_size
        self.bilstm = nn.LSTM(inner_size, inner_size // 2, batch_first=True, bidirectional=True)
        self.label_embeddings = nn.Parameter(clusters_embeddings.clone().to(device), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)
        self.ffnn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(inner_size, ffnn_size),
                nn.GELU(),
                nn.Linear(ffnn_size, inner_size),
                nn.LayerNorm(inner_size)
            )
            for _ in self.layer_size
        ])
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()
        self.mask = nn.Parameter(torch.eye(head_size), requires_grad=False)
        self.ones = nn.Parameter(torch.ones(head_size), requires_grad=False)
        self.output = nn.Parameter(clusters_embeddings.clone().to(device).T, requires_grad=True)
        
        if copy_model is not None:
            self.copy_weights(copy_model)
    
    def copy_weights(self, model):
        
        self.bilstm.load_state_dict(model.bilstm.state_dict())
    
    def apply_attention(self, query, key, value):
    
        scores = query @ key.mT
        weights = self.softmax(scores)
        result = value.mT @ weights.mT
        result = result.mT

        return result    
    
    def forward(self, input_sequence):
        
        features = self.bilstm(input_sequence)[0]
        query = self.label_embeddings
        query = query.repeat(input_sequence.size(0), 1, 1)
        
        for layer, ffnn in zip(self.layer_size, self.ffnn):
            
            after_self_attention = self.apply_attention(query, query, query)
            after_cross_attention = self.apply_attention(after_self_attention, features, features)
            query = ffnn(after_cross_attention)

        output_logits = query @ self.output
        output_logits = output_logits * self.mask
        output_logits = output_logits @ self.ones
        
        return output_logits
    

class AttentionXML(nn.Module):
    
    def __init__(self, clusters_embeddings, inner_size, density_layer_size):
        super(AttentionXML, self).__init__()
        
        head_size = clusters_embeddings.size(0)
        self.bilstm = nn.LSTM(inner_size, inner_size // 2, bidirectional=True, batch_first=True)
        self.attention = nn.Linear(inner_size, head_size, bias=False)
        self.attention.weight.data = clusters_embeddings.clone()
        self.softmax = nn.Softmax(dim=-1)
        self.density = nn.Linear(inner_size, density_layer_size)
        self.output = nn.Linear(density_layer_size, 1)
        
    def forward(self, input_sequence, mask):
        
        hidden_representations = self.bilstm(input_sequence)[0]
        scores = self.attention(hidden_representations)
        scores_masked = (scores.permute(1, 0, 2) + mask).permute(1, 0, 2)
        weights = self.softmax(scores_masked)
        attentioned = (hidden_representations.mT @ weights).mT
        dense = self.density(attentioned)
        output_logits = self.output(dense)
        
        return output_logits