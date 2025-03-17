import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.init import kaiming_uniform_
import torch.nn.functional as F

def get_param(*args):
    param = Parameter(torch.empty(args))
    kaiming_uniform_(param.data, mode='fan_in', nonlinearity='relu')
    return param
    
class PlaceConsumeLayer(nn.Module):
    def __init__(self, transition_num, dimension, dropout):
        super(PlaceConsumeLayer, self).__init__()
        self.transition_num = transition_num
        self.dimension = dimension
        self.C = nn.Embedding(transition_num, dimension) # Consume Embedding
        self.T = nn.Embedding(transition_num, 4)
        self.relu = nn.LeakyReLU()
        self.place_condition = nn.Sequential(
                nn.Linear(2*(dimension+4), dimension*2),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(dimension*2, dimension),
            )
        self.softmax = nn.Softmax(dim=1)
    
    def cal_embedding_weights(self, place_embeddings, transition_seq):
        """ Args:
        place_embeddings: (B', max_trace_len+1, K+4)
        transition_seq: (B')
        Returns:
            embedding_weights (B', max_trace_len+1, K).
        """
        consumed_conditions = F.relu(self.C(transition_seq)).unsqueeze(1).expand(-1, place_embeddings.shape[1], -1)
        consumed_time_conditions = self.T(transition_seq).unsqueeze(1).expand(-1, place_embeddings.shape[1], -1)
        # (B', max_trace_len+1, 2*(K+4))
        concatenated = torch.cat((place_embeddings, consumed_conditions, consumed_time_conditions), dim=2)
        
        # (B', max_trace_len+1, K)
        embedding_weights = self.place_condition(concatenated.view(-1, place_embeddings.shape[2]+self.dimension+4))
        embedding_weights = embedding_weights.view(place_embeddings.shape[0], place_embeddings.shape[1], self.dimension)
        
        mask = torch.all(place_embeddings == 0, dim=2)
        embedding_weights[mask] = torch.tensor(-1e9, device=place_embeddings.device)
        embedding_weights = self.softmax(embedding_weights)
        return embedding_weights
        
        
    def forward(self, marking_state, transition_seq):
        """ Args:
        marking_state: (B', max_trace_len+1, K+4)
        transition_seq: (B')
        Returns:
            consumed_marking_state `(B', max_trace_len+1, K+4)`.
        """
        transition_seq = transition_seq-1
        # embedding_weights (B', max_trace_len+1, K)
        embedding_weights = self.cal_embedding_weights(marking_state, transition_seq)
        consumed_place_embedding = torch.mul(embedding_weights, 
                                             F.relu(self.C(transition_seq)).unsqueeze(1).repeat(1, marking_state.shape[1], 1))
        return consumed_place_embedding
    
class TransitionJudgeLayer(nn.Module):
    def __init__(self, transition_num, dimension, dropout, beta, alpha=1e-10):
        super(TransitionJudgeLayer, self).__init__()
        self.transition_num = transition_num
        self.dimension = dimension
        
        self.condition_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dimension+4, dimension*2),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
            ) for _ in range(transition_num)
        ])
        self.fire_judge = nn.Sequential(
                nn.Linear(dimension*2, 1),
            )
        
        self.time_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dimension+4, dimension*2),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(dimension*2, 1),
            ) for _ in range(transition_num)
        ])
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.beta = beta
        self.alpha = alpha
        
    def forward(self, marking_state):
        """ Args:
        marking_state: (B, max_trace_len+1, K+4)
        Returns:
            transition_scores (B, T).
        """
        mask = torch.all(marking_state == 0, dim=2)
        # scores :(B, T)
        scores = torch.empty((marking_state.shape[0], self.transition_num), device=marking_state.device)
        for i, layer in enumerate(self.condition_layers):
            # condition:(B, max_trace_len+1, K)
            condition = layer(marking_state)
            condition = condition.masked_fill(mask.unsqueeze(-1).expand_as(condition), 0.0)
            
            # resource score (B,)
            resource_score = self.sigmoid(self.fire_judge(condition.sum(dim=1))).squeeze(-1)
            resource_score = torch.clamp(resource_score, min=1e-5)

            # time_condition (B, max_trace_len+1)
            time_condition = self.time_layers[i](marking_state).squeeze(-1)
            time_score, _ = torch.max(time_condition, dim=1)

            scores[:, i] = resource_score*time_score-self.beta*torch.exp(resource_score)
        
        return scores
    
class TransitionPlaceEmbeddingModel(nn.Module):
    def __init__(self, transition_num, dimension, dropout, beta):
        super(TransitionPlaceEmbeddingModel, self).__init__()
        self.transition_num = transition_num
        self.dimension = dimension
        self.G = nn.Embedding(transition_num+1, dimension) # Generate Embedding, the first is init_place
        self.place_consume = PlaceConsumeLayer(transition_num, dimension,dropout)
        self.judge_layer = TransitionJudgeLayer(transition_num, dimension, dropout, beta)
        
    def marking_update(self, marking, activity_seq, time_seq):
        """ Args:
        marking: (B, max_trace_len+1, K+4)
        activity_seq: (B, max_trace_len)
        time_seq: (B, 4, max_trace_len)
        Returns:
            Tensor output shape of `(B, max_trace_len+1, K+4)`.
        """
        for i in range(activity_seq.shape[1]):
            ongoing_idx = torch.nonzero(activity_seq[:,i]).view(-1)
            if len(ongoing_idx) > 0:
                generated_seq = activity_seq[ongoing_idx,i]
                unseen_idx = generated_seq == (self.transition_num + 1)
                if unseen_idx.any():
                    generated_seq = generated_seq[~unseen_idx]
                    ongoing_idx = ongoing_idx[~unseen_idx]
                # generated_marking (B', K+4)
                generated_emebdding = torch.cat((F.relu(self.G(generated_seq)), 
                                                 time_seq[ongoing_idx, :, i]), dim=1)
                # consumed_marking (B', max_trace_len+1, K)
                ongoing_marking = marking[ongoing_idx]
                consumed_place_embedding = self.place_consume(ongoing_marking, generated_seq)
                # marking_update
                marking[ongoing_idx, : , :self.dimension] =  F.relu(marking[ongoing_idx, : , :self.dimension] - consumed_place_embedding)
                marking[ongoing_idx, i+1] = generated_emebdding
        return marking
    
        
    def forward(self, batch_data):
        # activity_seq:(B, max_len)
        # time_seq:(B, 4, max_len)
        activity_seq = batch_data[:, 0, :].long()
        time_seq = batch_data[:, 1:, :]
        # marking (B, max_len+1, K+4)
        initial_marking = torch.zeros((activity_seq.shape[0], activity_seq.shape[1]+1, self.dimension+4),device=batch_data.device)
        initial_marking[:, 0, :self.dimension] = F.relu(self.G(torch.tensor([0],device=batch_data.device))).repeat(batch_data.shape[0], 1)
        
        marking_current= self.marking_update(initial_marking, activity_seq, time_seq)
        scores = self.judge_layer(marking_current)
        return scores