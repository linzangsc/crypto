'''
define the network of crypto prediction
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class sub_graph(nn.Module):
    def __init__(self, input_channel, hidden_dim, history_window):
        super().__init__()
        self._input_channel = input_channel
        self._history_window = history_window
        self._hidden_dim = hidden_dim
        self._encoder = nn.Sequential(
            nn.Linear(input_channel, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

    def forward(self, data):
        # data shape: [batch_size, history_window, input_channel]
        node_feature = self._encoder(data)
        # node_feature shape: [batch_size, history_window, hidden_dim]
        aggregate_feature = torch.max(node_feature, dim=1, keepdim=True)[0]
        # aggregate_feature shape: [batch_size, 1, hidden_dim]
        aggregate_feature = aggregate_feature.repeat(1, self._history_window, 1)
        # aggregate_feature shape: [batch_size, history_window, hidden_dim]
        relation_feature = torch.cat([node_feature, aggregate_feature], dim=-1)
        # relation_feature shape: [batch_size, history_window, 2*hidden_dim]
        return relation_feature

class classification_node(nn.Module):
    def __init__(self, input_channel, output_channel, hidden_dim) -> None:
        super().__init__()
        self._hidden_layer = nn.Sequential(
            nn.Linear(input_channel, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.LayerNorm(hidden_dim // 8),
            nn.ReLU()
        )
        self._classification_head = nn.Linear(hidden_dim // 8, output_channel)

    def forward(self, feature):
        # fearure shape: [batch_size, hidden_dim]
        decode_feature = self._hidden_layer(feature)
        # decode_feature shape: [batch_szie, hidden_dim // 4]
        classification_output = self._classification_head(decode_feature)
        # classification_output shape: [batch_szie, output_channel]
        return F.sigmoid(classification_output)

class crypto_network(nn.Module):
    def __init__(self, input_channel, output_channel, hidden_dim, history_window):
        super().__init__()
        self._input_channel = input_channel
        self._output_channel = output_channel
        self._history_window = history_window
        self._hidden_dim = hidden_dim
        
        self._sub_graph_list = nn.Sequential(
            sub_graph(input_channel, hidden_dim, history_window),
            sub_graph(2*hidden_dim, 2*hidden_dim, history_window),
            sub_graph(4*hidden_dim, 4*hidden_dim, history_window)
        )
        # not share weight between rise and fall classification node
        self._rise_classification_node = \
            classification_node(8*hidden_dim, output_channel, hidden_dim)
        self._fall_classification_node = \
            classification_node(8*hidden_dim, output_channel, hidden_dim)

    def forward(self, data):
        # data shape: [batch_size, history_window, feature_dim]
        sequence_feature = self._sub_graph_list(data)
        # seq_feature shape: [batch_size, history_window, output_dim]
        sequence_feature = torch.max(sequence_feature, dim=1)[0]
        # seq_feature shape: [batch_size, output_dim]
        rise_prediction = self._rise_classification_node(sequence_feature)
        fall_prediction = self._fall_classification_node(sequence_feature)
        return rise_prediction, fall_prediction

if __name__ == "__main__":
    network = crypto_network(input_channel=6, output_channel=2, hidden_dim=16, history_window=3)
    data = np.arange(36).reshape((2, 3, 6))
    tensor_data = torch.Tensor(data)
    output = network(tensor_data)
    print(output)