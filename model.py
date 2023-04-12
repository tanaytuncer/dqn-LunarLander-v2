"""
Actor Policy Model
Convolutional Neural Net with 3 conv layers and two linear layers
Author: Tanay Tunçer
"""

import torch
import torch.nn as nn


class DQN(self):
    """Convolutional Neural Net with 3 conv layers and two linear layers"""

    def __init__(self, state_size, action_size):
        """
        Intitialization of parameters and build model structure.
        ===
        Params:
            
        """

        super(DQN, self).__init__()
        self.seed = torch.manual_seed(2023)
        self.conv_1d = nn.Sequential(

            nn.Conv1d(state_size, 32, kernel_size = 8, stride = 4),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size = 4, stride = 2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size = 3, stride = 1),
            nn.ReLU()
        )

        conv_1d_output_size = self.n_size(self.conv_1d())
        self.fc = nn.Sequential(
            nn.Linear(conv_1d_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )

    def n_size(self):
        n_size = self.conv_1d().shape
        return n_size[0]

    def forward(self, s):
        x = self.conv_1d(s).view(s.size()[0], -1) 
        return self.fc(x)
