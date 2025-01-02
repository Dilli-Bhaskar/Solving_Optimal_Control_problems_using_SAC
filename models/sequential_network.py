import torch
from torch import nn


class Seq_Network(nn.Module):
    def __init__(self, layers, hidden_activation, output_activation=None):
        super().__init__()
        hidden_layers = layers[:-1]
        network = [nn.Sequential(nn.Linear(i, o), hidden_activation) for i, o in
                   zip(hidden_layers, hidden_layers[1:])]
        network.append(nn.Linear(layers[-2], layers[-1]))
        if output_activation:
            network.append(output_activation)
        self.network = nn.Sequential(*network)
        self.apply(self._init_weights_)

    def forward(self, tensor):
        return self.network(tensor)

    @staticmethod
    def _init_weights_(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01)
            
            
class Mu_Network(nn.Module):
    def __init__(self, layers, hidden_activation, output_activation=None):
        super().__init__()
        hidden_layers = layers[:-1]
        shared_layers = [nn.Sequential(nn.Linear(i, o), hidden_activation) for i, o in
                         zip(hidden_layers, hidden_layers[1:])]
        
        self.mean_layers = nn.Sequential(*shared_layers, nn.Linear(layers[-2], layers[-1]))
        self.log_std_layers = nn.Sequential(*shared_layers, nn.Linear(layers[-2], layers[-1]))
        self.apply(self._init_weights_)      
        
    def forward(self, tensor):
        mean = self.mean_layers(tensor)
        log_std = torch.clamp(self.log_std_layers(tensor), min=-20, max=2)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)  # Squash the output
        action = y_t
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mean = torch.tanh(mean) 
        return action, log_prob, mean

    @staticmethod
    def _init_weights_(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01)

LOG_STD_MIN = -5
LOG_STD_MAX = -0.5

class SNAF_Mu_Network(nn.Module):
    def __init__(self, layers, hidden_activation, output_activation=None):
        super().__init__()
        hidden_layers = layers[:-1]
        shared_layers = [nn.Sequential(nn.Linear(i, o), hidden_activation) for i, o in
                         zip(hidden_layers, hidden_layers[1:])]
        
        self.mean_layers = nn.Sequential(*shared_layers, nn.Linear(layers[-2], layers[-1]))
        self.log_std_layers = nn.Sequential(*shared_layers, nn.Linear(layers[-2], layers[-1]))
        
        self.P_layers = nn.Sequential(*shared_layers, nn.Linear(layers[-2], layers[-1]))
        
        
        self.apply(self._init_weights_)      
        
    def forward(self, tensor):
        mean = self.mean_layers(tensor)
        log_std = torch.clamp(self.log_std_layers(tensor), min=-20, max=2)
        # log_std = torch.tanh(log_std)
        # log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        std = log_std.exp()
        
        
        P = torch.clamp(self.P_layers(tensor), min=-20, max=2)
        # P = torch.tanh(P)
        # P = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (P + 1)  # From SpinUp / Denis Yarats
        
        P = P.exp()
        
        
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)  # Squash the output
        action = y_t
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mean = torch.tanh(mean) 
        return action, log_prob, mean, P

    @staticmethod
    def _init_weights_(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01)
