import torch.nn as nn
import torch
from torch.nn import functional as F
import math

BADGE_THRESHOLD = 500

# Reconstruction + KL divergence losses summed over all elements and batch
def BCE_loss_function(recon_x, x, mu, logvar, data_shape, act_choice=5):

    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    # BCE = l1_loss(recon_x, outcome)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

class BaselineVAE(nn.Module):
    def __init__(self, obsdim, outdim, **kwargs):

        super(BaselineVAE, self).__init__()

        self.latent_dim = kwargs.get('latent_dim', 20)
        self.num_prediction_days = kwargs.get('num_prediction_days', 7)
        self.num_kernel_weights = kwargs.get('num_kernel_weights', 1)
        self.baseline_date_of_badge = kwargs.get('baseline_date_of_badge', outdim//2)
        self.device = kwargs.get('device', torch.device('cpu'))
        self.badge_threshold = kwargs.get('badge_threshold', 500)
        self.action_ix = kwargs.get('action_ix', 5)
        self.proximity_to_badge = kwargs.get('proximity_to_badge', False)

        self.obs_dim = obsdim
        self.out_dim = outdim

        # encoder (inference network)
        n_out = obsdim+outdim*self.proximity_to_badge
        self.encoder1 = nn.Linear(n_out, 400)
        self.encoder2 = nn.Linear(400, 200)
        self.encoder3_mu = nn.Linear(200, self.latent_dim)
        self.encoder3_var = nn.Linear(200, self.latent_dim)

        self.n_out = int(math.ceil(self.out_dim/self.num_prediction_days))

        # decoder (model network)
        self.decoder1 = nn.Linear(self.latent_dim, 200)
        self.decoder2 = nn.Linear(200, 400)
        self.decoder3 = nn.Linear(400, self.num_prediction_days)

        # kernel weights
        self.badge_param = nn.Parameter(torch.randn(self.num_kernel_weights, requires_grad=True).float())
        self.badge_param_bias = nn.Parameter(torch.tensor([0.0,0.0], requires_grad=True).float())

    @property
    def positive_badge_param(self):
        return F.softplus(self.badge_param_bias[0] + self.badge_param)

    def encode(self, x, **kwargs):
        if self.proximity_to_badge:
            h0 = torch.cat((x, kwargs['prox_to_badge']), dim=1)
        else:
            h0 = x
        h = F.relu(self.encoder1(h0))
        h1 = F.relu(self.encoder2(h))
        return self.encoder3_mu(h1), self.encoder3_var(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, **kwargs):
        h = F.relu(self.decoder1(z))
        h1 = F.relu(self.decoder2(h))
        prob_of_act = self.decoder3(h1).repeat(1, self.n_out)[:,:self.out_dim]
        prob_of_act = prob_of_act + kwargs['kernel']

        return torch.sigmoid(prob_of_act)

    # define the encoder and decoder here!
    def forward(self, x, **kwargs):
        mu, logvar = self.encode(x.view(-1, self.obs_dim), **kwargs)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, kernel=self.kernel(z, x, **kwargs)), mu, logvar

    def kernel(self, z, x, **kwargs):
        kernel_features = torch.zeros(size=(2 * self.out_dim,)).float().to(self.device)
        return kernel_features[kwargs['kernel_data'].long().view(-1,self.out_dim)]


class LinearParametricVAE(BaselineVAE):
    def __init__(self, obsdim, outdim, **kwargs):
        super(LinearParametricVAE, self).__init__(obsdim, outdim, num_kernel_weights=2, **kwargs)

    def kernel(self, z, x, **kwargs):
        kernel_features = torch.zeros(size=(2 * self.out_dim,)).float().to(self.device)
        kernel_features[self.out_dim + 1:] = -1
        kernel_features[:self.out_dim + 1] = torch.arange(1, self.out_dim + 2).float() / (self.out_dim + 2)
        kernel_features[self.out_dim + 1:] *= self.positive_badge_param[0]
        kernel_features[:self.out_dim + 1] *= self.positive_badge_param[1]
        return kernel_features[kwargs['kernel_data'].long().view(-1,self.out_dim)]


class AddSteeringParameter(BaselineVAE):
    def __init__(self, obsdim, outdim, **kwargs):
        super(AddSteeringParameter, self).__init__(obsdim, outdim, **kwargs)
        self.decoder1 = nn.Linear(self.latent_dim-1, 200)
        self.steer_weight = nn.Parameter(torch.randn(1).float(), requires_grad=True)

    @property
    def positive_steer_weight(self):
        return F.softplus(self.steer_weight)

    def decode(self, z, **kwargs):
        h = F.relu(self.decoder1(z[:,:self.latent_dim-1]))
        h1 = F.relu(self.decoder2(h))
        prob_of_act = self.decoder3(h1).repeat(1, self.n_out)[:, :self.out_dim]
        prob_of_act = prob_of_act + torch.sigmoid(self.positive_steer_weight*z[:,-1].view(-1,1))*kwargs['kernel']
        return torch.sigmoid(prob_of_act)


class LinearParametricPlusSteerParamVAE(AddSteeringParameter, LinearParametricVAE):
    def __init__(self, obsdim, outdim, **kwargs):
        super(LinearParametricPlusSteerParamVAE, self).__init__(obsdim, outdim, **kwargs)


class FullParameterisedVAE(BaselineVAE):
    def __init__(self, obsdim, outdim, **kwargs):
        super(FullParameterisedVAE, self).__init__(obsdim, outdim, num_kernel_weights=2*outdim, **kwargs)

    def __invert_tensor(self, tensor):
        inv_idx = torch.arange(tensor.size(0) - 1, -1, -1).long().to(self.device)
        inv_tensor = tensor.index_select(0, inv_idx)
        return inv_tensor

    def kernel(self, z, x, **kwargs):

        kernel_features = torch.ones(size=(2 * self.out_dim,)).float().to(self.device)
        kernel_features[:self.out_dim + 1] =  F.softplus(self.badge_param_bias[0]+self.badge_param[:self.out_dim + 1])
        kernel_features[self.out_dim + 1:] = -F.softplus(self.badge_param_bias[1]+self.badge_param[self.out_dim + 1:])

        to_return = kernel_features[kwargs['kernel_data'].long().view(-1, self.out_dim)]
        return to_return


class FullParameterisedPlusSteerParamVAE(AddSteeringParameter, FullParameterisedVAE):
    def __init__(self, obsdim, outdim, **kwargs):
        super(FullParameterisedPlusSteerParamVAE, self).__init__(obsdim, outdim, **kwargs)

# class FlexibleLinearParametricVAE(BaselineVAE):
#     def __init__(self, obsdim, outdim, **kwargs):
#         super(FlexibleLinearParametricVAE, self).__init__(obsdim, outdim, num_kernel_weights=3, **kwargs)
#
#     def kernel(self, z, x, **kwargs):
#         kernel_features = torch.zeros(size=(2 * self.out_dim,)).float().to(self.device)
#         kernel_features[self.out_dim + 1:] = -1
#         kernel_features[:self.out_dim + 1] = F.softplus(torch.arange(1, self.out_dim + 2).float().to(self.device) + self.badge_param[2])/(self.out_dim + 2)
#         kernel_features[self.out_dim + 1:] *= self.positive_badge_param[0]
#         kernel_features[:self.out_dim + 1] *= self.positive_badge_param[1]
#         return kernel_features[kwargs['kernel_data'].long().view(-1, self.out_dim)]
#
#
# class FlexibleLinearPlusSteerParamVAE(AddSteeringParameter, FlexibleLinearParametricVAE):
#     def __init__(self, obsdim, outdim, **kwargs):
#         super(FlexibleLinearPlusSteerParamVAE, self).__init__(obsdim, outdim, **kwargs)
