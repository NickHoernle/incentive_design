import torch.nn as nn
import torch
from torch.nn import functional as F
import math
import torch.distributions as distrib
from torch.distributions import Poisson
from normalizing_flows import NormalizingFlow
from normalizing_flows import PlanarFlow

BADGE_THRESHOLD = 500
poisson_loss = nn.PoissonNLLLoss(reduction='sum', log_input=False)

# Reconstruction + KL divergence losses summed over all elements and batch
def BCE_loss_function(recon_x, x, KLD, data_shape, act_choice=5):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    return BCE + KLD

def ZeroInflatedPoisson_loss_function(recon_x, x, latent_loss, data_shape, act_choice=5):

    x_shape = x.size()
    # if x == 0
    recon_x_0_bin = recon_x[0]
    recon_x_0_count = recon_x[1]
    poisson_0 = (x==0).float()*Poisson(recon_x_0_count).log_prob(x)

    # else if x > 0
    recon_x_greater0_bin = recon_x[0]
    recon_x_greater0_count = recon_x[1]
    poisson_greater0 = (x>0).float()*Poisson(recon_x_greater0_count).log_prob(x)

    zero_inf = torch.cat((
        torch.log(1-recon_x_0_bin).view(x_shape[0], x_shape[1], -1),
        poisson_0.view(x_shape[0], x_shape[1], -1)
    ), dim=2)

    log_l = (x==0).float()*torch.logsumexp(zero_inf, dim=2)
    log_l += (x>0).float()*(torch.log(recon_x_0_bin)+poisson_greater0)

    # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return -torch.sum(log_l) + latent_loss


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

        self.hard_tan = nn.Hardtanh(min_val=1e-5, max_val=15)

        # encoder (inference network)
        num_in_dim = obsdim+(self.obs_dim//7)*self.proximity_to_badge
        self.encoder_dims = 50
        self.encoder = kwargs.get('encoder', BaselineVAE.default_network(num_in_dim, [200, 400, 200], self.encoder_dims))
        self.mu = nn.Linear(self.encoder_dims, self.latent_dim)
        self.log_var = nn.Linear(self.encoder_dims, self.latent_dim)

        self.n_out = int(math.ceil(self.out_dim/self.num_prediction_days))

        # decoder (model network)
        self.decoder = kwargs.get('decoder', BaselineVAE.default_network(
                self.latent_dim, [200, 400, 200], self.num_prediction_days, encoder=False))

        # kernel weights
        self.badge_param = nn.Parameter(torch.randn(self.num_kernel_weights), requires_grad=True)
        self.badge_param_bias = nn.Parameter(torch.tensor([0.0,0.0], requires_grad=True).float())

        # weights to control for bump
        self.badge_bump_param_ = nn.Parameter(torch.tensor([0.0, 0.0], requires_grad=True).float())

        self.zeros = torch.zeros(self.latent_dim).to(self.device)
        self.ones = torch.ones(self.latent_dim).to(self.device)


    def make_pos(self, input):
        return self.hard_tan(F.softplus(input))

    @property
    def badge_bump_param(self):
        return self.make_pos(self.badge_bump_param_)

    @staticmethod
    def default_network(in_dim, hidden_dims, out_dim, encoder=True):

        layers = [nn.Linear(in_dim, hidden_dims[0])]
        layers.append(nn.ReLU(True))
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.ReLU(True))

        layers.append(nn.Linear(hidden_dims[-1], out_dim))

        if encoder:
            layers.append(nn.ReLU(True))

        return nn.Sequential(*layers)

    def encode(self, x, **kwargs):
        if self.proximity_to_badge:
            h0 = torch.cat((x, kwargs['prox_to_badge']), dim=1)
        else:
            h0 = x
        h = self.encoder(h0)
        return self.mu(h), self.log_var(h)

    def decode(self, z, **kwargs):
        h = self.decoder(z)

        prob_of_act = h.repeat(1, self.n_out)[:,:self.out_dim]
        prob_of_act = prob_of_act + kwargs['kernel']

        return torch.sigmoid(prob_of_act)

    def latent_loss(self, x, z_params):
        n_batch = x.size(0)

        # Retrieve mean and var
        mu, log_var = z_params

        sigma = torch.exp(0.5 * log_var)

        # Re-parametrize
        q = distrib.Normal(self.zeros, self.ones)
        z = (sigma * q.sample((n_batch,))) + mu

        # Compute KL divergence
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return z, kl_div

    def forward(self, x, **kwargs):
        zparams = self.encode(x.view(-1, self.obs_dim), **kwargs)
        z, latent_loss = self.latent_loss(x, zparams)
        return self.decode(z, kernel=self.kernel(z, x, **kwargs)), latent_loss

    def get_z(self, x, **kwargs):
        zparams = self.encode(x.view(-1, self.obs_dim), **kwargs)
        z, latent_loss = self.latent_loss(x, zparams)
        return z

    def kernel(self, z, x, **kwargs):
        kernel_features = torch.zeros(size=(2 * self.out_dim,)).float().to(self.device)

        kernel_features2 = torch.zeros(size=(2 * self.out_dim,)).float().to(self.device)
        kernel_features2[self.out_dim-1:self.out_dim+1] = self.badge_bump_param

        return kernel_features[kwargs['kernel_data'].long().view(-1,self.out_dim)] + \
               kernel_features2[kwargs['kernel_data'].long().view(-1,self.out_dim)]


class LinearParametricVAE(BaselineVAE):
    def __init__(self, obsdim, outdim, **kwargs):
        super(LinearParametricVAE, self).__init__(obsdim, outdim, num_kernel_weights=2, **kwargs)

    def kernel(self, z, x, **kwargs):

        kernel_features = torch.zeros(size=(2 * self.out_dim,)).float().to(self.device)
        kernel_features[self.out_dim + 1:] = -torch.arange(1, self.out_dim).float() / (self.out_dim)
        kernel_features[:self.out_dim + 1] = torch.arange(1, self.out_dim + 2).float() / (self.out_dim + 2)
        kernel_features[self.out_dim + 1:] *= self.make_pos(self.badge_param[0])
        kernel_features[:self.out_dim + 1] *= self.make_pos(self.badge_param[1])

        # badge bump
        kernel_features2 = torch.zeros(size=(2 * self.out_dim,)).float().to(self.device)
        kernel_features2[self.out_dim - 1:self.out_dim + 1] = self.badge_bump_param - \
                                                              kernel_features[self.out_dim - 1:self.out_dim + 1]

        return kernel_features[kwargs['kernel_data'].long().view(-1, self.out_dim)] + \
               kernel_features2[kwargs['kernel_data'].long().view(-1, self.out_dim)]


class AddSteeringParameter(BaselineVAE):
    def __init__(self, obsdim, outdim, **kwargs):
        super(AddSteeringParameter, self).__init__(obsdim, outdim, **kwargs)
        self.decoder = kwargs.get('decoder', BaselineVAE.default_network(self.latent_dim-1, [200, 400, 200],
                                                                         self.num_prediction_days, encoder=False))
        self.steer_weight_ = nn.Parameter(torch.randn(1).float(), requires_grad=True)

    @property
    def steer_weight(self):
        return self.make_pos(self.steer_weight_)

    def decode(self, z, **kwargs):
        h = self.decoder(z[:,:self.latent_dim-1])
        prob_of_act = h.repeat(1, self.n_out)[:, :self.out_dim]
        prob_of_act = prob_of_act + torch.sigmoid(self.steer_weight*z[:,-1].view(-1,1))*kwargs['kernel']
        return torch.sigmoid(prob_of_act)


class AddNormalizingFlow(AddSteeringParameter):
    def __init__(self, obsdim, outdim, **kwargs):
        super(AddNormalizingFlow, self).__init__(obsdim, outdim, **kwargs)

        self.K = 12
        self.flow_params = nn.Linear(self.encoder_dims, self.K*(self.latent_dim*2+1))
        # block_planar = [PlanarFlow]
        # self.flow = NormalizingFlow(dim=self.latent_dim,
        #                             blocks=block_planar,
        #                             flow_length=12,
        #                             density=distrib.MultivariateNormal(self.zeros, torch.eye(self.latent_dim)))
        self.flow = NormalizingFlow(K=self.K, D=self.latent_dim)

    def encode(self, x, **kwargs):
        if self.proximity_to_badge:
            h0 = torch.cat((x, kwargs['prox_to_badge']), dim=1)
        else:
            h0 = x
        h = self.encoder(h0)
        return self.mu(h), self.log_var(h), self.flow_params(h)

    def latent_loss(self, x, z_params):
        n_batch = x.size(0)

        # Retrieve set of parameters
        mu, log_var, flow_params = z_params

        # Re-parametrize a Normal distribution
        q = distrib.Normal(torch.zeros(mu.shape[1]), torch.ones(log_var.shape[1]))

        sigma = torch.exp(0.5 * log_var)

        # Obtain our first set of latent points
        z_0 = (sigma * q.sample((n_batch,)).to(self.device)) + mu

        # Complexify posterior with flows
        z_k, list_ladj = self.flow(z_0, flow_params.chunk(self.K, dim=1))

        # ln q(z_0)
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # ladj = torch.cat(list_ladj)
        kl_div -= torch.sum(list_ladj)
        # ln p(z_k)
        # log_p_zk = -0.5 * z_k * z_k
        #
        # #  ln q(z_0) - ln p(z_k)
        # logs = (log_q_z0 - log_p_zk).sum()
        #
        # # Add log determinants
        # ladj = torch.cat(list_ladj)
        #
        # # ln q(z_0) - ln p(z_k) - sum[log det]
        # logs -= torch.sum(ladj)

        return z_k, kl_div


class LinearParametricPlusSteerParamVAE(AddSteeringParameter, LinearParametricVAE):
    def __init__(self, obsdim, outdim, **kwargs):
        super(LinearParametricPlusSteerParamVAE, self).__init__(obsdim, outdim, **kwargs)


class FullParameterisedVAE(BaselineVAE):
    def __init__(self, obsdim, outdim, **kwargs):
        super(FullParameterisedVAE, self).__init__(obsdim, outdim, num_kernel_weights=2*outdim, **kwargs)

    def kernel(self, z, x, **kwargs):

        kernel_features = torch.ones(size=(2 * self.out_dim,)).float().to(self.device)
        kernel_features[:self.out_dim + 1] =  self.make_pos(self.badge_param_bias[0] + self.badge_param[:self.out_dim + 1])
        kernel_features[self.out_dim + 1:] = -self.make_pos(self.badge_param_bias[1] + self.badge_param[self.out_dim + 1:])

        kernel_features2 = torch.zeros(size=(2 * self.out_dim,)).float().to(self.device)
        kernel_features2[self.out_dim - 1:self.out_dim + 1] = self.badge_bump_param

        to_return = kernel_features[kwargs['kernel_data'].long().view(-1, self.out_dim)] + \
               kernel_features2[kwargs['kernel_data'].long().view(-1, self.out_dim)]
        return to_return


class FullParameterisedPlusSteerParamVAE(AddSteeringParameter, FullParameterisedVAE):
    def __init__(self, obsdim, outdim, **kwargs):
        super(FullParameterisedPlusSteerParamVAE, self).__init__(obsdim, outdim, **kwargs)

class FullParameterisedPlusSteerPlusNormParamVAE(AddNormalizingFlow, FullParameterisedPlusSteerParamVAE):
    def __init__(self, obsdim, outdim, **kwargs):
        super(FullParameterisedPlusSteerPlusNormParamVAE, self).__init__(obsdim, outdim, **kwargs)



###############################################################################################
# Modeling the actual count data
###############################################################################################

class BaselineVAECount(BaselineVAE):
    def __init__(self, obsdim, outdim, **kwargs):
        super(BaselineVAECount, self).__init__(obsdim, outdim, **kwargs)

        # decoder (model network)
        self.decoder_count = kwargs.get('decoder', BaselineVAE.default_network(self.latent_dim, [200, 400, 200],
                                                                         self.num_prediction_days, encoder=False))

        # kernel weights
        self.badge_param_count = nn.Parameter(torch.randn(self.num_kernel_weights, requires_grad=True).float())
        self.badge_param_bias_count = nn.Parameter(torch.tensor([0.0, 0.0], requires_grad=True).float())

        # weights to control for bump
        self.badge_bump_param_count_ = nn.Parameter(torch.tensor([0.0, 0.0], requires_grad=True).float())

    @property
    def badge_bump_param_count(self):
        return self.make_pos(self.badge_bump_param_count_)

    def decode(self, z, **kwargs):
        h = self.decoder(z)
        prob_of_act = h.repeat(1, self.n_out)[:,:self.out_dim]
        prob_of_act = prob_of_act + kwargs['kernel']

        hc = self.decoder_count(z)
        prob_of_act_count = hc.repeat(1, self.n_out)[:, :self.out_dim]
        prob_of_act_count = prob_of_act_count + kwargs['kernel_count']

        return (torch.sigmoid(prob_of_act), F.softplus(prob_of_act_count))

    def forward(self, x, **kwargs):
        # same encoder that was used before
        z_params = self.encode(x.view(-1, self.obs_dim), **kwargs)
        z, latent_loss = self.latent_loss(x, z_params)
        return self.decode(z, kernel=self.kernel(z, x, **kwargs), kernel_count=self.kernel_count(z, x, **kwargs)), latent_loss

    def kernel_count(self, z, x, **kwargs):
        kernel_features = torch.zeros(size=(2 * self.out_dim,)).float().to(self.device)

        kernel_features2 = torch.zeros(size=(2 * self.out_dim,)).float().to(self.device)
        kernel_features2[self.out_dim - 1:self.out_dim + 1] = self.badge_bump_param_count

        return kernel_features[kwargs['kernel_data'].long().view(-1, self.out_dim)] + \
               kernel_features2[kwargs['kernel_data'].long().view(-1, self.out_dim)]

class AddSteeringParameterCount(BaselineVAECount, AddSteeringParameter):
    def __init__(self, obsdim, outdim, **kwargs):
        super(AddSteeringParameterCount, self).__init__(obsdim, outdim, **kwargs)

        self.steer_weight_ = nn.Parameter(torch.randn(2).float(), requires_grad=True)
        self.decoder = AddSteeringParameterCount.default_network(self.latent_dim-2, [200, 400, 200],
                                                                         self.num_prediction_days, encoder=False)
        self.decoder_count = AddSteeringParameterCount.default_network(self.latent_dim-2, [200, 400, 200],
                                                                         self.num_prediction_days, encoder=False)

    def decode(self, z, **kwargs):
        h = self.decoder(z[:, :self.latent_dim - 2])
        prob_of_act = h.repeat(1, self.n_out)[:, :self.out_dim]
        prob_of_act = prob_of_act + torch.sigmoid(self.steer_weight[0] * z[:, -1].view(-1, 1)) * kwargs['kernel']

        hc = self.decoder_count(z[:, :self.latent_dim - 2])
        prob_of_act_count = hc.repeat(1, self.n_out)[:, :self.out_dim]
        prob_of_act_count = prob_of_act_count + torch.sigmoid(self.steer_weight[1] * z[:, -2].view(-1, 1)) * kwargs['kernel_count']

        return (torch.sigmoid(prob_of_act), F.softplus(prob_of_act_count))


# class AddNormalizingFlowCount(AddSteeringParameterCount, AddNormalizingFlow):
#     pass


class LinearParametricVAECount(LinearParametricVAE, BaselineVAECount):
    def kernel_count(self, z, x, **kwargs):
        kernel_features = torch.zeros(size=(2 * self.out_dim,)).float().to(self.device)
        kernel_features[self.out_dim + 1:] = -torch.arange(1, self.out_dim).float() / (self.out_dim)
        kernel_features[:self.out_dim + 1] = torch.arange(1, self.out_dim + 2).float() / (self.out_dim + 2)
        kernel_features[self.out_dim + 1:] *= self.make_pos(self.badge_param_count[0])
        kernel_features[:self.out_dim + 1] *= self.make_pos(self.badge_param_count[1])

        kernel_features2 = torch.zeros(size=(2 * self.out_dim,)).float().to(self.device)
        kernel_features2[self.out_dim - 1:self.out_dim + 1] = self.badge_bump_param_count - \
                                                              kernel_features[self.out_dim - 1:self.out_dim + 1]

        return kernel_features[kwargs['kernel_data'].long().view(-1, self.out_dim)] + \
               kernel_features2[kwargs['kernel_data'].long().view(-1, self.out_dim)]


class LinearParametricPlusSteerParamVAECount(AddSteeringParameterCount, LinearParametricVAECount):
    pass


class FullParameterisedVAECount(FullParameterisedVAE, BaselineVAECount):
    def __init__(self, obsdim, outdim, **kwargs):
        super(FullParameterisedVAECount, self).__init__(obsdim, outdim, **kwargs)

    def kernel_count(self, z, x, **kwargs):

        kernel_features = torch.ones(size=(2 * self.out_dim,)).float().to(self.device)

        kernel_features[:self.out_dim + 1] = self.make_pos(
            self.badge_param_bias_count[0] + self.badge_param_count[:self.out_dim + 1])
        kernel_features[self.out_dim + 1:] = -self.make_pos(
            self.badge_param_bias_count[1] + self.badge_param_count[self.out_dim + 1:])

        kernel_features2 = torch.zeros(size=(2 * self.out_dim,)).float().to(self.device)
        kernel_features2[self.out_dim - 1:self.out_dim + 1] = self.badge_bump_param_count - \
                                                              kernel_features[self.out_dim - 1:self.out_dim + 1]
        return kernel_features[kwargs['kernel_data'].long().view(-1, self.out_dim)] + \
               kernel_features2[kwargs['kernel_data'].long().view(-1, self.out_dim)]


class FullParameterisedPlusSteerParamVAECount(AddSteeringParameterCount, FullParameterisedVAECount):
    pass


class NormalizingFlowFP_PlusSteer(AddNormalizingFlow, FullParameterisedPlusSteerParamVAECount):
    pass