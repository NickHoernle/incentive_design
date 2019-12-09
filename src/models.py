import torch.nn as nn
import torch
from torch.nn import functional as F
import math
from torch.distributions import Poisson

BADGE_THRESHOLD = 500
poisson_loss = nn.PoissonNLLLoss(reduction='sum', log_input=False)

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


def Poisson_loss_function(recon_x, x, mu, logvar, data_shape, act_choice=5):

    BCE = poisson_loss(recon_x, x)

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def ZeroInflatedPoisson_loss_function(recon_x, x, mu, logvar, data_shape, act_choice=5):

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

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return -torch.sum(log_l) + KLD


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
        n_out = obsdim+(self.obs_dim//7)*self.proximity_to_badge
        print(n_out)
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
        ones = torch.ones(self.num_kernel_weights)
        ones[69:71] = 0
        self.badge_param = nn.Parameter(ones*torch.randn(self.num_kernel_weights), requires_grad=True)
        # print(self.badge_param)
        self.badge_param_bias = nn.Parameter(torch.tensor([0.0,0.0], requires_grad=True).float())

        # weights to control for bump
        self.badge_bump_param = nn.Parameter(torch.tensor([0.0, 0.0], requires_grad=True).float())

    @property
    def positive_badge_param(self):
        return F.softplus(self.badge_param)

    @property
    def positive_badge_bump_param(self):
        return F.softplus(self.badge_bump_param)

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

        kernel_features2 = torch.zeros(size=(2 * self.out_dim,)).float().to(self.device)
        kernel_features2[self.out_dim-1:self.out_dim+1] = self.positive_badge_bump_param

        return kernel_features[kwargs['kernel_data'].long().view(-1,self.out_dim)] + \
               kernel_features2[kwargs['kernel_data'].long().view(-1,self.out_dim)]


class LinearParametricVAE(BaselineVAE):
    def __init__(self, obsdim, outdim, **kwargs):
        super(LinearParametricVAE, self).__init__(obsdim, outdim, num_kernel_weights=2, **kwargs)

    def kernel(self, z, x, **kwargs):
        kernel_features = torch.zeros(size=(2 * self.out_dim,)).float().to(self.device)
        kernel_features[self.out_dim + 1:] = -torch.arange(1, self.out_dim).float() / (self.out_dim)
        kernel_features[:self.out_dim + 1] = torch.arange(1, self.out_dim + 2).float() / (self.out_dim + 2)
        kernel_features[self.out_dim + 1:] *= self.positive_badge_param[0]
        kernel_features[:self.out_dim + 1] *= self.positive_badge_param[1]

        kernel_features2 = torch.zeros(size=(2 * self.out_dim,)).float().to(self.device)
        kernel_features2[self.out_dim - 1:self.out_dim + 1] = self.positive_badge_bump_param - \
                                                              kernel_features[self.out_dim - 1:self.out_dim + 1]

        return kernel_features[kwargs['kernel_data'].long().view(-1, self.out_dim)] + \
               kernel_features2[kwargs['kernel_data'].long().view(-1, self.out_dim)]


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

    def kernel(self, z, x, **kwargs):

        kernel_features = torch.ones(size=(2 * self.out_dim,)).float().to(self.device)
        kernel_features[:self.out_dim + 1] = F.softplus(self.badge_param_bias[0]+self.badge_param[:self.out_dim + 1])
        kernel_features[self.out_dim + 1:] = -F.softplus(self.badge_param_bias[1]+self.badge_param[self.out_dim + 1:])

        kernel_features2 = torch.zeros(size=(2 * self.out_dim,)).float().to(self.device)
        kernel_features2[self.out_dim - 1:self.out_dim + 1] = self.positive_badge_bump_param

        to_return = kernel_features[kwargs['kernel_data'].long().view(-1, self.out_dim)] + \
               kernel_features2[kwargs['kernel_data'].long().view(-1, self.out_dim)]
        return to_return


class FullParameterisedPlusSteerParamVAE(AddSteeringParameter, FullParameterisedVAE):
    def __init__(self, obsdim, outdim, **kwargs):
        super(FullParameterisedPlusSteerParamVAE, self).__init__(obsdim, outdim, **kwargs)

class FlexibleLinearParametricVAE(BaselineVAE):
    def __init__(self, obsdim, outdim, **kwargs):
        super(FlexibleLinearParametricVAE, self).__init__(obsdim, outdim, num_kernel_weights=4, **kwargs)
        # initialise away from 0
        self.badge_param = nn.Parameter(torch.tensor([-5.0,-0.0,-5.0,-0.0,0.0,0.0], requires_grad=True).float())

    def kernel(self, z, x, **kwargs):
        kernel_features = torch.zeros(size=(2 * self.out_dim,)).float().to(self.device)

        kernel_features[:self.out_dim+1] = F.softplus(
            self.badge_param[0]*self.positive_badge_param[4]+self.positive_badge_param[1] * torch.arange(0, self.out_dim+1).float().to(self.device))
        kernel_features[self.out_dim + 1:] = -F.softplus(
            self.badge_param[2]*self.positive_badge_param[5]+self.positive_badge_param[3] * torch.arange(0, self.out_dim-1).float().to(self.device))

        kernel_features2 = torch.zeros(size=(2 * self.out_dim,)).float().to(self.device)
        kernel_features2[self.out_dim - 1:self.out_dim + 1] = self.positive_badge_bump_param

        return kernel_features[kwargs['kernel_data'].long().view(-1, self.out_dim)] + \
               kernel_features2[kwargs['kernel_data'].long().view(-1, self.out_dim)]
#
#
class FlexibleLinearPlusSteerParamVAE(AddSteeringParameter, FlexibleLinearParametricVAE):
    def __init__(self, obsdim, outdim, **kwargs):
        super(FlexibleLinearPlusSteerParamVAE, self).__init__(obsdim, outdim, **kwargs)



###############################################################################################
# Modeling the actual count data
###############################################################################################

class BaselineVAECount(BaselineVAE):
    def __init__(self, obsdim, outdim, **kwargs):
        super(BaselineVAECount, self).__init__(obsdim, outdim, **kwargs)

        # decoder (model network)
        self.decoder_count1 = nn.Linear(self.latent_dim, 200)
        self.decoder_count2 = nn.Linear(200, 400)
        self.decoder_count3 = nn.Linear(400, self.num_prediction_days)

        # kernel weights
        self.badge_param_count = nn.Parameter(torch.randn(self.num_kernel_weights, requires_grad=True).float())

        self.badge_param_bias_count = nn.Parameter(torch.tensor([0.0, 0.0], requires_grad=True).float())

        # weights to control for bump
        self.badge_bump_param_count = nn.Parameter(torch.tensor([0.0, 0.0], requires_grad=True).float())

    @property
    def positive_badge_count_param(self):
        return F.softplus(self.badge_param_count)

    @property
    def positive_badge_bump_param_count(self):
        return F.softplus(self.badge_bump_param_count)

    def decode(self, z, **kwargs):
        h = F.relu(self.decoder1(z))
        h1 = F.relu(self.decoder2(h))
        prob_of_act = self.decoder3(h1).repeat(1, self.n_out)[:,:self.out_dim]
        prob_of_act = prob_of_act + kwargs['kernel']

        hc = F.relu(self.decoder_count1(z))
        h1c = F.relu(self.decoder_count2(hc))
        prob_of_act_count = self.decoder_count3(h1c).repeat(1, self.n_out)[:, :self.out_dim]
        prob_of_act_count = prob_of_act_count + kwargs['kernel_count']

        return (torch.sigmoid(prob_of_act), F.softplus(prob_of_act_count))

    def forward(self, x, **kwargs):
        mu, logvar = self.encode(x.view(-1, self.obs_dim), **kwargs)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, kernel=self.kernel(z, x, **kwargs), kernel_count=self.kernel_count(z, x, **kwargs)), mu, logvar

    def kernel_count(self, z, x, **kwargs):
        kernel_features = torch.zeros(size=(2 * self.out_dim,)).float().to(self.device)

        kernel_features2 = torch.zeros(size=(2 * self.out_dim,)).float().to(self.device)
        kernel_features2[self.out_dim - 1:self.out_dim + 1] = self.positive_badge_bump_param_count

        return kernel_features[kwargs['kernel_data'].long().view(-1, self.out_dim)] + \
               kernel_features2[kwargs['kernel_data'].long().view(-1, self.out_dim)]

class AddSteeringParameterCount(BaselineVAECount):
    def __init__(self, obsdim, outdim, **kwargs):
        super(AddSteeringParameterCount, self).__init__(obsdim, outdim, **kwargs)
        self.decoder1 = nn.Linear(self.latent_dim-2, 200)
        self.decoder_count1 = nn.Linear(self.latent_dim-2, 200)
        self.steer_weight = nn.Parameter(torch.randn(2).float(), requires_grad=True)

    @property
    def positive_steer_weight(self):
        return F.softplus(self.steer_weight)

    def decode(self, z, **kwargs):
        h = F.relu(self.decoder1(z[:,:self.latent_dim-2]))
        h1 = F.relu(self.decoder2(h))
        prob_of_act = self.decoder3(h1).repeat(1, self.n_out)[:,:self.out_dim]
        prob_of_act = prob_of_act + torch.sigmoid(self.positive_steer_weight[0]*z[:,-1].view(-1,1))*kwargs['kernel']

        hc = F.relu(self.decoder_count1(z[:,:self.latent_dim-2]))
        h1c = F.relu(self.decoder_count2(hc))
        prob_of_act_count = self.decoder_count3(h1c).repeat(1, self.n_out)[:, :self.out_dim]
        prob_of_act_count = prob_of_act_count + torch.sigmoid(self.positive_steer_weight[1]*z[:,-2].view(-1,1))*kwargs['kernel_count']

        return (torch.sigmoid(prob_of_act), F.softplus(prob_of_act_count))


class LinearParametricVAECount(LinearParametricVAE, BaselineVAECount):
    def kernel_count(self, z, x, **kwargs):
        kernel_features = torch.zeros(size=(2 * self.out_dim,)).float().to(self.device)
        kernel_features[self.out_dim + 1:] = -torch.arange(1, self.out_dim).float() / (self.out_dim)
        kernel_features[:self.out_dim + 1] = torch.arange(1, self.out_dim + 2).float() / (self.out_dim + 2)
        kernel_features[self.out_dim + 1:] *= self.positive_badge_count_param[0]
        kernel_features[:self.out_dim + 1] *= self.positive_badge_count_param[1]

        kernel_features2 = torch.zeros(size=(2 * self.out_dim,)).float().to(self.device)
        kernel_features2[self.out_dim - 1:self.out_dim + 1] = self.positive_badge_bump_param_count - \
                                                              kernel_features[self.out_dim - 1:self.out_dim + 1]

        return kernel_features[kwargs['kernel_data'].long().view(-1, self.out_dim)] + \
               kernel_features2[kwargs['kernel_data'].long().view(-1, self.out_dim)]


class LinearParametricPlusSteerParamVAECount(AddSteeringParameterCount, LinearParametricVAECount):
    def __init__(self, obsdim, outdim, **kwargs):
        super(LinearParametricPlusSteerParamVAECount, self).__init__(obsdim, outdim, **kwargs)


class FullParameterisedVAECount(FullParameterisedVAE, BaselineVAECount):
    def __init__(self, obsdim, outdim, **kwargs):
        super(FullParameterisedVAECount, self).__init__(obsdim, outdim, **kwargs)

    def kernel_count(self, z, x, **kwargs):

        kernel_features = torch.ones(size=(2 * self.out_dim,)).float().to(self.device)

        kernel_features[:self.out_dim + 1] = F.softplus(self.badge_param_bias_count[0] + self.badge_param_count[:self.out_dim + 1])
        kernel_features[self.out_dim + 1:] = -F.softplus(self.badge_param_bias_count[1] + self.badge_param_count[self.out_dim + 1:])

        kernel_features2 = torch.zeros(size=(2 * self.out_dim,)).float().to(self.device)
        kernel_features2[self.out_dim - 1:self.out_dim + 1] = self.positive_badge_bump_param_count - \
                                                              kernel_features[self.out_dim - 1:self.out_dim + 1]
        return kernel_features[kwargs['kernel_data'].long().view(-1, self.out_dim)] + \
               kernel_features2[kwargs['kernel_data'].long().view(-1, self.out_dim)]


class FullParameterisedPlusSteerParamVAECount(AddSteeringParameterCount, FullParameterisedVAECount):
    def __init__(self, obsdim, outdim, **kwargs):
        super(FullParameterisedPlusSteerParamVAECount, self).__init__(obsdim, outdim, **kwargs)