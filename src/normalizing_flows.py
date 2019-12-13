import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distrib
import torch.distributions.transforms as transform


# class Flow(transform.Transform, nn.Module):
#     def __init__(self):
#         transform.Transform.__init__(self)
#         nn.Module.__init__(self)
#
#     # Init all parameters
#     def init_parameters(self):
#         for param in self.parameters():
#             param.data.uniform_(-0.01, 0.01)
#
#     # Hacky hash bypass
#     def __hash__(self):
#         return nn.Module.__hash__(self)
#
#
# class NormalizingFlow(nn.Module):
#
#     def __init__(self, dim, blocks, flow_length, density):
#         super().__init__()
#         biject = []
#         for f in range(flow_length):
#             for b_flow in blocks:
#                 biject.append(b_flow(dim))
#         self.transforms = transform.ComposeTransform(biject)
#         self.bijectors = nn.ModuleList(biject)
#         self.base_density = density
#         self.final_density = distrib.TransformedDistribution(density, self.transforms)
#         self.log_det = []
#
#     def forward(self, z):
#         self.log_det = []
#         # Applies series of flows
#         for b in range(len(self.bijectors)):
#             self.log_det.append(self.bijectors[b].log_abs_det_jacobian(z))
#             z = self.bijectors[b](z)
#         return z, self.log_det
#
#
# class PlanarFlowPReLu(Flow):
#     '''
#     f(x) = x for x > 0 and alpha for x < 0
#     '''
#
#     def __init__(self, dim):
#         super(PlanarFlowPReLu, self).__init__()
#         self.weight = nn.Parameter(torch.Tensor(1, dim))
#         self.scale = nn.Parameter(torch.Tensor(1, dim))
#         self.bias = nn.Parameter(torch.Tensor(1))
#         self.alpha_ = nn.Parameter(torch.Tensor(1))
#         self.init_parameters()
#
#     @property
#     def alpha(self):
#         return F.softplus(self.alpha)
#
#     def _call(self, z):
#         f_z = F.linear(z, self.weight, self.bias)
#         print(f_z.shape)
#         return z + self.scale * torch.tanh(f_z)
#
#     def log_abs_det_jacobian(self, z):
#         f_z = F.linear(z, self.weight, self.bias)
#         psi = (1 - torch.tanh(f_z) ** 2) * self.weight
#         det_grad = 1 + torch.mm(psi, self.scale.t())
#         return torch.log(det_grad.abs() + 1e-9)
#
# class PlanarFlow(Flow):
#
#     def __init__(self, dim):
#         super(PlanarFlow, self).__init__()
#         self.weight = nn.Parameter(torch.Tensor(1, dim))
#         self.scale = nn.Parameter(torch.Tensor(1, dim))
#         self.bias = nn.Parameter(torch.Tensor(1))
#         self.init_parameters()
#
#     def _call(self, z):
#         f_z = F.linear(z, self.weight, self.bias)
#         return z + self.scale * torch.tanh(f_z)
#
#     def log_abs_det_jacobian(self, z):
#         f_z = F.linear(z, self.weight, self.bias)
#         psi = (1 - torch.tanh(f_z) ** 2) * self.weight
#         det_grad = 1 + torch.mm(psi, self.scale.t())
#         return torch.log(det_grad.abs() + 1e-9)

class PlanarFlow(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.D = D

    def forward(self, z, lamda):
        '''
        z - latents from prev layer
        lambda - Flow parameters (b, w, u)
        b - scalar
        w - vector
        u - vector
        '''
        b = lamda[:, :1]
        w, u = lamda[:, 1:].chunk(2, dim=1)

        # Forward
        # f(z) = z + u tanh(w^T z + b)
        transf = torch.tanh(
            z.unsqueeze(1).bmm(w.unsqueeze(2))[:, 0] + b
        )
        f_z = z + u * transf

        # Inverse
        # psi_z = tanh' (w^T z + b) w
        psi_z = (1 - transf ** 2) * w
        log_abs_det_jacobian = torch.log(
            (1 + psi_z.unsqueeze(1).bmm(u.unsqueeze(2))).abs()
        )

        return f_z, log_abs_det_jacobian


class NormalizingFlow(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.flows = nn.ModuleList([PlanarFlow(D) for i in range(K)])

    def forward(self, z_k, flow_params):
        # ladj -> log abs det jacobian
        sum_ladj = 0
        for i, flow in enumerate(self.flows):
            z_k, ladj_k = flow(z_k, flow_params[i])
            sum_ladj += ladj_k

        return z_k, sum_ladj