"""
All functions related to loss computation and optimization.
"""

import torch
import torch.optim as optim
import numpy as np
from sde_lib import SDE, VESDE
from torch_runstats.scatter import scatter
import functools
from tqdm import tqdm, trange
from sampling import get_predictor, get_corrector

from e3_layers.utils import saveMol
import wandb
import logging


def get_optimizer(config, params):
  """Returns a flax optimizer object based on `config`."""
  if config.optim.optimizer == 'Adam':
    optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                           weight_decay=config.optim.weight_decay)
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer


class VPSDE():
  def __init__(self, diffusion_keys, beta_min=0.1, beta_max=20, N=1000):
    """Construct a Variance Preserving SDE.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N
    self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
    self.alphas = 1. - self.discrete_betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
    self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
    self.irreps = diffusion_keys # a dict {diffused_key: dim}

  @property
  def T(self):
    return 1

  def marginal(self, batch, return_std=False):
    t = batch['t'][batch.nodeSegment()]
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    if return_std:
      return std
    zs = {}
    for key in self.irreps.keys():
      mean = torch.exp(log_mean_coeff) * batch[key] 
      z = torch.randn_like(batch[key])
      batch[key] = mean + std*z
      zs[key] = z
    return batch, {'zs':zs, 'std':std}

  def sde(self, batch, dt=None):
    if dt is None:
      dt = 1. / self.N
    t = batch['t']
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    diffusion = torch.sqrt(beta_t)
    for key in self.irreps.keys():
      x = batch[key]
      drift = -0.5 * beta_t * x
      x_mean = x + drift * dt
      z = torch.randn_like(x)
      x = x_mean + diffusion * np.sqrt(abs(dt)) * z
      batch[key] = x
    return batch.to(batch.device)
  
  def prior_sampling(self, batch):
    for key, dim in self.irreps.items():
      batch[key] = torch.randn((batch['_n_nodes'].sum(), dim))
    return batch
  
  def reverse(self, score_fn):
    """Create the reverse-time SDE/ODE.

    Args:
      score_fn: A time-dependent score-based model that takes x and t and returns the score.
      probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
    """
    N = self.N
    T = self.T
    sde_fn = self.sde
    beta_0, beta_1 = self.beta_0, self.beta_1
    irreps = self.irreps

    # Build the class for reverse-time SDE.
    class RSDE(self.__class__):
      def __init__(self):
        self.N = N

      @property
      def T(self):
        return T

      def sde(self, batch):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""
        scores = score_fn(batch)
        t = batch['t'][batch.nodeSegment()]
        beta_t = beta_0 + t * (beta_1 - beta_0)
        diffusion = torch.sqrt(beta_t)
        dt = -1. / self.N
        batch = sde_fn(batch, dt)
        for key in irreps:          
          batch[key] = batch[key] - dt * diffusion ** 2 * scores[f'score_{key}']
        batch = batch.to(batch.device)
        return batch

    return RSDE()

def get_sde_loss_fn(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5):
  """Create a loss function for training with arbirary SDEs.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.

  Returns:
    A loss function.
  """
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    """Compute the loss function.

    Args:
      model: A score model.
      batch: A mini-batch of training data.

    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    
    t = torch.rand(len(batch), device=batch.device) * (sde.T - eps) + eps
    score_fn = get_score_fn(sde, model, train)
    
    batch_perturbed = batch.clone()
    batch_perturbed.attrs['t'] = ('graph', '1x0e')
    batch_perturbed['t'] = t
    batch_perturbed, misc = sde.marginal(batch_perturbed)
    
    scores = score_fn(batch_perturbed)
    losses = {}
    for key in sde.irreps.keys():
      loss = torch.square(scores[f'score_{key}']*misc['std'] + misc['zs'][key])
      loss = reduce_op(loss.reshape(loss.shape[0], -1), dim=-1)
      loss = torch.mean(loss)
      losses[key] = loss
    total_loss = sum(losses.values())
    losses['total'] = total_loss
    return total_loss, losses

  return loss_fn


def get_score_fn(sde, model, train=False):
  def score_fn(batch):
    if train:
      model.train()
    else:
      model.eval()
    result = model(batch)
    std = sde.marginal(batch, return_std=True)
    for key in sde.irreps.keys():
      result[f'score_{key}'] = -result[f'score_{key}'] / std - batch[key]
    return result
  return score_fn


def get_step_fn(sde, train, optimizer=None, reduce_mean=False, continuous=True,
                likelihood_weighting=False, grad_clid_norm=None, grad_acc=1):
  """Create a one-step training/evaluation function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    optimizer: An optimizer.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses according to
      https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

  Returns:
    A one-step function for training or evaluation.
  """
  if continuous:
    loss_fn = get_sde_loss_fn(sde, train, reduce_mean=reduce_mean,
                              continuous=True, likelihood_weighting=likelihood_weighting)
  else:
    assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
    if isinstance(sde, VESDE):
      loss_fn = get_smld_loss_fn(sde, train, reduce_mean=reduce_mean)
    elif isinstance(sde, VPSDE):
      loss_fn = get_ddpm_loss_fn(sde, train, reduce_mean=reduce_mean)
    else:
      raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")

  def step_fn(state, batch):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      state: A dictionary of training information, containing the score model, optimizer,
       EMA status, and number of optimization steps.
      batch: A mini-batch of training/evaluation data.

    Returns:
      loss: The average loss value of this state.
    """
    model = state['model']
    if train:
      optimizer = state['optimizer']
      loss, losses = loss_fn(model, batch)
      loss.backward()
      if not grad_clid_norm is None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clid_norm)
      if not state['step']==0 and state['step']%grad_acc==0:
        flag = True
        for name, param in model.named_parameters():
          if (param.grad).isnan().any() or (param.grad).isinf().any():
            flag = False
            logging.warning("Gradient is None, skipping optim step.")
        if flag:
          optimizer.step()
        optimizer.zero_grad(set_to_none=True)
      state['step'] += 1
      state['ema'].update(model.parameters())
    else:
      ema = state['ema']
      ema.store(model.parameters())
      ema.copy_to(model.parameters())
      loss, losses = loss_fn(model, batch)
      ema.restore(model.parameters())

    return loss.item(), {key:value.item() for key, value in losses.items()}

  return step_fn