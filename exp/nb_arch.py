
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: dev_nb/arch_lib.ipynb

import IPython.core.debugger as db
from functools import partial

import torch
from torch import tensor
from torch import nn
from torch.nn import init
from torch import optim
import torch.nn.functional as F

#torch.set_num_threads(2)


from nb_util import get_mnist_data, normalize_tr_val, DebugRand
from nb_hooks import Hooks
from nb_data import MultiDimDataset, get_dls
from nb_training import Trainer
from nb_optimiser import get_optimiser


#----------------------------------------------------
# nn.Module layer that wraps a basic function. This layer is now a regular layer that
# you can put in nn.Sequential.
#----------------------------------------------------
class FuncLayer(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x): return self.func(x)


#----------------------------------------------------
# Basic functions which we will convert into layers. We will make use of these layers
# in our CNN model
#
# NB: We can't pass in a lambda function into the FuncLayer below, because the model won't pickle and
# you won't be able to save it with PyTorch. So it's best to use named functions with the FuncLayer.
#----------------------------------------------------
def flatten(x): return x.view(x.shape[0], -1)
# Reshape a flat vector of size (batch_size x 784) into a batch_size of 28x28 images with 1 channel
def mnist_resize(x): return x.view(-1, 1, 28, 28)

#----------------------------------------------------
# Get MNIST train and valid data and prepare the data loaders
#
# Set the 'repro' flag to allow fetching of data in a known order for reproducibility during debugging
#----------------------------------------------------
def prepare_mnist_data (repro):
  # Load MNIST data and normalise it
  x_train,y_train,x_valid,y_valid = get_mnist_data()
  x_train,x_valid = normalize_tr_val(x_train,x_valid)

  # Create Dataset and Data Loader with batch size
  train_ds,valid_ds = MultiDimDataset(x_train, y_train),MultiDimDataset(x_valid, y_valid)
  train_dl,valid_dl = get_dls(train_ds, valid_ds, bs=512, repro=repro)

  # Number of output classes
  n_classes = y_train.max().item()+1

  return (train_dl, valid_dl, n_classes)

#----------------------------------------------------
# Create a CNN architecture with a series of paired Conv2D and Relu layers. Each Relu layer is optionally
# followed by a BatchNorm layer. In other words, the last layer of each Conv set can be either a
# Relu or a BatchNorm. In addition, before and after these Conv sets, there are some
# pre and post layers that reshape the data as needed.
#
# We are given some optional parameters:
#    a list of output filter sizes for each Conv layer
#    a relu creation function which we use to create the Relu layer instead of the built-in Relu function
#    a batch norm creation function we use to create a BatchNorm layer
#    an initialisation function for the Conv layer weights and biases
#
# We have a flag to allow reproducibility of results for debugging. This uses the DebugRand class to
# set the Random Number Generator to a known state before initialising any layer weights/biases
#
# To use this class, first initialise it with all the necessary parameters, and then call() it.
#     arch = CNNArch(...)
#     model, hook_layers = arch(n_classes)
#
# We also return a tuple of two lists of layers ie. ([list of all model layers], [list of last layer in each Conv set])
#----------------------------------------------------
class CNNArch():
  def __init__(self, filter_list=None, relu_fn=None, bn_fn=None, init_conv_fn=None):
    # eg. with mnist 28x28 images, these 4 conv layers will output images of size
    # 14x14, 7x7, 4x4 and 2x2 respectively
    default_filters = [8, 16, 32, 64]

    self.filter_list=filter_list if filter_list is not None else default_filters
    self.relu_fn, self.bn_fn, self.init_conv_fn = relu_fn, bn_fn, init_conv_fn

  # ----------------------------
  # Creates the architecture based on the parameters provided during initialisation
  # ----------------------------
  def __call__(self, n_classes, repro=False):
    # For reproducibility during debugging
    DebugRand.repro = repro

    # Number of conv layers
    filter_list=self.filter_list
    num_conv_layers = len(filter_list)

    # We prepare two lists - a list of Conv + Relu + BatchNorm layer sets, and
    # a list of the last layer of each set
    conv_layers = []
    end_layers = []
    for i in range(num_conv_layers):

      # The first layer always has 1 input filter. All subsequent layers have the same number of input filters
      # as the number of output filters of the previous layer. The first layer also has a different kernel size
      # than all subsequent layers
      if (i == 0):
        num_inp_filters = 1
        kernel_size = 5
      else:
        num_inp_filters = filter_list[i - 1]
        kernel_size = 3

      # Create a set consisting of a Conv layer, a Relu layer and an optional BatchNorm layer.
      # All these layers are added to the list of layers

      # The Conv layer has no bias if we are using BatchNorm
      bias = True if self.bn_fn is None else False
      conv = nn.Conv2d (num_inp_filters, filter_list[i], kernel_size, padding=kernel_size//2, stride=2, bias=bias)
      relu = nn.ReLU() if self.relu_fn is None else self.relu_fn()
      conv_layers.extend([conv, relu])

      # Create a BatchNorm layer if a creation function is provided
      if (self.bn_fn is not None):
        conv_layers.append (self.bn_fn(filter_list[i]))

      # Add the last layer in the set
      end_layers.append(conv_layers[-1])

      # Initialise the weights (and biases) of the Conv layer
      if (self.init_conv_fn):
        DebugRand.set_seed(555)                   # Set RNG to a known state
        self.init_conv_fn(conv)
        DebugRand.show_state(conv.weight.shape, conv.weight.mean())

    # Create the architecture. We have a pre-layer that reshapes the flat MNIST input from 784 to 28x28 images
    # and some post layers for classification.
    DebugRand.set_seed(555)                       # Set RNG to a known state
    lin = nn.Linear(filter_list[-1], n_classes)
    DebugRand.show_state(lin.weight.shape, lin.weight.mean())
    final_layers = [nn.AdaptiveAvgPool2d(1), FuncLayer(flatten), lin]
    arch = nn.Sequential(
        FuncLayer(mnist_resize),
        *conv_layers,
        *final_layers
    )

    return (arch, (conv_layers + final_layers, end_layers))

#----------------------------------------------------
# Run a full MNIST CNN model end-to-end
#
# We have a lot of flexibility to run this:
#    With different CNN architectures by passing in an architecture creation function
#    With different optimisers by passing in an optimiser function
#    We can provide callbacks and hooks to be attached
#----------------------------------------------------
def run_cnn_mnist(num_epochs, arch_fn, opt_func=optim.SGD, opt_groups=None, lr=0.4, cbs=[], hook_cls=None, repro=False):
  # Prepare the data
  train_dl, valid_dl, n_classes = prepare_mnist_data (repro)

  # Create the CNN architecture
  arch, hook_layers = arch_fn(n_classes, repro)

  # Loss Function
  loss_func = F.cross_entropy

  # Optimiser
  opt = get_optimiser(arch, lr, opt_func, opt_groups)

  # Create the training loop and run it for the given number of epochs
  loop = Trainer(train_dl, valid_dl, arch, opt, loss_func, cbs)

  # If a list of hook classes has been provided, add those hooks to the model layers
  if (hook_cls):
    # Add hooks to the hook_layers and save a list of all the hooks
    loop.hooks = Hooks(hook_layers, hook_cls)

  # Run the training loop
  loop.fit(num_epochs)

  return(loop)


#----------------------------------------------------
# Initialise the weights and bias of a Conv layer using either Kaiming Normal
# or Kaiming Uniform.
#
# To use it:
#    init_fn = InitConv(..)
#    init_fn(conv_layer)
#----------------------------------------------------
class InitConv():
  def __init__(self, type, a=None):
    # 'Type' is either 'uniform' or 'normal'
    self.type=type
    # 'a' is an initialisation parameter used by the Kaiming function
    self.a=a

  # ----------------------------
  # Initialise the weights and biases
  # ----------------------------
  def __call__(self, conv):
    # Select the initialisation function
    if (self.type == "uniform"):
      init_fn = init.kaiming_uniform_
    elif (self.type == "normal"):
      init_fn = init.kaiming_normal_

    # Set the weights using the 'a' parameter if one is given
    if (self.a is None):
       init_fn(conv.weight)
    else:
      init_fn(conv.weight, a=self.a)

    # Initialise the bias if the conv layer has biases
    if getattr(conv, 'bias', None) is not None:
      conv.bias.data.zero_()


#----------------------------------------------------
# Generalised Relu module which can:
# Leaky - use a leaky Relu instead of a regular Relu
# Subtract - subtract a fixed value from the calculated Relu value
# Max - Force a max value for the Relu value
#----------------------------------------------------
class GeneralRelu(nn.Module):
    def __init__(self, leak=None, sub=None, maxv=None):
        super().__init__()
        self.leak,self.sub,self.maxv = leak,sub,maxv

    def forward(self, x):
        x = F.leaky_relu(x,self.leak) if self.leak is not None else F.relu(x)
        if self.sub is not None: x.sub_(self.sub)
        if self.maxv is not None: x.clamp_max_(self.maxv)
        return x


class BatchNorm(nn.Module):
    def __init__(self, nf, mom=0.1, eps=1e-5):
        super().__init__()
        # NB: pytorch bn mom is opposite of what you'd expect
        self.mom,self.eps = mom,eps
        self.mults = nn.Parameter(torch.ones (nf,1,1))
        self.adds  = nn.Parameter(torch.zeros(nf,1,1))
        self.register_buffer('vars',  torch.ones(1,nf,1,1))
        self.register_buffer('means', torch.zeros(1,nf,1,1))

    def update_stats(self, x):
        m = x.mean((0,2,3), keepdim=True)
        v = x.var ((0,2,3), keepdim=True)
        self.means.lerp_(m, self.mom)
        self.vars.lerp_ (v, self.mom)
        return m,v

    def forward(self, x):
        if self.training:
            with torch.no_grad(): m,v = self.update_stats(x)
        else: m,v = self.means,self.vars
        x = (x-m) / (v+self.eps).sqrt()
        #print('BN', m.mean(), v.mean(), x.mean())
        return x*self.mults + self.adds


class ArchBase:
  def __init__(self):
    self.model = None

  # ----------------------------
  # Show summary of the model with output sizes of each layer, for a given
  # input size
  # ----------------------------
  def summary(self, input_sz):
    from torchsummary import summary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = self.model.to(device)
    summary(model, input_size=input_sz)

  # ----------------------------
  # Load previously saved model weights
  # ----------------------------
  def load_weights(self, weights_path):
    self.model.load_state_dict(torch.load(weights_path))

  # ----------------------------
  # Save the model weights (after training)
  # ----------------------------
  def save_weights(self, weights_path):
    torch.save(self.model.state_dict(), weights_path)

  # ----------------------------
  # Freeze some layers of the model
  # ----------------------------
  def freeze(self, module=None, on=False):
    module = module if module is not None else self.model
    rg = not on

    # Get learnable parameters of all sub-modules in 'module'
    for m in module.modules():
      # We always leave BatchNorm layers unfrozen, so we never touch
      # them while freezing or unfreezing
      if not isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        for p in m.parameters():
          p.requires_grad_(rg)

  # ----------------------------
  # Define the module-layer groups to split the model for discriminative LRs.
  # The parameters from each module group will be put into separate parameter
  # groups.
  # ----------------------------
  def module_groups(self):
    lr_groups = [self.model]
    return lr_groups

  # ----------------------------
  # Create the model
  # ----------------------------
  def create_model(self):
    pass