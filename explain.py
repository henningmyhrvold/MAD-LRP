import os
import sys
import torch
import pickle
from torch.nn import Sequential, Conv2d, Linear
import pathlib
import argparse
import torchvision
from torchvision import datasets, transforms
import configparser
import numpy as np
import matplotlib.pyplot as plt
import lrp
from lrp.patterns import fit_patternnet, fit_patternnet_positive # PatternNet patterns
from utils import store_patterns, load_patterns
from visualization import project, clip_quantile, heatmap_grid, grid

def compute_and_plot_explanation(rule, ax_, patterns=None, plt_fn=heatmap_grid): 
    # Forward pass
    y_hat_lrp = lrp_vgg.forward(x, explain=True, rule=rule, pattern=patterns)

    # Choose argmax
    y_hat_lrp = y_hat_lrp[torch.arange(x.shape[0]), y_hat_lrp.max(1)[1]]
    y_hat_lrp = y_hat_lrp.sum()

    # Backward pass (compute explanation)
    y_hat_lrp.backward()
    attr = x.grad

    # Plot
    attr = plt_fn(attr)
    ax_.imshow(attr)
    ax_.set_title(rule)
    ax_.axis('off')

# PatternNet is typically handled a bit different, when visualized.
def signal_fn(X):
    if X.shape[1] in [1, 3]: X = X.permute(0, 2, 3, 1).detach().cpu().numpy()
    X = clip_quantile(X)
    X = project(X)
    X = grid(X)
    return X

# base path
base_path = os.getcwd()

# Top level data directory. Here we assume the format of the directory conforms
# to the ImageFolder structure
data_dir = "./Morph_Data/"

# Models to choose from
model_name = "vgg"

# Number of classes in the dataset
num_classes = 2

# Batch size for testing (change depending on how much memory you have)
batch_size = 12
input_size = 224

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


torch.manual_seed(1)


data_transforms = {
    'test': transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Normalization as expected by pytorch vgg models
_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view((1, 3, 1, 1))
_std  = torch.tensor([0.229, 0.224, 0.225], device=device).view((1, 3, 1, 1))

def unnormalize(x):
    return x * _std + _mean

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
image_datasets = {label_dir: datasets.ImageFolder(os.path.join(data_dir, 'test_explain', label_dir), data_transforms['test']) for label_dir in ['bonafide_only','morph_only']}

# Create training and validation dataloaders
dataloaders_dict = {label_dir: torch.utils.data.DataLoader(image_datasets[label_dir], batch_size=batch_size, shuffle=True, num_workers=4) for label_dir in ['bonafide_only','morph_only']}


# # # # # VGG model
vgg_num = 19 

vgg = torch.load('model_ft.pth').to(device)
vgg.eval()

print("Loaded vgg-%i" % vgg_num)

lrp_vgg = lrp.convert_vgg(vgg).to(device)
# # # # #

# Check that the vgg and lrp_vgg models does the same thing
for x, y in dataloaders_dict['morph_only']: break
print(y)
x = x.to(device)
x.requires_grad_(True)

y_hat = vgg(x)
y_hat_lrp = lrp_vgg.forward(x)

assert torch.allclose(y_hat, y_hat_lrp, atol=1e-4, rtol=1e-4), "\n\n%s\n%s\n%s" % (str(y_hat.view(-1)[:10]), str(y_hat_lrp.view(-1)[:10]), str((torch.abs(y_hat - y_hat_lrp)).max()))
print("Done testing")
# # # # #

# # # # # Patterns for PatternNet and PatternAttribution
patterns_path = os.path.join(base_path,'test_results','vgg19_pattern_pos_morph.pkl')
if not os.path.exists(patterns_path):
    patterns = fit_patternnet_positive(lrp_vgg, dataloaders_dict['morph_only'], device=device)
    store_patterns(patterns_path, patterns)
else:
    patterns = [torch.tensor(p).to(device) for p in load_patterns(patterns_path)]

print("Loaded patterns")

# # # # # Plotting 
explanations = [
        # rule                  Pattern     plt_fn          Fig. pos
        ('alpha2beta1',         None,       heatmap_grid,   (0, 1)), 
        ('epsilon',             None,       heatmap_grid,   (0, 2)), 
        ('gamma+epsilon',       None,       heatmap_grid,   (1, 0)), 
        ('patternnet',          patterns,   heatmap_grid,   (1, 1)),
        ('patternattribution',  patterns,   heatmap_grid,   (1, 2)),
    ]

#signal_fn

fig, ax = plt.subplots(2, 3, figsize=(12, 8))
print("Plotting")

# Plot inputs
input_to_plot = unnormalize(x).permute(0, 2, 3, 1).contiguous().detach().cpu().numpy()
input_to_plot = grid(input_to_plot, 3, 1.)
ax[0, 0].imshow(input_to_plot)
ax[0, 0].set_title("Input")
ax[0, 0].axis('off')

# Plot explanations
for i, (rule, pattern, fn, (p, q) ) in enumerate(explanations): 
    compute_and_plot_explanation(rule, ax[p, q], patterns=pattern, plt_fn=fn)

fig.tight_layout()
fig.savefig(os.path.join(base_path,'test_results','vgg19_explanations_morph.png'), dpi=280)




# Check that the vgg and lrp_vgg models does the same thing
for x, y in dataloaders_dict['bonafide_only']: break
x = x.to(device)
x.requires_grad_(True)

y_hat = vgg(x)
y_hat_lrp = lrp_vgg.forward(x)

assert torch.allclose(y_hat, y_hat_lrp, atol=1e-4, rtol=1e-4), "\n\n%s\n%s\n%s" % (str(y_hat.view(-1)[:10]), str(y_hat_lrp.view(-1)[:10]), str((torch.abs(y_hat - y_hat_lrp)).max()))
print("Done testing")
# # # # #

# # # # # Patterns for PatternNet and PatternAttribution
patterns_path = os.path.join(base_path,'test_results','vgg19_pattern_pos_bonafide.pkl')
if not os.path.exists(patterns_path):
    patterns = fit_patternnet_positive(lrp_vgg, dataloaders_dict['bonafide_only'], device=device)
    store_patterns(patterns_path, patterns)
else:
    patterns = [torch.tensor(p).to(device) for p in load_patterns(patterns_path)]

print("Loaded patterns")

# # # # # Plotting 
explanations = [
        # rule                  Pattern     plt_fn          Fig. pos
        ('alpha2beta1',         None,       heatmap_grid,   (0, 1)), 
        ('epsilon',             None,       heatmap_grid,   (0, 2)), 
        ('gamma+epsilon',       None,       heatmap_grid,   (1, 0)), 
        ('patternnet',          patterns,   heatmap_grid,   (1, 1)),
        ('patternattribution',  patterns,   heatmap_grid,   (1, 2)),
    ]

fig, ax = plt.subplots(2, 3, figsize=(12, 8))
print("Plotting")

# Plot inputs
input_to_plot = unnormalize(x).permute(0, 2, 3, 1).contiguous().detach().cpu().numpy()
input_to_plot = grid(input_to_plot, 3, 1.)
ax[0, 0].imshow(input_to_plot)
ax[0, 0].set_title("Input")
ax[0, 0].axis('off')

# Plot explanations
for i, (rule, pattern, fn, (p, q) ) in enumerate(explanations): 
    compute_and_plot_explanation(rule, ax[p, q], patterns=pattern, plt_fn=fn)

fig.tight_layout()
fig.savefig(os.path.join(base_path,'test_results','vgg19_explanations_bonafide.png'), dpi=280)







