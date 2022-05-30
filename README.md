This repository is based on the code from Frederik Hvilshøj GitHub repository TorchLRP as well as the beginner tutorial from pytorch.org authored by Nathan Inkawhich for fine-tuning torchvision models. I got help from my course advisor Haoyu Zhang on finding this code. He has also debugged the code eariler and connected the code written by Hvilshøj with the code from pythorch.org so it worked for S-MAD. I have added some small changes in order to calculate True Positiv, True Negative, False Positive, False Negative. I have also added the confusion_matrix functionallity from sklearn. 

Available from:

https://github.com/fhvilshoj/TorchLRP

https://pytorch.org/tutorials/beginner/finetuning\_torchvision\_models\_tutorial.html

# Implementation of LRP for pytorch
PyTorch implementation of some of the Layer-Wise Relevance Propagation (LRP)
rules for linear layers and convolutional layers.

The modules decorates `torch.nn.Sequential`, `torch.nn.Linear`, and
`torch.nn.Conv2d` to be able to use `autograd` backprop algorithm to compute
explanations.

## Installation
To install requirements, refer to the [`requirements.yml`](requirements.yml)
file.

If you use `conda`, then you can install an environment called `torchlrp` by
executing the following command: 

```bash
> conda env create -f requirements.yml
```

To be able to import `lrp` as below, make sure that the `TorchLRP` directory is
included in your path.


**Implemented rules:**
|Rule 							|Key 					| Note 												|
|:------------------------------|:----------------------|:--------------------------------------------------|
|epsilon-rule					| "epsilon" 			| Implemented but epsilon fixed to `1e-1` 			|
|gamma-rule						| "gamma" 				| Implemented but gamma fixed to `1e-1`				|
|epsilon-rule					| "epsilon" 			| gamma and epsilon fixed to `1e-1`					|
|alpha=1 beta=0 				| "alpha1beta0" 		| 													|
|alpha=2 beta=1 				| "alpha2beta1" 		| 													|
|PatternAttribution (all) 		| "patternattribution" 	| Use additional argument `pattern=patterns_all` 	|
|PatternAttribution (positive) 	| "patternattribution" 	| Use additional argument `pattern=patterns_pos` 	|
|PatternNet (all) 				| "patternnet" 			| Use additional argument `pattern=patterns_all` 	|
|PatternNet (positive) 			| "patternnet" 			| Use additional argument `pattern=patterns_pos` 	|

To compute patterns for the two `PatternAttribution` methods, import
`lrp.patterns` and call
```python 
import lrp.patterns.*
patterns_all = fit_patternnet(model, train_loader)
patterns_pos = fit_patternnet_positive(model, train_loader)
```
_Note:_ Biases are currently ignored in the alphabeta-rule implementations.


### Trace intermediate relevances
Thanks to [francescomalandrino](https://github.com/francescomalandrino), you can now also
trace the intermediate relevances by enabling traces:

```python
... 
lrp.trace.enable_and_clean()
y_hat.backward()
all_relevances=lrp.trace.collect_and_disable()

for i,t in enumerate(all_relevances):
    print(i,t.shape)
```

## VGG
It is also possible to use this code for pretrained vgg models from `torchvision`,
by using the `lrp.convert_vgg` function to convert `torch.nn.Conv2d` and `torch.nn.Linear` layers to `lrp.Conv2d` and `lrp.Linear`, respectively. 

The most interesting parts is converting the torch vgg models, such that they can be
explained. To do so, do as follows:

```python 
vgg = torchvision.models.vgg16(pretrained=True).to(device)
vgg.eval()
lrp_vgg = lrp.convert_vgg(vgg).to(device)
```

The `lrp_vgg` model will then have the same parameters as the original network.
Afterwards, explanations can be produced.


# LRP for S-MAD

*Note from Haoyu:*

To install pytorch and torchvision, check your OS(windows/linux/...), installation source (conda/pip/...), compute platform (cpu/cuda10/cuda11/...) and required version (Pytorch==1.6, ) in [`requirements.yml`](requirements.yml).

Then find corresponding installation command at https://pytorch.org/get-started/previous-versions/.

I was debugging in pytorch==1.7.1 + torchvision==0.8 and it also works.


## Finetune a S-MAD model
 ```bash
python finetune.py
```
There are some hardcoded arguments at line 176. By switching the 'feature_extract' parameter you can choose to either fintune the whole model or only the last layer.

The finetuned model is stored as 'model_ft.pth'.

## Explain finetuned model with LRP
 ```bash
python explain.py 
```

This script will load the fintuned S-MAD model and then run the LRP explaination based on their own example scripts. The results are stored at './test_results' folder.

You may need to modify the script to generate your own figures for experiments and illustrations.


# Calculating True Positive, True Negative, False Positive, False Negative 

*Notes from Henning*

To the finetune.py file I added the sklearn confusion_matrix. By creating two variables, y_true and y_prediction, I could count these up during the training and validating phase. As of now, this only works if batch_size = 1 on line 180. If you wish to run with a higher batch_size for higher accuracy, I recommend to comment out the changes related to TP, TN. FP, FN.   


## References
[1] Bach, S., Binder, A., Montavon, G., Klauschen, F., Müller, K.R. and Samek, W., 2015. On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation. PloS one, 10(7), p.e0130140.  
[2] Kindermans, P.J., Schütt, K.T., Alber, M., Müller, K.R., Erhan, D., Kim, B. and Dähne, S., 2017. Learning how to explain neural networks: Patternnet and patternattribution. arXiv preprint arXiv:1705.05598.  
[3] Montavon, G., Binder, A., Lapuschkin, S., Samek, W. and Müller, K.R., 2019. Layer-wise relevance propagation: an overview. In Explainable AI: interpreting, explaining and visualizing deep learning (pp. 193-209). Springer, Cham.  
[4] Hvilshøj, Frederik, Implementation of LRP for pytorch, Available from: https://github.com/fhvilshoj/TorchLRP, Accessed: 14.04.2022 
[5] Inkawhich, Nathan, FINETUNING TORCHVISION MODELS, Available from: https://pytorch.org/tutorials/beginner/finetuning\_torchvision\_models\_tutorial.html, Accessed: 14.04.2022

