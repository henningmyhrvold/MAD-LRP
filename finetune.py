from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import sklearn
import math
from tqdm import tqdm
from os import listdir
from sklearn.metrics import confusion_matrix
import fnmatch


print('Scikit-learn version is {}.'.format(sklearn.__version__))
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


# Counts all the .jpg and .png in a specific folder. This is done to speed up the training with using preallocated arrays with a fixed size.
def count_images(my_path):
	num_images = len(fnmatch.filter(os.listdir(my_path),'*.png')) + len(fnmatch.filter(os.listdir(my_path),'*.jpg'))
	return num_images


# Hardcoded paths to the datasets different folders.
path_train_bonafide = r'/home/ubuntu/Desktop/TorchLRP-master/Morph_Data/train/bonafide'
path_train_morph = r'/home/ubuntu/Desktop/TorchLRP-master/Morph_Data/train/morph'
path_validation_bonafide = r'/home/ubuntu/Desktop/TorchLRP-master/Morph_Data/validation/bonafide'
path_validation_morph = r'/home/ubuntu/Desktop/TorchLRP-master/Morph_Data/validation/morph'


# Finds the number of images in the training set and validation set. 
number_train_images = count_images(path_train_bonafide) + count_images(path_train_morph)
number_validation_images = count_images(path_validation_bonafide) + count_images(path_validation_morph)

print(number_train_images)
print(number_validation_images)



def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Add these paramters for the TP, TN, FP, FN calculations. 
    # NOTE! If you are going to use batch_size > 1, comment out all TP, TN, FP, FN code
    
    y_true = []
    y_prediction = []
    
    for epoch in range(num_epochs):
        print()
        print('#' * 40)
        print('#' * 40)
        print()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()  # Set model to training mode
		
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                                
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    # mode we calculate the loss by summing the final output and the auxiliary output
                    # but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                
                # These paramters are added for the TP, TN, FP, FN calculations. 
                
                #print('-------------------------------------------------')
                #print('SELF-ADDED: PREDICTION SUM: {}'.format(torch.sum(preds)))
                #print('SELF-ADDED: PREDICTION LIST: {}'.format(preds.tolist()))
                #print('SELF-ADDED: TRUE LABELS SUM: {}'.format(torch.sum(labels.data)))
                #print('SELF-ADDED: TRUE LABELS LIST: {}'.format(labels.data.tolist()))
                #print('SELF-ADDED: EQUAL: {}'.format(torch.sum(preds == labels.data)))
                
                y_prediction.extend(preds.tolist())
                y_true.extend(labels.data.tolist())
                
                #print('SELF-ADDED: y_prediction running: {}'.format(y_prediction))
                #print('SELF-ADDED: y_true running:       {}'.format(y_true))
                
                # Running corrects = TP + TN
                running_corrects += torch.sum(preds == labels.data)
                
                #print('SELF-ADDED: RunningCorrect: {}'.format(running_corrects))
                
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            
            # Add these paramters for the TP, TN, FP, FN calculations.
            # The confusion matrix contains all the TP, TN, FP, FN values in a 2 dimentional array
            # Clears all values before starting new phase or epoch
            print('SELF-ADDED: PHASE {} '.format(phase))
            cnf_matrix = confusion_matrix(y_true, y_prediction)
            
            print(cnf_matrix)
            
            tp, fp, fn, tn = cnf_matrix.ravel()
            print('True Positive: {},  False Positive: {},  False Negative: {},  True Negative: {}\n'.format(tp, fp, fn, tn))
            
            y_true = []
            y_prediction = []
            
            print()
            print('SELF-ADDED: ANTALL {} Loss: {:.4f} Acc: {:.4f}'.format(phase, running_loss, running_corrects))
            print()
            print('ORIGINAL: PHASE {} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            with open("/home/ubuntu/Desktop/TorchLRP-master/results_output.txt","a") as file:
            	file.write("\n")
            	file.write("\n")
            	file.write("##############################################\n")
            	file.write("\n")
            	file.write("\n")
            	file.write('Epoch {}/{}\n'.format(epoch, num_epochs - 1))
            	file.write('----------------------------------------------\n')
            	file.write('SELF-ADDED: PHASE {}\n'.format(phase))
            	file.write('True Positive: {},  False Positive: {},  False Negative: {},  True Negative: {}\n'.format(tp, fp, fn, tn))
            	file.write('\n')
            	file.write('SELF-ADDED: ANTALL {} Loss: {:.4f} Acc: {:.4f}\n'.format(phase, running_loss, running_corrects))
            	file.write('\n')
            	file.write('ORIGINAL: PHASE {} Loss: {:.4f} Acc: {:.4f}\n'.format(phase, epoch_loss, epoch_acc))
            	file.close()


            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'validation':
                val_acc_history.append(epoch_acc)

        print()
           

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    with open("/home/ubuntu/Desktop/TorchLRP-master/results_output.txt","a") as file:
    	file.write('\n')
    	file.write('\n')
    	file.write('Training complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
    	file.write('Best val Acc: {:4f}\n'.format(best_acc))
    	file.write('\n')
    	file.write('----------------------------------------------\n')
    	file.close()

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables.
    model_ft = None
    input_size = 0

    if model_name == "vgg":
        """ VGG19_bn
        """
        model_ft = models.vgg19_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size



# Top level data directory. Here we assume the format of the directory conforms
# to the ImageFolder structure
data_dir = "./Morph_Data/"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "vgg"

# Number of classes in the dataset
num_classes = 2

# Batch size for training (change depending on how much memory you have)
batch_size = 128

# Number of epochs to train for
num_epochs = 64

# Flag for feature extracting. When False, we finetune the whole model,
# when True we only update the reshaped layer params
feature_extract = False

# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Print the model we just instantiated
#print(model_ft)



# Data augmentation and normalization for training, resize images.
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validation': transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'validation']}

# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'validation']}

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Send the model to GPU
model_ft = model_ft.to(device)



#  Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)



# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(params_to_update, lr=0.001, betas=(0.9, 0.999), eps=1e-08)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Train and evaluate
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))

#Save model
torch.save(model_ft, 'model_ft.pth')







