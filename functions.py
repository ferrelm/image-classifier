# Imports here
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import json
import numpy as np
import os, random
import time
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import argparse


def load_data(data_dir):

	# Data location
	#data_dir = 'flowers'
	train_dir = data_dir + '/train'
	valid_dir = data_dir + '/valid'
	test_dir = data_dir + '/test'

	# TODO: Define your transforms for the training, validation, and testing sets
	# data_transforms
	train_transform = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

	valid_transform = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

	test_transform = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


	# TODO: Load the datasets with ImageFolder
	# image_datasets
	train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
	valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transform)
	test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

	# TODO: Using the image datasets and the trainforms, define the dataloaders
	# dataloaders
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
	valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=True)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

	# Ensamble
	dataset = [train_dataset, valid_dataset, test_dataset]
	loader = [train_loader, valid_loader, test_loader]
	print("\nload data completed")
    
	return dataset, loader

	
def build_model(arch, hidden_units, learning_rate, gpu):
	# TODO: Build and train your network
	# Use GPU if it's available

	device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")

	if arch == "vgg13":
		model = models.vgg13(pretrained=True)
	elif arch == "vgg16":
		model = models.vgg16(pretrained=True)
	else:
		print("wrong model")

	# Freeze parameters so we don't backprop through them
	for param in model.parameters():
		param.requires_grad = False
    
	from collections import OrderedDict
	classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('drop', nn.Dropout(0.5)), 
                          ('fc2', nn.Linear(hidden_units, 102)),   
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
	model.classifier = classifier


	# Define criterion and optimizer
	criterion = nn.NLLLoss()
	# Only train the classifier parameters, feature parameters are frozen

	optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

	model.to(device)

	print("\nbuild model completed")

	return model, classifier, criterion, optimizer



def train_model(loader, model, criterion, optimizer, epochs, gpu):
	# Run the model
	device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
	print(f"\nInitiating Training the model. Processor: {device}")
	#epochs = 2
	steps = 0
	running_loss = 0

	print_every = 10

	start = time.time()
	time_batch = start
    
	train_loader = loader[0]
	valid_loader = loader[1]

	for epoch in range(epochs):
		for inputs, labels in train_loader:
			steps += 1
            
			# Move input and label tensors to the default device
			inputs, labels = inputs.to(device), labels.to(device)
        
			optimizer.zero_grad()
        
			logps = model.forward(inputs)
			loss = criterion(logps, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.item()
        
			if steps % print_every == 0:
				valid_loss = 0
				accuracy = 0
				model.eval()
				with torch.no_grad():
					for inputs, labels in valid_loader:
						inputs, labels = inputs.to(device), labels.to(device)
						logps = model.forward(inputs)
						batch_loss = criterion(logps, labels)
                    
						valid_loss += batch_loss.item()
                    
						# Calculate accuracy
						ps = torch.exp(logps)
						top_p, top_class = ps.topk(1, dim=1)
						equals = top_class == labels.view(*top_class.shape)
						accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

				time_elapsed = time.time() - time_batch                    
            
				# print
                
				training_loss = running_loss/print_every
				validation_loss = valid_loss/len(valid_loader)
				acc = accuracy/len(valid_loader)
                
				print(f"\nStep {steps}. "
					f"Epoch {epoch+1}/{epochs}. "
					f"Train loss: {running_loss/print_every:.3f}. "
					f"Validation loss: {valid_loss/len(valid_loader):.3f}. "
					f"Validation accuracy: {accuracy/len(valid_loader):.3f}."            
					f"\nDevice = {device}. "
					f"Time per batch: {time_elapsed:.3f} seconds.")                    
                    
                                           
				running_loss = 0
				model.train()
            
				time_batch = time.time()
            
	time_elapsed = time.time() - start
	print(f"\nDevice = {device}; Total time: {time_elapsed//60:.0f}m:{time_elapsed%60:.0f}s")
	#model
    
	return model, training_loss, validation_loss, acc, time_elapsed



def test_model(loader, model, criterion, gpu):
	# TODO: Do validation on the test set

	device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")

	print(f"\nInitiating Testing the model. Processor: {device}")
    
	start = time.time()
	time_batch = start

	test_loss = 0
	accuracy = 0
	model.eval()
    
	test_loader = loader[2]    

	with torch.no_grad():
		for inputs, labels in test_loader:
			inputs, labels = inputs.to(device), labels.to(device)
			logps = model.forward(inputs)
			batch_loss = criterion(logps, labels)
                    
			test_loss += batch_loss.item()
                    
			# Calculate accuracy
			ps = torch.exp(logps)
			top_p, top_class = ps.topk(1, dim=1)
			equals = top_class == labels.view(*top_class.shape)
			accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            
            
	time_elapsed = time.time() - start
	print(f"\nTest loss: {test_loss/len(test_loader):.3f}. "
          f"Test accuracy: {accuracy/len(test_loader):.3f}."
        f"\nDevice = {device}; Total time: {time_elapsed//60:.0f}m:{time_elapsed%60:.0f}s")         
            

       

def save_checkpoint(save_dir, dataset, model, arch, classifier, optimizer, hidden_units, learning_rate, epochs, train_loss, valid_loss, accuracy, time_elapsed):
	# TODO: Save the checkpoint 
	train_dataset = dataset[0]
	model.class_to_idx = train_dataset.class_to_idx

	checkpoint = {'model_type': arch,
              'classifier' : classifier,
              'hidden_layer': hidden_units,
              'epochs': epochs,
              'optimizer': optim.Adam(model.classifier.parameters(), lr=learning_rate),
              'optimizer_state_dict': optimizer.state_dict(),
              'model_state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx,
              'train_loss': train_loss,
              'valid_loss': valid_loss,
              'accuracy': accuracy,
              'time_elapsed': time_elapsed,                  
                 }

	torch.save(checkpoint, save_dir)
	print(f"\nSaving the model completed.")
          
          
          
          

def load_checkpoint(filepath):
# TODO: Write a function that loads a checkpoint and rebuilds the model
    
    print(f"\nLoading the model...")
    
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'

    checkpoint = torch.load(filepath, map_location=map_location)
    
    arch = checkpoint['model_type']
    
    #model = models.vgg16(pretrained=True)
    if arch == "vgg13":
        model = models.vgg13(pretrained=True)
    elif arch == "vgg16":
        model = models.vgg16(pretrained=True)
        
        
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    optimizer = checkpoint['optimizer']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    
    train_loss = checkpoint['train_loss']
    valid_loss = checkpoint['valid_loss']
    accuracy = checkpoint['accuracy']
    time_elapsed = checkpoint['time_elapsed']
    
   
    print(f"\nTraining loss = {train_loss:.3f}."
      f"\nValidation loss = {valid_loss:.3f}"
      f"\naccuracy = {accuracy:.3f}"
      f"\ntime elapsed = {time_elapsed:0f}")

    return model, optimizer



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''    
    # TODO: Process a PIL image for use in a PyTorch model

    
    if image.height > image.width:
        image.thumbnail((256,100000))
    else:
        image.thumbnail((100000,256))
    
    edge = (256-224)/2
    box = (edge, edge, edge + 224, edge + 224)
    image = image.crop(box)
    image = np.array(image)/255
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean)/std
    
    return image.transpose(2,0,1)



def imshow2(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension

    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax



def predict(image_path, model, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    #model.cpu()    
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
    model.to(device)    
    model.eval()

    image = process_image(image_path)
    #print(image.shape)
    
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.unsqueeze(0)

    #Will only work if converted back to cpu
    model.cpu()
    
    logps = model.forward(image)
    ps = torch.exp(logps)
    probs, classes = ps.topk(topk, dim=1)
    
    probs = probs.data.numpy()[0]
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    
    labels = []
    for label in classes.numpy()[0]:
        labels.append(idx_to_class[label])
    
    return probs, labels



# TODO: Display an image along with the top 5 classes
#def sanity_check(image_path)
def sanity_check(model, image_path, topk, cat_to_name, gpu):
        
    string = image_path.split("/")
    #print(string)
    
    title = cat_to_name[string[3]]
    
    image_path = Image.open(image_path)

    probs, classes = predict(image_path, model, topk, gpu)
    #print(image_path)
    print(f"\nProbability vector: {probs}")
    #print(classes)
    names = [cat_to_name[x] for x in classes]
    print(f"\nFlower Names: {names}")

    
    print(f"\n\n\nFlower Name: {cat_to_name[string[3]]} \nClass Probability: {probs[0]:.5f}\n")

'''    
    plt.imshow(image_path)
    plt.figure(figsize = (6, 10))
    ax = plt.subplot(2, 1, 1)
    ax.imshow(image_path)
    ax.axis('off')
    ax.set_title(title)
    
    plt.subplot(2, 1, 2)
    sns.barplot(x=probs, y=names, color=sns.color_palette()[0]);
    plt.show()
'''
    
