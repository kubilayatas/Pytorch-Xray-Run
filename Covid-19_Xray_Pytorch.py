import argparse
###############################################################################
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset which includes 'images' folder and 'metadata.csv' file")
ap.add_argument("-p", "--plot", type=str, default="Training Results.png",
	help="path to output loss/accuracy graphical plot file with extention '.png'")
ap.add_argument("-m", "--savemodel", type=str, default="./",
	help="path to output final trained model")
ap.add_argument("-e", "--epochs", type=int, default=50,
	help="An integer input to Epoch number")
ap.add_argument("-b", "--batchSize", type=int, default=16,
	help="An integer input to Batch Size")
#ap.add_argument("-l", "--learningRate", type=float, default=1e-3,
#	help="A floating point input to LearningRate, default is 1e-3")
ap.add_argument("-t", "--topmodel", type=str, default="resnet18",
	help="Input to select pretrained top model, default is 'resnet18'. alexnet, squeezenet1_0, squeezenet1_1,\
	vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn, densenet121, densenet169, densenet161,\
	densenet201, inception, googlenet, shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5,\
	shufflenet_v2_x2_0, mobilenet_v2, resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d,\
	resnext101_32x8d, wide_resnet50_2, wide_resnet101_2, mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3,\
	and efficientnet-b0 are available.")
args = vars(ap.parse_args()) 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os, random, torch, time, copy
import torch.nn.functional as F
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset
from PIL import Image, ImageOps, ImageFilter
from skimage.filters import threshold_local

from torch.autograd import Variable

covid_dataset_path = args["dataset"]
plotfigure = args["plot"]
ModelSavePath = args["savemodel"]
EpoCHs = args["epochs"]
BatChSize = args["batchSize"]
TopModelName = args["topmodel"]
##################################
print("########[ARGS INFO]########")
print("DATASET PATH: {}".format(covid_dataset_path))
print("MODEL SAVE PATH: {}".format(ModelSavePath))
print("EPOCHS: {}".format(EpoCHs))
print("BATCH SIZE: {}".format(BatChSize))
print("TOP MODEL NAME: {}".format(TopModelName))
print("######[END ARGS INFO]#######")
##################################
classes = ['no-covid', 'covid']

class image_dataset(Dataset):
    """Class creator for the x-ray dataset."""

    def __init__(self, csv_path, root_dir, transform=None, phase=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform
        # If not a PA view, drop the line 
        self.df.drop(self.df[self.df.view != 'PA'].index, inplace=True)
        self.phase = phase

    def __len__(self):
        
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if self.df['finding'].iloc[idx] != 'COVID-19':
            finding = 0
            img_path = os.path.sep.join([covid_dataset_path, 'images', self.df['filename'].iloc[idx]])
            image = Image.open(img_path)
            sample = {'image': image, 'finding': finding}
            
            if self.transform:
                sample = {'image': self.transform[self.phase](sample['image']), 'finding': finding}

        else:
            finding = 1
            img_path = os.path.sep.join([covid_dataset_path, 'images', self.df['filename'].iloc[idx]])
            image = Image.open(img_path)
            sample = {'image': image, 'finding': finding}

            if self.transform:
                sample = {'image': self.transform[self.phase](sample['image']), 'finding': finding}

        return sample

xray_dataset = image_dataset(csv_path=os.path.sep.join([covid_dataset_path, 'metadata.csv']), root_dir=covid_dataset_path)

class HistEqualization(object):
    """Image pre-processing.

    Equalize the image historgram
    """
    
    def __call__(self,image):
        
        return ImageOps.equalize(image, mask = None)

class ContrastBrightness(object):
    """Image pre-processing.

    alpha = 1.0 # Simple contrast control [1.0-3.0]
    beta = 0    # Simple brightness control [0-100]
    """
    
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
    
    def __call__(self,image,):
        image = np.array(image)
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                image[y,x] = np.clip(self.alpha*image[y,x] + self.beta, 0, 255)

                return Image.fromarray(np.uint8(image)*255)

class SmothImage(object):
    """Image pre-processing.

    Smooth the image
    """
    def __call__(self,image):
        
        return image.filter(ImageFilter.SMOOTH_MORE)

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Grayscale(1),
        transforms.RandomRotation(30, fill=(0,)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        ContrastBrightness(1.2,25),
        HistEqualization(),
        SmothImage(),
        transforms.ToTensor(),
		transforms.Normalize(mean=[0.456], std=[0.224])
		#transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #transforms.Normalize([0.5],[0.25])
    ]),
    'test': transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(240),
        transforms.CenterCrop(224),
        ContrastBrightness(1.2,25),
        HistEqualization(),
        SmothImage(),
        transforms.ToTensor(),
		transforms.Normalize(mean=[0.456], std=[0.224])
		#transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #transforms.Normalize([0.5], [0.25])
    ]),
}

scale = transforms.Resize(256)
crop = transforms.CenterCrop(400)
composed = transforms.Compose([
    transforms.Grayscale(1),
    HistEqualization(),
    transforms.RandomRotation(30, fill=(0,)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224),    
])

xray_transform = image_dataset(
    csv_path=os.path.sep.join([covid_dataset_path, 'metadata.csv']),
    root_dir=covid_dataset_path,
    transform=data_transforms,
    phase='train'
)

image_datasets = {
    x: image_dataset(
        csv_path=os.path.sep.join([covid_dataset_path, 'metadata.csv']),
        root_dir=covid_dataset_path,
        transform=data_transforms,
        phase=x)
    for x in ['train', 'test']
}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BatChSize,
                                              shuffle=True, num_workers=8)
              for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("[INFO] DEVICE: {}".format(device))

def train_model(model, criterion, optimizer, scheduler, num_epochs=25, device="cpu"):
	"""
	Support function for model training.
	Args:
	model: Model to be trained
	criterion: Optimization criterion (loss)
	optimizer: Optimizer to use for training
	scheduler: Instance of ``torch.optim.lr_scheduler``
	num_epochs: Number of epochs
	device: Device to run the training on. Must be 'cpu' or 'cuda'
	"""
	since = time.time()
	TrainHistory={'val_acc':[],'val_loss':[],'train_acc':[],'train_loss':[]}
	print("[INFO] Training begins with {} device...".format(device))
	
	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0
	
	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)
		
		# Each epoch has a training and validation phase
		for phase in ['train', 'test']:
			if phase == 'train':
				model.train()  # Set model to training mode
			else:
				model.eval()   # Set model to evaluate mode
			
			running_loss = 0.0
			running_corrects = 0
			# Iterate over data.
			for data in dataloaders[phase]:
				inputs = data['image']
				labels = data['finding']
				inputs = inputs.to(device)
				labels = labels.to(device)
				# zero the parameter gradients
				optimizer.zero_grad()
				# forward
				# track history if only in train
				with torch.set_grad_enabled(phase == 'train'):
					outputs = model(inputs)
					_, preds = torch.max(outputs, 1)
					loss = criterion(outputs, labels)
					# backward + optimize only if in training phase
					if phase == 'train':
						loss.backward()
						optimizer.step()
				# statistics
				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)
			
			if phase == 'train':
				scheduler.step()
			
			epoch_loss = running_loss / dataset_sizes[phase]
			epoch_acc = running_corrects.double() / dataset_sizes[phase]
			print('{} Loss: {:.4f} Acc: {:.4f}'.format(
			phase, epoch_loss, epoch_acc))
			# deep copy the model
			if phase == 'test':
				TrainHistory["val_acc"].append(epoch_acc.item())
				TrainHistory["val_loss"].append(epoch_loss)
			if phase == 'train':
				TrainHistory["train_acc"].append(epoch_acc.item())
				TrainHistory["train_loss"].append(epoch_loss)
            
			if phase == 'test' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())

		print()
	
	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
	time_elapsed // 60, time_elapsed % 60))
	print('Best validation Acc: {:4f}'.format(best_acc))
	# Saving History of train and test
	pattth = os.path.sep.join([ModelSavePath,TopModelName])
	os.makedirs(pattth, exist_ok=True)
	pattth = os.path.join(pattth,TopModelName+"_TrainHistory")
	print("[INFO] Saving history to: {}".format(pattth))
	text_file = open(pattth, "w")
	text_file.write(str(TrainHistory))
	text_file.close()
	# Plotting
	N = EpoCHs
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(np.arange(0, N), TrainHistory["train_loss"], label="train_loss")
	plt.plot(np.arange(0, N), TrainHistory["val_loss"], label="val_loss")
	plt.plot(np.arange(0, N), TrainHistory["train_acc"], label="train_acc")
	plt.plot(np.arange(0, N), TrainHistory["val_acc"], label="val_acc")
	plt.title("Training Loss and Accuracy on COVID-19 Dataset")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	pattth = os.path.sep.join([ModelSavePath,TopModelName])
	pattth = os.path.join(pattth,TopModelName+"_historyfigure.svg")
	plt.savefig(pattth,format='svg', dpi=1200)
	# load best model weights
	model.load_state_dict(best_model_wts)
	# save best model weights
	pattth = os.path.sep.join([ModelSavePath,TopModelName])
	pattth = os.path.join(pattth,TopModelName+"_best_model.state_dict")
	torch.save(model.state_dict(), pattth)
	# For load
	#model = TheModelClass(*args, **kwargs)
	#model.load_state_dict(torch.load(PATH))
	#model.eval()
	return model
##################################################
class Bizim_Model(nn.Module):
	def __init__(self, my_pretrained_model):
		super(Bizim_Model, self).__init__()
		self.firstlayer = nn.Conv2d(1, 3, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1))
		self.pretrained = my_pretrained_model
		self.my_new_layers = nn.Sequential(
		nn.Linear(1000, 500),
		nn.Dropout(0.5),
		nn.ReLU(),
		nn.Linear(500,128),
		nn.ReLU(),
		nn.Linear(128, 2)
		)

	def forward(self, x):
			x = self.firstlayer(x)
			x = self.pretrained(x)
			x = self.my_new_layers(x)
			return x


test=False
##################################################
if TopModelName=="alexnet":
	pre_model = models.alexnet(pretrained=True, progress=True)
	
elif TopModelName=="squeezenet1_0":
	pre_model=models.squeezenet1_0(pretrained=True,progress=True)
elif TopModelName=="squeezenet1_1":
	pre_model=models.squeezenet1_1(pretrained=True,progress=True)

####### VGG's ###########
elif TopModelName=="vgg11":
	pre_model=models.vgg11(pretrained=True, progress=True)
elif TopModelName=="vgg11_bn":
	pre_model=models.vgg11_bn(pretrained=True, progress=True)
elif TopModelName=="vgg13":
	pre_model=models.vgg13(pretrained=True, progress=True)
elif TopModelName=="vgg13_bn":
	pre_model=models.vgg13_bn(pretrained=True, progress=True)
elif TopModelName=="vgg16":
	pre_model=models.vgg16(pretrained=True, progress=True)
elif TopModelName=="vgg16_bn":
	pre_model=models.vgg16_bn(pretrained=True, progress=True)
elif TopModelName=="vgg19":
	pre_model=models.vgg19(pretrained=True, progress=True)
elif TopModelName=="vgg19_bn":
	pre_model=models.vgg19_bn(pretrained=True, progress=True)
############################

elif TopModelName=="densenet121":
	pre_model = models.densenet121(pretrained=True, progress=True)
elif TopModelName=="densenet169":
	pre_model = models.densenet169(pretrained=True, progress=True)
elif TopModelName=="densenet161":
	pre_model = models.densenet161(pretrained=True, progress=True)
elif TopModelName=="densenet201":
	pre_model = models.densenet201(pretrained=True, progress=True)
	
elif TopModelName=="inception":
	pre_model = models.inception_v3(pretrained=True, progress=True)

elif TopModelName=="googlenet":
	pre_model = models.googlenet(pretrained=True, progress=True)

elif TopModelName=="shufflenet_v2_x0_5":
	pre_model = models.shufflenet_v2_x0_5(pretrained=True, progress=True)
elif TopModelName=="shufflenet_v2_x1_0":
	pre_model = models.shufflenet_v2_x0_5(pretrained=True, progress=True)
elif TopModelName=="shufflenet_v2_x1_5":
	pre_model = models.shufflenet_v2_x0_5(pretrained=True, progress=True)
elif TopModelName=="shufflenet_v2_x2_0":
	pre_model = models.shufflenet_v2_x0_5(pretrained=True, progress=True)

elif TopModelName=="mobilenet_v2":
	pre_model = models.mobilenet_v2(pretrained=True, progress=True)

######### ResNet's ###########
elif TopModelName=='resnet18':
	pre_model = models.resnet18(pretrained=True, progress=True)
elif TopModelName=='resnet34':
	pre_model = models.resnet34(pretrained=True, progress=True)
elif TopModelName=='resnet50':
	pre_model = models.resnet50(pretrained=True, progress=True)
elif TopModelName=='resnet101':
	pre_model = models.resnet101(pretrained=True, progress=True)
elif TopModelName=='resnet152':
	pre_model = models.resnet152(pretrained=True, progress=True)
elif TopModelName=="resnext50_32x4d":
	pre_model = models.resnext50_32x4d(pretrained=True, progress=True)
elif TopModelName=='resnext101_32x8d':
	pre_model = models.resnext101_32x8d(pretrained=True, progress=True)
elif TopModelName=="wide_resnet50_2":
	pre_model = models.wide_resnet50_2(pretrained=True, progress=True)
elif TopModelName=='wide_resnet101_2':
	pre_model = models.wide_resnet101_2(pretrained=True, progress=True)
##############################

elif TopModelName=="mnasnet0_5":
	pre_model = models.mnasnet0_5(pretrained=True, progress=True)
elif TopModelName=="mnasnet0_75":
	pre_model = models.mnasnet0_75(pretrained=True, progress=True)
elif TopModelName=="mnasnet1_0":
	pre_model = models.mnasnet1_0(pretrained=True, progress=True)
elif TopModelName=="mnasnet1_3":
	pre_model = models.mnasnet1_3(pretrained=True, progress=True)
	
elif TopModelName=="efficientnet-b0":
	from efficientnet_pytorch import EfficientNet
	pre_model = EfficientNet.from_pretrained('efficientnet-b0')
######################### Testing Models #################################
elif TopModelName=="resnext50_32x4d_test":
	model = models.resnext50_32x4d(pretrained=True, progress=True)
	model.conv1=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
	model.fc.out_features=2
	test=True
elif TopModelName=="vgg16_test":
	model=models.vgg16(pretrained=True, progress=True)
	model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	model.classifier[6].out_features = 2
	test=True

else:
	pre_model = models.resnet18(pretrained=True, progress=True)



if test:
	print("[TEST MODE]")
else:
	for param in pre_model.parameters():
		param.requires_grad = False
	model = Bizim_Model(my_pretrained_model=pre_model)

criterion = nn.CrossEntropyLoss()

# Note that we are only training the head.
optimizer_ft = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.1)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.3)

model_ft = train_model(model.to(device), criterion, optimizer_ft, exp_lr_scheduler, num_epochs = EpoCHs, device = device)

#Evaluation
correct = 0
total = 0
with torch.no_grad():
    for data in dataloaders['test']:
        inputs = data['image']
        labels = data['finding']
        outputs = model_ft(inputs.float().to(device))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.to(device)).sum().item()

print('Accuracy of the network: %d %%' % (
    100 * correct / total))

###################################################
class_correct = list(0. for i in range(2))
class_total = list(0. for i in range(2))


with torch.no_grad():
    for data in dataloaders['test']:
        images = data['image']
        labels = data['finding']
        outputs = model_ft(images.to(device))
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels.to(device)).squeeze()
        for i in range(images.shape[0]):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(2):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
######################################################
dataloader = torch.utils.data.DataLoader(image_datasets['test'], batch_size=(len(image_datasets['test'])), num_workers=8)

dataiter = iter(dataloader)
data = dataiter.next()
images = data['image']
labels = data['finding']

model_ft.to('cpu')

output = torch.tensor(model_ft(images).detach().numpy())

########################################################

from sklearn.metrics import classification_report
# show a nicely formatted classification report
print(classification_report(labels, np.argmax(output,1), target_names=classes))

###########################################################
from sklearn.metrics import confusion_matrix

# compute the confusion matrix and and use it to derive the raw
# accuracy, sensitivity, and specificity
cm = confusion_matrix(labels, np.argmax(output,1))
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
# show the confusion matrix, accuracy, sensitivity, and specificity
print(cm)
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))