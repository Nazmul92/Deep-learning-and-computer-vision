## Performance evaluation of multi-layer perceptrons (MLP) and convolutional neural networks (CNN) for image classification.
 In this project, I focused on the performance evaluation of MLP and CNN, saved the best model based on the validation loss and training loss, and finally tested each model with the testing set.
## Dataset
A well-known image classification dataset: [MNIST](https://www.kaggle.com/competitions/digit-recognizer/data) is used in this project.
## Load dataset
MNIST is a default dataset in the torchvision datasets class of PyTorch.
```bash
from torchvision import datasets

transform = transforms.ToTensor()
train_data = datasets.MNIST(root='data',train=True,download=True,
                            transform=transform)

test_data = datasets.MNIST(root='data', train=False, download=True,
                           transform=transform)
```
## Dataset split into training set and validation set
```bash
from torch.utils.data.sampler import SubsetRandomSampler

num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)

split = int(np.floor(valid_size*num_train))                            
train_idx, valid_idx = indices[split:], indices[:split]

trainSampler = SubsetRandomSampler(train_idx)
validSampler = SubsetRandomSampler(valid_idx)
```
## Dataloader to access samples
```bash
train_loader = th.utils.data.DataLoader(train_data,batch_size=batch_size,             
                                        sampler=trainSampler,num_workers=0)

valid_loader = th.utils.data.DataLoader(train_data,batch_size=batch_size,             
                                       sampler=validSampler, num_workers=0)

test_loader = th.utils.data.DataLoader(test_data,batch_size=batch_size, num_workers=0) 
```
