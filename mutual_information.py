from itertools import count
import torch
import torch.nn as nn 
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
import math


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 784 # 28x28
hidden_size = 100 
num_classes = 10 # 0-9
num_epochs = 2  # eteration
batch_size = 100
learning_rate = 0.001 

# Import MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='./mnist', 
                                            train=True, 
                                       transform=transforms.ToTensor(),  
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root='./mnist', 
                                           train=False, 
                                           transform=transforms.ToTensor())
                
# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=False) 

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def count_mtx(labels, outputs):
    _, predicted = torch.max(outputs.data, 1)
    count_mtrx= torch.zeros((10,10))
    for i in range(len(predicted)):
        count_mtrx[labels[i]][predicted[i]] +=1
        # count +=1 
    return count_mtrx

def mI(matrix, data_size):
      ml = 0
      matrix_joint = torch.div(matrix, data_size)
      p_label = torch.sum(matrix_joint,0)
      p_pred = torch.sum(matrix_joint,1)
      for x in range(len(p_pred)):
        for y in range(len(p_label)):
          if matrix_joint[y][x] == 0:
            ml +=0
          else:
            ml += matrix_joint[y][x] * (math.log(matrix_joint[y][x] /(p_label[y] * p_pred[x])))
      return ml

x_axis = []
mutual_information = []
idx = 0
n_total_steps = len(train_loader)
for epoch in range(num_epochs):

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28).to(device) # in different vidio tutorial, data = data.reshape(data.shape[0]. -1)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        count_matrix = count_mtx(labels, outputs)

        # Backward and optimize
        optimizer.zero_grad()  # doesn't store the back propagation from the previous forward propagation
        loss.backward()
        optimizer.step() # gradient descent or adam setp

        datasize = 100
        if (batch_idx+1) % datasize == 0:
            idx +=1
            # print(f"mutual infromation : {mI(count_matrix, datasize)}")
            # print (f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            x_axis.append(idx)
            mutual_information.append(mI(count_matrix,datasize))

plt.scatter(x_axis, mutual_information)
# plt.xlabel("epoch")
plt.xlabel("batch")
plt.ylabel("mutual information")