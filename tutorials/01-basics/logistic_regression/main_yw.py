import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Hyper parameters
input_size = 28 * 28
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 1e-3

# MNIST dataset (images and labels)
train_dataset = torchvision.datasets.MNIST(root='../../data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader (input pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# Model
model = nn.Linear(input_size, num_classes)
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# Train
total_step = len(train_loader)
for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
                # Convert image data
                images = images.reshape(-1, input_size)
                # forward
                predicts = model(images)
                loss = criterion(predicts, labels)
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (i+1) % 100 == 0:
                        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                                .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Eval
with torch.no_grad():
        total = 0
        correct = 0
        for images, labels in test_loader:
                images = images.reshape(-1, input_size)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += len(labels)
                correct += (labels == predicted).sum().item()
        print(f'the correct/total pct is {correct/total * 100}')

# Store
torch.save(model.state_dict, 'logistic_regression')