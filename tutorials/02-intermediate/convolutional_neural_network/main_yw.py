from tkinter.constants import SEL_FIRST
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
    print(f"PC has {torch.cuda.device_count()} GPUs!")

# Hyper parameters
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 5e-4

# data
# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='/data/',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='/data/',
                                          train=False, 
                                          transform=transforms.ToTensor())
# Torch data loader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model
# CNN
class myCNN(nn.Module):
        def __init__(self, num_classes = num_classes) -> None:
                super(myCNN, self).__init__()
                # Conv 1
                self.conv1 = nn.Sequential(
                        nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2), # (28 + 2*2 - 5)/1 + 1 = 28
                        nn.BatchNorm2d(16),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride = 2) # (28 - 2)/2 + 1 = 14
                )
                # Conv 2
                self.conv2 = nn.Sequential(
                        nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2), # (14 + 4 -5) / 1 + 1 = 14
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2) # (14 -2) / 2 + 1 =7
                )
                # FC 
                self.fc = nn.Linear(32 * 7 * 7, num_classes)
        
        def forward(self, input):
                output = self.conv1(input)
                output = self.conv2(output)
                output = output.reshape(output.size(0), -1)
                output = self.fc(output)
                return output

model = myCNN(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Train 
epoch_losses = []
num_batches = len(train_loader)
for epoch in range(num_epochs):
        epoch_loss = 0
        for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                # forward
                predicted = model(images)
                loss = criterion(predicted, labels)
                epoch_loss += loss.item()
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # LOG
                if (i+1) % 100 ==0:
                        print('Epoch [{}/{}], step [{}/{}], loss is: {:.4f}'.format(epoch+1, num_epochs, i+1, num_batches, loss.item()))
        epoch_losses.append(epoch_loss/num_batches)

# Plotting the epoch vs. loss graph
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o')
plt.title('Epoch vs. Loss')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.grid(True)
plt.savefig('epoch_loss_plot.png')
plt.show()

# Eval
model.eval() # for batch norm.
with torch.no_grad():
        total = 0
        correct = 0
        for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += len(labels.size(0))
                correct += (labels == predicted).sum().item()
        print(f'the correctness pct is {correct/total * 100}')

# save model
torch.save(model.state_dict(), 'CNNmodel.ckpt')