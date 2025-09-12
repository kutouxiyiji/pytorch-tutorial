# Inplement the same model as main.py, for practice.
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'current device: {device}')

# Hyper-parameters
input_size = 28 * 28 # 784
hidden_size = 500
hidden_size2 = 100
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 1e-3

# For plotting
epoch_losses = []

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='/data', train=True, transform = transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='/data', train=False, transform = transforms.ToTensor(), download=False)

# data loader
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

# Model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

model = NeuralNet(input_size, hidden_size, hidden_size2, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

# Train the model
total_num_batch = len(train_loader)
for epoch in range(num_epochs):
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader):
                # Move the tensors to config device
                images = images.reshape(-1, input_size).to(device)
                labels = labels.to(device)

                # Forward
                outputs = model.forward(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                # print samples
                if i == 0:
                        print(f'images shape is {images.shape}')
                        print(f'labels shape is {labels.shape}')
                        print(f'outputs shape is {outputs.shape}')
                        print(f'and loss is {loss}')

                # Backward
                optimizer.zero_grad() # we need to clear the grads for each batch, each batch will update the weights independently
                loss.backward()
                optimizer.step()

                if (i+1) % 100 == 0:
                        print('Epoch [{}/{}], batch step [{}/{}], loss{:.4f}'.format(epoch, num_epochs, i + 1, total_num_batch, loss.item()))
        # Store average loss for the epoch
        epoch_losses.append(total_loss / total_num_batch)

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
                images = images.reshape(-1, input_size).to(device)
                labels = labels.to(device)
                outputs = model.forward(images)
                _, predicts = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicts == labels).sum().item()
        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'mnist_model.ckpt')

# Plotting the epoch vs. loss graph
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o')
plt.title('Epoch vs. Loss')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.grid(True)
plt.savefig('epoch_loss_plot.png')
plt.show()