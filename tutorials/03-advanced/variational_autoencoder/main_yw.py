import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import optimizer
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Create a directory if not exists
sample_dir = 'samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# Hyper paramteres
input_size = 28*28
h_dim = 400
z_dim = 20
num_epoches = 20
batch_size = 100
learning_rate = 1e-3

# Data
# MNIST dataset
dataset = torchvision.datasets.MNIST(root='../../data',
                                     train=True,
                                     transform=transforms.ToTensor(),
                                     download=True)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=batch_size, 
                                          shuffle=True)

# VAE model. Variational Autoencoder.
class VAE(nn.Module):
        def __init__(self, image_size=784, h_dim=400, z_dim=20) -> None:
              super().__init__()
              self.fc1 = nn.Linear(image_size, h_dim)
              self.fc2 = nn.Linear(h_dim, z_dim)
              self.fc3 = nn.Linear(h_dim, z_dim)
              self.fc4 = nn.Linear(z_dim, h_dim)
              self.fc5 = nn.Linear(h_dim, image_size)
        
        def encoder(self, x):
                h = F.relu(self.fc1(x))
                return self.fc2(h), self.fc3(h)
        
        def reparameterize(self, mu, log_var):
                std = torch.exp(log_var/2)
                eps = torch.randn_like(log_var)
                return mu + eps * std

        def decoder(self, z):
                h = self.fc4(z)
                return F.sigmoid(self.fc5(h))

        def forward(self, x):
                mu, log_var = self.encoder(x)
                z = self.reparameterize(mu, log_var)
                output = self.decoder(z)
                return output, mu, log_var

# Model and Optimizer
model = VAE().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

# Train (including two losses) and eval per batch
for epoch in range(num_epoches):
        for i, (images, _) in enumerate(data_loader):
                images = images.to(device).view(-1, input_size)
                x_reconstruct, mu, log_var = model(images)

                reconstruct_loss = F.binary_cross_entropy(x_reconstruct, images, size_average=False)
                # KL(N(μ,σ^2)∣∣N(0,1))
                kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = reconstruct_loss + kl_div

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % 100 == 0:
                        print ("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}, total Loss:{:.4f}" 
                                .format(epoch+1, num_epoches, i+1, len(data_loader), reconstruct_loss.item(), kl_div.item(), loss.item()))

        with torch.no_grad():
                # Save the sampled images
                z = torch.randn(batch_size, z_dim).to(device)
                out = model.decoder(z).view(-1, 1, 28, 28)
                save_image(out, os.path.join(sample_dir, 'sampled-{}.png'.format(epoch+1)))

                # Save the reconstructed images
                out, _, _ = model(images)
                x_concat = torch.cat([images.view(-1, 1, 28, 28), out.view(-1, 1, 28, 28)], dim=3)
                save_image(x_concat, os.path.join(sample_dir, 'reconst-{}.png'.format(epoch+1)))