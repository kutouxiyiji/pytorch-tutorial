import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 1
output_size = 1
num_epochs = 150
learning_rate = 0.001

# Toy dataset
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 
                    [9.779], [6.182], [7.59], [2.167], [7.042], 
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 
                    [3.366], [2.596], [2.53], [1.221], [2.827], 
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

# model
model = nn.Linear(input_size, output_size).to(device)

# loss and optimizer
criterion = nn.MSELoss()
optimzer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# train
for epoch in range(num_epochs):
        # data
        x = torch.from_numpy(x_train).to(device)
        y = torch.from_numpy(y_train).to(device)
        # forward
        predicts = model(x)
        loss = criterion(predicts, y)
        # backward
        optimzer.zero_grad()
        loss.backward()
        optimzer.step()
        if epoch % 5 == 0:
                print('epoch {}/{}, loss: {}'.format(epoch, num_epochs, loss.item()))

# Eval
predicts = model(torch.from_numpy(x_train).to(device)).detach().cpu().numpy()

plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicts, label='Fitted line')
plt.legend()
plt.show()

# Save the model checkpoint
torch.save(model.state_dict(), 'linear_model.ckpt')