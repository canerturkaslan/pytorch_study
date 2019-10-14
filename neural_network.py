import torch
import torch.optim as optim
from torchvision import transforms, datasets

train = datasets.MNIST('', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))

test = datasets.MNIST('', train=False, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor()
                      ]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)

import torch.nn as nn
import torch.nn.functional as F


# nn.Module'den miras alınır (inherits) init fonksiyonunun nn.Module' class'indan calistirilmasi icin 'super' kullanilir.
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


net = Net()
print(net)
X = torch.rand((28, 28))
print(X)
X = X.view(-1, 28 * 28)
output = net(X)
print(output)

optimizer = optim.Adam(net.parameters(), lr=0.001)
EPOCHS = 3

for epoch in range(EPOCHS):
    for data in trainset:
        X, y = data
        net.zero_grad()
        output = net(X.view(-1, 28 * 28))
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()
    print(loss)
correct = 0
total = 0

# model.train()" and "model.eval()" activates and deactivates Dropout and BatchNorm, so it is quite important.
# "with torch.no_grad()" only deactivates gradient calculations, but doesn't turn off Dropout and BatchNorm.
# Your model accuracy will therefore be lower if you don't use model.eval() when evaluating the model.


with torch.no_grad():
    for data in testset:
        X, y = data
        output = net(X.view(-1, 784))
        #        print(output)
        for idx, i in enumerate(output):
            #           print(torch.argmax(i),y[idx])
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1
print("Accuracy: ", round(correct / total, 3))

import matplotlib.pyplot as plt
plt.imshow(X[0].view(28, 28))
plt.show()
print(torch.argmax(net(X[0].view(-1,784))[0]))
