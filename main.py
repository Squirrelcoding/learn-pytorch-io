# Code for creating a spiral dataset from CS231n
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import from_numpy, nn
from sklearn.model_selection import train_test_split
from helper_functions import plot_decision_boundary

RANDOM_STATE = 42

N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.LongTensor)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

class SpiralClassificationModel(nn.Module):
    def __init__(self, dimensionality, num_classes) -> None:
        super().__init__()
        self.layer1 = nn.Linear(in_features=dimensionality, out_features=10)
        self.layer2 = nn.Linear(in_features=10, out_features=10)
        self.layer3 = nn.Linear(in_features=10, out_features=num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

model = SpiralClassificationModel(D, K)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epochs = 4000

for epoch in range(epochs):

    # Put the model in training mode
    model.train()

    train_logits = model(X_train)
    train_results = torch.softmax(train_logits, dim=1).argmax(dim=1)

    loss = loss_fn(train_logits, y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=train_results)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():

        test_logits = model(X_test)
        test_probs = torch.softmax(test_logits, dim=1).argmax(dim=1)

        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_probs)

        if epoch % 10 == 0:
            print(f"epoch: {epoch} | train loss: {loss}, train accuracy: {acc} | test loss: {test_loss}, test accuracy: {test_acc}")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model, X_test, y_test)
plt.show()
