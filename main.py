import torch
from torch import from_numpy, nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from helper_functions import plot_predictions, plot_decision_boundary

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

def tanh(x):
    return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

samples = 100
RANDOM_STATE = 42

X, y = make_moons(n_samples=samples, noise=0.03, random_state=RANDOM_STATE)

X = from_numpy(X).type(torch.float)
y = from_numpy(y).type(torch.float)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

plt.scatter(x=X[:, 0], y=X[:, 1], c=y, cmap=plt.cm.RdYlBu) #pyright: ignore

class MoonClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(in_features=2, out_features=10)
        self.layer2 = nn.Linear(in_features=10, out_features=10)
        self.layer3 = nn.Linear(in_features=10, out_features=10)
        self.layer4 = nn.Linear(in_features=10, out_features=1)
        self.relu = tanh
    def forward(self, x):
        return self.layer4(self.relu(self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))))

model = MoonClassifier()


loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

epochs = 2500

for epoch in range(epochs):

    model.train()
    # 1. Do the forward pass
    train_logits = model(X_train).squeeze(1)
    train_pred_probs = torch.sigmoid(train_logits)
    train_preds = torch.round(train_pred_probs)

    loss = loss_fn(train_logits, y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=train_preds)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_logits = model(X_test).squeeze(1)
        test_pred_probs = torch.sigmoid(test_logits)
        test_preds = torch.round(test_pred_probs)

        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_preds)
        if epoch % 25 == 0:
            print(f"epoch: {epoch} | train loss: {loss}, train accuracy: {acc} | test loss: {test_loss}, test accuracy: {test_acc}")

model.eval()

with torch.inference_mode():
    logits = model(X_test).squeeze(1)
    pred_probs = torch.sigmoid(logits)
    preds = torch.round(pred_probs)
    loss = loss_fn(logits, y_test)
    acc = accuracy_fn(y_true=y_test, y_pred=preds)
    print(preds[:50])
    print(y_test[:50])
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model, X_test, y_test)
plt.show()
