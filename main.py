import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from models import SimpleNN, DropoutBatchNormNN
from utils import train_with_plot, evaluate_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

if __name__ == "__main__":
    model_simple = SimpleNN().to(device)
    optimizer = torch.optim.Adam(model_simple.parameters(), lr=0.001)
    train_loss_hist_simple = train_with_plot(model_simple, train_loader, criterion, optimizer)

    model_dropout_bn = DropoutBatchNormNN().to(device)
    optimizer = torch.optim.Adam(model_dropout_bn.parameters(), lr=0.001)
    train_loss_hist_dropout_bn = train_with_plot(model_dropout_bn, train_loader, criterion, optimizer)

    evaluate_model(model_simple, test_loader)
    evaluate_model(model_dropout_bn, test_loader)
