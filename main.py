from data import dataloaders, class_names
from model import build_model
from train import train_model
from config import device
from visualize import imshow, visualize_model
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import torch
import torchvision

cudnn.benchmark = True
plt.ion()

def main():
    # inputs, classes = next(iter(dataloaders['train']))
    # out = torchvision.utils.make_grid(inputs)
    # imshow(out, title=[class_names[x] for x in classes])
    # plt.show(block=True)

    model_ft = build_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_schedule = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)

    trained_model = train_model(model_ft, criterion, optimizer_ft, exp_lr_schedule, num_epochs=5)

    torch.save(trained_model.state_dict(), 'model_weights.pth')

    visualize_model(trained_model, num_images=6)
    plt.show(block=True)

if __name__ == "__main__":
    main()