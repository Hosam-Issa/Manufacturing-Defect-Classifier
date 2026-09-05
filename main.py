from data import dataloaders, dataset_sizes, class_names
from model import build_model
from train import train_model
from visualize import imshow, visualize_model
import torchvision
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
plt.ion()

def main():
    inputs, classes = next(iter(dataloaders['train']))
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])
    plt.show(block=True)

    # model = build_model()
    # trained_model = train_model(model, )
    # visualize_model(trained_model)

if __name__ == "__main__":
    main()