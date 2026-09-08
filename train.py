import time
import torch
import os
from tempfile import TemporaryDirectory
from data import dataloaders, dataset_sizes
from config import device

def train_model(model, criterion, optimizer, scheduler, num_epochs=5):
    elapsed = time.time()

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        # Loop through each epoch
        for epoch in range(num_epochs):
            print(f"Epoch {epoch}/{num_epochs - 1}")
            print("-" * 10)

            for phase in ['train', 'test']:
                # set to train/eval mode
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0.0

                for inputs, labels in dataloaders[phase]:
                    # Send inputs/labels to device
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Zero parameter gradients
                    optimizer.zero_grad()

                    # Forward
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, pred = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # Backwards if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(pred == labels.data)

                if phase == 'train':
                    scheduler.step()

                # Epoch stats
                epoch_loss = running_loss/dataset_sizes[phase]
                epoch_acc = running_corrects.double()/dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # Copy model if there's a new best acc
                if phase == 'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

        total_time = time.time() - elapsed
        print(f'Training complete in {total_time // 60:.0f}m {total_time % 60:.0f}s')
        print(f'Best test acc: {best_acc:4f}')

        # Load best model weights
        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))

    return model