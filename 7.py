import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import time
import os
import numpy as np
from matplotlib import pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights
from torch.utils.tensorboard import SummaryWriter

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
DATA_DIRECTORY = 'hymenoptera_data'

# TASK f: Change this value to 0.0001, 1, or 100 to run your experiments
CURRENT_LEARNING_RATE = 0.001

INPUT_SIZE = (224, 224)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DATA_TRANSFORM = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(npimg.transpose((1, 2, 0)))


if __name__ == '__main__':

    # Setup train dataset
    image_dataset = datasets.ImageFolder(os.path.join(DATA_DIRECTORY, 'train'), DATA_TRANSFORM)
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=4, shuffle=True, num_workers=1)

    n_images = len(image_dataset)
    class_names = image_dataset.classes

    # TASK d & e: Initialize TensorBoard SummaryWriter with a dynamic name based on the learning rate
    log_dir = f'runs/resnet18_lr_{CURRENT_LEARNING_RATE}'
    writer = SummaryWriter(log_dir)

    # Get a batch of training images to log images and trace graph
    dataiter = iter(dataloader)
    images, labels = next(dataiter)

    # Log the image grid to TensorBoard
    img_grid = torchvision.utils.make_grid(images)
    matplotlib_imshow(img_grid, one_channel=False)
    writer.add_image('four_hymenoptera_images', img_grid)

    # Local visualization loop
    images_vis, class_indices = next(iter(dataloader))
    for image, class_index in zip(images_vis, class_indices):
        image = image.numpy().transpose((1, 2, 0))
        image = STD * image + MEAN
        image = np.clip(image, 0, 1)
        class_name = class_names[class_index]
        plt.title(class_name)
        plt.imshow(image)
        plt.show()

    # Load pretrained resnet model
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, 2)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print('Backend: {}'.format(device))
    model = model.to(device)

    # TASK d: Write the network layout graph to TensorBoard
    graph_inputs = images.to(device)
    writer.add_graph(model, graph_inputs)

    # Setup Loss and Optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=CURRENT_LEARNING_RATE, momentum=0.9)
    lr_scheduler_step = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    n_epochs = 30
    model.train()

    start = time.time()
    for epoch in range(n_epochs):
        print('# Epoch {}/{} #'.format(epoch, n_epochs - 1))

        sum_loss = 0.0
        n_correct = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

            sum_loss += loss.item() * inputs.size(0)
            n_correct += torch.sum(preds == labels.data)

        lr_scheduler_step.step()

        overall_loss = sum_loss / n_images
        accuracy = n_correct / n_images

        print('train loss = {:.4f}, accuracy = {:.4f}'.format(overall_loss, accuracy))

        # TASK e: Write metrics to TensorBoard after each epoch execution loop
        writer.add_scalar('Loss/train', overall_loss, epoch)
        writer.add_scalar('Accuracy/train', accuracy, epoch)

    time_elapsed = time.time() - start
    print('training finished in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))