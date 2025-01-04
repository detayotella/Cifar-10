# CIFAR-10 Image Classification with PyTorch

This project focuses on predicting image classes using a Convolutional Neural Network (CNN) built with PyTorch on the CIFAR-10 dataset. The CIFAR-10 dataset is directly loaded using PyTorch's `torchvision` library.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Training & Evaluation](#training-&-evaluation)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview
The CIFAR-10 dataset contains 60,000 color images, each with dimensions of 32x32 pixels. The images are evenly distributed across 10 different classes, with 6,000 images per class. This project's objective is to develop and train a Convolutional Neural Network (CNN) to accurately classify these images into their respective categories.

## Dataset
The CIFAR-10 dataset includes the following 10 classes:
1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

## Dependencies
- Python 3.6+
- PyTorch
- torchvision
- numpy
- matplotlib

## Installation
1. Clone this repository:
   ```sh
   git clone https://github.com/yourusername/cifar10-pytorch.git
   cd cifar10-pytorch
   ```
2. Install the required packages:
   ```sh
   pip install torch torchvision numpy matplotlib
   ```

## Model Architecture
The CNN architecture used in this project consists of:
- Two convolutional layers
- Max-pooling layers after each convolutional layer
- Two fully connected layers
- ReLU activations

Here is the model architecture:
```python
import torch.nn as nn
import torch.nn.functional as F

class TinyVGG_V2(nn.Module): 
    def __init__(self): 
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, 
                      padding=1), 
            nn.ReLU(), 
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, 
                      stride=1, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2)
        ) 
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, 
                      stride=1, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2)
        ) 
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, 
                      stride=1, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, 
                      padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2)
        ) 
        self.classifier = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(in_features=256*4*4, 
                      out_features=1024), 
            nn.ReLU(), 
            nn.Linear(in_features=1024, out_features=512), 
            nn.ReLU(), 
            nn.Linear(in_features=512, out_features=10)
        )

    def forward(self, x): 
        return self.classifier(self.conv_block_3(self.conv_block_2(self.conv_block_1(x))))     
```

## Training & Evaluation 
The training process involves:
1. Loading the CIFAR-10 dataset.
2. Preprocessing the data.
3. Defining the CNN model.
4. Defining the loss function and optimizer.
5. Training the model for a number of epochs.
6. Saving the trained model.

Here is an example of the training loop:
```python
import torch
import torchvision
import torchvision.transforms as transforms
torch.manual_seed(42)
torch.cuda.manual_seed(42)

NUM_EPOCHS = 5

model_0 = TinyVGG(input_shape=3,
                  hidden_units=10,
                  output_shape=len(train_data.classes)).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(),
                             lr=1e-3)

from timeit import default_timer as timer
start_timer = timer()

model_0_results = train(model=model_0,
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=NUM_EPOCHS)
end_timer = timer()
print(f"Total training time: {end_timer - start_timer:.3f} seconds")

## Results
After training the model for a few epochs, you should see an improvement in accuracy. The expected accuracy will depend on the number of epochs, learning rate, and other hyperparameters.

## Usage
To use the trained model for predictions:
1. Load the trained model.
2. Preprocess the input image.
3. Pass the image through the model to get the prediction.

Example:
```python
img_batch, label_batch = next(iter(train_dataloader))

img_single, label_single = img_batch[0].unsqueeze(dim=0), label_batch[0]
print(f"Single image shape: {img_single.shape}")

model_0.eval()
with torch.inference_mode():
    pred = model_1(img_single.to(device))


print(f"Output logits: \n{pred}\n")
print(f"Output prediction probabilites:\n{torch.softmax(pred, dim=1)}\n")
print(f"Output prediction label: \n{torch.argmax(torch.softmax(pred, dim=1), dim=1)}\n")
print(f"Actual label:\n{label_single}")
```

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

This readme provides a comprehensive guide to setting up and running a CIFAR-10 image classification project using PyTorch. It covers everything from installation to model architecture, training, evaluation, and usage.
