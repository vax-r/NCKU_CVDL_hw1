import torch
import torchvision
import torchvision.transforms as v2
import torchvision.models as models
import torchsummary
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import os
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F


def acc_and_loss(model, dataloader, device):
    correct_pred, num_examples = 0, 0
    cross_entropy = 0.0

    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        cross_entropy += F.cross_entropy(outputs, labels).item()
        _, pred_labels = torch.max(outputs.data, 1)
        num_examples += labels.size(0)
        correct_pred += (pred_labels == labels).sum()
    return float(correct_pred / num_examples) * 100, cross_entropy / num_examples


# global variables
PATH = './cifar_net.pth'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# load and normalize dataset
train_transform = v2.Compose(
    [v2.ToTensor(),
     v2.RandomHorizontalFlip(),
     v2.RandomVerticalFlip(),
     v2.RandomRotation(30),     
     v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_transform = v2.Compose(
    [
        v2.ToTensor(),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

batch_size = 64

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                        shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                    download=False, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                        shuffle=False, num_workers=2)



train_acc_list = []
valid_acc_list = []
train_loss_list = []
valid_loss_list = []


def Train():

    VD_cnn = models.vgg19_bn(weights="VGG19_BN_Weights.DEFAULT", num_classes=1000)

    # change the classifier layer of the model
    VD_cnn.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 10) # change output features to 10
        )
    
    VD_cnn.to(device)

    # define criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(VD_cnn.parameters(), lr=0.003)
    optimizer = optim.SGD(VD_cnn.parameters(), lr=0.001, momentum=0.9)

    NUM_EPOCHS = 40

    for epoch in range(NUM_EPOCHS):
        
        VD_cnn.train()

        for ind, (inputs, labels) in tqdm(enumerate(trainloader), total=len(trainloader)):
            inputs, labels = inputs.to(device), labels.to(device)

            logits = VD_cnn(inputs)
            cost = F.cross_entropy(logits, labels)

            optimizer.zero_grad()

            cost.backward()

            optimizer.step()

        VD_cnn.eval()
        with torch.no_grad():
            train_acc, train_loss = acc_and_loss(VD_cnn, trainloader, device=device)
            valid_acc, valid_loss = acc_and_loss(VD_cnn, testloader, device=device)

            # save the model with best validation accuracy
            if len(valid_acc_list) > 0 and valid_acc > max(valid_acc_list):
                torch.save(VD_cnn.state_dict(), PATH)

            train_acc_list.append(train_acc)
            train_loss_list.append(train_loss)
            valid_acc_list.append(valid_acc)
            valid_loss_list.append(valid_loss)

            print(f"Epoch: {epoch + 1} / {NUM_EPOCHS} Train Acc : {train_acc: .2f}% | Validation Acc : {valid_acc: .2f}%")
            print(f"Epoch: {epoch + 1} / {NUM_EPOCHS} Train Loss : {train_loss: .2f}% | Validation Loss : {valid_loss: .2f}%")

    print("Finish training")

    # show and save loss and acc figure
    plt.title("Loss")
    plt.plot([*range(NUM_EPOCHS)], train_loss_list, label = "train_loss")
    plt.plot([*range(NUM_EPOCHS)], valid_loss_list, label = "val_loss")
    plt.legend()
    # plt.show()
    plt.savefig("Loss.png")

    plt.title("Accuracy")
    plt.plot([*range(NUM_EPOCHS)], train_acc_list, label = "train_acc")
    plt.plot([*range(NUM_EPOCHS)], valid_acc_list, label = "val_acc")
    plt.legend()
    # plt.show()
    plt.savefig("Accuracy.png")

    



# Print model structure
def Print_model_struct():

    model = models.vgg19_bn(num_classes=10)
    model = model.to(device)

    print(torchsummary.summary(model, input_size=(3 ,32, 32)))

def Test():
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

    model = models.vgg19_bn(num_classes=10)
    model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))

    outputs = model(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                  for j in range(4)))
    
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in tqdm(testloader):
            images, labels = data
            print(images, labels)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            print(images, labels)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')




if __name__ == "__main__":

    # Train()
    Test()
    # Print_model_struct()