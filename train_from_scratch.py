from codes.dataloader import load_my_data
from codes.model import resnet18_revise_model
from codes.test_model import test_my_model
from codes.train_model import train_my_model
import torch.nn as nn
import torch.optim as optim
import torch


def train_scratch_model(dataloaders, dataset_sizes, epoch=25, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    device = "cuda"
    model = resnet18_revise_model(pretrained=False)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr, momentum=0.9)
    model = train_my_model(model, criterion, optimizer, dataloaders, dataset_sizes, epoch)
    
    return model

if __name__ == "__main__":
    my_dataloaders, my_datasizes = load_my_data("./CUB_200_2011/CUB_200_2011/CUB_200_2011/images")
    model_base = train_scratch_model(my_dataloaders, my_datasizes, epoch=50, lr=0.001)
    torch.save(model_base.state_dict(), './resnet18_fromscratch.pth')
    test_accuracy = test_my_model(my_dataloaders, "./resnet18_fromscratch.pth")
    print(test_accuracy)




