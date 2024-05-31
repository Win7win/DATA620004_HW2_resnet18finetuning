from dataloader import load_my_data
from model import resnet18_revise_model
from test_model import test_my_model
from codes.train_model import train_my_model
import torch.nn as nn
import torch.optim as optim
import torch

def train_fintune_model(dataloaders, dataset_sizes, epoch=25, lr_fc=0.001, lr_pre=0.0001):
    criterion = nn.CrossEntropyLoss()
    device = "cuda"
    model = resnet18_revise_model(pretrained=True)
    model = model.to(device)
    fc_params_id = list(map(id, model.fc.parameters()))     # 返回的是parameters的内存地址
    base_params = filter(lambda p: id(p) not in fc_params_id, model.parameters())

    # 采用不同学习率设置
    optimizer = optim.SGD([
        {'params': base_params, 'lr': lr_pre},   
        {'params': model.fc.parameters(), 'lr': lr_fc}], momentum=0.9)

    model = train_my_model(model, criterion, optimizer, dataloaders, dataset_sizes, epoch)
    
    return model

if __name__ == "__main__":
    my_dataloaders, my_datasizes = load_my_data("./CUB_200_2011/CUB_200_2011/CUB_200_2011/images")
    
    # 参数搜索
    lr_fc = [0.001, 0.005, 0.0005]
    lr_pre = [0.0001, 0.0005, 0.00005]
    epochs = [50]

    for lr in lr_fc:
        for lre in lr_pre:
            for epoch in epochs:
                    model_finetuned, bs_epoch = train_fintune_model(my_dataloaders, my_datasizes, epoch=epoch, lr_fc=lr, lr_pre=lre)
                    torch.save(model_finetuned.state_dict(), f'./task1_output/resnet18_finetuned_{epoch}_{lr}_{lre}.pth')
                    test_accuracy = test_my_model(my_dataloaders, f'./task1_output/resnet18_finetuned_{epoch}_{lr}_{lre}.pth')
                    # 记录测试结果
                    with open("fintuned_model_test_result.txt", "a+", encoding="utf-8") as f:
                        f.write(f"参数：{lr}，{lre}，最佳epoch：{bs_epoch+1}。测试集上accuracy为：{test_accuracy}")
                        f.write("\n")



