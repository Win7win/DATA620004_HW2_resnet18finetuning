from model import resnet18_revise_model
import torch
from dataloader import load_my_data



def test_my_model(dataloaders, model_path):
    device = "cuda"
    model = resnet18_revise_model(pretrained=False)  # 初始化修改后的模型
    model.load_state_dict(torch.load(model_path))    # 加载训练好的模型权重
    model = model.to(device)
    model.eval()  # 设置模型为评估模式

    running_corrects = 0
    total_samples = 0

    # 遍历测试数据集
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 前向传播，计算预测结果
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
        
        # 更新统计数据
        running_corrects += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)

    # 计算整体准确率
    accuracy = running_corrects.double() / total_samples
    print(f'Test Accuracy: {accuracy:.4f}')

    return accuracy

if __name__ == "__main__":
    model_path = 'resnet18_finetuned.pth'
    my_dataloaders, _ = load_my_data("/remote-home/yqwu/cub_200_bird/CUB_200_2011_data/CUB_200_2011/images")
    print(test_my_model(my_dataloaders, model_path))
