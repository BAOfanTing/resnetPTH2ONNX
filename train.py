import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from model import resnet34
from PIL import Image
from torch.utils.data import Dataset


# 定义数据集路径
train_txt_path = "./train.txt"
val_txt_path = "./val.txt"
image_path = "./"  # 图片存放路径
class_txt_path = "./classes.txt"

# 读取类别文件，获取类别列表
with open(class_txt_path, 'r', encoding='utf-8') as f:
    classes = [line.strip() for line in f.readlines()]

# 构建类别到索引的映射字典
class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}


#
class MyDataset(Dataset):
    def __init__(self, text_path, transform=None):
        super(MyDataset, self).__init__()
        self.root = text_path
        f = open(self.root, 'r', encoding='utf-8')
        data = f.readlines()

        imgs = []
        labels = []

        for line in data:
            img_path, class_name = line.strip().split(',')
            # 将图像路径添加到列表中
            imgs.append(os.path.join("./", img_path))
            # 将中文类别转换为数字类别并添加到列表中
            labels.append(class_to_idx[class_name])
        self.img = imgs
        self.label = labels
        self.transform = transform

    def __len__(self):
        """
        返回数据集的长度
        """
        return len(self.label)

    def __getitem__(self, item):
        """
        根据索引获取数据集中的样本
        """
        img = self.img[item]
        label = self.label[item]
        img = Image.open(img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    batch_size = 128
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))


    train_dataset = MyDataset(train_txt_path,transform=data_transform["train"])
    val_dataset = MyDataset(val_txt_path, transform=data_transform["val"])
    
    # 创建训练数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size, shuffle=True,
                                            num_workers=nw)

    # 创建验证数据加载器
    validate_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=batch_size, shuffle=False,
                                                num_workers=nw)
    

    net = resnet34()
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model_weight_path = "./resnet34-pre.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 14)
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    epochs = 100
    best_acc = 0.0
    save_path = './resNet34-bird.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_num = len(validate_loader.dataset)
        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracyccuracy: %.3f' %
            (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
