import torch.nn as nn
import torch

class ConvNet(nn.Module):
    def __init__(self, num_class=10):
        super(ConvNet, self).__init__()
        # ----------TODO------------
        # define a network 
        # ----------TODO------------
        class Residual(nn.Module):
            def __init__(self, input_channel, num_channel):

                super().__init__()
                self.conv1 = nn.Conv2d(input_channel, num_channel, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(num_channel, num_channel, kernel_size=3, padding=1)
                self.conv3 = nn.Conv2d(input_channel, num_channel, kernel_size=1)
                self.bn1 = nn.BatchNorm2d(num_channel)
                self.bn2 = nn.BatchNorm2d(num_channel) 
            def forward(self, x):
                y = torch.nn.functional.relu(self.bn1(self.conv1(x)))
                y = self.bn2(self.conv2(y))
                x_ = self.conv3(x)
                y += x_
                return torch.nn.functional.relu(y)

        block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        def resnet_block(input_channels, num_channels, num_residuals):
            blk = []
            for i in range(num_residuals):
                if i == 0:
                    blk.append(Residual(input_channels, num_channels))
                else:
                    blk.append(Residual(num_channels, num_channels))
            return blk
        block2 = nn.Sequential(*resnet_block(32, 64, 2))
        block3 = nn.Sequential(*resnet_block(64, 128, 2))
        block4 = nn.Sequential(*resnet_block(128, 256, 2))
        block5 = nn.Sequential(*resnet_block(256, 512, 2))
        self.model = nn.Sequential(block1, block2, block3, block4 ,block5,
                                   nn.AdaptiveAvgPool2d((1, 1)),
                                   nn.Flatten(),
                                   nn.Linear(512, 10))


    def forward(self, x):

        # ----------TODO------------
        # network forwarding 
        # ----------TODO------------

        return self.model(x)


if __name__ == '__main__':
    import torch
    from torch.utils.tensorboard  import SummaryWriter
    from dataset import CIFAR10
    writer = SummaryWriter(log_dir='../experiments/network_structure')
    net = ConvNet()
    train_dataset = CIFAR10()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=False, num_workers=2)
    # Write a CNN graph. 
    # Please save a figure/screenshot to '../results' for submission.
    for imgs, labels in train_loader:
        writer.add_graph(net, imgs)
        writer.close()
        break 
