import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class s3fd(nn.Module):
    def __init__(self, num_classes, batch_size):
        
        super(s3fd, self).__init__()
        self.bs = batch_size
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.fc6     = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=3)
        self.fc7     = nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0)

        self.conv6_1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.conv6_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.conv7_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.conv7_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.fc1 = nn.Linear(2304,num_classes)

    def forward(self, x):
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h)); f3_3 = h
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h)); f4_3 = h
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h)); f5_3 = h
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h));     ffc7 = h
        h = F.relu(self.conv6_1(h))
        h = F.relu(self.conv6_2(h)); f6_2 = h
        h = F.relu(self.conv7_1(h))
        h = F.relu(self.conv7_2(h)); f7_2 = h
        m = f7_2.view(1,-1)
        op = self.fc1(m)
        return F.softmax(op) 
