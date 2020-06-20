import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from Cifar10 import Cifar10
import torch


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.far = Cifar10()
        self.far.read()
        self.classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')




    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        for epoch in range(2):
            running_loss = 0.0
            for i, data in enumerate(net.far.trainloader, 0):
                inputs, labels = data
                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()


                running_loss += loss.item()
                if i % 2000 == 1999:
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training')
        ff = "./cifar_net.pth"
        torch.save(net.state_dict(), ff)

    def test(self):
        dataiter = iter(self.far.testloader)
        images, labels = dataiter.next()

        # print images
        imshow(torchvision.utils.make_grid(images))
        print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

        #def predict(self, image):



if __name__ == '__main__':
    ff = "./cifar_net.pth"
    net = Net()
    net.train()

    net.load_state_dict(torch.load(ff))

    it  = iter(net.far.testloader)
    images, labels = it.next()
    outputs = net(images)
    print(outputs)
    _, predicted = torch.max(outputs, 1)
    print(predicted)
    print('Predicted: ', ' '.join('%5s' % net.classes[predicted[j]]
                              for j in range(4)))


    correct = 0
    total = 0
    with torch.no_grad():
        for data in net.far.testloader:        
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in net.far.testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            net.classes[i], 100 * class_correct[i] / class_total[i]))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)




