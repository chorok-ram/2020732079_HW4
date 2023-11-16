# 2020732079_HW4
test basic  commit하고 remote 수정  더 수정
---
* 팀원의 repository를 fork한후 이를 local로 옮겨  
수정한 내용입니다.
수정
import torch

class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # L1 입력 형태=(?, 3, 64, 128)
        #    합성곱     -> (?, 32, 64, 128)
        #    풀링     -> (?, 32, 32, 64)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # L2 입력 형태=(?, 32, 32, 64)
        #    합성곱      ->(?, 64, 32, 64)
        #    풀링      ->(?, 64, 16, 32)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # 전역 평균 풀링
        self.global_avg_pooling = torch.nn.AdaptiveAvgPool2d(1)
        # L3 선형 레이어: 64x1x1 입력 -> 10 출력
        self.fc = torch.nn.Linear(64, 10, bias=True)
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.global_avg_pooling(out)
        out = out.view(out.size(0), -1)   # 완전 연결 계층을 위해 펼침
        out = self.fc(out)
        return out



# Construct model
model = CNN()

for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader:
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print(f'Epoch: {epoch+1}, Cost: {avg_cost}')

print('Training finished')

# test
trans = transforms.Compose([transforms.Resize((64,128)),transforms.ToTensor()])
test_data = dsets.ImageFolder(root='/content/drive/MyDrive/test_data', transform=trans)
test_set = DataLoader(dataset=test_data, batch_size=len(test_data))
with torch.no_grad():
    for X, Y in test_set:
        X = X.to(device)
        Y = Y.to(device)

        prediction = model(X)
        correct_prediction = torch.argmax(prediction, 1) == Y
        accuracy = correct_prediction.float().mean()
        print(f'Accuracy: {accuracy.item()}')


