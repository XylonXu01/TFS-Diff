import torch

class SE_block(torch.nn.Module):
    def __init__(self,in_channel,ratio):
        super(SE_block, self).__init__()
        self.avepool = torch.nn.AdaptiveAvgPool2d(1)
        self.linear1 = torch.nn.Linear(in_channel,in_channel//ratio)
        self.linear2 = torch.nn.Linear(in_channel//ratio,in_channel)
        self.sigmoid = torch.nn.Sigmoid()
        self.Relu = torch.nn.ReLU()

    def forward(self,input):
        b,c,w,h = input.shape
        x = self.avepool(input)
        x = x.view([b,c])
        x = self.linear1(x)
        x = self.Relu(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        x = x.view([b,c,1,1])

        return input*x

if __name__ == "__main__":
    input = torch.randn((8, 12, 256, 256))
    model = SE_block(in_channel=12,ratio=8)
    output = model(input)
    print(output.shape)

