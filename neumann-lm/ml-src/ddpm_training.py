from ddpm import DenoiseModel, UNet
import torchvision
import torch
from torch.utils.data import DataLoader 
from torch.optim import Adam

class DDPMTraining:
    def __init__(self, img_channels=1, img_sz=32, n_samples=16, n_steps=1000,
                 nepoch=100, batch_sz=32, lr=0.001):
        """
        Here comes the training. 
        """
        self.img_channels = img_channels
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.img_sz = img_sz
        self.nepoch = nepoch
        self.eps_model = UNet(image_channels=1)
        self.dif_model = DenoiseModel(self.eps_model, self.n_steps)
        self.dataset = self.get_dataset()
        self.batch_sz=batch_sz
        self.lr = lr
        self.dl = DataLoader(self.dataset, self.batch_sz, shuffle=True,
                             pin_memory=True)
        self.optimizer = Adam(self.eps_model.parameters(), lr=self.lr)


    def get_dataset(self, img_sz=224):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_sz),
            torchvision.transforms.ToTensor(),
        ])
        return torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)

    def sample(self):
        with torch.no_grad():
            x = torch.randn([self.n_samples, self.img_channels, self.img_sz,
                             self.img_sz])
            for _t in range(self.n_steps):
                t = self.n_steps - _t - 1
                x = self.dif_model.p_sample(x, x.new_full((self.n_samples,), t, dtype=torch.long))

    def training(self):
        for epoch in range(self.nepoch):
            for data in self.dl:
                _data = data[0].to("cpu")
                self.optimizer.zero_grad()
                loss = self.dif_model.loss(_data, None)
                loss.backward()
                self.optimizer.step()
                if epoch >= 10 and epoch % 20 == 0:
                    print(f"Current loss is: {loss}")
            self.sample()


