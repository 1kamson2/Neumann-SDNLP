from ddpm import DenoiseModel, UNet
import torchvision
import torch
from torch.utils.data import DataLoader 
from torch.optim import Adam
import time

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DDPMTraining:
    def __init__(self, img_channels=1, img_sz=32, n_samples=16, n_steps=1000,
                 nepoch=100, batch_sz=32, lr=0.001):
        """
        Here comes the training. 
        """
        print(f"[INFO]: EPOCH INFORMATION: INITIALIZING...")
        self.img_channels = img_channels
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.img_sz = img_sz
        self.nepoch = nepoch
        self.batch_sz=batch_sz
        self.lr = lr
        print(f"[INFO] EPOCH: INITIALIZED")
        print(f"[INFO] NOISE MODEL: INITIALIZING...")
        self.eps_model = UNet(image_channels=img_channels).to(_device) 
        print(f"[INFO] NOISE MODEL: INITIALIZED")
        print(f"[INFO] DIFFUSION MODEL: INITIALIZING...")
        self.dif_model = DenoiseModel(self.eps_model, self.n_steps).to(_device) 
        print(f"[INFO] DIFFUSION MODEL: INITIALIZED")
        print(f"[INFO] DATASET: INITIALIZING...")
        self.dataset = self.get_dataset()
        print(f"[INFO] DATASET: INITIALIZED")
        print(f"[INFO] DATALOADER: INITIALIZING...")
        self.dl = DataLoader(self.dataset, self.batch_sz, shuffle=True,
                             pin_memory=True)
        print(f"[INFO] DATALOADER: INITIALIZED")
        print(f"[INFO] OPTIMIZER: INITIALIZING...")
        self.optimizer = Adam(self.eps_model.parameters(), lr=self.lr)
        print(f"[INFO] OPTIMIZER: INITIALIZED")


    def get_dataset(self, img_sz=448):
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

    def run_epoch(self, epoch): 
        for data in self.dl:
            _data = data[0].to(_device)
            self.optimizer.zero_grad()
            loss = self.dif_model.loss(_data, None) 
            loss.backward()
            self.optimizer.step()
            if epoch >= 10 and epoch % 20 == 0:
                print(f"Current loss is: {loss}")

    def training(self):
        _time = time.time()
        print("[INFO] TRAINING: BEGINS")
        for epoch in range(self.nepoch):
            self.run_epoch(epoch)
            self.sample()
        print(f"[INFO]: TRAINING: ENDED, TIME ELAPSED:"
              f"{time.time() - _time:0.3f}")


