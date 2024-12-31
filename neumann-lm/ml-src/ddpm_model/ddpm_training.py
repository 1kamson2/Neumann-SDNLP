from ddpm_model.ddpm import DenoiseModel, UNet
import torchvision
from torchvision.utils import save_image
import torch
from torch.utils.data import DataLoader 
from torch.optim import Adam
import time

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DDPMApp:
    def __init__(self, img_channels=3, img_sz=32, n_channels=64,
                 channels_mul=(1,2,2,4), is_attn = (False, False, False, True),
                 n_samples=16, n_steps=1000, nepoch=3, batch_sz=64, lr=2e-5):
        """
        Here comes the training. 
        """
        self.img_channels = img_channels
        self.img_sz = img_sz
        self.n_channels = n_channels
        self.channels_mul = channels_mul
        self.is_attn = is_attn
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.nepoch = nepoch
        self.batch_sz=batch_sz
        self.lr = lr
        self.eps_model = UNet(image_channels=img_channels,
                              n_channels=self.n_channels,
                              ch_mults=self.channels_mul, is_attn=self.is_attn).to(_device) 
        self.dif_model = DenoiseModel(noise_model=self.eps_model,
                                      steps=self.n_steps, batch_sz=self.batch_sz).to(_device) 
        self.dataset = self.get_dataset()
        self.dl = DataLoader(self.dataset, self.batch_sz, shuffle=True,
                             pin_memory=True)
        self.optimizer = Adam(self.eps_model.parameters(), lr=self.lr)


    def get_dataset(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.img_sz),
            torchvision.transforms.Grayscale(3), 
            torchvision.transforms.ToTensor(),
        ])
        return torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)

    def sample(self):
        with torch.no_grad():
            x = torch.randn([self.n_samples, self.img_channels, self.img_sz,
                             self.img_sz], device=_device)
            for _t in range(self.n_steps):
                t = self.n_steps - _t - 1
                x = self.dif_model.p_sample(x, x.new_full((self.n_samples,), t,
                                                          device=_device, dtype=torch.long))

    def run_epoch(self, epoch): 
        loss = 0
        print(f"[INFO] EPOCH: {epoch}")
        for data in self.dl:
            _data = data[0].to(_device)
            self.optimizer.zero_grad()
            loss = self.dif_model.loss(_data) 
            loss.backward()
            self.optimizer.step()
        try:
            print(f"Current loss is: {loss}")
        except IOError as e:
            print(e)


    def training(self):
        print("[INFO] TRAINING: BEGINS")
        _time = time.time()
        for epoch in range(self.nepoch):
            self.run_epoch(epoch)
            self.sample()
        print(f"[INFO]: TRAINING: ENDED, TIME ELAPSED:"
              f"{time.time() - _time:0.3f}")

        torch.save(self.eps_model.state_dict(), "./weights/UNET_WEIGHTS.pth")
        torch.save(self.dif_model.state_dict(), "./weights/DDPM_WEIGHTS.pth")

    def evaluate(self):
        xt = torch.randn([self.n_samples, self.img_channels, self.img_sz,
                          self.img_sz], device=_device)
        # --- RELOAD MODEL WITH NEW WEIGHTS --- #
        self.eps_model.load_state_dict(torch.load('./weights/UNET_WEIGHTS.pth',
        weights_only=True, map_location=_device))
        self.dif_model.load_state_dict(torch.load('./weights/DDPM_WEIGHTS.pth',
        weights_only=True, map_location=_device))
        self.dif_model.eval()
        with torch.no_grad():
            for _t in range(self.n_steps):
                t = self.n_steps - _t - 1
                xt = self.dif_model.p_sample(xt, xt.new_full((self.n_samples,), t,
                                                              device=_device, dtype=torch.long))

        for i in range(self.n_samples):
           save_image(xt[i], f"./images/img{i}.png") 
