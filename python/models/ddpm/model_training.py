from models.ddpm.model import DenoiseModel, UNet
import torchvision
from torchvision.utils import save_image
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import time
import logging
from dataset.dataset import FileManager

logger = logging.getLogger(__name__)
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPMApp:
  def __init__(
    self,
    channels_mul=(1, 2, 2, 4),
    is_attn=(False, False, False, True),
    **config,
  ):
    """
    This is the module where you should consider training your model. User's
    config is given by arguments provided while running Python.
    """
    print(
      "Denoise Model started initialization. For more info about model"
      "parameters check help."
    )

    logger.info(
      "Denoise Model started initialization. For more info about model"
      "parameters check help."
    )
    self.img_channels = config["imgchannels"]
    self.img_sz = config["imgsize"]
    self.n_channels = config["nchannels"]
    self.channels_mul = channels_mul
    self.is_attn = is_attn
    self.n_samples = config["nsamples"]
    self.n_steps = config["nsteps"]
    self.nepoch = config["nepoch"]
    assert config.get("nepoch", None) is not None, "Not implemented"
    self.batch_sz = config["batchsize"]
    self.lr = config["lr"]
    self.eps_model = UNet(
      image_channels=self.img_channels,
      n_channels=self.n_channels,
      ch_mults=self.channels_mul,
      is_attn=self.is_attn,
    ).to(_device)
    self.dif_model = DenoiseModel(
      noise_model=self.eps_model, steps=self.n_steps, batch_sz=self.batch_sz
    ).to(_device)
    self.dataset = self.get_dataset()
    self.dl = DataLoader(
      self.dataset, self.batch_sz, shuffle=True, pin_memory=True
    )
    self.optimizer = Adam(self.eps_model.parameters(), lr=self.lr)
    self.unet_weights = FileManager().unet_weights_path
    self.ddpm_weights = FileManager().ddpm_weights_path
    print("Transformer initialization done.")
    logger.info("Transformer initialization done.")

  def get_dataset(self):
    transform = torchvision.transforms.Compose(
      [
        torchvision.transforms.Resize(self.img_sz),
        torchvision.transforms.Grayscale(3),
        torchvision.transforms.ToTensor(),
      ]
    )
    return torchvision.datasets.MNIST(
      root="./data", train=True, download=True, transform=transform
    )

  def sample(self):
    with torch.no_grad():
      x = torch.randn(
        [self.n_samples, self.img_channels, self.img_sz, self.img_sz],
        device=_device,
      )
      for _t in range(self.n_steps):
        t = self.n_steps - _t - 1
        x = self.dif_model.p_sample(
          x, x.new_full((self.n_samples,), t, device=_device, dtype=torch.long)
        )

  # @epoch_wrapper(["[INFO] EPOCH: {epoch}"], [])
  def run_epoch(self, epoch):
    loss = 0
    for data in self.dl:
      _data = data[0].to(_device)
      self.optimizer.zero_grad()
      loss = self.dif_model.loss(_data)
      loss.backward()
      self.optimizer.step()
    print(f"Current loss is: {loss}")

  def training(self):
    print("Denoise Model currently running training.")
    logger.info("Denoise Model currently running training.")
    _time = time.time()
    for epoch in range(self.nepoch):
      self.run_epoch(epoch)
      self.sample()
    print(f"TRAINING: ENDED, TIME ELAPSED:{time.time() - _time:0.3f}")
    logger.info(f"TRAINING: ENDED, TIME ELAPSED:{time.time() - _time:0.3f}")

    torch.save(self.eps_model.state_dict(), self.unet_weights)
    torch.save(self.dif_model.state_dict(), self.ddpm_weights)

  def evaluate(self):
    xt = torch.randn(
      [self.n_samples, self.img_channels, self.img_sz, self.img_sz],
      device=_device,
    )
    # --- RELOAD MODEL WITH NEW WEIGHTS --- #
    self.eps_model.load_state_dict(
      torch.load(self.unet_weights, weights_only=True, map_location=_device)
    )
    self.dif_model.load_state_dict(
      torch.load(self.ddpm_weights, weights_only=True, map_location=_device)
    )
    self.dif_model.eval()
    with torch.no_grad():
      for _t in range(self.n_steps):
        t = self.n_steps - _t - 1
        xt = self.dif_model.p_sample(
          xt,
          xt.new_full((self.n_samples,), t, device=_device, dtype=torch.long),
        )

    for i in range(self.n_samples):
      save_image(xt[i], f"./images/img{i}.png")
