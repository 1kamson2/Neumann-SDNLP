from ddpm_model.ddpm import * 
import unittest
import torch
HEADER_LENGTH = 60 
UNET_INFO_HEADER = "[UNET INFO]"
DDPM_INFO_HEADER = "[DDPM INFO]"

class DDPMTest(unittest.TestCase):
    def setUp(self):
        self.steps = 1000
        self.noise_model = UNet()
        self.test_ddpm = DenoiseModel(self.noise_model, self.steps)

    @unittest.skip("There is no need for so much info")
    def test_setup_info(self): 
        # --- Print noise model's information --- #
        print('-' * (HEADER_LENGTH // 2) + UNET_INFO_HEADER + '-' * (HEADER_LENGTH 
                                                                  // 2))
        print(self.noise_model)
        print(f"{'-' * (HEADER_LENGTH + len(UNET_INFO_HEADER))}")
        # --- Print ddpm model's information --- #
        print('-' * (HEADER_LENGTH // 2) + DDPM_INFO_HEADER + '-' * (HEADER_LENGTH
                                                                  // 2))
        print(self.test_ddpm)
        print(f"{'-' * (HEADER_LENGTH + len(DDPM_INFO_HEADER))}")

        



if __name__ == "__main__":
    unittest.main()
