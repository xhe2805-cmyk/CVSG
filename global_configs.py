"""Input feature dimensions and default computation device."""
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TEXT_DIM = 768
ACOUSTIC_DIM = 74
VISUAL_DIM = 47
