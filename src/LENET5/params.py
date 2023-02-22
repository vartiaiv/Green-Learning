import torch

# check device
if torch.has_mps:
    DEVICE = 'mps'
elif torch.has_cuda:
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'
print(f'Process on {DEVICE}', '\n\n')

if (DEVICE == 'cuda'):
    print(f'Device name: {torch.cuda.get_device_name(0)}', '\n\n')

# Hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
N_EPOCHS = 50

IMG_SIZE = 32
N_CLASSES = 10

PATIENCE = 10