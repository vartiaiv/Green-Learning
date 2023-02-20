import torch
import numpy as np
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils import clip_grad_value_
from sklearn.metrics import accuracy_score


def forward_backward_pass(lenet_model: Module, 
                        optim: Optimizer, 
                        dataloader: DataLoader, 
                        device: str, 
                        do_grad_clip: bool = False
                        ) -> tuple(Module, np.float64, np.float64):

    iteration_loss = []

    image_targets = []
    image_predictions = []

    if optim is not None:
        # Indicate that we are in training mode
        lenet_model.train()
    else:
        # Indicate that we are in evaluation mode
        lenet_model.eval()

    for batch in dataloader:
        # Zero the gradient of the optimizer.
        if optim is not None:
            optim.zero_grad()

        # Get the batches.
        images, y_true = batch

        # Give them to the appropriate device & reshape the audio.
        images = images.float().unsqueeze(dim=1).to(device)
        y_hat = y_hat.to(device)
        label = label.to(device)

        # Get the prediction of the model
        y_hat = lenet_model(images)

        # Calculate the loss of our model.
        loss = torch.nn.functional.cross_entropy(input=y_hat, target=y_true)

        if optim is not None:
            # Do the backward pass
            loss.backward()

            # Gradient clipping (if applicable)
            if do_grad_clip:
                clip_grad_value_(lenet_model.parameters(), 1)

            # Do an update of the weights (i.e. a step of the optimizer)
            optim.step()

        # Loss the loss of the batch
        iteration_loss.append(loss.item())

        image_targets.append(y_true.cpu())
        image_predictions.append(y_hat.detach().cpu())


    image_targets = torch.cat(image_targets)
    image_predictions = torch.cat(image_predictions)

    # Calculate accuracy metrics of the batch
    image_predictions = torch.argmax(torch.nn.functional.softmax(image_predictions, dim=1), dim=1)
    accuracy = accuracy_score(y_true=image_targets.numpy(), y_pred=image_predictions.numpy())

    return lenet_model, np.mean(iteration_loss), accuracy