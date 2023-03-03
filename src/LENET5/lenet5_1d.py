import torch
from torch.nn import Module, Conv2d, MaxPool2d, Linear, ReLU, Softmax, Sequential, BatchNorm2d, Dropout2d
from torch import Tensor
from typing import Union, Tuple, Optional
from pytorch_model_summary import summary
from torchscan import summary as torchscan_summary


class LeNet5_1D(Module):
    def __init__(self,
                conv_1_output_dim: int = 6,
                conv_2_output_dim: int = 16,
                conv_1_kernel_size: Union[int, Tuple[int, int]] = 5,
                conv_2_kernel_size: Union[int, Tuple[int, int]] = 5,
                pooling_1_kernel_size: Union[int, Tuple[int, int]] = 2,
                pooling_2_kernel_size: Union[int, Tuple[int, int]] = 2,
                num_classes: int = 10,
                conv_1_stride: Optional[Union[int, Tuple[int, int]]] = 1,
                conv_2_stride: Optional[Union[int, Tuple[int, int]]] = 1,
                conv_1_padding: Optional[Union[int, Tuple[int, int]]] = 0,
                conv_2_padding: Optional[Union[int, Tuple[int, int]]] = 0,
                pooling_1_stride: Optional[Union[int, Tuple[int, int]]] = 2,
                pooling_2_stride: Optional[Union[int, Tuple[int, int]]] = 2,
                flatten_size = 400,
                linear_1_output_dim: int = 120,
                linear_2_output_dim: int = 84,
    ) -> None:

        super().__init__()

        self.block_1 = Sequential(
            Conv2d(in_channels=1,
                   out_channels=conv_1_output_dim,
                   kernel_size=conv_1_kernel_size,
                   stride=conv_1_stride,
                   padding=conv_1_padding),
            ReLU(),
            BatchNorm2d(num_features=conv_1_output_dim),
            MaxPool2d(kernel_size=pooling_1_kernel_size,
                      stride=pooling_1_stride),
        )

        self.block_2 = Sequential(
            Conv2d(in_channels=conv_1_output_dim,
                   out_channels=conv_2_output_dim,
                   kernel_size=conv_2_kernel_size,
                   stride=conv_2_stride,
                   padding=conv_2_padding),
            ReLU(),
            BatchNorm2d(num_features=conv_2_output_dim),
            MaxPool2d(kernel_size=pooling_2_kernel_size,
                      stride=pooling_2_stride),
        )

        # self.block_3 = Sequential(
        #     Conv2d(in_channels=conv_2_output_dim,
        #            out_channels=conv_3_output_dim,
        #            kernel_size=conv_3_kernel_size,
        #            stride=conv_3_stride,
        #            padding=conv_3_padding),
        #     ReLU()
        # )

        self.flatten_size = flatten_size

        self.fc1 = Sequential(
            Linear(in_features=flatten_size, out_features=linear_1_output_dim), 
            ReLU()
        )

        self.fc2 = Sequential(
            Linear(in_features=linear_1_output_dim, out_features=linear_2_output_dim),
            ReLU()
        )

        self.fc3 = Linear(in_features=linear_2_output_dim, out_features=num_classes)

    def forward(self, X: Tensor) -> Tensor:
        y = self.block_1(X)
        y = self.block_2(y)
        # y = self.block_3(y)

        y = y.flatten(start_dim=1)
        y = self.fc1(y)
        y = self.fc2(y)
        y = self.fc3(y)

        return y


def main():
    conv_1_output_dim = 6
    conv_2_output_dim = 16
    conv_3_output_dim = 120

    conv_1_kernel_size = 5
    conv_2_kernel_size = 5
    conv_3_kernel_size = 5

    pooling_1_kernel_size = 2
    pooling_2_kernel_size = 2

    linear_1_output_dim = 84
    num_classes = 10

    model = LeNet5_1D()

    # print(summary(model, torch.rand(4,1,32,32).float()))
    torchscan_summary(model, (1,32,32), receptive_field=True)


if __name__ == "__main__":
    main()