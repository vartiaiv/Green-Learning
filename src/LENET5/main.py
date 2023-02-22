from train_lenet import train_lenet
from test_lenet import test_lenet
import sys
import torch


def main():
    # Get the job index from the command line
    job_idx = int(sys.argv[1])
    print("Job index: ", job_idx)
    random_seed = job_idx
    torch.manual_seed(random_seed)

    # Get the dataset from the command line
    dataset_name = str(sys.argv[2])
    train_lenet(dataset_name=dataset_name)
    test_lenet(dataset_name=dataset_name)

if __name__ == "__main__":
    main()