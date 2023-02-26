from main import main as training_main
from train_lenet import train_lenet
from test_lenet import test_lenet
from torch.utils.benchmark import Timer

import torch
import sys


def main():
    # Get the job index from the command line
    job_idx = int(sys.argv[1])
    print("Job index: ", job_idx)
    num_threads = torch.get_num_threads()

    # Get the dataset from the command line
    dataset_name = str(sys.argv[2])

    # Timers
    training_timer = Timer(
        stmt="train_lenet(dataset_name)", 
        setup="from __main__ import train_lenet", 
        globals={"dataset_name": dataset_name},
        num_threads=num_threads,
        label="Time the optimization process")
    testing_timer = Timer(
        stmt="test_lenet(dataset_name)", 
        setup="from __main__ import test_lenet", 
        globals={"dataset_name": dataset_name},
        num_threads=num_threads,
        label="Time the prediction process")

    # Run the timers
    # print(training_timer.timeit(10))
    print(testing_timer.timeit(10))

if __name__ == "__main__":
    main()
