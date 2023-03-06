import os
from absl import app
from params_ffcnn import FLAGS
from absl import logging

import getkernel, getfeature, getweight
from utils.perf import mytimer

@mytimer
def main(argv):
    print("--------Training --------\n")
    
    getkernel.main(argv)
    getfeature.main(argv)
    getweight.main(argv)

    print("--------Training done --------")

if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass