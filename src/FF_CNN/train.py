import os
# NOTE Comment the next line and see if any warnings come from KMeans
# If needed, do this BEFORE importing numpy and KMeans or any other module using them
# Avoid memory leak with KMeans on Windows with MKL, when less chunks than available threads
# - getweight.py uses KMeans
os.environ["OMP_NUM_THREADS"] = '1'  


from absl import app
from params_ffcnn import FLAGS
from absl import logging

import getkernel
import getfeature
import getweight

def main(argv):
    getkernel.main(argv)
    getfeature.main(argv)
    getweight.main(argv)


if __name__ == "__main__":
    app.run(main)