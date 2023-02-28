from absl import flags

# NOTE change these values if prefer not using command line
DATASETS_ROOT = r".\datasets"  # shared between both networks
MODELS_ROOT = r".\models\ffcnn"  # separate folder for each network
USE_DATASET = "mnist"  # the dataset to be used, MNIST by default, other option is 'cifar10'
USE_PORTION = 1.0  # float percentage of the dataset to be used, 1.0 does not take subset
USE_CLASSES = "0-9"  # classes to be used, 0-9 is for all 10 classes to be used (like MNIST)
KERNEL_SIZES = "5,5"  # convolutional kernel size this was used in the original paper
NUM_KERNELS = "31,63"  # number of kernels for stages 0 and 1 respectively (these amounts in original code)
ENERGY_PERCENT = None
USE_NUM_IMAGES = -1  # -1 to use all images in getkernel.py


# ***************************************************************************************************************

# define command line flags and default values
flags.DEFINE_string('datasets_root', DATASETS_ROOT, "Datasets root location")
flags.DEFINE_string('models_root', MODELS_ROOT, "Where ffcnn models trained on different datasets are stored")
flags.DEFINE_string('use_dataset', USE_DATASET, "Name of a multiclass dataset")
flags.DEFINE_float('use_portion', USE_PORTION, "fraction of the dataset to be used. 1.0 for all")  # debug
flags.DEFINE_string('use_classes', USE_CLASSES, "Supported format: '0,1,5-9'")
flags.DEFINE_string('kernel_sizes', KERNEL_SIZES, "Kernels size for each stage. Format: '3,3'")
flags.DEFINE_string('num_kernels', NUM_KERNELS, "Num of kernels for each stage. Format: '4,10'")
flags.DEFINE_float('energy_percent', ENERGY_PERCENT, "Energy to be preserved in each stage." )
flags.DEFINE_integer('use_num_images', USE_NUM_IMAGES, "Num of images used for training. (-1 for all)")

FLAGS = flags.FLAGS  # this contains flags and default values, import in needed scripts