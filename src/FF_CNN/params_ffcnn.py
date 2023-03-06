from absl import flags

# NOTE change dataset name here, other parameters are selected depending on it
USE_DATASET = 'cifar10'  # options 'mnist', 'cifar10'

# NOTE this is the number of sample images (used in getkernel.py)
# this speeds up the training process a lot
USE_NUM_IMAGES = 200  # number of sample images in getkernel.py (-1 to use all images) 

# NOTE portion of training images (used in getweight.py)
TRAIN_IMAGES_PORTION = 0.05  # 1.0 use entire dataset for weights; size subset as a percentage of original dataset. 

# NOTE Change for debug purposes only. This should be 1.0 when doing evaluation!
TEST_IMAGES_PORTION = 1.0  

# Other, no need to change
USE_SEED = None  # int seed
DATASETS_ROOT = r".\datasets"  # shared between both networks
MODELS_ROOT = r".\models\ffcnn"  # separate folder for each network
USE_CLASSES = '0-9'  # classes to be used, 0-9 is for all 10 classes to be used (like MNIST)
KERNEL_SIZES = '5,5'  # convolutional kernel size this was used in the original code
# number of conv kernels: write 1 less than wanted number (6->5 and 16->15)
NUM_KERNELS = '5,15' if USE_DATASET == 'mnist' else '31,63'
# fully connected layer dims: originally 120,84,10 for mnist and "200,100,10" for cifar10
NUM_CLUSTERS = '120,84,10' if USE_DATASET == 'mnist' else '200,100,10'
ENERGY_PERCENT = None


# NOTE ignore the flags below if you dont want to use command line
# ***************************************************************************************************************

# define command line flags and default values
flags.DEFINE_string('use_dataset', USE_DATASET, "Name of a multiclass dataset. E.g. 'mnist'")
flags.DEFINE_float('train_images_portion', TRAIN_IMAGES_PORTION, "fraction of the dataset to be used. 1.0 for all")  # debug
flags.DEFINE_float('test_images_portion', TEST_IMAGES_PORTION, "fraction of the dataset to be used. 1.0 for all")  # debug
flags.DEFINE_string('datasets_root', DATASETS_ROOT, "Datasets root location")
flags.DEFINE_string('models_root', MODELS_ROOT, "Where ffcnn models trained on different datasets are stored")
flags.DEFINE_string('use_classes', USE_CLASSES, "Supported format: '0,1,5-9'")
flags.DEFINE_string('kernel_sizes', KERNEL_SIZES, "Kernels size for each stage. Format: '3,3'")
flags.DEFINE_integer('use_seed', USE_SEED, "seed")  # debug
flags.DEFINE_string('num_kernels', NUM_KERNELS, "Num of kernels for each stage. Format: '4,10'")
flags.DEFINE_string('num_clusters', NUM_CLUSTERS, "Num of clusters. Format: '120,84,10'")
flags.DEFINE_float('energy_percent', ENERGY_PERCENT, "Energy to be preserved in each stage." )
flags.DEFINE_integer('use_num_images', USE_NUM_IMAGES, "Num of images used for training. (-1 for all)")

FLAGS = flags.FLAGS  # this contains flags and default values, import in needed scripts