from absl import flags

# NOTE change dataset name here, other parameters are selected depending on it
USE_DATASET = 'cifar10'  # ('mnist', 'cifar10')

# NOTE this is the number of samples used in training (getkernel.py)
# This number must be greater than or equal to number of classes
USE_NUM_IMAGES = 10  # (-1 to use all images) 

# NOTE this is the number of features used in training (getfeature.py)
# This number must be greater than or equal to 1st FC layer nodes (e.g., 120 for normal LeNet-5)
# Preferably thousands of features will be obtained for LLSR training
USE_NUM_FEATURES = 200  # (-1 to use all images) 

# Other params, no need to change
USE_SEED = None  # rng seed, should be None when doing real experiments
DATASETS_ROOT = r".\datasets"  # data location shared between all networks
MODELS_ROOT = r".\models\ffcnn"  # FFCNN base folder, branches into folders based on dataset selection
KERNEL_SIZES = '5,5'  # convolutional kernel size this was used in the original code
# number of AC filters: (without including the DC kernel)
NUM_KERNELS = '5,15' if USE_DATASET == 'mnist' else '31,63'
# fully connected layer dims: originally 120,84,10 for mnist and "200,100,10" for cifar10
NUM_CLUSTERS = '120,84,10' if USE_DATASET == 'mnist' else '200,100,10'
ENERGY_PERCENT = None


# NOTE ignore the flags below if you dont want to use command line
# ***************************************************************************************************************

# define command line flags and default values
flags.DEFINE_string('use_dataset', USE_DATASET, "Name of a multiclass dataset. E.g. 'mnist'")
flags.DEFINE_integer('use_num_images', USE_NUM_IMAGES, "Num of images used for training. (-1 for all)")
flags.DEFINE_integer('use_num_features', USE_NUM_FEATURES, "Num of features used for training. (-1 for all)")
flags.DEFINE_integer('use_seed', USE_SEED, "seed")  # debug
flags.DEFINE_string('datasets_root', DATASETS_ROOT, "Datasets root location")
flags.DEFINE_string('models_root', MODELS_ROOT, "Where ffcnn models trained on different datasets are stored")
flags.DEFINE_string('kernel_sizes', KERNEL_SIZES, "Kernels size for each stage. Format: '3,3'")
flags.DEFINE_string('num_kernels', NUM_KERNELS, "Num of kernels for each stage. Format: '4,10'")
flags.DEFINE_string('num_clusters', NUM_CLUSTERS, "Num of clusters. Format: '120,84,10'")
flags.DEFINE_float('energy_percent', ENERGY_PERCENT, "Energy to be preserved in each stage." )

FLAGS = flags.FLAGS  # this contains flags and default values, import in needed scripts