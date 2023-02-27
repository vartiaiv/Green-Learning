from absl import flags

DATASETS_ROOT = r".\datasets"  # shared between both networks
MODELS_ROOT = r".\models\ffcnn"  # separate folder for each network

# define flags and default values
flags.DEFINE_string('datasets_path', None, "The output dir to save params.")  # unused
flags.DEFINE_string('models_path', None, "The output dir to save params.")  # unused
flags.DEFINE_string('use_classes', "0-9", "Supported format: '0,1,5-9'")
flags.DEFINE_string('kernel_sizes', "3,3", "Kernels size for each stage. Format: '5,5'")
flags.DEFINE_string('num_kernels', "4,10", "Num of kernels for each stage. Format: '31,63'")
flags.DEFINE_float('energy_percent', None, "Energy to be preserved in each stage." )
flags.DEFINE_integer('use_num_images', -1, "Num of images used for training. (-1 for all)")
flags.DEFINE_string('use_dataset', "cifar10", "Name of a multiclass dataset, e.g. 'cifar10'")
flags.DEFINE_float('use_portion', 0.1, "fraction of the dataset to be used. 1.0 for all")

FLAGS = flags.FLAGS  # this contains flags and default values, import in needed scripts