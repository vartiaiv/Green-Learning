from absl import flags
# from absl.flags import FLAGS

# define flags and default values
flags.DEFINE_string('output_path', None, "The output dir to save params.")                   # str
flags.DEFINE_string('use_classes', "0-9", "Supported format: '0,1,5-9'")                     # str
flags.DEFINE_string('kernel_sizes', "3,3", "Kernels size for each stage. Format: '5,5'")     # str
flags.DEFINE_string('num_kernels', "4,10", "Num of kernels for each stage. Format: '31,63'")  # str
flags.DEFINE_float('energy_percent', None, "Energy to be preserved in each stage." )         # float
flags.DEFINE_integer('use_num_images', 100, "Num of images used for training. (-1 for all)")  # int
flags.DEFINE_string('use_dataset', "mnist", "Name of a multiclass dataset, e.g. 'cifar10'")  # str
flags.DEFINE_float('use_portion', 0.05, "fraction of datasets to be used. 1.0 for all")  # float

FLAGS = flags.FLAGS  # this contains flags and default values, import in needed scripts