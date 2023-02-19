from absl import flags
from absl.flags import FLAGS  # for importing

# define flags and default values
flags.DEFINE_string('output_path', None, "The output dir to save params")
flags.DEFINE_string('use_classes', "0-9", "Supported format: '0,1,5-9'")
flags.DEFINE_string('kernel_sizes', "5,5", "Kernels size for each stage. Format: '3,3'")
flags.DEFINE_string('num_kernels', "4,10", "Num of kernels for each stage. Format: '4,10'")
flags.DEFINE_float('energy_percent', None, "Energy to be preserved in each stage")
flags.DEFINE_integer('use_num_images', 40, "Num of images used for training (-1 for all)")
flags.DEFINE_string('use_dataset', "mnist", "Name of a multiclass dataset, e.g. 'cifar10' or 'mnist'")
