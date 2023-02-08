import data
import saab

from absl import app
from absl import flags
from absl.flags import FLAGS
from absl import logging
import pickle

from utils.timer import timeit

# define flags
flags.DEFINE_string('output_path', None, "The output dir to save params")
flags.DEFINE_string('use_classes', "0-9", "Supported format: '0,1,5-9'")
flags.DEFINE_string('kernel_sizes', "5,5", "Kernels size for each stage. Format: '3,3'")
flags.DEFINE_string('num_kernels', "4,10", "Num of kernels for each stage. Format: '4,10'")
flags.DEFINE_float('energy_percent', None, "Energy to be preserved in each stage")
flags.DEFINE_integer('use_num_images', 100, "Num of images used for training (-1 for all)")
flags.DEFINE_string('use_dataset', "cifar10", "Name of a multiclass dataset, e.g. 'cifar10' or 'mnist'")

@timeit
def main(argv):
    # read data
    use_classes = FLAGS.use_classes
    use_dataset = FLAGS.use_dataset

    train_images, train_labels, _, _, class_list = data.import_data(use_classes, use_dataset)

    kernel_sizes = saab.parse_list_string(FLAGS.kernel_sizes)
    if FLAGS.num_kernels:
        num_kernels = saab.parse_list_string(FLAGS.num_kernels)
    else:
        num_kernels = None
    energy_percent = FLAGS.energy_percent
    use_num_images = FLAGS.use_num_images
    use_dataset = FLAGS.use_dataset

    print('Parameters:')
    print('use_classes:', use_classes, '=>', class_list)
    print('kernel_sizes:', kernel_sizes)
    print('number_kernels:', num_kernels)
    print('energy_percent:', energy_percent)
    print('use_num_images:', use_num_images)
    print('use_dataset:', use_dataset)

    pca_params = saab.multi_Saab_transform(train_images, train_labels,
                         kernel_sizes=kernel_sizes,
                         num_kernels=num_kernels,
                         energy_percent=energy_percent,
                         use_num_images=use_num_images,
                         use_classes=class_list)
    # save data
    with open(r'./CIFAR_FF/pca_params.pkl','wb') as fw:
        pickle.dump(pca_params, fw)

if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass