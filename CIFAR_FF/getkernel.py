import data
import saab

from absl import app
from absl import flags
from absl.flags import FLAGS
from absl import logging
import pickle

# define flags
flags.DEFINE_string('output_path', None, "The output dir to save params")
flags.DEFINE_string('use_classes', "0-9", "Supported format: 0,1,5-9")
flags.DEFINE_string('kernel_sizes', "5,5", "Kernels size for each stage. Format: '3,3'")
flags.DEFINE_string('num_kernels', "31,63", "Num of kernels for each stage. Format: '4,10'")
flags.DEFINE_float('energy_percent', None, "Energy to be preserved in each stage")
flags.DEFINE_integer('use_num_images', -1, "Num of images used for training")

def main(argv):
    # read data
    train_images, train_labels, test_images, _, class_list = data.import_data(FLAGS.use_classes)
    print('Training image size:', train_images.shape)
    print('Testing_image size:', test_images.shape)

    kernel_sizes = saab.parse_list_string(FLAGS.kernel_sizes)
    if FLAGS.num_kernels:
        num_kernels = saab.parse_list_string(FLAGS.num_kernels)
    else:
        num_kernels = None
    energy_percent = FLAGS.energy_percent
    use_num_images = FLAGS.use_num_images
    print('Parameters:')
    print('use_classes:', class_list)
    print('Kernel_sizes:', kernel_sizes)
    print('Number_kernels:', num_kernels)
    print('Energy_percent:', energy_percent)
    print('Number_use_images:', use_num_images)

    pca_params = saab.multi_Saab_transform(train_images, train_labels,
                         kernel_sizes=kernel_sizes,
                         num_kernels=num_kernels,
                         energy_percent=energy_percent,
                         use_num_images=use_num_images,
                         use_classes=class_list)
    # save data
    with open('pca_params.pkl','wb') as fw:
        pickle.dump(pca_params, fw)

if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass