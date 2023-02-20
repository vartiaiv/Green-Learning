import data
import saab

from absl import app
from absl import logging
from defaultflags import FLAGS
from src.utils.timer import timeit

import os
from src.utils.io import save

# io paths
here = os.path.dirname(os.path.abspath(__file__))
savepath = os.path.join(here, "model", "pca_params.pkl")

@timeit
def main(argv):   
    # read data
    train_images, train_labels, _, _, class_list = data.import_data()

    kernel_sizes = saab.parse_list_string(FLAGS.kernel_sizes)
    if FLAGS.num_kernels:
        num_kernels = saab.parse_list_string(FLAGS.num_kernels)
    else:
        num_kernels = None
    energy_percent = FLAGS.energy_percent
    use_num_images = FLAGS.use_num_images
    if use_num_images > len(train_images):
        use_num_images = len(train_images)

    print('Saab parameters:')
    print('kernel_sizes:', kernel_sizes)
    print('number_kernels:', num_kernels)
    print('energy_percent:', energy_percent)
    print('use_num_images:', use_num_images)
    print('class_list:', class_list)

    pca_params = saab.multi_Saab_transform(train_images, train_labels,
                         kernel_sizes=kernel_sizes,
                         num_kernels=num_kernels,
                         energy_percent=energy_percent,
                         use_num_images=use_num_images,
                         class_list=class_list)
    
    # save params
    save(savepath, pca_params)       

if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass