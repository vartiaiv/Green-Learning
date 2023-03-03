import os
from absl import app
from params_ffcnn import FLAGS
from absl import logging

import saab
import data_ffcnn as data_ffcnn
from utils.io import save_params
from utils.perf import mytimer


@mytimer
def main(argv):   
    # io paths
    modelpath = os.path.join(FLAGS.models_root, f"ffcnn_{FLAGS.use_dataset}")

    # read data
    train_images, train_labels, _, _, class_list = data_ffcnn.import_data()

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
    print('use_num_images:', use_num_images if use_num_images > 0 else 'all')
    print('class_list:', class_list)

    pca_params = saab.multi_Saab_transform(train_images, train_labels,
                         kernel_sizes=kernel_sizes,
                         num_kernels=num_kernels,
                         energy_percent=energy_percent,
                         use_num_images=use_num_images,
                         class_list=class_list)

    save_params(modelpath, "pca_params.pkl", pca_params)


if __name__ == "__main__":      
    try:
        app.run(main)
    except SystemExit:
        pass