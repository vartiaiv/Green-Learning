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
    print('--------PCA training --------')
    # io paths
    modelpath = os.path.join(FLAGS.models_root, f"ffcnn_{FLAGS.use_dataset}")
    print("Model name:", os.path.basename(modelpath))

    use_num_images = FLAGS.use_num_images

    num_kernels = saab.parse_list_string(FLAGS.num_kernels)
    kernel_sizes = saab.parse_list_string(FLAGS.kernel_sizes)
    energy_percent = FLAGS.energy_percent
    print("- USE_NUM_IMAGES:", use_num_images)

    # load train data
    train_images, train_labels, class_list = data_ffcnn.import_data()
    print("Train images size:", train_images.shape)

    pca_params = saab.multi_Saab_transform(train_images, train_labels,
                         kernel_sizes=kernel_sizes,
                         num_kernels=num_kernels,
                         energy_percent=energy_percent,
                         use_num_images=use_num_images,
                         class_list=class_list)
    print('--------Finish PCA training --------')
    save_params(modelpath, "pca_params.pkl", pca_params)


if __name__ == "__main__":      
    try:
        app.run(main)
    except SystemExit:
        pass