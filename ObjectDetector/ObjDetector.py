import time
import os
import sys

import cv2
import numpy as np
import re

current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir, 'models'))
sys.path.append(os.path.join(current_dir, 'models', 'research'))
sys.path.append(os.path.join(current_dir, 'models', 'research', 'object_detection'))
from . import Models


from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.builders import model_builder
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pathlib
import tensorflow as tf

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# PATH_TO_MODEL_DIR = os.path.join(current_dir, 'weights/efficientdet_d2_coco17_tpu-32')

np.random.seed(12345)
COLORS = 255*np.random.rand(90, 3)


class ObjectDetector:
    def __init__(self, name, label_map):
        assert name in Models, "Model is not supported"
        self.category_index = label_map_util.create_category_index_from_labelmap(label_map, use_display_name=True)
        print(self.category_index)
        self.start_time = time.time()
        self.initialized = False
        des = Models[name]
        PATH_TO_MODEL_DIR = os.path.join(current_dir, 'weights', des['folder'])
        self.PATH_TO_CFG = os.path.join(PATH_TO_MODEL_DIR,'pipeline.config')
        self.PATH_TO_CKPT = os.path.join(PATH_TO_MODEL_DIR, 'checkpoint', des['ckpt'])
        self.detection = self.load_model()
    def load_model(self):
        print('Loading model... ', end='')

        # Load pipeline config and build a detection model
        configs = config_util.get_configs_from_pipeline_file(self.PATH_TO_CFG)
        model_config = configs['model']
        detection_model = model_builder.build(model_config=model_config, is_training=False)

        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(self.PATH_TO_CKPT).expect_partial()

        @tf.function
        def detect_fn(image):
            """Detect objects in image."""

            image, shapes = detection_model.preprocess(image)
            prediction_dict = detection_model.predict(image, shapes)
            detections = detection_model.postprocess(prediction_dict, shapes)

            return detections
        return detect_fn

    def __call__(self, image_np, threshold = 0.5):
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        # input_tensor = input_tensor[tf.newaxis, ...]

        # input_tensor = np.expand_dims(image_np, 0)
        fstart = time.time()
        detections = self.detection(input_tensor)

        if not self.initialized:
            end_time = time.time()
            elapsed_time = end_time - self.start_time
            print('Done! Took {} seconds'.format(elapsed_time))
            self.initialized = True
        else:
            print('[+] FPS {:.2f}'.format(1 / (time.time() - fstart)))

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        label_id_offset = 1
        detections['detection_classes'] = detections['detection_classes'].astype(np.int32) + label_id_offset

        image_np_with_detections = image_np.copy()
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            self.category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.50,
            agnostic_mode=False)

        return image_np_with_detections



