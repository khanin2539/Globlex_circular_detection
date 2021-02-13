import os
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_list('images', './data/images/kite.jpg', 'path to input image')
flags.DEFINE_string('output', './detections/', 'path to output folder')
flags.DEFINE_float('iou', 0.5, 'iou threshold')
flags.DEFINE_float('score', 0.5, 'score threshold')
flags.DEFINE_boolean('count', True, 'count objects within images')
flags.DEFINE_boolean('dont_show', False, 'dont show image output')
flags.DEFINE_boolean('info', False, 'print info on detections')
flags.DEFINE_boolean('crop', False, 'crop detections from images')
flags.DEFINE_boolean('ocr', False, 'perform generic OCR on detection regions')
flags.DEFINE_boolean('plate', False, 'perform license plate recognition')

def main(image):
    
    #set all parameters rather than flag arguments.
    print("hi")
    framework = 'tf'
    weights = './checkpoints/yolov4-416'
    size =  416
    tiny =  False
    model = 'yolov4'
    output =  './detections/'
    iou = 0.5
    score = 0.5
    count = True
    dont_show = False
    info = False
    #images = ['./data/images/assa.jpg']
    print("here")
    
    #set date time
    dt = str(utils.datetime.datetime.now()) 
    
    #set an image counter
    count_image = 2
    
    # Creating Tensorflow sesstion
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config()
    input_size = 416
    
    #retrieve an image path from streamlit and put it in the list
    #images = []
    #images.append(image)
    # load model
  
    saved_model_loaded = tf.saved_model.load(weights, tags=[tag_constants.SERVING])
    is_non_empty= bool(saved_model_loaded)
    print(is_non_empty)
    print("model done")
    #print(images)
    #type(images)
    #is_non_empty_image= bool(images)
    #print(is_non_empty_image)
    # loop through images in list and run Yolov4 model on each
    #for count, image_path in enumerate(images, 1):
        #original_image = cv2.imread(image_path)
        #original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    image = np.array(image.convert('RGB'))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #important
    image_data = cv2.resize(image, (input_size, input_size))
    image_data = image_data / 255.
    
    # get image name by using split method
    #image_name = image_path.split('/')[-1]
    #image_name = image_name.split('.')[0]

    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)

    
    infer = saved_model_loaded.signatures['serving_default']
    batch_data = tf.constant(images_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    # run non max suppression on detections
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=200,
        max_total_size=200,
        iou_threshold=iou,
        score_threshold=score
    )

    # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
    original_h, original_w, _ = image.shape
    bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)
    
    # hold all detection data in one variable
    pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

    # read in all class names from config
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)

    # by default allow all classes in .names file
    allowed_classes = list(class_names.values())
    
    # custom allowed classes (uncomment line below to allow detections for only people)
    #allowed_classes = ['person']

    

    

    
    # count objects found
    counted_classes = count_objects(pred_bbox, by_class = True, allowed_classes=allowed_classes)
    # loop through dict and print
    for key, value in counted_classes.items():
        print("Number of {}s: {}".format(key, value))
        print("Timestamp", dt)
    image_1 = utils.draw_bbox(image, pred_bbox, info, counted_classes, allowed_classes=allowed_classes)
    
    
    image_1 = Image.fromarray(image_1.astype(np.uint8))
    
    image_1 = cv2.cvtColor(np.array(image_1), cv2.COLOR_BGR2RGB)
    #cv2.imwrite(FLAGS.output + 'detection' + str(count) + '.jpg', image)
    #count = count+1
    #print(count)
#cv2.imwrite(output + 'detection' + str(count_image) + '.jpg', image)
    count_image += 1
    return image, counted_classes;
    

