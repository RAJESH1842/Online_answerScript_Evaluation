import os
# Ensure TensorFlow does not use oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pickle
import numpy as np
import re
import cv2
from PIL import Image as im
from word_detector import detect, prepare_img, sort_multiline
from path import Path
from typing import List
import argparse



# Create a StringLookup layer
lookup = tf.keras.layers.StringLookup()

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=Path, default=Path('C:/testtes'))
parser.add_argument('--kernel_size', type=int, default=25)
parser.add_argument('--sigma', type=float, default=11)
parser.add_argument('--theta', type=float, default=7)
parser.add_argument('--min_area', type=int, default=100)
parser.add_argument('--img_height', type=int, default=1000)
parsed = parser.parse_args()

print("File path: ", parsed.data)

def get_img_files(data_dir: Path) -> List[Path]:
    res = []
    for ext in ['*.png', '*.jpg', '*.bmp']:
        res += Path(data_dir).files(ext)
    return res

def save_image_names_to_text_files():
    list_img_names_serial = []

    for fn_img in get_img_files(parsed.data):
        print(f'Processing file {fn_img}')

        # load image and process it
        img = prepare_img(cv2.imread(fn_img), parsed.img_height)
        detections = detect(img, kernel_size=parsed.kernel_size, sigma=parsed.sigma, theta=parsed.theta, min_area=parsed.min_area)

        # sort detections: cluster into lines, then sort each line
        lines = sort_multiline(detections)

        # plot results
        plt.imshow(img, cmap='gray')
        num_colors = 7
        colors = plt.cm.rainbow(np.linspace(0, 1, num_colors))
        
        for line_idx, line in enumerate(lines):
            color = colors[line_idx % num_colors]
            for word_idx, det in enumerate(line):
                xs = [det.bbox.x, det.bbox.x, det.bbox.x + det.bbox.w, det.bbox.x + det.bbox.w, det.bbox.x]
                ys = [det.bbox.y, det.bbox.y + det.bbox.h, det.bbox.y + det.bbox.h, det.bbox.y, det.bbox.y]
                plt.plot(xs, ys, c=color)
                plt.text(det.bbox.x, det.bbox.y, f'{line_idx}/{word_idx}')
                crop_img = img[det.bbox.y:det.bbox.y + det.bbox.h, det.bbox.x:det.bbox.x + det.bbox.w]

                path = 'extracted_words'
                if not os.path.exists(path):
                    os.mkdir(path)

                img_path = os.path.join(path, f"line{line_idx}_word{word_idx}.jpg")
                cv2.imwrite(img_path, crop_img)
                list_img_names_serial.append(img_path)

        plt.show()

    with open("C:/testtes/abc.txt", "w") as textfile:
        for element in list_img_names_serial:
            textfile.write(element + "\n")

save_image_names_to_text_files()

# Verify the file path
characters_file_path = "./characters.pkl"
if not os.path.exists(characters_file_path):
    raise FileNotFoundError(f"The file {characters_file_path} does not exist. Please check the path and try again.")

# Load characters for OCR
with open(characters_file_path, "rb") as fp:
    characters = pickle.load(fp)
    print(characters)

# Create lookup layers for characters
AUTOTUNE = tf.data.AUTOTUNE
char_to_num = lookup(vocabulary=characters, mask_token=None)
num_to_chars = lookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

# Parameters for image processing and model
batch_size = 64
image_width = 128
image_height = 32
max_len = 21

# Functions for image preprocessing
def distortion_free_resize(image, img_size):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    if pad_height % 2 != 0:
        pad_height_top = pad_height // 2 + 1
        pad_height_bottom = pad_height // 2
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        pad_width_left = pad_width // 2 + 1
        pad_width_right = pad_width // 2
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(image, paddings=[[pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0, 0]])
    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image

def preprocess_image(image_path, img_size=(image_width, image_height)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, 1)
    image = distortion_free_resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image

def process_images_2(image_path):
    image = preprocess_image(image_path)
    return {"image": image}

def prepare_test_images(image_paths):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths)).map(process_images_2, num_parallel_calls=AUTOTUNE)
    return dataset.batch(batch_size).cache().prefetch(AUTOTUNE)

# Loading the detected word images
with open("C:/testtes/abc.txt", "r") as f:
    t_images = [line.strip() for line in f]

inf_images = prepare_test_images(t_images)

# Define the CTCLayer for the model
class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred

def build_model():
    input_img = keras.Input(shape=(image_width, image_height, 1), name="image")
    labels = keras.layers.Input(name="label", shape=(None,))

    x = keras.layers.Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv1")(input_img)
    x = keras.layers.MaxPooling2D((2,2), name="pool1")(x)
    x = keras.layers.Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv2")(x)
    x = keras.layers.MaxPooling2D((2,2), name="pool2")(x)

    new_shape = ((image_width // 4), (image_height // 4) * 64)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(64, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True, dropout=0.25))(x)
    x = keras.layers.Dense(len(char_to_num.get_vocabulary()) + 2, activation="softmax", name="dense2")(x)

    output = CTCLayer(name="ctc_loss")(labels, x)

    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="handwriting_recognizer")
    model.compile(optimizer=keras.optimizers.Adam())
    return model

# Get the model
model = build_model()
model.summary()

# Load the trained model
custom_objects = {"CTCLayer": CTCLayer}
reconstructed_model = keras.models.load_model("./ocr_model_50_epoch.h5", custom_objects=custom_objects)
prediction_model = keras.models.Model(reconstructed_model.get_layer(name="image").input, reconstructed_model.get_layer(name="dense2").output)

# Function to decode predictions
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_len]

    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_chars(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

# Perform inference on the word images
pred_test_text = []
for batch in inf_images:
    batch_images = batch["image"]
    preds = prediction_model.predict(batch_images)
    pred_texts = decode_batch_predictions(preds)
    pred_test_text.extend(pred_texts)

# Print the extracted text
for i, text in enumerate(pred_test_text):
    print(f"Word {i+1}: {text}")

# Combine the text into a sentence (optional)
sentence = ' '.join(pred_test_text)
print("Extracted Sentence:", sentence)
