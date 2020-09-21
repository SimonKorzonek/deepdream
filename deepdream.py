from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
from keras.preprocessing.image import save_img
from IPython.display import clear_output

separator = "-------------------------------------------------------------------------------"

'''
    Gradient tape is TF API for auto-differentiation. Computing gradient
    respecting input values. Tensorflow records all operations executed inside
    of tf.GradientTape. Tape, and gradients associated with each recorded
    operation are needed for calculating gradients of whole "recorded"
    computation. Uses reverse mode defferentiation - reverse autodiff
'''
@tf.function
def deepdream(model, img, step_size):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = calc_loss(img, model)

    # reduce_std - reducing standard deviation
    gradients = tape.gradient(loss, img)
    gradients /= tf.math.reduce_std(input_tensor=gradients) + 1e-8  # 1e-8 - 0.00000001. Numbers in this notation are for reducing number of zeros to display
    img = img + gradients * step_size  # updating image by adding gradients * step size
    img = tf.clip_by_value(img, clip_value_min=-1, clip_value_max=1)  # cutting values outside acceptable ratio

    return loss, img

def get_image(target_size):
    image_path = tf.keras.utils.get_file(fname=FULLPATH, origin=DIRECTORY)  # downloads file from url
    img = tf.keras.preprocessing.image.load_img(path=image_path, target_size=target_size)  # loads image into PILLOW format
    return img

def normalize_img(img):
    img = 255 * (img + 1.0) / 2.0  # making sure that pixel values are... whatta fuk
    return tf.cast(img, tf.uint8)  # cast img tensor to uint8 datatype

def show(img):
    plt.figure(figsize=(12, 12))
    plt.grid(False)
    plt.axis('off')
    plt.imshow(img)
    plt.show()

def calc_loss(img_tile, model):
    img_batch = tf.expand_dims(img_tile, axis=0)  # inserts dimension of 1 at dimension axis indexed in tensor shape
    layer_activations = model(img_batch)  # passing tensor expantion operation to dream model
    losses = []

    for activation in layer_activations:
        # returned tensor will have just one element - mean of tensor elements
        loss = tf.math.reduce_mean(input_tensor=activation)
        losses.append(loss)

    # reducing tensor to one element - sum of losses
    return tf.math.reduce_sum(input_tensor=losses)

def run_deep_dream(model, img, steps, step_size, octave_counter, start_running):
    # Convert from uint8 to the range expected by the model.
    img = tf.keras.applications.inception_v3.preprocess_input(img)

    for step in range(steps):
        step_counter = step + 1
        runtime = datetime.utcnow() - start_running
        loss, img = deepdream(model, img, step_size=STEP_SIZE)
        print(f"Octave: {octave_counter}/{OCTAVES}    Step: {step_counter}/{steps}    Runtime: {runtime}    Loss: {loss}")

        if step % 10 == 0:
            clear_output(wait=True)

    result = normalize_img(img)
    clear_output(wait=True)

    return result

def sing():
    print(separator)
    print("I've had too much to dream last night")
    print("Too much to dream..")
    print("I'm not ready to face the light")
    print("I've had too much to dream..")
    print("Last night..")
    print(separator)
    print("Done.")


# T W E A K A B L E  D R E A M  P A R A M S:
DIRECTORY = "C:/Users/simon/Desktop/deepdream/source_images/"
FILENAME = "1.jpg"
OCTAVES = 10
OCTAVE_SCALE = 1.5
STEP_SIZE = 0.02
chosen_layers = [0, 0, 1, 1, 2, 2, 5, 5, 6, 7, 7, 9, 9]
target_size = [120, 160]  # 120 160
steps = 10
LAYER_NAMES = []
FULLPATH = DIRECTORY + FILENAME

for layer in chosen_layers:
    LAYER_NAMES.append(f"mixed{layer}")

#  D R E A M   M O D E L
base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
layers = [base_model.get_layer(name).output for name in LAYER_NAMES]
dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

#  G R A B  &  P R E P R O C E S S  S O U R C E  I M A G E
original_img = get_image(target_size)
original_img = np.array(original_img)
img = tf.constant(np.array(original_img))
base_shape = tf.cast(tf.shape(img)[:-1], tf.float32)
start_running = datetime.utcnow()

print(separator)
print(f"Dreaming about: {FILENAME.split('.')[0]}")


#  R U N  D E E P D R E A M
for octave in range(OCTAVES):

    new_shape = tf.cast(base_shape * (OCTAVE_SCALE ** octave), tf.int32)
    img = tf.image.resize(img, new_shape).numpy()
    print(separator)

    img = run_deep_dream(
        model=dream_model,
        img=img,
        steps=steps,
        step_size=STEP_SIZE,
        octave_counter=octave + 1,
        start_running=start_running)


#  S A V E   D R E A M
clear_output(wait=True)
save_img(f"{OCTAVES}-{STEP_SIZE}-{target_size}-{chosen_layers}-{FILENAME}", img)
show(original_img)
show(img)

sing()


'''
- R E A D  M E -


URL ----------> Path of file to dreamify


chosen_layers --> Layers of dream model selected to attempt in object recognition
                process. Lower layers will see most basic patterns. Higher ones
                will produce more complex patterns made from sum of simpler patterns.
                Named from 'mixed0' to 'mixed9'.

                - YOU HAVE TO CHOSE AT LEAST 2 LAYERS

                - HIGHER LAYERS ARE COMPUTIONALY HEAVIER AND ARE
                    MORE RAM CONSUMING


OCTAVES ------> Amount of iterations of gradient ascent applied on different scales.
                This allowes patterns generated at smaller scales to be incorporated
                into patterns at higher scales and filled with additional detail.
                Octaves allows to generate patterns within patterns

                Reccomended values: 5 - 20

OCTAVE_SCALE -> Ratio of image upscaling at each octave iteration.
                10 octaves with 1.3 scaling gives dream in 4k and requires 12GB RAM


STEP_SIZE ----> Higher value amplifies dreamyfied patterns intensivity,
                but this doesn't always mean that's gonna look better. Higher
                values work well with greater number of used dream layers.
                Complex-color source images are works better with smaller steps,
                they're not messing colors that much

                Recommanded values: 0.05 - 0.2


TARGET_SIZE --> Downsizing image will give abstraction on higher level
                of source_image composition - you can increase OCTAVE_SCALE
                to make autput bigger. Upsizing image will produce more
                detailed chaos, decrease OCTAVE_SCALE

                You have to respect proportions in original image resolution,
                so dreamified image doesn't stretch.

                    If source_photo is in x:y aspect ratio and x > y:
                        if vertical: x:y
                        if horizontal: y:x

                Higher values of target size will cause dream to attach more
                dreamified patterns to source image detail, but in InceptionV3
                model these patterns will get smaller than in case of smaller value
                Smaller target size will blur source_image,
                but it will get processed faster

'''
