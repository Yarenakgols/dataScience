from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import os
import cv2
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from skimage.transform import resize
from sklearn.model_selection import train_test_split

base_dir = '/content/drive/MyDrive/Yüksek Lisans/PNG'
images_dir ='/content/drive/MyDrive/Yüksek Lisans/PNG/png_image_path'
masks_dir = '/content/drive/MyDrive/Yüksek Lisans/PNG/png_mask_path'

images_listdir = os.listdir(images_dir)
masks_listdir = os.listdir(masks_dir)
random_images = np.random.choice(images_listdir, size = 9, replace = False)

print("İmages : ",len(images_listdir))
print("Masks : ",len(masks_listdir))

from google.colab import drive
drive.mount('/content/drive')

image_size=512
input_image_size=(512,512)

def read_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (image_size, image_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

import cv2
number=10
rows = 3
cols = 3
fig, ax = plt.subplots(rows, cols, figsize = (6,6))
for i, ax in enumerate(ax.flat):
    if i < len(random_images):
        img = read_image(f"{images_dir}/{random_images[i]}")
        ax.set_title(f"{random_images[i]}")
        ax.imshow(img)
        ax.axis('off')

fig, ax = plt.subplots(rows, cols, figsize = (6,6))
for i, ax in enumerate(ax.flat):
    if i < len(random_images):
        file=random_images[i]
        if os.path.exists(os.path.join(masks_dir,file)):
            img = read_image(f"{masks_dir}/{file}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ax.set_title(f"{random_images[i]}")
            ax.imshow(img)
            ax.axis('off')
        else:
            print('not exist')

""" bu kod, belirtilen dizinlerden görüntü ve maske dosyalarını okumayı, ön işlemeyi ve bunları numpy dizilerinde saklamayı amaçlar"""

MASKS=np.zeros((1,image_size, image_size, 1), dtype=bool)
IMAGES=np.zeros((1,image_size, image_size, 3),dtype=np.uint8)

for j,file in enumerate(images_listdir[0:number]):
    try:
        image = read_image(f"{images_dir}/{file}")
        image_ex = np.expand_dims(image, axis=0)
        IMAGES = np.vstack([IMAGES, image_ex])
        mask = read_image(f"{masks_dir}/{file}")
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = mask.reshape(512,512,1)
        mask_ex = np.expand_dims(mask, axis=0)
        MASKS = np.vstack([MASKS, mask_ex])
    except:
        print(file)
        continue

images=np.array(IMAGES)[1:number+1]
masks=np.array(MASKS)[1:number+1]
print(images.shape,masks.shape)

from sklearn.model_selection import train_test_split

images_train, images_test, masks_train, masks_test= train_test_split(images,masks, test_size=0.2, random_state=20)

print("Images_train : ",len(images_train),"Masks_train : " ,len(masks_train))

import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPool2D,
    Conv2DTranspose,
    Concatenate,
    Input,
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19

def double_conv_block(input, num_filters):
    x = Conv2D(num_filters, (3, 3), padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = double_conv_block(x, num_filters)
    return x

def build_vgg19_unet(input_shape):

    inputs = Input(input_shape)

    vgg19 = VGG19(include_top=False, weights="imagenet", input_tensor=inputs)

    s1 = vgg19.get_layer("block1_conv2").output
    s2 = vgg19.get_layer("block2_conv2").output
    s3 = vgg19.get_layer("block3_conv4").output
    s4 = vgg19.get_layer("block4_conv4").output

    b1 = vgg19.get_layer("block5_conv4").output

    d1 = decoder_block(b1, s4, 128)
    d2 = decoder_block(d1, s3, 64)
    d3 = decoder_block(d2, s2, 32)
    d4 = decoder_block(d3, s1, 16)

    #outputs = Conv2D(1, (1, 1), padding="same", activation="softmax")(d4)

    #outputs = Conv2D(1, (1, 1), padding="same", activation="tanh")(d4)

    outputs = Conv2D(1, (1, 1), padding="same", activation="relu"")(d4)

    model = Model(inputs, outputs, name="VGG19_U-Net")
    return model

if __name__ == "__main__":
    input_shape = (256, 256, 3)
    model = build_vgg19_unet(input_shape)
    model.summary()

from tensorflow.keras.optimizers import Adam

input_shape = (512, 512, 3)
model = build_vgg19_unet(input_shape)

model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    images_train,
    masks_train,
    validation_split=0.1,
    batch_size=2,
    epochs=20
)

model.save('vgg19_unet.h5')

train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = range(1, len(train_accuracy) + 1)

plt.plot(epochs, train_accuracy, 'bo', label='Eğitim Doğruluğu')
plt.plot(epochs, val_accuracy, 'b', label='Validation Doğruluğu')
plt.title('Eğitim ve Validation Doğruluk Değerleri')
plt.xlabel('Epochs')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()

import numpy as np

predictions = model.predict(images_test)

binary_predictions = (predictions > 0.5).astype(np.uint8)
binary_masks_test = (masks_test > 0.5).astype(np.uint8)

def jaccard_index(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    jaccard = intersection / union
    return jaccard

jaccard_scores = []
for i in range(len(binary_predictions)):
    jaccard = jaccard_index(binary_masks_test[i], binary_predictions[i])
    jaccard_scores.append(jaccard)

# Ortalama Jaccard İndeksini hesaplayın
average_jaccard = np.mean(jaccard_scores)
print("Ortalama Jaccard İndeksi:", average_jaccard)

def show_result(idx, og, unet, target, p):

    fig, axs = plt.subplots(1, 3, figsize=(12,12))
    axs[0].set_title("Original "+str(idx) )
    axs[0].imshow(og)
    axs[0].axis('off')

    axs[1].set_title("U-Net: p>"+str(p))
    axs[1].imshow(unet)
    axs[1].axis('off')

    axs[2].set_title("Ground Truth")
    axs[2].imshow(target)
    axs[2].axis('off')

    plt.show()

unet_predict = model.predict(images_test)

len(images_test)

r1,r2,r3,r4 = 0.5, 0.75, 0.95 , 0.9999

unet_predict1 = (unet_predict > r1).astype(np.uint8)
unet_predict2 = (unet_predict > r2).astype(np.uint8)
unet_predict3 = (unet_predict > r3).astype(np.uint8)
unet_predict4 = (unet_predict > r4).astype(np.uint8)

show_test_idx = random.sample(range(len(unet_predict)), 2)
for idx in show_test_idx:
    show_result(idx, images_test[idx], unet_predict1[idx], masks_test[idx], r1)
    show_result(idx, images_test[idx], unet_predict2[idx], masks_test[idx], r2)
    show_result(idx, images_test[idx], unet_predict3[idx], masks_test[idx], r3)
    show_result(idx, images_test[idx], unet_predict4[idx], masks_test[idx], r4)
    print()

import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score

y_true = (masks_test > 0).astype(int)
y_preds = [(unet_predict1 > r1).astype(int), (unet_predict2 > r2).astype(int),
           (unet_predict3 > r3).astype(int), (unet_predict4 > r4).astype(int)]

thresholds = [r1, r2, r3, r4]
iou_scores = []

for i, threshold in enumerate(thresholds):
    y_pred = y_preds[i]
    iou = jaccard_score(y_true.reshape(-1), y_pred.reshape(-1), average='binary')
    iou_scores.append(iou)
    print(f"IoU (Jaccard Score) for Threshold {i+1}: {iou:.4f}")

plt.plot(thresholds, iou_scores, marker='o', linestyle='-')
plt.xlabel('Eşik Değeri')
plt.ylabel('IoU (Jaccard Score)')
plt.title('Farklı Eşik Değerleri için IoU Değerleri')
plt.grid(True)
plt.show()
