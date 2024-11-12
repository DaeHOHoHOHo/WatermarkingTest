import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

model = load_model('high_accuracy_cnn_watermark_model.h5')

def preprocess_image(image_path):
    image = Image.open(image_path).resize((32, 32))
    image = np.array(image) / 255.0
    return image

def add_watermark_to_image(model, image_path, watermark_bit=1):
    image = preprocess_image(image_path)
    image_expanded = np.expand_dims(image, axis=0)
    
    importance_map = model.predict(image_expanded)[0]
    
    watermarked_image = np.copy(image * 255).astype(np.uint8)
    for i in range(watermarked_image.shape[0]):
        for j in range(watermarked_image.shape[1]):
            for k in range(3):
                if importance_map[i, j] > 0.5: 
                    watermarked_image[i, j, k] = (watermarked_image[i, j, k] & ~1) | watermark_bit

    return watermarked_image

def display_images(original, watermarked):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original.astype(np.uint8))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(watermarked)
    axes[1].set_title("Watermarked Image")
    axes[1].axis("off")

    plt.show()

image_path = 'path_to_your_image.jpg'
original_image = preprocess_image(image_path) * 255
watermarked_image = add_watermark_to_image(model, image_path)
display_images(original_image, watermarked_image)
