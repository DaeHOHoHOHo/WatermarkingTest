import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.datasets import cifar100
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def residual_block(x, filters):
    shortcut = x
    x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), padding='same')(shortcut)
    
    x = layers.add([x, shortcut])
    return x

def create_residual_cnn_model():
    inputs = layers.Input(shape=(32, 32, 3))
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)

    x = residual_block(x, 64)
    x = layers.MaxPooling2D((2, 2))(x)

    x = residual_block(x, 128)
    x = layers.MaxPooling2D((2, 2))(x)

    x = residual_block(x, 256)
    x = layers.MaxPooling2D((2, 2))(x)

    x = residual_block(x, 512)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(32 * 32, activation='sigmoid')(x)
    outputs = layers.Reshape((32, 32))(x)
    model = models.Model(inputs, outputs)
    return model

model = create_residual_cnn_model()
optimizer = optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

x_train_gray = x_train[..., 0]
x_test_gray = x_test[..., 0]

lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

batch_size = 128
epochs = 100
history = model.fit(
    x_train, x_train_gray,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test, x_test_gray),
    callbacks=[lr_scheduler, early_stopping]
)
model.save('high_accuracy_cnn_watermark_model.h5')
print("모델 학습 및 저장 완료")
