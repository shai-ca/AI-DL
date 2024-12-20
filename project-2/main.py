
# import os
# import random

# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt

# from tensorflow.keras import layers, models, datasets
# from tensorflow.keras.applications import MobileNet
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
# from keras.layers.experimental.preprocessing import Resizing
# from tensorflow.keras.optimizers import Adam


# def load_dataset(validation_split=0.2,dec_factor=10):
#     # Load the CIFAR-10 dataset
#     (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()


#     # Reduce the number of images by a factor of dec_factor
#     train_images = train_images[::dec_factor]  # Take every Nth image
#     train_labels = train_labels[::dec_factor]  # Corresponding labels
#     test_images = test_images[::dec_factor]
#     test_labels = test_labels[::dec_factor]


#     # Split the training data into training and validation sets
#     train_images, val_images, train_labels, val_labels = train_test_split(
#         train_images, train_labels, test_size=validation_split, random_state=42)

#     # Normalize pixel values to be between 0 and 1
#     train_images, val_images, test_images = train_images / 255.0, val_images / 255.0, test_images / 255.0
    
#     # Convert labels to one-hot encoding
#     train_labels = to_categorical(train_labels, 10)
#     val_labels = to_categorical(val_labels, 10)
#     test_labels = to_categorical(test_labels, 10)

#     return train_images, train_labels, val_images, val_labels, test_images, test_labels


# def create_model():
#     image_size = 128
#     base_model = MobileNet( input_shape = (image_size,image_size,3),
#                             include_top=False,
#                              weights='imagenet')
#     base_model.trainable = False  # Freeze the base model

#     model = models.Sequential([        
#         Resizing(image_size, image_size, interpolation="nearest", input_shape=train_images.shape[1:]),
#         base_model,
#         layers.GlobalAveragePooling2D(),
#         layers.Dense(10, activation='softmax')
#     ])

#     # Specify the learning rate
    
#     # Instantiate the Adam optimizer with the default learning rate
#     optimizer = Adam()

#     model.compile(optimizer=optimizer,
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])

#     return model

# def plot_train_vs_val_accuracy(history):
#     # Plot training & validation accuracy values
#     #=========== FILL IN THIS CODE SECTION

    
#     #===========
#     return



# # Set the random seeds
# os.environ['PYTHONHASHSEED'] = str(42)  # This variable influences the hash function's behavior in Python 3.3 and later.
# random.seed(42)
# np.random.seed(42)
# tf.random.set_seed(42)

# #Load the dataset
# train_images, train_labels, val_images, val_labels, test_images, test_labels = load_dataset()
# #Create the backbone model that will be used to train
# model = create_model()
# #Do the actual training
# history = model.fit(train_images, train_labels, epochs=5, validation_data=(val_images, val_labels))
# #Evaluate 
# test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
# print(f"Test accuracy: {test_acc}, Test loss: {test_loss}")

# plot_train_vs_val_accuracy(history)

# First install required packages
# pip install numpy
# pip install tensorflow-macos  # This includes the Metal backend

import numpy as np
import tensorflow as tf

# Enable Metal backend
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Example of GPU-accelerated operations
def gpu_matrix_multiplication(size=1000):
    # Create random matrices
    A = np.random.rand(size, size).astype(np.float32)
    B = np.random.rand(size, size).astype(np.float32)
    
    # Convert to tensorflow tensors (will use GPU automatically)
    A_gpu = tf.constant(A)
    B_gpu = tf.constant(B)
    
    # Perform matrix multiplication on GPU
    C_gpu = tf.matmul(A_gpu, B_gpu)
    
    return C_gpu.numpy()

# Benchmark comparison
def benchmark_comparison():
    import time
    
    size = 2000
    
    # CPU timing
    start = time.time()
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    C = np.dot(A, B)
    cpu_time = time.time() - start
    
    # GPU timing
    start = time.time()
    C_gpu = gpu_matrix_multiplication(size)
    gpu_time = time.time() - start
    
    print(f"CPU time: {cpu_time:.2f} seconds")
    print(f"GPU time: {gpu_time:.2f} seconds")
    print(f"Speedup: {cpu_time/gpu_time:.2f}x")

# Example usage
if __name__ == "__main__":
    benchmark_comparison()