import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(0)
import cv2
import rasterio

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError

####################### Functions to load and prepare satellite images ############################

def load_hr_images(aoi):
    """
    Load high-resolution images from the dataset.

    Args:
    - aoi: Name of the AOI folder.

    Returns:
    - List of loaded images.
    """
    images = []
    hr_image_path = os.path.join('dataset/hr_dataset', aoi, f"{aoi}_ps.tiff")
    if os.path.isfile(hr_image_path):
        try:
            with rasterio.open(hr_image_path) as src:
                image = src.read([1, 2, 3])  # Read RGB channels if available
                image = np.moveaxis(image, 0, -1)  # Convert to HWC format
                images.append(image)
        except Exception as e:
            print(f"Warning: Failed to read image {hr_image_path} with error {e}")
    else:
        print(f"Warning: High-resolution image path {hr_image_path} does not exist")
    return images

def preprocess_image(image, target_size):
    """
    Preprocess the image for SRCNN.

    Args:
    - image: Input image.
    - target_size: Tuple of target size (width, height).

    Returns:
    - Preprocessed image.
    """
    image = cv2.resize(image, target_size)
    if np.max(image) != 0:
        image = image / np.max(image)

    return image

def process_aois(aoi_names, target_size_hr, target_size_lr):
    """
    Load and preprocess images for multiple AOIs.

    Args:
    - aoi_names: List of AOI base names.
    - target_size_hr: Tuple of target size (width, height) for high-resolution images.
    - target_size_lr: Tuple of target size (width, height) for low-resolution images.

    Returns:
    - Dictionary with AOI names as keys and tuple of (LR images, HR images) as values.
    """
    data = {}
    for aoi_base_name in aoi_names:
        print('processing:', aoi_base_name)

        # print(f"Processing AOI: {aoi_base_name}")
        # lr_images = load_lr_images(aoi_base_name, num_revisits)
        hr_images = load_hr_images(aoi_base_name)

        # Preprocess images
        # lr_images = [preprocess_image(img, target_size_lr) for img in lr_images]
        hr_images = [preprocess_image(img, target_size_hr) for img in hr_images]

        # compress hr_images at scale of 5 to generate low res images
        lr_images = []
        for img in hr_images:
            lr_images.append(cv2.resize(img, target_size_lr, interpolation=cv2.INTER_AREA))

        # Convert lists to numpy arrays
        lr_images = np.array(lr_images)
        hr_images = np.array(hr_images)

        data[aoi_base_name] = (lr_images, hr_images)

    return data


############################ Functions to prepare training data ###################################
HR_DATASET_PATH = 'dataset/hr_dataset'
# LR_DATASET_PATH = 'dataset/lr_dataset'

def prepare_training_data():
    
    print('start preparing training data')
    print('clean up raw data')
    # clean the dataset
    metadata = pd.read_csv("dataset/metadata.csv")
    metadata.rename(columns={metadata.columns[0]: 'aoi_name'}, inplace=True)
    metadata.drop_duplicates(subset=['aoi_name'], keep='first', inplace=True)
    
    print('selecting AOIs')
    # Select AOIs with landcover and cloud cover less than 0.05
    df_select = metadata[(metadata['aoi_name'].str.contains('Landcover')) & (metadata['cloud_cover'] < 0.05)]

    # Select 1000 AOIs for model training
    landcover_folders = []
    aoi_names_set = set(df_select['aoi_name'])
    for folder_name in os.listdir(HR_DATASET_PATH):
        if folder_name in aoi_names_set:
            landcover_folders.append(folder_name)
        # Stop when 1000 folders have been added
        if len(landcover_folders) >= 1000:
            break
    
    print('loading images')
    # upscale the low-resolution images
    target_size_hr = (500, 500)
    target_size_lr = (100, 100)
    sample_images = process_aois(landcover_folders, target_size_hr, target_size_lr)

    # Collect all high-res and low-res images from sample_images
    all_high_images = []
    all_low_images = []
    
    print('finish loading images')

    for aoi_name in sample_images:
        all_low_images.extend(sample_images[aoi_name][0])  # Low-resolution images
        all_high_images.extend(sample_images[aoi_name][1])  # High-resolution images
    
    print('upscale low res images')
    # upscale the 100 x 100 low resolution image to 500 x 500 using bicubic interpolation
    for i in range(len(all_low_images)):
        upscaled_img = cv2.resize(all_low_images[i], (500, 500), interpolation=cv2.INTER_CUBIC)
        all_low_images[i] = upscaled_img

    # Convert lists to numpy arrays
    all_high_images = np.array(all_high_images)
    all_low_images = np.array(all_low_images)
    
    print('spliting data into train, validation, and test')
    # Split the data into train, validation, and test sets
    train_high_image = all_high_images[:800]
    train_low_image = all_low_images[:800]

    validation_high_image = all_high_images[800:900]
    validation_low_image = all_low_images[800:900]

    test_high_image = all_high_images[900:]
    test_low_image = all_low_images[900:]
    
    print('finish data preparation')
    return (
        train_high_image, train_low_image, validation_high_image, validation_low_image, test_high_image, test_low_image
    )


############################ Implementation of the SRCNN model ###################################
def srcnn_model():
    model = Sequential([
        Conv2D(128, (9, 9), activation='relu', padding='same', input_shape=(500, 500, 3)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        Conv2D(32, (1, 1), activation='relu', padding='same'),
        Conv2D(3, (5, 5), activation='relu', padding='same')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())

    return model


############################# Main function to train the SRCNN model ###############################
if __name__ == '__main__':
    # Prepare the training data
    train_high_image, train_low_image, validation_high_image, validation_low_image, test_high_image, test_low_image = prepare_training_data()
    
    print('define model')
    # Create the SRCNN model
    model = srcnn_model()
    
    print('training starts')
    # Train the model
    model.fit(
        train_low_image, train_high_image,
        batch_size = 10,
        epochs = 100,
        validation_data=(validation_low_image, validation_high_image),
        callbacks=[EarlyStopping(monitor='val_loss', patience=2)]
    )

    print('finished training and saving the weights')

    # Save the model
    model.save('srcnn_weights.h5')
