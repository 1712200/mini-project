import cv2
import os
import glob
import shutil
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.models import Model
import numpy as np
import cv2
import os
import glob
import shutil


# Input folders
input_folders = ['no', 'yes']

# Output folders for resized images
output_folders = ['resized_images_no', 'resized_images_yes']

# Remove existing output folders and create new ones
for folder in output_folders:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

# Resize images, apply sharpening filter, and save them to output folders
for folder, output_folder in zip(input_folders, output_folders):
    for img_path in glob.glob(os.path.join(folder, "*.png")):
        image = cv2.imread(img_path)
        img_resized = cv2.resize(image, (224, 224))

        # Convert to RGB
        img_resized_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # Apply sharpening filter
        kernel = np.array([[-1,-1,-1],
                           [-1, 9,-1],
                           [-1,-1,-1]], dtype=np.float32) # Sharpening kernel
        img_sharpened = cv2.filter2D(img_resized_rgb, -1, kernel)

        # Save the resized and sharpened RGB image to the output folder
        filename = os.path.basename(img_path)
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, cv2.cvtColor(img_sharpened, cv2.COLOR_RGB2BGR))
        
        


# Input folders
input_folders = ['resized_images_no', 'resized_images_yes']

# Output folders for feature vectors
output_folders = ['feature_vectors_no', 'feature_vectors_yes']

# Remove existing output folders and create new ones
for folder in output_folders:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

# Load VGG16 model pre-trained on ImageNet dataset
base_model = VGG16(weights='imagenet', include_top=True)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)  # Get feature vector from fc2 layer

# Extract features and save them to output folders
for folder, output_folder in zip(input_folders, output_folders):
    for img_path in glob.glob(os.path.join(folder, "*.png")):
        # Load and preprocess image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Get feature vector
        features = model.predict(img_array)

        # Save feature vector
        filename = os.path.basename(img_path).split('.')[0]  # Extract filename without extension
        np.savetxt(os.path.join(output_folder, f"{filename}.txt"), features)