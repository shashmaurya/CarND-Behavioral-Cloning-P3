
### This file is organized into two sections
### First section loads the data and prepares it
### Second section builds the keras model and trains it


### SECTION 1: Load the training data, and prepare it for use with the training model
###
# import the needed libraries
import csv
import numpy as np
import cv2

from scipy import ndimage

def populate_data():
    """
    This function reads the driving_log csv file, and imports the data to memory
    Reads the images, and the measurements to be used later for the model training
    Left and right camera images are also included in the training data, with 
    artificial data for their respective steering angles
    Applies lateral inversion to all the images and inverts steering angles for 
    augmented images 
    Return the set of image data as X array and steering angles as Y array for use 
    with the keras model in the next section
    
    """
    
    # Open the csv file and read the data on each line and store in the list
    csv_data = []
    with open('../../../opt/carnd_p3/data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            csv_data.append(line)
    print("Finished reading CSV file")
    
    ## Using the paths in the lines list, read the store images, measumrement data in a list
    # Initialize the lists that will store the data
    images = []
    measurements = []
    aug_images = []
    aug_measurements = []
    image_dir = '../../../opt/carnd_p3/data/IMG/'       # Image directory
    
    header_flag = 0         # Header flag to skip the first line with column names
    for row in csv_data:    # Process each row of data from the csv
        if header_flag == 0:
            print("Skipping Header")
            header_flag = 1
            continue

        # Get the correct file paths to the images
        filepath_center = image_dir + row[0].split('/')[-1]
        filepath_left = image_dir + row[1].split('/')[-1]
        filepath_right = image_dir + row[2].split('/')[-1]
        
        # Read the three images using paths defined above
        # Use ndimage which gets images in RGB, as used by drive.py
        image_center = ndimage.imread(filepath_center)
        image_left = ndimage.imread(filepath_left)        
        image_right = ndimage.imread(filepath_right)
        
        # Add the 3 new images to the main list
        images.extend([image_center, image_left, image_right])
        
        # Define steering correction value for the left and right images
        corr = 0.25

        # Get the steering measurement from the csv data
        measurement = float(row[3])
        
        # Use the above two values and get the manipulated data for left and right images
        # Append all 3 to the measurement list  
        measurements.extend([measurement, measurement+corr, measurement-corr])
    
    
    # Build an augmented data set, that included inverted images too
    # Iterate through the lists built so far, and augment each
    for image, measurement in zip(images, measurements):
        # First append the original
        aug_images.append(image)
        
        # Flip the image and add to the augmented list
        flipped_image = np.fliplr(image)
        aug_images.append(np.fliplr(image))

        # Add the original measurement, and the inverted one for the flipped image
        # to the augmented list
        aug_measurements.append(measurement)
        aug_measurements.append(measurement * -1)
    
    # Print a confirmation    
    print("Finished reading Image data")
    
    # Converted the final augmented lsits to arrays for use with keras
    X_train = np.array(aug_images)
    y_train = np.array(aug_measurements)    
    
    # return the
    return X_train, y_train

# Call the function populate_data to get the data arrays 
X_train, y_train = populate_data()
print("Data populated. Starting training")



### Section 2: Build The Model and Train it
### Save the trained model as model.h5 to test on the vehicle
###
# Import the necessary libraries
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D 
from keras.layers import Cropping2D

# Initalize a sequential model
model = Sequential()

# Layer 1: Use lambda function to normalize the data
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

# Layer 2: Crop the image from top and bottom
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))

# Layer 3: Add a convolutional Layer with ReLU Activation
model.add(Convolution2D(6,5,5, activation='relu'))

# Layer 4: Add a maxpool layer - Default 2x2
model.add(MaxPooling2D())

# Layer 5: Convolution Layer with ReLU Activation
model.add(Convolution2D(6,5,5, activation='relu'))

# Layer 6: Maxpooling Layer
model.add(MaxPooling2D())

# Layer 7, 8: Flatten the model, followed by fully connected layer
model.add(Flatten())
model.add(Dense(1))

# Compile the model. Use Adam optimizer for learning
model.compile(loss='mse', optimizer='adam')

# Run training on the model, with 80-20 split between the training and validation data
# Train for 5 EPOCHS
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

# Save the model for use with drive.py to run the car
model.save('model.h5')

