# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_data/Network.JPG "Network"
[image2]: ./writeup_data/center_camera.jpg "Center Camera"
[image3]: ./writeup_data/left_camera.jpg "Left Camera"
[image4]: ./writeup_data/right_camera.jpg "Right Camera"
[image5]: ./writeup_data/original_image.jpg "Original Image"
[image6]: ./writeup_data/flipped_image.jpg "Flipped Image"
[image7]: ./writeup_data/plot_loss.png "Training Loss"
[video1]: ./video.mp4 "Final Video"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

The project submission includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* test_run_final.mp4 video recorded demonstrating successful laps around the track
* plot_loss.py a script to plot loss data using output from model.py. Added later for documentation. Could be included in model.py, but wanted to avoid running the entire model again.

Folder test_run_final contains the data saved for the final run
Folder test_run_iterations contains data for some of the prior runs, both successful and unsuccessful, showing the progressive development of the model.
Folder writeup_data contains the data for the writeup


#### 2. Submission includes functional code

My code, implemented via model.py, provides the functional requirements for loading the data, creating the convolutional neural network, and training it.
The model is saved and can be used by drive.py to drive the vehicle autonomously around the track on the Udacity simulator.

```sh
python drive.py model.h5
```

The file model.py can be executed to create a new model and train it. It can be run to re-train the model.

 ```sh
python model.py
```

#### 3. Submission code is usable and readable

The code in model.py is functional and usable. The convolutional neural network is clearly outline and can be used as is, or edited for re-usability.

A generator was not needed for this application as the model was able to load all the data without any memory issues. 

File model.py has comments accompanying the code that describe it's function at each step of the process, and provide clarity for the reader.


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model used for this project was initializing a Keras sequential model. Next a Keras Lambda Layer was used to normalize the data. Values were transformed with magnitude between 0 to 1, and then centerd around 0.

It contains two convolutional network layers. They can be found on the lines 128 and 134 of the model.py file respectively
```
model.add(Convolution2D(6,5,5, activation='relu'))
```

Each one of them has a depth of 6 and kernel size of 5x5. They also include an activation using rectified linear unit. The ReLU provides the non-linearity to the model.



#### 2. Attempts to reduce overfitting in the model

In order to prevent overfitting, a validation subset was used for evaluation of loss, besides the one training data. The data set was split 80-20 for training and validation respectively. 
Also the the dataset was shuffled to ensure that the same sets, or order is not repeated. It prevents the model from trying to learn the sequence of inputs.

Code line 148 for reference

```
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
```

Performance on the simulator track indicates that the model can operate on new data correctly, consistently staying on the track for multiple laps.



#### 3. Model parameter tuning

In the compilation phase an adam optimizer was used, obviating the need for manual tuning. Mean squared error was used for loss loss calculation.

Other training parameters that were tuned are number of EPOCHS, correction factor, and crop size. The approach has been described in detail in the training documentation section.


#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 


The training data set provided with the project resources was the primary source of training data. Multiple attempts were made at running the car manually on the track, but the data generated with that was not good enough due to lack of good control over the car.
The provided datset, after using augmentation and the two extra camera angles, generated enough data to train the model successfully.

Augmentation, by lateral inversion, provided enough samples for right turns since most of the track needs steering to the left. Left and right camera angles, coupled with artificial steering data, simulate conditions for maneuvering needed to prevent from going off-track. Again more details on this are provided in the Documentation section next. 



### Architecture and Training Documentation

#### 1. Solution Design Approach


__Steps followed to arrive at final model and code__

One of the challenges of this project was that the code could only be tested on the GPU enabled workspace. Since the time on that is limited, writing the code and debugging wherever possible was to be done in the normal workspace. This presented the need of incremental and modular approach to get build the model network, before commencing parameter tuning for training.
Described below are some steps that were taken to arrive at the final model

1. Read and load the training data, verify execution
    Read the csv and images data. Print sample data lilke the arrays sizes to check if loaded data is as expected.
    Started with 2 simple basic flatten model, and fully connected layer
    Test the training pipleine for execution. Accuracy is not a concern yet. 
    Did a vehicle testrun on the simulator, it starts but keeps turning to go off track and runs in circles

2. Correct the image formatting
    Edited the image reading from cv2 to ndimage library, since drive.py uses RGB. 
    Some improvement in vehicle run, does not run in circles but goes off track right after the start.
    Now the execution proces was understood, and the pipleine proved. Time to add more layers

3. Started to implement the actual network architecture now. Added Convolution and Maxpooling layers to the network
    The final model overall resembles Lenet
    Chose this architecture for its proven capability for image processing. I have also used it in earlier project/ labs with good results. 
    Tested the pipeline, more focus on the accuracy of the model. Car drives better, but still goes off track on tricky turns.

4. Following steps added incremental improvement to performance, but the car still would be erratic at one or two specific turns
    Added lambda function for normalizing images
    Added the left and right images to the image array for training
    Implemented Augmented data to flip images and measurement
    Cropping the images to include only relevant data from each frame

5. Model parameters tuning process
    Tuning of some model parameters was done which yeilded the final solution
    This part is decribed in detail in the training process documenatation section

After the tuning process was successful, the car was able to run around the track consistenly staying on road, for multiple loops. A video was recorded for the final run.


__Problems Faced__

Two sections of the track where the car would get confused and go off track
    Turn after bridge where markings were not prominent on the right side
    Another spot where vehicle did not see the markings on the right side and went into the grass
These problems were resolved in the final model tuning    
    The parameters EPOCHS, correction factor, and crop size, were tuned as described in the documentation section.



#### 2. Final Model Architecture


As mentioned before, the final architecture is based on the LeNet architecture. This was a preferred choice since it has proved to work on images in earlier project and labwork

The final model structure is described below
* Layer 1: Lambda
* Layer 2: Cropping2D Layer
* Layer 3: Convolution 2D Layer: Depth 6, Width 5, Height 5, Defaults- Padding 'valid', Stride (1, 1), Activation ReLU
* Layer 4: MaxPooling2D Layer: Default pool size, 2x2
* Layer 5: Convolution 2D Layer: Depth 6, Width 5, Height 5, Defaults- Padding 'valid', Stride (1, 1), Activation ReLU
* Layer 6: MaxPooling2D Layer: Default pool size, 2x2
* Layer 7: Flatten Layer: Flatten the Network
* Layer 8: Dense: Fully connected layer


It's implemention in the code is from line 118 to line 144

```
model = Sequential()

# Use lambda function to linearize the data
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

# Crop the image from top and bottom
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))

# Add a convolutional Layer with ReLU Activation
model.add(Convolution2D(6,5,5, activation='relu'))

# Add a maxpool layer - Default 2x2
model.add(MaxPooling2D())

# Convolution Layer with ReLU Activation
model.add(Convolution2D(6,5,5, activation='relu'))

# Maxpooling Layer
model.add(MaxPooling2D())

# Flatten the model, followed by fully connected layer
model.add(Flatten())
model.add(Dense(1))

# Compile the model. Use Adam optimizer for learning
model.compile(loss='mse', optimizer='adam')

```

Below is a more visual representation of the model. Although, some details may not be clear owing to the large size of the model.


![alt text][image1]


#### 3. Creation of the Training Set & Training Process

This section will be described in two parts. First the creation of dataset and then the training process.


#### 3.1 Creation of Training Set

After multiple attempts to collect training data by manually running the vehicle on the Udacity simulator, I realized that using the keyboard controls does not result in a good enough driving behavior. The data recorded had vehicle not being in good control most of the times, and thus would not be useful for providing accurate training.
This led to my decision of using the driving data provided with project resources. Extrapolating the data set could yield more training data. By adding the left and right pictures, and some augmentation I could make the set sufficiently large to train the model. These steps are described in detail below

Original Data set size (Center Images): __8036__


__Appending left and right camera images__

Since we have images from 3 different cameras, adding them to the training set to broaden our scenarios would be useful.
Adding the left and right images tripled the size of original dataset size to: __24108__

A correction factor is applied to estimate how much steering angle would be needed to orient the center camera to that viewing angle. This value is a part of the parameters that were tuned during training to arrive at the final result.

Ref code: Line 67-74

```
# Define steering correction value for the left and right images
corr = 0.25

# Get the steering measurement from the csv data
measurement = float(row[3])

# Use the above two values and get the manipulated data for left and right images
# Append all 3 to the measurement list  
measurements.extend([measurement, measurement+corr, measurement-corr])
```

An example of the same vehicle position being captured by 3 cameras is given below.

Center Image

![alt text][image2]


Left Image

![alt text][image3]


Right Image

![alt text][image4]



__Augmentation__

Augmentation was done by laterally inverting the images. In principle, it would equivalent to driving the vehicle on the track in the opposite direction - clockwise.
Applying augmentation doubled the dataset again, bringing the number of training samples to: __48216__

Numpy function `fliplr()` was used to flip the imgage left to right.
Ref code: Line 84

`flipped_image = np.fliplr(image)`


An example of an augmented image is given below
Original

![alt text][image5]



Augmented

![alt text][image6]


Steering measurements were multiplied by -1 to match for the inverted image. This is because the steering angles are centered about 0. 
Ref code: Line 90
`aug_measurements.append(measurement * -1)`


__Final Datset size: 48216__ 

With a 80-20 split, this would mean 38572 images for training and 9644 for validation set. This seems to be enough data, both quantity and quality-wise for training.



#### 3.2. Training Process

__Training-Validation Split__
To prevent overfitting to the training set, a training-validation split of 80-20 was used. This ensures that the model can be evaluated for performance on images it has not seen before.


__Shuffle__
Shuffling the data set also ensures that the sequence of images gets altered each time. This prevents a possibility of the network learning the sequence of images in the training set, and making predictions based on that.


__EPOCHS__
Number of EPOCHS was varied between 3 and 7 for the training. In some cases, increasing the number of EPOCHS was counter-productive as it would start increasing the loss on the validation set. This was possibly due to overfitting on later EPOCHS.
The final result was achieved on 5 EPOCHS.


__Correction factor__
The correction factor is the estimate of the steering measurement for left and right cameras. This value is also critical in tuning because
the model would calibrate how much to steer on turns based on it.

It is define on line 68 of the code
`corr = 0.25`

Correction factor was varied between 0.15 to 0.30, in increments of 0.05.
The final value that yielded best results for this model was 0.25.


__Image Crop Size__

Another technique that helped improve performance was cropping the input image to only the part that contains the useful details. That means cropping out the top of the image with the sky and scenery, and the bottom part with the hood. This would lead the model learning only on the relevant details of the road section.

This value is defined in the cropping layer of the model on line 125. It is in the order `cropping = (top, bottom)`
`model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))`

Different combinations of values were used duing the trials, ranging from (50, 20) to (70, 25) in increments on 5 pixel for both the top and bottom crop. Value of (70, 25) seemed to yeild best results.


__Trials and Tuning__

Sample data from a training attempt is shown below. Progression of the error reduction, and the error difference between the training and validation set is an indication of the roubustness and generality of the model.

Output from model.py showing the epochwise progress on loss, for training and validation data.

```sh
38572/38572 [==============================] - 50s 1ms/step - loss: 0.0249 - val_loss: 0.0223
Epoch 2/5
38572/38572 [==============================] - 48s 1ms/step - loss: 0.0176 - val_loss: 0.0217
Epoch 3/5
38572/38572 [==============================] - 49s 1ms/step - loss: 0.0153 - val_loss: 0.0178
Epoch 4/5
38572/38572 [==============================] - 48s 1ms/step - loss: 0.0137 - val_loss: 0.0188
Epoch 5/5
38572/38572 [==============================] - 48s 1ms/step - loss: 0.0132 - val_loss: 0.0182
```

![alt text][image7]



### Simulation

#### 1. Car Able to Navigate Correctly

The car drives safely through the track with final model created and trained with the solution.
It ran multiple laps back to back, and does so consistently without going off-track. As stated in requirements, tires stay within the track lines, operating safely throughout conisdering passenger safefty.

The video of the final run (a little over 3 laps) is available below. 

__Video reference__

![alt text][video1]


