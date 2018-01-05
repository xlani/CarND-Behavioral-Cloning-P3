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

[image1]: ./training_loss.png "Training & Validation Loss"
[image2]: ./examples/center_image.png "Example image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup.md summarizing the results

#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py & model.h5 files, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

My main source for the code is the [Q&A video for project 3](https://review.udacity.com/#!/rubrics/432/view).

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is an adjusted form of the [NVIDIA E2E learning model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) and consists of five convolution layers with 5x5 & 3x3 filter sizes and depths between 24 and 48 (model.py lines 68-72).

The model includes the RELU activation function to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 66).

#### 2. Attempts to reduce overfitting in the model

The model contains one dropout layer in order to reduce overfitting (model.py lines 75). In addition I chose an early stopping approach (only 3 epochs), because of my good experience with that at the last project.

The model was trained and validated on different data sets (validation split = 80/20) to get some insight to the training regarding overfitting & underfitting (code line 83). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 83).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. It was used only one lap of driving, where the main focus was to stay centered in the road. I used the mouse to steer the car during recording the training data.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I started with the LeNet architecture and run the simulator to see how well the car was driving around track one. The car actually went left off the track right after the start. So I implemented using left and right camera images and flipping all images as well for training.

After that it still went off the track after the first sharp corner, so I decided to switch over to the more powerful NVIDIA E2E learning architecture. That actually lead right to a model that could drive autonomously and properly around the track.

I had a look at the training and validation loss (see final results in the following picture).

![alt text][image1]

The validation loss is higher than the training loss and increases a little bit over the three epochs. I tried a little bit around with a dropout layer. But I actually did not manage to bring the validation loss to decrease over the epochs.

Nevertheless at all tries the car was able to autonomously drive around track one.

#### 2. Final Model Architecture

The final model architecture is the [NVIDIA E2E learning model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) with one additional dropout layer after the first fully connected layer.

#### 3. Creation of the Training Set & Training Process

I recorded one lap on track one using center lane driving. Here is an example image of the data set: 

![alt text][image2]

To augment the data set, I used the images of the left and right camera as well and also flipped images and steering angles to increase robustness of the model.

After the collection process, I had 7248 number of data points. I then preprocessed this data by normalization with a Keras lambda layer. I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.

To avoid overfitting I used an early stopping approch and trained only for 3 epochs.
