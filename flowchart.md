# Flow-Chart 
Importing modules.

The cv2 module is "Computer Vision" version 2
which is used to capture the video of a person's
face, frame by frame.

Mediapipe is a pre-trained model by Google,
which can be installed with pip.
This particular model is used to detect faces and
irises.

The landmarks corresponding to the locations of the 
iris and pupil in a human eye, are:
> LEFT_IRIS = [474, 475, 476, 477]
> RIGHT_IRIS = [469, 470, 471, 472]
> left_pupil_indices = [468, 469, 470, 471]
> right_pupil_indices = [473, 474, 475, 476]

These landmarks are fed into the neural network in the form
of a python list.

The landmarks do not indicate the physical coordinates
of the iris or pupil as these values vary between
individuals; instead they act as keys telling the 
network what unique features to look for in the face.

> for the full list of landmarks, refer the application
programming interface which is available for many 
languages like JS



The solution for face in the image is obtained using 
mediapipe which uses a "Convolutional Network Model"
architecture.

* The distance from the person to the camera needs
to be constant so, this particular program alerts
the user if he/she is too close or too far, so as to 
get accurate readings.

