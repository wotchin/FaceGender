# FaceGender
FaceGender SaaS:
Judge the gender of face picture online with deep-learning.
Implement this model with Keras,dlib(>=19.9).
:)

# How to use
## Install python env.
Do recommend installing this environment with Anaconda.
Download Anaconda:
>https://www.anaconda.com/download/#linux

And you should input these in the shell(for linux):

```
conda create -n FaceGender python=3.5
conda install -c menpo dlib=19.9
conda install keras
conda install opencv
conda install flask
```
If you have GPU,you should install cuda(nvidia GPU).And you could input these in the shell:

```
conda install tensorflow-gpu
```
## WARNING
Before you use the project,you should change the PYTHON ENVIRONMENT!

```
source activate FaceGender
```

## Test the FaceGender

```
source activate FaceGender
python test.py
```

## Run the web program
```
python webGender.py
```

# About the project

## models from
>https://github.com/davisking/dlib
https://github.com/oarriaga/face_classification

## At the end
If you want to have a high recognition performance,you should train this model with a more image dataset.
