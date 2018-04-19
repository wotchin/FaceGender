#!/usr/bin/env python
import gender2 as gender
import sys

detector = gender.get_face_detector("./trained_models/face/mmod_human_face_detector.dat","./trained_models/face/shape_predictor_68_face_landmarks.dat")

classifier = gender.get_gender_classifier("./trained_models/gender/alex.27-0.946816.hdf5")

if len(sys.argv) > 1 :
    faces = detector(sys.argv[1])
else:
    faces = detector("test.jpeg")
print(classifier(faces[0]))

