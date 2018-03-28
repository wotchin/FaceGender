import gender
from time import time
detector = gender.get_face_detector("./trained_models/face/mmod_human_face_detector.dat","./trained_models/face/shape_predictor_68_face_landmarks.dat")

classifier = gender.get_gender_classifier("./trained_models/gender/simple_CNN.81-0.96.hdf5")

last = time()
faces = detector("./test.jpge")
print(classifier(faces[0]))
print(time() - last)
