# Face Recognition

Face recognition problems commonly fall into two categories:
Face Verification - "is this the claimed person?". For example, at some airports, you can pass through customs by letting a system scan your passport and then verifying that you (the person carrying the passport) are the correct person. A mobile phone that unlocks using your face is also using face verification. This is a 1:1 matching problem.
Face Recognition - "who is this person?". 

FaceNet learns a neural network that encodes a face image into a vector of 128 numbers. By comparing two such vectors, you can then determine if two pictures are of the same person.

# Steps
1- Implement the triplet loss function

2- Use a pretrained model to map face images into 128-dimensional encodings

3- Use these encodings to perform face verification and face recognition

#
Face verification solves an easier 1:1 matching problem; face recognition addresses a harder 1:K matching problem.
The triplet loss is an effective loss function for training a neural network to learn an encoding of a face image.
The same encoding can be used for verification and recognition. Measuring distances between two images' encodings allows you to determine whether they are pictures of the same person.

# References:
Florian Schroff, Dmitry Kalenichenko, James Philbin (2015). FaceNet: A Unified Embedding for Face Recognition and Clustering
Yaniv Taigman, Ming Yang, Marc'Aurelio Ranzato, Lior Wolf (2014). DeepFace: Closing the gap to human-level performance in face verification
The pretrained model we use is inspired by Victor Sy Wang's implementation and was loaded using his code: https://github.com/iwantooxxoox/Keras-OpenFace.
Our implementation also took a lot of inspiration from the official FaceNet github repository: https://github.com/davidsandberg/facenet
