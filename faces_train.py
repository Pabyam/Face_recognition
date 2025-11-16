import os
import cv2 as cv
import numpy as np

# List of the celebrities
people=[]
for i in os.listdir(r'C:\Slam Vision\face_recognition\faces'):
    people.append(i)
print(people)

# The path to the files containing the pictures
DIR=r'C:\Slam Vision\face_recognition\faces'

# Load haar_cascade
haar_cascade=cv.CascadeClassifier('haar_face.xml')

features=[]
labels=[]

def create_train():
    for person in people:
        path=os.path.join(DIR, person)
        label=people.index(person)

        for img in os.listdir(path):
            img_path=os.path.join(path,img)
            img_array=cv.imread(img_path)
             # Check if image loaded successfully
            if img_array is None:
                continue
            # Convert in gray
            gray=cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)
            # Detect the faces
            faces_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)

            # Put all the faces detected in features and the index of the person in label
            for (x,y,w,h) in faces_rect:
                faces_roi=gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print("Training Done ......")

#print(f'Length of the features={len(features)}')
#print(f'Length of the labels={len(labels)}')

# Convert to numpy for machine learning
features=np.array(features,dtype='object')
labels=np.array(labels)

# Create the model
face_recognition=cv.face.LBPHFaceRecognizer_create()

# Train the model
face_recognition.train(features,labels)

# Save the model, the features and labels
face_recognition.save('Face_trained.yml')
np.save('feature.npy',features)
np.save('label.npy',labels)