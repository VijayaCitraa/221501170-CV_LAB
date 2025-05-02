import cv2 
import matplotlib.pyplot as plt 
face_cascade =cv2.CascadeClassifier(cv2.data.haarcascades +  'haarcascade_frontalface_default.xml')
image_path = 'face_sample.jpg'  # Replace with your image path 
img = cv2.imread(image_path) 
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5) 
for (x, y, w, h) in faces: 
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2) 
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
plt.figure(figsize=(8, 6)) 
plt.imshow(img_rgb) 
plt.axis('off') 
plt.title('Detected Faces') 
plt.show()