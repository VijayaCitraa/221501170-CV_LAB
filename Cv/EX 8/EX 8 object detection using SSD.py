import torch 
import torchvision 
from torchvision import transforms 
import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True) 
model.eval()   
transform = transforms.Compose([ 
transforms.ToTensor(), 
]) 
image_path = 'image.jpg'  # Replace with your image path 
image = cv2.imread(image_path) 
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB 
image_resized = cv2.resize(image_rgb, (320, 320))  # Resize to match the SSD input size 
image_tensor = transform(image_resized) 
image_tensor = image_tensor.unsqueeze(0)   
with torch.no_grad(): 
prediction = model(image_tensor) 
boxes = prediction[0]['boxes'].cpu().numpy() 
labels = prediction[0]['labels'].cpu().numpy() 
scores = prediction[0]['scores'].cpu().numpy() 
threshold = 0.5 
boxes = boxes[scores > threshold] 
for box, label in zip(boxes, labels): 
x1, y1, x2, y2 = box.astype(int) 
cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2) 
cv2.putText(image, f'Class {label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
(255, 0, 0), 2) 
plt.figure(figsize=(12, 8)) 
plt.imshow(image_rgb) 
plt.axis('off') 
plt.title("SSD Object Detection") 
plt.show()