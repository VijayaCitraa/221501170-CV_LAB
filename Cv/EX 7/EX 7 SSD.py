import cv2 
import tensorflow as tf 
import matplotlib.pyplot as plt 
# Load pre-trained SSD model 
model = tf.saved_model.load('ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model') 
# Load test image 
image_path = 'car_scene.jpg' 
img = cv2.imread(image_path) 
input_tensor = tf.convert_to_tensor([img]) 
detections = model(input_tensor) 
# Visualize results 
for i in range(int(detections['num_detections'][0])): 
score = detections['detection_scores'][0][i].numpy() 
if score > 0.5: 
box = detections['detection_boxes'][0][i].numpy() 
y1, x1, y2, x2 = box 
(h, w) = img.shape[:2] 
cv2.rectangle(img, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), (0, 255, 0), 2) 
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) 
plt.axis('off') 
plt.title('Detected Objects using SSD') 
plt.show()