import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image

def load_images_from_folder(folder):
    images = []
    labels = []
    label_dict = {}  
    label_counter = 0
    
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        
        if os.path.isdir(subfolder_path):
            print(f"Loading images from {subfolder_path}")  
            
            if subfolder not in label_dict:
                label_dict[subfolder] = label_counter
                label_counter += 1
            label = label_dict[subfolder]
            
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.jpg') : 
                    img_path = os.path.join(subfolder_path, filename)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (100, 100)) 
                        images.append(img)
                        labels.append(label)
    
    return np.array(images), np.array(labels), label_dict

folder_path = r"C:\Users\AI_LAB\Downloads\archive (11)"
images, labels, label_dict = load_images_from_folder(folder_path)

print(f"Loaded {len(images)} images.")

if len(images) == 0:
    raise ValueError(f"No images were loaded from the directory: {folder_path}")

images = images.astype('float32') / 255.0 
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(np.unique(labels)), activation='softmax') 
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

def predict_and_display_image(img, model, label_dict):
    
    img_resized = cv2.resize(img, (100, 100))
    img_input = img_resized.astype('float32') / 255.0  
    img_input = np.expand_dims(img_input, axis=0)  
    
    prediction = model.predict(img_input)
    predicted_label = np.argmax(prediction)  
    predicted_name = list(label_dict.keys())[predicted_label] 
    
    cv2.imshow("Original Image", img)
    
    img_with_text = img.copy()
    cv2.putText(img_with_text, predicted_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    cv2.imshow("Recognized Image", img_with_text)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

sample_image = X_test[2] 
sample_image = cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR)  
predict_and_display_image(sample_image, model, label_dict)
