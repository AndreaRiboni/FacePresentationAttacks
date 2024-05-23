import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import ViTHybridImageProcessor, ViTHybridForImageClassification
from transformers import AutoImageProcessor, SwinForImageClassification
from transformers import ViTImageProcessor, ViTForImageClassification
from sklearn.metrics import roc_curve
from PIL import Image
import requests
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

excluded_dataset = 'msu' #msu done

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

swin_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
swin_model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

# Define a binary classification head
class BinaryClassificationHead(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BinaryClassificationHead, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Modify the Swin Transformer model for binary classification
swin_classifier_head = BinaryClassificationHead(768, 32)  # Adjust input size as needed
swin_model.classifier = swin_classifier_head

# Define loss function and optimizer
swin_criterion = nn.BCEWithLogitsLoss()
swin_optimizer = optim.AdamW(swin_model.parameters(), lr=1e-4)

vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
vit_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# Modify the Swin Transformer model for binary classification
vit_classifier_head = BinaryClassificationHead(768, 32)  # Adjust input size as needed
vit_model.classifier = vit_classifier_head

# Define loss function and optimizer
vit_criterion = nn.BCEWithLogitsLoss()
vit_optimizer = optim.AdamW(vit_model.parameters(), lr=1e-4)

import pandas as pd

train_val_df = pd.read_csv("no_msu/rotations/trainval_for_stacking.csv", index_col=False).drop(columns=['index'])
test_df = pd.read_csv("no_msu/rotations/test_for_stacking.csv", index_col=False).drop(columns=['index'])

train_val_df['swin_features'] = train_val_df['swin_features'].apply(lambda x: float(x.strip('[]')))
train_val_df['vit_features'] = train_val_df['vit_features'].apply(lambda x: float(x.strip('[]')))

test_df['swin_features'] = test_df['swin_features'].apply(lambda x: float(x.strip('[]')))
test_df['vit_features'] = test_df['vit_features'].apply(lambda x: float(x.strip('[]')))

# Function to calculate EER
def calculate_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return EER

def calculate_hter(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]

    threshold = eer_threshold
    predicted_labels = [1 if score > threshold else 0 for score in y_scores]

    false_acceptance = sum(1 for i in range(len(predicted_labels)) if predicted_labels[i] == 1 and y_true.iloc[i] == 0)
    false_rejection = sum(1 for i in range(len(predicted_labels)) if predicted_labels[i] == 0 and y_true.iloc[i] == 1)

    total_samples = len(y_true)
    hter = ((false_acceptance + false_rejection) / (2 * total_samples)) * 100
    return hter

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import roc_curve
import numpy as np
from sklearn.metrics import accuracy_score

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_val_df[['vit_features', 'swin_features']], 
                                                  train_val_df['label'], 
                                                  test_size=0.2, 
                                                  random_state=42)



import xgboost as xgb

xgb_classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_classifier.fit(X_train, y_train)

# Validate the model
y_val_scores = xgb_classifier.predict_proba(X_val)[:, 1]
eer = calculate_eer(y_val, y_val_scores)
print(f'Validation EER: {eer}')

# Test the model
X_test = test_df[['vit_features', 'swin_features']]
y_test = test_df['label']
y_test_scores = xgb_classifier.predict_proba(X_test)[:, 1]

# Calculate HTER
hter = calculate_hter(y_test, y_test_scores)
print(f"HTER using EER threshold: {hter:.2f}%")

swin_model.load_state_dict(torch.load('best_swin_model.pth'))
vit_model.load_state_dict(torch.load('best_vit_model.pth'))


##############################

import cv2
import torch
from torchvision import transforms
from PIL import Image

# Load your models here (swin_model and vit_model)
# Make sure they are in evaluation mode and loaded with the correct weights
# For example:
# swin_model.eval()
# vit_model.eval()

# Define your transform (the same as used for your training images)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def classify_frame(frame):
    """
    This function takes a single frame, applies the necessary transformations,
    and then uses the loaded models to classify the frame.
    """
       # Convert the frame to grayscale for the face detector
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # If no faces are detected, return None or an appropriate value
    if len(faces) == 0:
        return 1  # Or handle this scenario appropriately

    # For simplicity, use the first detected face
    x, y, w, h = faces[0]

    # Adjust the face region to make it square
    length = max(w, h)  # Find the max dimension
    center_x, center_y = x + w // 2, y + h // 2
    x_new = max(center_x - length // 2, 0)
    y_new = max(center_y - length // 2, 0)

    # Ensure the square region is within frame bounds
    if x_new + length > frame.shape[1]:
        length = frame.shape[1] - x_new
    if y_new + length > frame.shape[0]:
        length = frame.shape[0] - y_new

    # Extract the square face region
    square_face = frame[y_new:y_new+length, x_new:x_new+length]

    # Convert the square face region to PIL Image
    image = Image.fromarray(cv2.cvtColor(square_face, cv2.COLOR_BGR2RGB))
    image = transform(image).unsqueeze(0)  # Apply the transformation

    # Assuming the model and the image tensor are on the same device
    with torch.no_grad():
        # Use your model to classify the frame
        # For example, with swin_model:
        swin_output = swin_model(image).logits
        vit_output = vit_model(image).logits
        
        # Flatten the features to 1D vectors
        swin_features = swin_output.cpu().numpy().flatten()
        vit_features = vit_output.cpu().numpy().flatten()

        # Concatenate features from both models
        combined_features = np.concatenate([vit_features, swin_features])

        # Reshape for XGBoost (1 sample, -1 features)
        combined_features = combined_features.reshape(1, -1)

        # Predict using XGBoost
        xgb_prediction = xgb_classifier.predict_proba(combined_features)[:, 1]
        # binary_prediction = int(xgb_prediction > 0.5)

    return xgb_prediction

# Initialize webcam
cap = cv2.VideoCapture(0)
frame_counter = 0  # Initialize a frame counter
from collections import deque
window_size = 5  # Example size

prediction_buffer = deque(maxlen=window_size)

def majority_vote(predictions):
    """
    Determines the video classification based on majority voting.
    """
    count_0 = 0
    for prediction in predictions:
        if prediction == 0:
            count_0 += 1
    count_1 = len(predictions) - count_0
    if count_0 < count_1:
        return 0
    return 1



current_prediction = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Increment the frame counter
    frame_counter += 1

    # Classify one frame every 10 frames
    if frame_counter % 20 == 0:
        current_prediction = classify_frame(frame)

        # Display the resulting frame with the prediction
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f'Attack likelihood (single frame): {1-current_prediction}', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    prediction_buffer.append( 1 if current_prediction > 0.5 else 0)

    video_classification = majority_vote(list(prediction_buffer))
    classification_text = 'Real' if video_classification == 0 else 'Attack'
    cv2.putText(frame, f'Buffer classification: {classification_text}', (10, 80), font, 1, (0, 0, 255), 2, cv2.LINE_AA)


    # Always show the frame, but the prediction updates only every 10 frames
    cv2.imshow('frame', frame)

    # Break the loop with the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
