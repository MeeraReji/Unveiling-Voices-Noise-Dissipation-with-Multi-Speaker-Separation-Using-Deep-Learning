import os
import torch
import torchaudio
import numpy as np
import gradio as gr
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from huggingface_hub import hf_hub_download
from speechbrain.inference import SepformerSeparation as Separator

# Function to extract features (MFCCs) with fixed length
def extract_features(audio, sr=16000, n_mfcc=13, max_length=None):
    if isinstance(audio, str):
        y, sr = librosa.load(audio, sr=sr)
    else:
        y = audio
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs = librosa.util.normalize(mfccs)
    if max_length:
        mfccs = librosa.util.fix_length(mfccs, size=max_length, axis=1)
    return mfccs.T

# Paths to your audio files for Hemachandra and Vidhathri
hemachandra_files = [
    '/content/drive/MyDrive/H1.wav', '/content/drive/MyDrive/H2.wav', 
    '/content/drive/MyDrive/H3.wav', '/content/drive/MyDrive/H4.wav', 
    '/content/drive/MyDrive/H5.wav'
]
vidhathri_files = [
    '/content/drive/MyDrive/VV1.wav', '/content/drive/MyDrive/VV2.wav', 
    '/content/drive/MyDrive/VV3.wav', '/content/drive/MyDrive/VV4.wav', 
    '/content/drive/MyDrive/VV5.wav'
]

# Extract features and labels with fixed length
X = []
y = []
max_length = 500  # Adjust based on your dataset's characteristics

for file in hemachandra_files:
    features = extract_features(file, max_length=max_length)
    X.append(features)
    y.append('hemachandra')

for file in vidhathri_files:
    features = extract_features(file, max_length=max_length)
    X.append(features)
    y.append('vidhathri')

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Convert string labels to numeric labels
label_to_id = {'hemachandra': 0, 'vidhathri': 1}
y_numeric = np.array([label_to_id[label] for label in y])

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_numeric, test_size=0.2, random_state=42, stratify=y_numeric)

# Reshape for CNN input (assuming input shape required by CNN)
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]

# Download the trained model from Hugging Face
model_repo_id = 'Varshvansh/Speaker_Recognition'
model_filename = 'speaker_recognition_cnn.h5'
model_path = hf_hub_download(repo_id=model_repo_id, filename=model_filename)

# Load the trained CNN model
cnn_model = load_model(model_path)

# Load Sepformer model
sepformer_model = Separator.from_hparams(source="speechbrain/sepformer-wsj02mix", savedir='pretrained_models/sepformer-wsj02mix')

# Path to your mixed audio file
mixed_audio_file = '/content/drive/MyDrive/VIHEKE.wav'

# Separate audio using Sepformer
est_sources = sepformer_model.separate_file(path=mixed_audio_file)

# Function to classify speaker
def classify_speaker(mfcc_feature, cnn_model):
    mfcc_feature = np.expand_dims(mfcc_feature, axis=(0, -1))  # Add batch and channel dimensions
    prediction = cnn_model.predict(mfcc_feature)
    print("Prediction: ", prediction)  # Log prediction
    predicted_label = np.argmax(prediction)
    return 'hemachandra' if predicted_label == 0 else 'vidhathri'

# Resample to 8000 Hz and classify each separated source
os.makedirs("separated_audio", exist_ok=True)
for i in range(est_sources.shape[2]):
    source = est_sources[:, :, i].detach().cpu().numpy().squeeze()
    # Resample to 8000 Hz if needed
    source = librosa.resample(source, orig_sr=44100, target_sr=8000) if est_sources.shape[1] == 44100 else source
    # Extract features
    mfcc_feature = extract_features(source, sr=8000, max_length=500)
    # Log extracted features
    print("MFCC Feature Shape: ", mfcc_feature.shape)
    # Classify speaker
    speaker_name = classify_speaker(mfcc_feature, cnn_model)
    # Save the separated audio
    output_file = f'separated_audio/{speaker_name}_separated_{i+1}.wav'
    sf.write(output_file, source, 8000)
    print(f'Saved separated audio to {output_file}')

def gradio_interface(mixed_audio):
    return separate_and_classify(mixed_audio)

interface = gr.Interface(fn=gradio_interface, inputs="file", outputs="file", title="Speaker Separation and Classification", description="Upload a mixed audio file to separate and classify speakers.")

# Launch the Gradio interface
interface.launch()
