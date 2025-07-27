# ==========================================================
# Step 1: Extract Wav2Vec2 features from RAVDESS audio files
# ==========================================================

import os
import numpy as np
import librosa
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import Wav2Vec2Processor, Wav2Vec2Model


class Wav2Vec2FeatureExtractor:
    def __init__(self, dataset_path= r"C:\Users\Kevin\Downloads\RAVDESS", output_feature_path="extracted_features.npy", output_label_path="extracted_labels.npy"): # replace wiht your RAVDESS dataset path
        self.DATASET_PATH = dataset_path  
        self.OUTPUT_PATH = output_feature_path
        self.LABELS_PATH = output_label_path

        # Emotion code mapping based on RAVDESS
        self.emotion_map = {
            '01': 'neutral',
            '02': 'calm',
            '03': 'happy',
            '04': 'sad',
            '05': 'angry',
            '06': 'fearful',
            '07': 'disgust',
            '08': 'surprised'
        }

        # Load processor and model
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        if torch.cuda.is_available():
            self.model.to("cuda")
            print("Using CUDA for Wav2Vec2 model.")

    # Method to extract features and save them
    def extract_and_save_features(self):
        X = []
        y = []

        print("Extracting Wav2Vec2 features from RAVDESS audio...")
        for actor_folder in os.listdir(self.DATASET_PATH):
            actor_path = os.path.join(self.DATASET_PATH, actor_folder)
            if os.path.isdir(actor_path):
                for file in os.listdir(actor_path):
                    if file.endswith(".wav"):
                        file_parts = file.split('-')
                        if file_parts[0] == '03' and file_parts[1] == '01':
                            emotion_code = file_parts[2]
                            emotion = self.emotion_map.get(emotion_code)
                            if emotion:
                                full_path = os.path.join(actor_path, file)
                                waveform, sr = librosa.load(full_path, sr=16000)
                                inputs = self.processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
                                if torch.cuda.is_available():
                                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                                with torch.no_grad():
                                    outputs = self.model(**inputs)
                                    hidden_states = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                                X.append(hidden_states)
                                y.append(emotion)

        y_encoded = LabelEncoder().fit_transform(y)
        X = np.array(X)
        y = np.array(y_encoded)

        print("Wav2Vec2 feature extraction complete.")
        print("Feature shape:", X.shape)
        print("Labels shape:", y.shape)

        np.save(self.OUTPUT_PATH, X)
        np.save(self.LABELS_PATH, y)

        print(f"Extracted features saved to: {self.OUTPUT_PATH}")
        print(f"Encoded labels saved to: {self.LABELS_PATH}")
