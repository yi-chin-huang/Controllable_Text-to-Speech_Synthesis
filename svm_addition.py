import os
import pickle
import numpy as np
import soundfile as sf
import torch
import librosa
from pathlib import Path
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from encoder import inference as encoder
from encoder.params_model import model_embedding_size as speaker_embedding_size
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder

# Function to load data
def load_data(folder_path):
    X = []  
    y = []  
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if file_path.endswith('.pk'):
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                X.append(data['embed'])  
                y.append(data['gender'])  
    return np.array(X), np.array(y)

# Load male and female embeddings
folder_path = "./data/"
X_male, y_male = load_data(folder_path + 'male')
X_female, y_female = load_data(folder_path + 'female')

X = np.vstack((X_male, X_female))
y = np.concatenate((y_male, y_female))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Train SVM
clf = svm.SVC(kernel='linear')  # Ensure we get a linear hyperplane
clf.fit(X_train, y_train)

# Test SVM
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Extract the normal vector (weights)
normal_vector = clf.coef_[0]
unit_normal_vector = normal_vector / np.linalg.norm(normal_vector)

# Select a female embedding
female_embedding = X_female[0]  # You may choose any female embedding

# Adjust the embedding
adjusted_embedding = female_embedding - unit_normal_vector

# Synthesize voice
def synthesize_voice(embed, text, synthesizer, vocoder):
    texts = [text]
    embeds = [embed]
    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    spec = specs[0]
    generated_wav = vocoder.infer_waveform(spec)
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
    generated_wav = encoder.preprocess_wav(generated_wav)
    return generated_wav

if __name__ == '__main__':
    enc_model_fpath = Path("saved_models/default/encoder.pt")
    syn_model_fpath = Path("saved_models/default/synthesizer.pt")
    voc_model_fpath = Path("saved_models/default/vocoder.pt")

    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device_id)
        print(f"Found {torch.cuda.device_count()} GPUs available. Using GPU {device_id} ({gpu_properties.name}) with {gpu_properties.total_memory / 1e9:.1f}Gb total memory.")
    else:
        print("Using CPU for inference.")

    encoder.load_model(enc_model_fpath)
    synthesizer = Synthesizer(syn_model_fpath)
    vocoder.load_model(voc_model_fpath)

    text = "Since we do not know who has already completed the survey, we are sending reminders to all students on campus. Thank you if you have already completed the survey – please take this opportunity to encourage your friends to participate as well. It is critical that we have the participation of as many students as possible so that we can get the fullest and most accurate picture of student perspectives and experiences at Stanford."

    # Generate voice from original female embedding
    original_voice_wav = synthesize_voice(female_embedding, text, synthesizer, vocoder)
    original_filename = "synthesis_audio/original_female_voice.wav"
    sf.write(original_filename, original_voice_wav.astype(np.float32), synthesizer.sample_rate)
    print(f"Original voice saved at {original_filename}")

    # Generate voice from adjusted embedding
    adjusted_voice_wav = synthesize_voice(adjusted_embedding, text, synthesizer, vocoder)
    adjusted_filename = "synthesis_audio/adjusted_female_voice.wav"
    sf.write(adjusted_filename, adjusted_voice_wav.astype(np.float32), synthesizer.sample_rate)
    print(f"Adjusted voice saved at {adjusted_filename}")
