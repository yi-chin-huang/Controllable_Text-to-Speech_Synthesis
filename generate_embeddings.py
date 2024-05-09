from pathlib import Path
from utils.default_models import ensure_default_models
from encoder import inference
from encoder import audio
import numpy as np
import pandas as pd
import os
import pickle

csv_path='./dataset/VoxCeleb1/vox1_meta.csv'
dataset_path='./dataset/VoxCeleb1/wav/'

def fetch_ids(gender=None, nationality=None):
    df = pd.read_csv(csv_path, sep='\t')

    if nationality == None:
        filtered_df = df[df['Gender'] == gender]
    elif gender == None:
        filtered_df = df[df['Nationality'] == nationality]
    else:
        filtered_df = df[(df['Gender'] == gender) & (df['Nationality'] == nationality)]
    
    ids = [id for id in filtered_df['VoxCeleb1 ID'].tolist() if Path(dataset_path + id).is_dir()]
    return ids

def get_wav_paths(id):
    root_dir_path = Path(dataset_path + id)
    wav_paths = []

    for item in os.listdir(root_dir_path):
        item_path = os.path.join(root_dir_path, item)
        if os.path.isdir(item_path):
            for filename in os.listdir(item_path):
                if filename.endswith('.wav'):
                    filepath = os.path.join(item_path, filename)
                    wav_paths.append(filepath)
            
    return wav_paths

def get_gender(id):
    df = pd.read_csv(csv_path, sep='\t')
    row_index = df.index[df['VoxCeleb1 ID'] == id][0]
    gender = df.loc[row_index, 'Gender']
    return gender

def get_nationality(id):
    df = pd.read_csv(csv_path, sep='\t')
    row_index = df.index[df['VoxCeleb1 ID'] == id][0]
    nationality = df.loc[row_index, 'Nationality']
    return nationality

def generate_embeddings(input_path: Path):
    encoder_path = Path("saved_models/default/encoder.pt")
    ensure_default_models(Path("saved_models"))
    inference.load_model(encoder_path)
    preprocessed_wav = inference.preprocess_wav(input_path)
    # TODO: try using_partials=False
    embeddings = inference.embed_utterance(preprocessed_wav, using_partials=True, return_partials=False)
    embeddings /= np.linalg.norm(embeddings)
    return embeddings

def get_embeddings(pickle_filepath):
    with open(pickle_filepath, 'rb') as pickle_file:
        label_dict = pickle.load(pickle_file)
    return label_dict


if __name__ == '__main__':
    embeddings_path = 'encoder/embeddings/'

    gender = 'f'
    ids = fetch_ids(gender)
    
    for id in ids:
        wav_paths = get_wav_paths(id)
        
        for wav_path in wav_paths:
            labels = wav_path.split("/")
            subfolder = labels[-2]
            file = labels[-1][:-4]
            
            label_dict = {}
            label_dict['id'] = id
            label_dict['gender'] = get_gender(id)
            label_dict['nationality'] = get_nationality(id)
            label_dict['embed'] = generate_embeddings(wav_path)
            label_dict['wav_path'] = wav_path

            pickle_filepath = embeddings_path + id + "_" + label_dict['gender'] + "_" + label_dict['nationality'] + "_" + subfolder + "_" + file + ".pk"

            # Open the file in binary write mode
            with open(pickle_filepath, 'wb') as file:
                pickle.dump(label_dict, file)