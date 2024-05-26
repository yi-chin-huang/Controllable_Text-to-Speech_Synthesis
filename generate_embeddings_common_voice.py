from pathlib import Path
from utils.default_models import ensure_default_models
from encoder import inference
from encoder import audio
import numpy as np
import pandas as pd
import os
import pickle

csv_path='dataset/cv-corpus-17.0-delta-2024-03-15/en/validated.tsv'
dataset_path='dataset/cv-corpus-17.0-delta-2024-03-15/en/clips/'

def fetch_filenames(gender=None, age=None):
    df = pd.read_csv(csv_path, sep='\t')

    if gender == None:
        filtered_df = df[df['age'] == age]
    elif age == None:
        filtered_df = df[df['gender'] == gender]
    else:
        filtered_df = df[(df['gender'] == gender) & (df['age'] == age)]
    
    filenames = [filename for filename in filtered_df['path'].tolist() if Path(dataset_path + filename).is_file()]
    return filenames

def get_gender(filename):
    df = pd.read_csv(csv_path, sep='\t')
    row_index = df.index[df['path'] == filename][0]
    gender = df.loc[row_index, 'gender']
    return gender

def get_age(filename):
    df = pd.read_csv(csv_path, sep='\t')
    row_index = df.index[df['path'] == filename][0]
    age = df.loc[row_index, 'age']
    return age

def generate_embeddings(input_path):
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
    
    # age_list = ["teens", "twenties", "thirties", "fourties", "fifties", "sixties", "seventies"]
    age_list = ["thirties", "fourties", "fifties", "sixties", "seventies"]
    
    for age in age_list:
        print(age)

        embeddings_path = 'embeddings/' + age + "/"

        if not os.path.exists(embeddings_path):
            os.makedirs(embeddings_path)
            print(f"Folder '{embeddings_path}' created.")
        else:
            print(f"Folder '{embeddings_path}' already exists.")

        filenames = fetch_filenames(age=age)
        print(filenames)
    
        for filename in filenames:
            audio_path = dataset_path + filename
            
            label_dict = {}
            label_dict['filename'] = filename
            label_dict['gender'] = get_gender(filename)
            label_dict['age'] = get_age(filename)
            label_dict['embed'] = generate_embeddings(audio_path)
                  
            pickle_filepath = embeddings_path + filename.split('.mp3')[0] + "_" + str(label_dict['gender']) + "_" + str(label_dict['age']) + ".pk"

            # Open the file in binary write mode
            with open(pickle_filepath, 'wb') as file:
                pickle.dump(label_dict, file)
