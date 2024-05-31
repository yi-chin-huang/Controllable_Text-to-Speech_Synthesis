from pathlib import Path
from utils.default_models import ensure_default_models
from encoder import inference as encoder
from vocoder import inference as vocoder
from synthesizer.inference import Synthesizer
import numpy as np
import pandas as pd
import os
import pickle
import soundfile as sf


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
    encoder.load_model(encoder_path)
    preprocessed_wav = encoder.preprocess_wav(input_path)
    embeddings = encoder.embed_utterance(preprocessed_wav, using_partials=True, return_partials=False)
    return embeddings


def load_embeddings(pickle_filepath):
    with open(pickle_filepath, 'rb') as pickle_file:
        label_dict = pickle.load(pickle_file)
    return label_dict


def run_inference(text, embed_pickle):
    embed = load_embeddings(embed_pickle)['embed']
    texts = [text]
    embeds = [embed]

    synthesizer = Synthesizer(Path("saved_models/default/synthesizer.pt"))
    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    spec = specs[0]
    print("Created the mel spectrogram")

    ## Generating the waveform
    print("Synthesizing the waveform:")
    vocoder.load_model(Path("saved_models/default/vocoder.pt"))
    generated_wav = vocoder.infer_waveform(spec)

    ## Post-generation
    # There's a bug with sounddevice that makes the audio cut one second earlier, so we
    # pad it.
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

    # Trim excess silences to compensate for gaps in spectrograms (issue #53)
    generated_wav = encoder.preprocess_wav(generated_wav)

    # Save it on the disk
    filename = embed_pickle.split('.pk')[0] + ".wav"
    sf.write(filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
    print("\nSaved output as %s\n\n" % filename)


if __name__ == '__main__':

    ensure_default_models(Path("saved_models"))
    
    # generate embeddings per age category
    age_list = ["teens", "twenties", "thirties", "fourties", "fifties", "sixties", "seventies"]
    
    for age in age_list:
        print(age)

        embeddings_path = 'embeddings/'

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

    # run inference
    # text = "Hello! This is a sixty-year old male speaking. How are you doing today?"
    text = "Romeo, take me somewhere we can be alone I'll be waiting, all there's left to do is run You'll be the prince and I'll be the princess It's a love story, baby, just say yes"
    # embed_pickle = "embeddings/sixties/common_voice_en_39587041_male_masculine_sixties.pk"
    embed_pickle = "embeddings/teens/common_voice_en_39645706_female_feminine_teens.pk"
    run_inference(text, embed_pickle)
