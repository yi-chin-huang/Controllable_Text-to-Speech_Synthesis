import numpy as np
import pandas as pd
import pickle

# paths on GCP
csv_path = "/home/dataset/CommonVoice/cv-corpus-17.0-2024-03-15/en/validated.tsv"
dataset_path = "/home/dataset/CommonVoice/cv-corpus-17.0-2024-03-15/en/clips/"

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


# df = pd.read_csv(csv_path, sep='\t')
# us_condition = (df['accents'].str.contains('United States', na=False)) & (df['gender'].notna()) & (df['age'].notna())
# uk_condition = (df['accents'].str.contains('England', na=False)) & (df['gender'].notna()) & (df['age'].notna())
# us_xor_uk_filtered_df = df[us_condition ^ uk_condition]

# young = ['teens', 'twenties']
# young_df = us_xor_uk_filtered_df[us_xor_uk_filtered_df['age'].isin(young)]

# old = ['sixties', 'seventies', 'eighties', 'nineties']
# old_df = us_xor_uk_filtered_df[us_xor_uk_filtered_df['age'].isin(old)]

# young_us_client_ids, old_us_client_ids = young_df[us_condition]['client_id'].unique(), old_df[us_condition]['client_id'].unique()
# young_uk_client_ids, old_uk_client_ids = young_df[uk_condition]['client_id'].unique(), old_df[uk_condition]['client_id'].unique()
# print(young_us_client_ids.shape, old_us_client_ids.shape)
# print(young_uk_client_ids.shape, old_uk_client_ids.shape)



from pathlib import Path
from utils.default_models import ensure_default_models
from encoder import inference as encoder
from vocoder import inference as vocoder
from synthesizer.inference import Synthesizer
import numpy as np
import pandas as pd
import pickle
import soundfile as sf
import audioread
from tqdm import tqdm
import warnings

# paths locally
# csv_path = 'dataset/cv-corpus-17.0-delta-2024-03-15/en/validated.tsv'
# dataset_path = 'dataset/cv-corpus-17.0-delta-2024-03-15/en/clips/'

# paths on GCP
csv_path = "../../dataset/CommonVoice/cv-corpus-17.0-2024-03-15/en/validated.tsv"
dataset_path = "../../dataset/CommonVoice/cv-corpus-17.0-2024-03-15/en/clips/"

# load encoder
encoder_path = Path("saved_models/default/encoder.pt")
encoder.load_model(encoder_path)

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


def calculate_avg_embeddings(client_ids):
    avg_embeddings, labels = None, None
    for client_id in tqdm(client_ids):
        try:
            filtered_rows = us_xor_uk_filtered_df[us_xor_uk_filtered_df['client_id'] == client_id][:10]

            labels_dict = {}
            labels_dict['gender'] = filtered_rows['gender'].iloc[0]
            labels_dict['age'] = filtered_rows['age'].iloc[0]
            labels_dict['accent'] = 'US' if 'United States' in filtered_rows['accents'].iloc[0] else 'UK'
            labels_dict['filenames'] = []
            labels_dict['avg_duration'] = 0
            labels_dict['num_utterances'] = len(filtered_rows)

            embeddings_per_client = None
            
            for _, row in filtered_rows.iterrows():
                filename = row['path']
                audio_path = dataset_path + filename

                if not Path(audio_path).is_file():
                    raise FileNotFoundError("The given audio file at" + audio_path + "was not found")
                
                embeddings = np.reshape(generate_embeddings(audio_path), (-1, 1))

                if embeddings_per_client is None:
                    embeddings_per_client = embeddings
                else:
                    embeddings_per_client = np.append(embeddings_per_client, embeddings, axis=1)

                labels_dict['filenames'].append(filename)
                labels_dict['avg_duration'] += get_audio_duration(audio_path) / labels_dict['num_utterances']
        
            avg_embeddings_per_client = np.mean(embeddings_per_client, axis=1).reshape((-1, 1))

            if avg_embeddings is None:
                avg_embeddings = avg_embeddings_per_client
            else:
                avg_embeddings = np.append(avg_embeddings, avg_embeddings_per_client, axis=1)
            labels = np.append(labels, labels_dict)

        except:
            pass

    return avg_embeddings, labels

def get_audio_duration(file_path):
    try:
        with audioread.audio_open(file_path) as audio_file:
            duration = audio_file.duration
            return duration
    except audioread.DecodeError:
        print("Failed to decode the audio file.")
        return None


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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preprocessed_wav = encoder.preprocess_wav(input_path)
    embeddings = encoder.embed_utterance(preprocessed_wav, using_partials=True, return_partials=False)
    return embeddings


def load_embeddings(pickle_filepath):
    with open(pickle_filepath, 'rb') as pickle_file:
        label_dict = pickle.load(pickle_file)
    return label_dict


def run_inference_from_embeddings(text, embed):
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
    filename = "demo.wav"
    sf.write(filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
    print("\nSaved output as %s\n\n" % filename)


if __name__ == '__main__':
    avg_embeddings = None
    labels = np.array([])

    df = pd.read_csv(csv_path, sep='\t')
    us_condition = (df['accents'].str.contains('United States', na=False)) & (df['gender'].notna()) & (df['age'].notna())
    uk_condition = (df['accents'].str.contains('England', na=False)) & (df['gender'].notna()) & (df['age'].notna())
    us_xor_uk_filtered_df = df[us_condition ^ uk_condition]

    young = ['teens', 'twenties']
    young_df = us_xor_uk_filtered_df[us_xor_uk_filtered_df['age'].isin(young)]

    old = ['sixties', 'seventies', 'eighties', 'nineties']
    old_df = us_xor_uk_filtered_df[us_xor_uk_filtered_df['age'].isin(old)]

    old_us_client_ids = old_df[us_condition]['client_id'].unique()
    # old_uk_client_ids = old_df[uk_condition]['client_id'].unique()
    # young_us_client_ids, old_us_client_ids = young_df[us_condition]['client_id'].unique()[:1000], old_df[us_condition]['client_id'].unique()
    # young_uk_client_ids, old_uk_client_ids = young_df[uk_condition]['client_id'].unique(), old_df[uk_condition]['client_id'].unique()
    # print("US speaker young/old:", young_us_client_ids.shape, old_us_client_ids.shape)
    # print("UK speaker young/old:", young_uk_client_ids.shape, old_uk_client_ids.shape)

    # avg_embeddings, labels = calculate_avg_embeddings_per_client()
    avg_embeddings, labels = calculate_avg_embeddings(old_us_client_ids)
    
    embeddings_pickle = "/home/dataset/CommonVoice/common_voice_embeddings_old_us.pk"

    with open(embeddings_pickle, 'wb') as file:
        pickle.dump((avg_embeddings, labels), file)
    
    # with open(labels_pickle, 'wb') as file:
    #     pickle.dump(labels, file)