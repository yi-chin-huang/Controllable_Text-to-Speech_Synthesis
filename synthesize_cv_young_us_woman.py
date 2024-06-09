import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from encoder import inference as encoder
from encoder.params_model import model_embedding_size as speaker_embedding_size
from synthesizer.inference import Synthesizer
from utils.argutils import print_args
from utils.default_models import ensure_default_models
from vocoder import inference as vocoder

import wave
import pickle
import pandas as pd

def load_data_from_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "--enc_model_fpath", type=Path,
                        default="saved_models/default/encoder.pt",
                        help="Path to a saved encoder")
    parser.add_argument("-s", "--syn_model_fpath", type=Path,
                        default="saved_models/default/synthesizer.pt",
                        help="Path to a saved synthesizer")
    parser.add_argument("-v", "--voc_model_fpath", type=Path,
                        default="saved_models/default/vocoder.pt",
                        help="Path to a saved vocoder")
    parser.add_argument("--cpu", action="store_true", help=\
        "If True, processing is done on CPU, even when a GPU is available.")
    parser.add_argument("--no_sound", action="store_true", help=\
        "If True, audio won't be played.")
    parser.add_argument("--seed", type=int, default=None, help=\
        "Optional random number seed value to make toolbox deterministic.")
    args = parser.parse_args()
    arg_dict = vars(args)
    print_args(args, parser)

    # Hide GPUs from Pytorch to force CPU processing
    if arg_dict.pop("cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    print("Running a test of your configuration...\n")

    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device_id)
        ## Print some environment information (for debugging purposes)
        print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
            "%.1fGb total memory.\n" %
            (torch.cuda.device_count(),
            device_id,
            gpu_properties.name,
            gpu_properties.major,
            gpu_properties.minor,
            gpu_properties.total_memory / 1e9))
    else:
        print("Using CPU for inference.\n")


    ## Load embed
    embeddings_path = "/home/dataset/CommonVoice/common_voice_embeddings_elders_augmented.pk"

    X, labels = load_data_from_pickle(embeddings_path)
    
    # Convert labels to a DataFrame
    df = pd.DataFrame(labels)

    # Ensure X has the correct shape
    X = np.array(X)
    if X.shape[0] != df.shape[0]:
        X = X.T

    df = pd.json_normalize(df[0])
    df['embed'] = list(X)

    average_embeds = []
    # Gender
    age_groups = ['teens', 'twenties']
    filtered_df = df[(df['gender'] == 'male_masculine') & (df['age'].isin(age_groups)) & (df['accent'] == 'US')]
    average_embed = np.mean(np.vstack(filtered_df['embed'].values), axis=0)
    average_embeds.append(average_embed)
    
    # Age
    age_groups = ['sixties', 'seventies', 'eighties', 'nineties']
    filtered_df = df[(df['gender'] == 'female_feminine') & (df['age'].isin(age_groups)) & (df['accent'] == 'US')]
    average_embed = np.mean(np.vstack(filtered_df['embed'].values), axis=0)
    average_embeds.append(average_embed)

    # Accent
    age_groups = ['teens', 'twenties']
    filtered_df = df[(df['gender'] == 'female_feminine') & (df['age'].isin(age_groups)) & (df['accent'] == 'UK')]
    average_embed = np.mean(np.vstack(filtered_df['embed'].values), axis=0)
    average_embeds.append(average_embed)

    # change 2 factors
    # Gender age
    age_groups = ['sixties', 'seventies', 'eighties', 'nineties']
    filtered_df = df[(df['gender'] == 'male_masculine') & (df['age'].isin(age_groups)) & (df['accent'] == 'US')]
    average_embed = np.mean(np.vstack(filtered_df['embed'].values), axis=0)
    average_embeds.append(average_embed)
    
    # Gender accent
    age_groups = ['teens', 'twenties']
    filtered_df = df[(df['gender'] == 'male_masculine') & (df['age'].isin(age_groups)) & (df['accent'] == 'UK')]
    average_embed = np.mean(np.vstack(filtered_df['embed'].values), axis=0)
    average_embeds.append(average_embed)

    # Age accent
    age_groups = ['sixties', 'seventies', 'eighties', 'nineties']
    filtered_df = df[(df['gender'] == 'female_feminine') & (df['age'].isin(age_groups)) & (df['accent'] == 'UK')]
    average_embed = np.mean(np.vstack(filtered_df['embed'].values), axis=0)
    average_embeds.append(average_embed)
    
    # Change 3 factors
    # Gender age accent
    age_groups = ['sixties', 'seventies', 'eighties', 'nineties']
    filtered_df = df[(df['gender'] == 'male_masculine') & (df['age'].isin(age_groups)) & (df['accent'] == 'UK')]
    average_embed = np.mean(np.vstack(filtered_df['embed'].values), axis=0)
    average_embeds.append(average_embed)

    ## Load the models one by one.
    print("Preparing the encoder, the synthesizer and the vocoder...")
    ensure_default_models(Path("saved_models"))
    encoder.load_model(args.enc_model_fpath)
    synthesizer = Synthesizer(args.syn_model_fpath)
    vocoder.load_model(args.voc_model_fpath)

    ## Interactive speech generation
    print("This is a GUI-less example of interface to SV2TTS. The purpose of this script is to "
          "show how you can interface this project easily with your own. See the source code for "
          "an explanation of what is happening.\n")
    
    def load_data_from_pickle(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    num_generated = 0
    input_voice_file = "/home/dataset/CommonVoice/cv-corpus-17.0-2024-03-15/en/clips/common_voice_en_380475.mp3" # young US woman A
    note_dict = ['gender', 'age', 'accent', 'gender_age', 'gender_accent', 'age_accent', 'gender_age_accent']
    for (i, average_embed) in enumerate(average_embeds):
        try:
            # Get the reference audio filepath
            message = "Reference voice: enter an audio filepath of a voice to be cloned (mp3, " \
                        "wav, m4a, flac, ...):\n"
            in_fpath = input_voice_file

            ## Computing the embedding
            # First, we load the wav using the function that the speaker encoder provides. This is
            # important: there is preprocessing that must be applied.

            # The following two methods are equivalent:
            # - Directly load from the filepath:
            preprocessed_wav = encoder.preprocess_wav(in_fpath)
            # - If the wav is already loaded:
            original_wav, sampling_rate = librosa.load(str(in_fpath))
            preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
            print("Loaded file %s succesfully" % in_fpath)

            # Then we derive the embedding. There are many functions and parameters that the
            # speaker encoder interfaces. These are mostly for in-depth research. You will typically
            # only use this function (with its default parameters):
            embed = encoder.embed_utterance(preprocessed_wav)
            # print(embed.shape, avg_female_embed.shape)
            print("Created the embedding")
            
            # modified_embed = np.mean( np.array([embed, average_embed]), axis=0)
            modified_embed = 0.4 * embed + 0.6 * average_embed
            # avg_embed /= np.linalg.norm(avg_embed)

            ## Generating the spectrogram
            text = "My father's car is a jaguar, he drives it faster than I do."
            
            # If seed is specified, reset torch seed and force synthesizer reload
            if args.seed is not None:
                torch.manual_seed(args.seed)
                synthesizer = Synthesizer(args.syn_model_fpath)

            # The synthesizer works in batch, so you need to put your data in a list or numpy array
            texts = [text]
            embeds = [modified_embed]
            # If you know what the attention layer alignments are, you can retrieve them here by
            # passing return_alignments=True
            specs = synthesizer.synthesize_spectrograms(texts, embeds)
            spec = specs[0]
            print("Created the mel spectrogram")


            ## Generating the waveform
            print("Synthesizing the waveform:")

            # If seed is specified, reset torch seed and reload vocoder
            if args.seed is not None:
                torch.manual_seed(args.seed)
                vocoder.load_model(args.voc_model_fpath)

            # Synthesizing the waveform is fairly straightforward. Remember that the longer the
            # spectrogram, the more time-efficient the vocoder.
            generated_wav = vocoder.infer_waveform(spec)


            ## Post-generation
            # There's a bug with sounddevice that makes the audio cut one second earlier, so we
            # pad it.
            generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

            # Trim excess silences to compensate for gaps in spectrograms (issue #53)
            generated_wav = encoder.preprocess_wav(generated_wav)


            # Save it on the disk
            avg_file_name = '' + input_voice_file.split('/')[-1][:-4]
            filename = "synthesis_audio/common_voice/young_us_womanA/" + note_dict[i] + "_changed.wav"
            print(generated_wav.dtype)
            sf.write(filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
            num_generated += 1
            print("\nSaved output as %s\n\n" % filename)


        except Exception as e:
            print("Caught exception: %s" % repr(e))
            print("Restarting\n")
