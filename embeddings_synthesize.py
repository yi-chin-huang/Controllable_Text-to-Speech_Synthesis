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

import pickle
import datetime;

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

    
    def load_data_from_pickle(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    embeddings_path = "/home/Yi-Chin/Controllable_Text-to-Speech_Synthesis/embeddings/"

    Xf_avg, labelsf_avg = load_data_from_pickle(embeddings_path + 'female_avg_embeddings.pickle')

    # embeddings_path2 = "../../embeddings/female/id10496_f_Canada_9-ZedlAVdLM_00016.pk"
    # one_embed = load_data_from_pickle(embeddings_path2)['embed']
    # print(one_embed)
    # Xm_avg_loaded, labelsm_avg_loaded = load_data_from_pickle(embeddings_path + 'male_avg_embeddings.pickle')
    # print(Xm_avg_loaded.shape)

    ## Load the models one by one.
    print("Preparing the synthesizer and the vocoder...")
    ensure_default_models(Path("saved_models"))
    encoder.load_model(args.enc_model_fpath)
    synthesizer = Synthesizer(args.syn_model_fpath)
    vocoder.load_model(args.voc_model_fpath)


    ## Run a test
    print("Testing your configuration with small inputs.")
    # Forward an audio waveform of zeroes that lasts 1 second. Notice how we can get the encoder's
    # sampling rate, which may differ.
    # If you're unfamiliar with digital audio, know that it is encoded as an array of floats
    # (or sometimes integers, but mostly floats in this projects) ranging from -1 to 1.
    # The sampling rate is the number of values (samples) recorded per second, it is set to
    # 16000 for the encoder. Creating an array of length <sampling_rate> will always correspond
    # to an audio of 1 second.
    # print("\tTesting the encoder...")
    # encoder.embed_utterance(np.zeros(encoder.sampling_rate))

    for i in range(2, 4):
        for j in range(i + 1, 4):
            embed0 = Xf_avg[i]
            embed1 = Xf_avg[j]
            embed0 /= np.linalg.norm(embed0)
            embed1 /= np.linalg.norm(embed1)

            label0 = labelsf_avg[i]['gender'] +  '_' + labelsf_avg[i]['nationality'] + '_' + labelsf_avg[i]['id']
            print(label0)
            label1 = labelsf_avg[j]['gender'] +  '_' + labelsf_avg[j]['nationality'] + '_' + labelsf_avg[j]['id']
            avg_label = 'avg_' + label0 + label1
            avg_embed = np.mean( np.array([ embed0, embed1 ]), axis=0 )
                    
            # # The synthesizer can handle multiple inputs with batching. Let's create another embedding to
            # # illustrate that
            # embeds = [embed0, avg_embed]
            # texts = ["test 1", "test 2"]
            # print("\tTesting the synthesizer... (loading the model will output a lot of text)")
            # mels = synthesizer.synthesize_spectrograms(texts, embeds)

            # # The vocoder synthesizes one waveform at a time, but it's more efficient for long ones. We
            # # can concatenate the mel spectrograms to a single one.
            # mel = np.concatenate(mels, axis=1)
            # # The vocoder can take a callback function to display the generation. More on that later. For
            # # now we'll simply hide it like this:
            # no_action = lambda *args: None
            # print("\tTesting the vocoder...")
            # # For the sake of making this test short, we'll pass a short target length. The target length
            # # is the length of the wav segments that are processed in parallel. E.g. for audio sampled
            # # at 16000 Hertz, a target length of 8000 means that the target audio will be cut in chunks of
            # # 0.5 seconds which will all be generated together. The parameters here are absurdly short, and
            # # that has a detrimental effect on the quality of the audio. The default parameters are
            # # recommended in general.
            # vocoder.infer_waveform(mel, target=200, overlap=50, progress_callback=no_action)

            # print("All test passed! You can now synthesize speech.\n\n")


            # ## Interactive speech generation
            # print("This is a GUI-less example of interface to SV2TTS. The purpose of this script is to "
            #         "show how you can interface this project easily with your own. See the source code for "
            #         "an explanation of what is happening.\n")

            # num_generated = 0
            ## Generating the spectrogram
            text = "This course is designed around lectures, assignments, and a course project to give students practical experience building spoken language systems."

            # If seed is specified, reset torch seed and force synthesizer reload
            if args.seed is not None:
                torch.manual_seed(args.seed)
                synthesizer = Synthesizer(args.syn_model_fpath)

        
            # The synthesizer works in batch, so you need to put your data in a list or numpy array
            texts = [text, text, text]
            embeds = [embed0, embed1, avg_embed]
            labels = [label0, label1, avg_label]
            
            # If you know what the attention layer alignments are, you can retrieve them here by
            # passing return_alignments=True
            specs = synthesizer.synthesize_spectrograms(texts, embeds)

            for k in range(3):
                spec = specs[k]
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
                current_time = datetime.datetime.now()
                description = label1
                filename = "synthesis_audio/demo_output_%s.wav" % labels[k]
                print(generated_wav.dtype)
                sf.write(filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
                # num_generated += 1
                print("\nSaved output as %s\n\n" % filename)

