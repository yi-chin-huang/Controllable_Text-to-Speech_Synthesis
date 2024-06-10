import os
import pickle
import numpy as np

def load_data_from_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

embeddings_path = "/home/Yi-Chin/Controllable_Text-to-Speech_Synthesis/embeddings/"

Xf_avg_loaded, labelsf_avg_loaded = load_data_from_pickle(embeddings_path + 'female_avg_embeddings.pickle')
Xm_avg_loaded, labelsm_avg_loaded = load_data_from_pickle(embeddings_path + 'male_avg_embeddings.pickle')

print(Xf_avg_loaded.shape)
print(Xm_avg_loaded.shape)