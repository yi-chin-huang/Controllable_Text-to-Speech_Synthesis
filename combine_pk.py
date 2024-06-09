import pickle
import pandas as pd
import numpy as np

def load_data_from_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

## Load embed
embeddings_path = "/home/dataset/CommonVoice/common_voice_avg_embeddings_1000.pk"
labels_path = "/home/dataset/CommonVoice/common_voice_labels_1000.pk"

labels = load_data_from_pickle(labels_path)
X = load_data_from_pickle(embeddings_path)

# Convert labels to a DataFrame
labels_df = pd.DataFrame(labels)
labels_df = pd.json_normalize(labels_df[0])

# Filter for ages
age_groups = ['teens', 'twenties', 'thirties', 'fourties', 'fifties']
filtered_labels_df = labels_df[labels_df['age'].isin(age_groups)]

# Get the indices of the filtered labels
filtered_indices = filtered_labels_df.index

# Filter the embeddings using the indices
filtered_X = X[:, filtered_indices]
filtered_labels_list = filtered_labels_df.to_dict(orient='records')

old_us = "/home/dataset/CommonVoice/common_voice_embeddings_old_us.pk"
old_uk = "/home/dataset/CommonVoice/common_voice_embeddings_old_uk.pk"
old_us_emb, old_us_labels = load_data_from_pickle(old_us)
old_uk_emb, old_uk_labels = load_data_from_pickle(old_uk)

concatenated_embeddings = np.concatenate((filtered_X, old_uk_emb), axis=1)
concatenated_embeddings = np.concatenate((concatenated_embeddings, old_us_emb), axis=1)

concatenated_labels = np.concatenate((filtered_labels_list, old_uk_labels[1:]), axis=0)
concatenated_labels = np.concatenate((concatenated_labels, old_us_labels[1:]), axis=0)

print(concatenated_embeddings.shape, concatenated_labels.shape)
out_pickle = "/home/dataset/CommonVoice/common_voice_embeddings_elders_augmented.pk"

with open(out_pickle, 'wb') as file:
    pickle.dump((concatenated_embeddings, concatenated_labels), file)
    