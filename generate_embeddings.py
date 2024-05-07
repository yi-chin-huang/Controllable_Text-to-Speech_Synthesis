from pathlib import Path
from utils.default_models import ensure_default_models
from encoder import inference
from encoder import audio
import numpy as np

encoder_path = Path("saved_models/default/encoder.pt")
input_path = Path("data/Jumana_English.wav")

ensure_default_models(Path("saved_models"))
inference.load_model(encoder_path)
preprocessed_wav = inference.preprocess_wav(input_path)
# TODO: try using_partials=False
embeddings = inference.embed_utterance(preprocessed_wav, using_partials=True, return_partials=False)
embeddings /= np.linalg.norm(embeddings)
print(embeddings)
print(embeddings.shape)