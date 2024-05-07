import pandas as pd
from pathlib import Path

def fetch_ids(file_path, nationality, gender):
    # Load the CSV data into a DataFrame
    df = pd.read_csv(file_path, sep='\t')
    # Filter the DataFrame based on nationality and gender
    filtered_df = df[(df['Nationality'] == nationality) & (df['Gender'] == gender)]
    
    wav_path = './dataset/VoxCeleb1/wav/'
    existing_dirs = [wav_path + id for id in filtered_df['VoxCeleb1 ID'].tolist() if Path(wav_path + id).is_dir()]
    print(existing_dirs)
    return existing_dirs

if __name__ == '__main__':
    file_path = './dataset/VoxCeleb1/vox1_meta.csv'  # Update this with the path to your CSV file
    nationality = 'Ireland'
    gender = 'm'
    results = fetch_ids(file_path, nationality, gender)
    # print("Matching IDs:", results)
