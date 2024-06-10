import numpy as np
import pandas as pd
import pickle

# paths on GCP
csv_path = "/home/dataset/CommonVoice/cv-corpus-17.0-2024-03-15/en/validated.tsv"

df = pd.read_csv(csv_path, sep='\t')
us_condition = (df['accents'].str.contains('United States', na=False)) & (df['gender'].notna()) & (df['age'].notna())
uk_condition = (df['accents'].str.contains('England', na=False)) & (df['gender'].notna()) & (df['age'].notna())
us_xor_uk_filtered_df = df[us_condition ^ uk_condition]

young = ['teens', 'twenties']
young_df = us_xor_uk_filtered_df[us_xor_uk_filtered_df['age'].isin(young)]

old = ['sixties', 'seventies', 'eighties', 'nineties']
old_df = us_xor_uk_filtered_df[us_xor_uk_filtered_df['age'].isin(old)]

# # Combine the dataframes
# combined_df = pd.concat([young_df, old_df])

# # Export combined dataframe to CSV
# output_csv_path = "/home/dataset/CommonVoice/tsv/common_voice_accent_age_filtered.csv"
# combined_df.to_csv(output_csv_path, index=False)


# Function to count unique male and female client IDs
def count_gender(client_ids, df):
    male_clients = df[(df['client_id'].isin(client_ids)) & (df['gender'] == 'male_masculine')]['client_id'].nunique()
    female_clients = df[(df['client_id'].isin(client_ids)) & (df['gender'] == 'female_feminine')]['client_id'].nunique()
    return male_clients, female_clients

# Get unique client IDs
young_us_client_ids = young_df[us_condition]['client_id'].unique()[:1000]
old_us_client_ids = old_df[us_condition]['client_id'].unique()
young_uk_client_ids = young_df[uk_condition]['client_id'].unique()
old_uk_client_ids = old_df[uk_condition]['client_id'].unique()

# Count male and female clients for each group
young_us_male, young_us_female = count_gender(young_us_client_ids, young_df)
old_us_male, old_us_female = count_gender(old_us_client_ids, old_df)
young_uk_male, young_uk_female = count_gender(young_uk_client_ids, young_df)
old_uk_male, old_uk_female = count_gender(old_uk_client_ids, old_df)

# Display the counts
gender_counts = {
    'young_us': {'male_masculine': young_us_male, 'female_feminine': young_us_female},
    'old_us': {'male_masculine': old_us_male, 'female_feminine': old_us_female},
    'young_uk': {'male_masculine': young_uk_male, 'female_feminine': young_uk_female},
    'old_uk': {'male_masculine': old_uk_male, 'female_feminine': old_uk_female},
}

print(gender_counts)