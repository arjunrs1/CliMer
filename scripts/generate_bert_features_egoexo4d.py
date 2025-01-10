import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import os
from tqdm import tqdm

# Set up BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to compute BERT features for a sentence
def compute_bert_features(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', max_length=512, truncation=True)
    outputs = model(**inputs)
    features = outputs.last_hidden_state[:, 1:, :]  # Use all token representations except CLS
    num_tokens = inputs['input_ids'].shape[1]  # Subtract 2 for CLS and SEP tokens
    return features, num_tokens

save_dir = '/private/home/arjunrs1/egoexo4d_features/narration_features/bert_features'
keysteps_save_dir = '/private/home/arjunrs1/egoexo4d_features/narration_features/keystep_features/bert_features'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if not os.path.exists(keysteps_save_dir):
    os.makedirs(keysteps_save_dir)

# Load data from CSV files
splits = ['train', 'val', 'test']
for split in splits:
    csv_file = f'/private/home/arjunrs1/exo_narration_grounding/data_processing/time_interval_annotation_files/narration_annotations/{split}.csv'
    keystep_csv_file = f'/private/home/arjunrs1/exo_narration_grounding/data_processing/time_interval_annotation_files/keystep_annotations/{split}.csv'
    #df = pd.read_csv(csv_file)
    keysteps_df = pd.read_csv(keystep_csv_file)

    """ # Compute BERT features for each narration
    print(f"[Narrations]: Computing {split} BERT features...")
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        narration_id = row['unique_narration_id']
        save_file = os.path.join(save_dir, f'{narration_id}.pt')
        if not os.path.exists(save_file):
            sentence = row['narration']
            features, num_tokens = compute_bert_features(sentence)
            # Save features and num_tokens to file
            torch.save({'features': features.squeeze(0), 'num_tokens': num_tokens}, save_file)
        #print(f'Saved features for {narration_id} to {save_file}') """

    # Compute BERT features for each keystep
    print(f"[Keysteps]: Computing {split} BERT features...")
    for index, row in tqdm(keysteps_df.iterrows(), total=keysteps_df.shape[0]):
        narration_id = row['unique_narration_id']
        save_file = os.path.join(keysteps_save_dir, f'{narration_id}.pt')
        if not os.path.exists(save_file):
            sentence = row['narration']
            try:
                features, num_tokens = compute_bert_features(sentence)
            except:
                print(sentence)
                print(narration_id)
            # Save features and num_tokens to file
            torch.save({'features': features.squeeze(0), 'num_tokens': num_tokens}, save_file)
        #print(f'Saved features for {narration_id} to {save_file}')