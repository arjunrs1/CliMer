import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import os
from tqdm import tqdm

# Initialize NLTK data needed for POS tagging
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

def parse_sentence(sentence):
    print(sentence)
    tokens = word_tokenize(sentence)
    tagged_tokens = nltk.pos_tag(tokens)

    verbs = []
    nouns = []

    for word, pos in tagged_tokens:
        if pos.startswith('VB'):  # Verb
            verbs.append(word.lower())
        elif pos.startswith('NN'):  # Noun
            if word != "C":
                nouns.append(word.lower())

    return verbs, nouns

def get_class_index(class_list, class_dict):
    class_index = [class_dict[class_item] for class_item in class_list]
    return class_list #TODO: Change this back to class_index after generating the groundingSAM values

def main():
    data_dir_narrations = '/private/home/arjunrs1/exo_narration_grounding/data_processing/time_interval_annotation_files/narration_annotations/'
    data_dir_keysteps = '/private/home/arjunrs1/exo_narration_grounding/data_processing/time_interval_annotation_files/keystep_annotations/'
    output_dir = '/private/home/arjunrs1/CliMer/data/egoexo4d/'
    takes_dir = '/datasets01/egoexo4d/v2/takes/'
    all_verbs_narrations = set()
    all_nouns_narrations = set()
    all_verbs_keysteps = set()
    all_nouns_keysteps = set()

    for split in ['val', 'train', 'test']:
        df_narrations = pd.read_csv(data_dir_narrations + f'{split}.csv')
        narrations = df_narrations['narration'].tolist()
        for narration in narrations:
            verbs, nouns = parse_sentence(narration)
            all_verbs_narrations.update(verbs)
            all_nouns_narrations.update(nouns)
        df_keysteps = pd.read_csv(data_dir_keysteps + f'{split}.csv')
        keysteps = df_keysteps['narration'].tolist()
        for keystep in keysteps:
            verbs, nouns = parse_sentence(keystep)
            all_verbs_keysteps.update(verbs)
            all_nouns_keysteps.update(nouns)

    verb_dict_narrations = {verb: i for i, verb in enumerate(sorted(list(all_verbs_narrations)))}
    noun_dict_narrations = {noun: i for i, noun in enumerate(sorted(list(all_nouns_narrations)))}
    verb_dict_keysteps = {verb: i for i, verb in enumerate(sorted(list(all_verbs_keysteps)))}
    noun_dict_keysteps = {noun: i for i, noun in enumerate(sorted(list(all_nouns_keysteps)))}

    for split in ['train', 'val', 'test']:
        df_narrations = pd.read_csv(data_dir_narrations + f'{split}.csv')
        df_keysteps = pd.read_csv(data_dir_keysteps + f'{split}.csv')
        exo_rows_narrations = []
        ego_rows_narrations = []
        exo_rows_keysteps = []
        ego_rows_keysteps = []
        """ print(f"[Narrations]: Processing {split} annotations...")
        for index, row in tqdm(df_narrations.iterrows(), total=df_narrations.shape[0]):
            take_uid = row['take_uid']
            take_dir = os.path.join(takes_dir, take_uid, "frame_aligned_videos")
            exo_camera_names = [os.path.splitext(filename)[0] for filename in os.listdir(take_dir) if filename.endswith('.mp4')]
            exo_camera_names = [cam for cam in exo_camera_names if "aria" not in cam]
            ego_camera_name = row['ego_camera_path'].split("/")[-1].split(".")[0]
            
            for exo_camera_name in exo_camera_names:
                video_id = f'{take_uid}_{exo_camera_name}'
                video_duration = row['duration_sec']
                narration_timestamp = row['narration_frame']
                narration = row['narration']
                verbs, nouns = parse_sentence(narration)
                verb_class = get_class_index(verbs, verb_dict_narrations)
                noun_class = get_class_index(nouns, noun_dict_narrations)
                row_dict = {
                    'video_id': video_id,
                    'duration': video_duration,
                    'narration_frame': narration_timestamp,
                    'narration': narration,
                    'verb_class': verb_class,
                    'noun_class': noun_class
                }
                if split != "train":
                    row_dict['start_frame'] = row['start_frame']
                    row_dict['end_frame'] = row['end_frame']
                exo_rows_narrations.append(row_dict)
                
            video_id = f'{take_uid}_{ego_camera_name}'
            video_duration = row['duration_sec']
            narration_timestamp = row['narration_frame']
            narration = row['narration']
            verbs, nouns = parse_sentence(narration)
            verb_class = get_class_index(verbs, verb_dict_narrations)
            noun_class = get_class_index(nouns, noun_dict_narrations)
            row_dict = {
                'video_id': video_id,
                'duration': video_duration,
                'narration_frame': narration_timestamp,
                'narration': narration,
                'verb_class': verb_class,
                'noun_class': noun_class
            }
            if split != "train":
                    row_dict['start_frame'] = row['start_frame']
                    row_dict['end_frame'] = row['end_frame']
            ego_rows_narrations.append(row_dict) """

        print(f"[Keysteps]: Processing {split} annotations...")
        for index, row in tqdm(df_keysteps.iterrows(), total=df_keysteps.shape[0]):
            take_uid = row['take_uid']
            take_dir = os.path.join(takes_dir, take_uid, "frame_aligned_videos")
            exo_camera_names = [os.path.splitext(filename)[0] for filename in os.listdir(take_dir) if filename.endswith('.mp4')]
            exo_camera_names = [cam for cam in exo_camera_names if "aria" not in cam]
            ego_camera_name = row['ego_camera_path'].split("/")[-1].split(".")[0]
            for exo_camera_name in exo_camera_names:
                video_id = f'{take_uid}_{exo_camera_name}'

                video_duration = row['duration_sec']
                keystep_timestamp = row['narration_frame']
                keystep = row['narration']
                verbs, nouns = parse_sentence(keystep)
                verb_class = get_class_index(verbs, verb_dict_keysteps)
                noun_class = get_class_index(nouns, noun_dict_keysteps)
                row_dict = {
                    'video_id': video_id,
                    'duration': video_duration,
                    'narration_frame': keystep_timestamp,
                    'narration': keystep,
                    'verb_class': verb_class,
                    'noun_class': noun_class,
                }
                if split != "train":
                    row_dict['start_frame'] = row['start_frame']
                    row_dict['end_frame'] = row['end_frame']
                exo_rows_keysteps.append(row_dict)

            video_id = f'{take_uid}_{ego_camera_name}'
            video_duration = row['duration_sec']
            keystep_timestamp = row['narration_frame']
            keystep = row['narration']
            verbs, nouns = parse_sentence(keystep)
            verb_class = get_class_index(verbs, verb_dict_keysteps)
            noun_class = get_class_index(nouns, noun_dict_keysteps)
            row_dict = {
                'video_id': video_id,
                'duration': video_duration,
                'narration_frame': keystep_timestamp,
                'narration': keystep,
                'verb_class': verb_class,
                'noun_class': noun_class,
            }
            if split != "train":
                    row_dict['start_frame'] = row['start_frame']
                    row_dict['end_frame'] = row['end_frame']
            ego_rows_keysteps.append(row_dict)

        """ exo_df_narrations = pd.DataFrame(exo_rows_narrations)
        exo_df_narrations = exo_df_narrations.sort_values(by=['video_id', 'narration_frame'])
        exo_df_narrations['clip_id'] = exo_df_narrations['video_id'] + '_' + exo_df_narrations.groupby('video_id').cumcount().astype(str)
        exo_df_narrations.to_csv(output_dir + f'egoexo4d_exos_narrations_{split}.csv', index=False)

        ego_df_narrations = pd.DataFrame(ego_rows_narrations)
        ego_df_narrations = ego_df_narrations.sort_values(by=['video_id', 'narration_frame'])
        ego_df_narrations['clip_id'] = ego_df_narrations['video_id'] + '_' + ego_df_narrations.groupby('video_id').cumcount().astype(str)
        ego_df_narrations.to_csv(output_dir + f'egoexo4d_egos_narrations_{split}.csv', index=False)

        all_views_df_narrations = pd.concat([exo_df_narrations, ego_df_narrations])
        all_views_df_narrations.to_csv(output_dir + f'egoexo4d_all_views_narrations_{split}.csv', index=False) """

        exo_df_keysteps = pd.DataFrame(exo_rows_keysteps)
        exo_df_keysteps = exo_df_keysteps.sort_values(by=['video_id', 'narration_frame'])
        exo_df_keysteps['clip_id'] = exo_df_keysteps['video_id'] + '_' + exo_df_keysteps.groupby('video_id').cumcount().astype(str)
        exo_df_keysteps.to_csv(output_dir + f'egoexo4d_exos_keysteps_{split}.csv', index=False)

        ego_df_keysteps = pd.DataFrame(ego_rows_keysteps)
        ego_df_keysteps = ego_df_keysteps.sort_values(by=['video_id', 'narration_frame'])
        ego_df_keysteps['clip_id'] = ego_df_keysteps['video_id'] + '_' + ego_df_keysteps.groupby('video_id').cumcount().astype(str)
        ego_df_keysteps.to_csv(output_dir + f'egoexo4d_egos_keysteps_{split}.csv', index=False)

        all_views_df_keysteps = pd.concat([exo_df_keysteps, ego_df_keysteps])
        all_views_df_keysteps.to_csv(output_dir + f'egoexo4d_all_views_keysteps_{split}.csv', index=False)
    
if __name__ == '__main__':
    main()