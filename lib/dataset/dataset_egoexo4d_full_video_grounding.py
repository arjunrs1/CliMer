import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

import pandas as pd
import os
from gulpio2 import GulpDirectory
import random
import numpy as np
import lmdb
import pickle
import time
from natsort import natsorted
import math
import json


class EgoExo4DDatasetGrounding(Dataset):
    def __init__(self, annotations_file, captions_file, same_vid_sampling,
                 metadata, fps, feature_stride, use_keysteps, device):
        self.data = pd.read_csv(annotations_file)
        self.full_metadata = pd.read_csv(metadata)
        self.captions_file = captions_file  # path to the lmdb containing the bert features
        self.device = device
        self.same_vid_sampling = same_vid_sampling
        self.fps = fps
        self.feature_stride = feature_stride
        self.use_keysteps = use_keysteps
        
        self.text_annotations_file = f'/private/home/arjunrs1/exo_narration_grounding/data_processing/time_interval_annotation_files/keystep_annotations/test.csv'
        self.text_annotations = pd.read_csv(self.text_annotations_file)
        self.mapping_file_path = "/private/home/arjunrs1/CliMer/data/egoexo4d/egoexo4d_all_views_keysteps_test.csv"
        self.mapping_file = pd.read_csv(self.mapping_file_path)
        self.test_path = f"/private/home/arjunrs1/exo_narration_grounding/splits/egoexo4d_splits/test.csv"
        self.test_data = pd.read_csv(self.test_path)
        self.base_path = '/private/home/arjunrs1/egoexo4d_features'
        self.camera_rankings_path = os.path.join(self.base_path, "all_camera_rankings.json")
        self.valid_take_uid_list_path = "/private/home/arjunrs1/egoexo4d_features/takes_to_remove.npy"
        valid_take_uid_list = list(np.load(self.valid_take_uid_list_path))

        # Function to extract take_uid from video_id
        def extract_take_uid(video_id):
            rsplit_idx = 2 if "aria" in video_id else 1
            return video_id.rsplit('_', rsplit_idx)[0]
        
        #self.data['take_uid'] = self.data['video_id'].apply(extract_take_uid)
        #self.data = self.data[self.data['take_uid'].isin(valid_take_uid_list)]
        #print(self.data.columns)
        #self.data['narration_frame'] = self.data['narration_frame'].astype(int)
        #self.text_annotations['narration_frame'] = self.text_annotations['narration_frame'].astype(int)
        #self.data['take_uid'] = self.data['take_uid'].str.strip().str.lower()
        #self.text_annotations['take_uid'] = self.text_annotations['take_uid'].str.strip().str.lower()
        #merged_data = pd.merge(self.data, self.text_annotations, on=['narration_frame', 'take_uid'], how='inner', suffixes=('', '_y'), indicator=True)
        #climer_data_filtered = merged_data[merged_data['_merge'] == 'both'].drop(columns=['_merge'])
        #self.clip_id_to_unique_narr_id_map = climer_data_filtered[['clip_id', 'unique_narration_id']]
        #self.data = climer_data_filtered

        with open(self.camera_rankings_path, "rb") as f:
            self.camera_rankings = json.load(f)

    def __len__(self):
        return len(self.data)

    def find_key_by_value(self, data, search_value):
        if data:
            for key, value in data.items():
                if value == search_value:
                    return key
        return "unk"

    def __getitem__(self, idx):
        #video_id,exo_cam,ego_cam,start_sec,end_sec,narration_ids
        #georgiatech_bike_06_10,cam01,aria05_214-1,0,64,"georgiatech_bike_06_10_ks_0,georgiatech_bike_06_10_ks_1,georgiatech_bike_06_10_ks_2,georgiatech_bike_06_10_ks_11,georgiatech_bike_06_10_ks_12,georgiatech_bike_06_10_ks_13"
        clip1 = self.data.iloc[idx]
        narration_clip = "test" #TODO: Replace back to this ===> self.data['narration'][idx]
        feature_idxs, feature_times, labels, clip_start_stop_time, clip_id, video_id, view_rank= self.get_feature_idxs(clip1, use_egovlp_ids=True)
        bert_features, num_tokens = self.get_egovlp_features(clip_id)

        return feature_idxs, feature_times, bert_features, num_tokens, labels, narration_clip, clip_start_stop_time, \
               video_id, view_rank

    def get_feature_idxs(self, window, use_egovlp_ids=True):
        """Get the feature indices for a randomly selected 64-second chunk that overlaps with the narration interval."""
        narr_ids = window['narration_ids'].split(',')
        exo_cam = window['exo_cam']
        ego_cam = window['ego_cam']
        video_id_raw = window['video_id']
        take_uid = self.test_data[self.test_data['take_name'] == video_id_raw]['take_uid'].iloc[0]
        video_id = f"{window['video_id']}_{exo_cam}"
        chunk_start_time = window['start_sec']
        chunk_stop_time = window['end_sec']
        #clip_ids = [id.replace("ks", exo_cam) for id in narr_ids]
        clip_ids = []
        # take_ego_id = f"{video_id}_{ego_cam}"
        # narration_ids = window['narration_ids'].split(',')
        # exo_cams = ast.literal_eval(exo_cams) if self.multi_view else [exo_cams]
        #clip_id = self.data['clip_id'][idx] #formatted as: georgiatech_bike_06_10_cam01_0 -> Verify that it is same as georgiatech_bike_06_10_ks_0
        clip_start_times = []
        clip_stop_times = []

        target = self.camera_rankings[take_uid]
        per_second_views = []
        for t in range(chunk_start_time, chunk_stop_time):
            tth_second_rank = target[str(t)]
            assert tth_second_rank is not None
            curr_view_rank = "ego" if ego_cam == exo_cam else self.find_key_by_value(tth_second_rank, exo_cam)
            per_second_views.append(curr_view_rank)
        view_rank = max(set(per_second_views), key=per_second_views.count)

        narrations = self.text_annotations[self.text_annotations['unique_narration_id'].isin(narr_ids)]
        for _, row in narrations.iterrows():
            narration_frame = row['narration_frame']
            start_frame = row['start_frame']
            end_frame = row['end_frame']
            matching_rows = self.mapping_file[
                (self.mapping_file['narration_frame'] == narration_frame) &
                (self.mapping_file['start_frame'] == start_frame) &
                (self.mapping_file['end_frame'] == end_frame) &
                (self.mapping_file['video_id'] == video_id)
            ]
            assert len(matching_rows) == 1, "Expected exactly one matching row, but found {}".format(len(matching_rows))
            # Get the clip_id from the matching row
            clip_id = matching_rows.iloc[0]['clip_id']
            clip_ids.append(clip_id)
            start = max(chunk_start_time, (int(row['start_frame']) / self.fps))
            stop = min(chunk_stop_time, (int(row['end_frame']) / self.fps))
            clip_start_times.append(start)
            clip_stop_times.append(stop)

        if use_egovlp_ids:
            clip_ids = narr_ids
        
        # Calculate the feature indices and times for the chunk
        feature_idx_step = int(1)
        chunk_start_feature_idx = np.floor(chunk_start_time / (self.feature_stride / self.fps))
        chunk_end_feature_idx = np.floor(chunk_stop_time / (self.feature_stride / self.fps))
        feature_idxs = np.arange(chunk_start_feature_idx, chunk_end_feature_idx, feature_idx_step)
        feature_times = (feature_idxs + 1) * (self.feature_stride / self.fps)

        # Create labels for the chunk
        num_frames_in_chunk = len(feature_idxs)
        uncombined_labels_list = []
        clip_start_stop_times = []
        for clip_s, clip_e in zip(clip_start_times, clip_stop_times):
            uncombined_labels = torch.zeros(num_frames_in_chunk, dtype=torch.float32)
            for i, feature_time in enumerate(feature_times):
                if clip_s <= feature_time <= clip_e:
                    uncombined_labels[i] = 1
            uncombined_labels_list.append(uncombined_labels)
            clip_start_stop_time = [clip_s, clip_e]
            clip_start_stop_times.append(clip_start_stop_time)
        return feature_idxs, feature_times, uncombined_labels_list, clip_start_stop_times, clip_ids, video_id, view_rank
    
    def get_egovlp_features(self, clip_ids):
        features = []
        num_tokens = []
        max_length = 20
        for clip_id in clip_ids:
            print("CLIP ID:")
            print(clip_id)
            features_1 = torch.load(os.path.join(self.captions_file, clip_id.split("_ks")[0], f"{clip_id}.pt"), map_location='cpu')
            features.append(features_1.expand(max_length,-1))
            num_tokens.append(3)

        return features, num_tokens

    def get_bert_features(self, clip_ids):
        features = []
        num_tokens = []
        max_length = 20
        for clip_id in clip_ids:
            take_sep_ind_1 = 3 if "aria" in clip_id.lower() else 2
            narr_postfix_1 = clip_id.rsplit('_', 2)[-1]

            if self.use_keysteps:
                narr_postfix_1 = "ks_" + narr_postfix_1

            narr_id_1 = f"{clip_id.rsplit('_', take_sep_ind_1)[0]}_{narr_postfix_1}.pt"
            features_1_full = torch.load(os.path.join(self.captions_file, narr_id_1), map_location='cpu')

            features_1 = features_1_full['features'].detach().squeeze(0).type(torch.float32)
            num_tokens_1 = features_1_full['num_tokens'] - max(features_1.shape[0] - max_length, 0)
            if features_1.shape[0] < max_length:
                features_1 = torch.cat((features_1, torch.zeros(max_length - features_1.shape[0], 768)), dim=0)
            else:
                features_1 = features_1[:max_length]
            features.append(features_1)
            num_tokens.append(num_tokens_1) 

        return features, num_tokens
