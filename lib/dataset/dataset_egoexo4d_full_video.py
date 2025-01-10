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
import math


class EgoExo4DDatasetFullVideo(Dataset):
    def __init__(self, annotations_file, captions_file, same_vid_sampling,
                 metadata, fps, feature_stride, use_keysteps, device):
        self.data = pd.read_csv(annotations_file)
        print(f"annotations_file: {annotations_file}")
        self.full_metadata = pd.read_csv(metadata)
        self.captions_file = captions_file  # path to the lmdb containing the bert features
        self.device = device
        self.same_vid_sampling = same_vid_sampling
        self.fps = fps
        self.feature_stride = feature_stride
        self.use_keysteps = use_keysteps

        self.split = "train" if "train" in annotations_file else ("val" if "val" in annotations_file else "test")
        self.text_annotations_file = f'/private/home/arjunrs1/exo_narration_grounding/data_processing/time_interval_annotation_files/keystep_annotations/{self.split}.csv'
        self.text_annotations = pd.read_csv(self.text_annotations_file)
        self.valid_take_uid_list_path = "/private/home/arjunrs1/egoexo4d_features/takes_to_remove.npy"
        valid_take_uid_list = list(np.load(self.valid_take_uid_list_path))

        # Function to extract take_uid from video_id
        def extract_take_uid(video_id):
            rsplit_idx = 2 if "aria" in video_id else 1
            return video_id.rsplit('_', rsplit_idx)[0]
        
        self.data['take_uid'] = self.data['video_id'].apply(extract_take_uid)
        self.data = self.data[self.data['take_uid'].isin(valid_take_uid_list)]
        self.data['narration_frame'] = self.data['narration_frame'].astype(int)
        self.text_annotations['narration_frame'] = self.text_annotations['narration_frame'].astype(int)
        self.data['take_uid'] = self.data['take_uid'].str.strip().str.lower()
        self.text_annotations['take_uid'] = self.text_annotations['take_uid'].str.strip().str.lower()
        merged_data = pd.merge(self.data, self.text_annotations, on=['narration_frame', 'take_uid'], how='inner', suffixes=('', '_y'), indicator=True)
        climer_data_filtered = merged_data[merged_data['_merge'] == 'both'].drop(columns=['_merge'])
        self.clip_id_to_unique_narr_id_map = climer_data_filtered[['clip_id', 'unique_narration_id']]
        self.data = climer_data_filtered

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        clip1_idx = idx
        narration_clip = self.data['narration'][idx]
        feature_idxs, feature_times, labels, clip_start_stop_time, clip_id, video_id = self.get_feature_idxs(clip1_idx)
        bert_features, num_tokens = self.get_egovlp_features(clip_id)

        return feature_idxs, feature_times, bert_features, num_tokens, labels, narration_clip, clip_start_stop_time, \
               video_id

    def get_feature_idxs_old(self, idx):
        """Get the file names of the relevant frames for a clip and add the background frame file names
        to the list
        Returns a list of frame filenames"""
        video_id = self.data['video_id'][idx]
        clip_id = self.data['clip_id'][idx]
        clip_start_frame = self.data['start_frame'][idx]
        clip_stop_frame = self.data['end_frame'][idx]
        clip_start_time = self.data['start_frame'][idx] / self.fps
        clip_stop_time = self.data['end_frame'][idx] / self.fps
        video_durations = self.data['duration']

        video_duration = video_durations[idx]
        feature_idx_step = int(1)
        video_end_feature_idx = np.floor(video_duration / (self.feature_stride / self.fps))

        feature_idxs = np.arange(0, video_end_feature_idx, feature_idx_step)
        feature_times = (feature_idxs + 1) * (self.feature_stride / self.fps)

        num_frames = np.floor(video_duration * self.fps)

        feature_locs = np.arange(self.feature_stride - 1, num_frames, self.feature_stride)

        uncombined_labels = []
        for f in feature_locs:
            if f > (clip_stop_frame - 1) or f < (clip_start_frame - 1):
                uncombined_labels.append(0)
            else:
                uncombined_labels.append(1)

        uncombined_labels = torch.tensor(uncombined_labels, dtype=torch.float32)

        clip_start_stop_time = [clip_start_time, clip_stop_time]

        return feature_idxs, feature_times, uncombined_labels, clip_start_stop_time, clip_id, video_id

    def get_feature_idxs(self, idx):
        """Get the feature indices for a randomly selected 64-second chunk that overlaps with the narration interval."""
        video_id = self.data['video_id'][idx]
        clip_id = self.data['clip_id'][idx]
        clip_start_time = self.data['start_frame'][idx] / self.fps
        clip_stop_time = self.data['end_frame'][idx] / self.fps
        video_duration = self.data['duration'][idx]
        # Define the duration of the chunk in seconds
        chunk_duration = 64
        # Calculate the valid start range to ensure at least some overlap with the narration interval
        valid_start_min = max(0, clip_stop_time - chunk_duration)
        valid_start_max = min(clip_start_time, video_duration - chunk_duration)
        
        if valid_start_max <= valid_start_min:
            # If the range is invalid, default to starting at the clip start time or as close as possible
            chunk_start_time = max(0, min(valid_start_min, video_duration - chunk_duration))
        else:
            # Randomly select the start time of the chunk within the valid range
            chunk_start_time = np.random.uniform(valid_start_min, valid_start_max)
        chunk_stop_time = chunk_start_time + chunk_duration
        # Calculate the feature indices and times for the chunk
        feature_idx_step = int(1)
        chunk_start_feature_idx = np.floor(chunk_start_time / (self.feature_stride / self.fps))
        chunk_end_feature_idx = np.floor(chunk_stop_time / (self.feature_stride / self.fps))
        feature_idxs = np.arange(chunk_start_feature_idx, chunk_end_feature_idx, feature_idx_step)
        feature_times = (feature_idxs + 1) * (self.feature_stride / self.fps)
        # Create labels for the chunk
        num_frames_in_chunk = len(feature_idxs)
        uncombined_labels = torch.zeros(num_frames_in_chunk, dtype=torch.float32)
        for i, feature_time in enumerate(feature_times):
            if clip_start_time <= feature_time <= clip_stop_time:
                uncombined_labels[i] = 1
        clip_start_stop_time = [clip_start_time, clip_stop_time]
        return feature_idxs, feature_times, uncombined_labels, clip_start_stop_time, clip_id, video_id

    def get_bert_features(self, clip_id):
        max_length = 20

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

        return features_1, num_tokens_1

    def get_egovlp_features(self, clip_id):
        #get clip1 features
        narr_id_1_row = self.clip_id_to_unique_narr_id_map.loc[self.clip_id_to_unique_narr_id_map['clip_id'] == clip_id]
        narr_id_1 = narr_id_1_row['unique_narration_id'].iloc[0]
        print(narr_id_1)
        print(self.captions_file)
        exit()
        features_1 = torch.load(os.path.join(self.captions_file, narr_id_1.split("_ks")[0], f"{narr_id_1}.pt"), map_location='cpu')

        return features_1.expand(20,-1), 3

#TODO: Switch to loading in from the test grounding windows directly
#TODO: Switch get_feature_idxs() fn to load in the window, use the annotated chunk (instead of computing the chunk as we do here)
#TODO: Switch to altering the clip_start_stop_time to be a list of lists ((start, stop) for EACH keystep). Adjust boundaries if necessary (clip to beg/end)
#TODO: In get_bert_features(), return LIST of features_1, num_tokens_1
#TODO: In evaluation.py, simply pass in each feature SEPARATELY along with the chunks to the grounding.
#TODO: Then, make sure that we compute IoU metrics across ALL the narrations in the chunk. May require separate fn to do this.
