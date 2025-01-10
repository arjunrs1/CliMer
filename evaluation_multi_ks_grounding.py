import torch
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import argparse
from multiprocessing import cpu_count, set_start_method

from torch import nn
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import cv2

from args import create_parser
from lib.model.model_x import TemporalGroundingCross
from lib.model.model_lightweight import TemporalGrounding
from lib.dataset.dataset_ego4d_full_video import Ego4dDatasetFullVideo
from lib.dataset.dataset_epic_full_video import EpicDatasetFullVideo
from lib.dataset.dataset_egoexo4d_full_video import EgoExo4DDatasetFullVideo
from lib.dataset.dataset_egoexo4d_full_video_grounding import EgoExo4DDatasetGrounding
from utils.data_utils import load_omnivore_clip_features, load_all_omnivore_features, load_all_egovlpv2_features
from utils.utils import iou_stats, iou_time, iou_time_based, get_hard_preds, split_features, recombine_preds

# Define an argument parser to allow relevant parameters for training to be set
parser = create_parser()

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print('device', DEVICE)

def main(args):
    if args.dataset == "ego4d":
        test_dataset = Ego4dDatasetFullVideo(args.caption_data_val, args.bert_features,
                                             args.same_vid_sampling, args.ego4d_metadata, args.fps, args.feature_stride,
                                             DEVICE)
    elif args.dataset == "egoexo4d":
        test_dataset = EgoExo4DDatasetGrounding(args.window_data_test, args.bert_features,
                                             args.same_vid_sampling, args.egoexo4d_test_metadata, args.fps, args.feature_stride,
                                             use_keysteps=True, device=DEVICE)
    else:
        test_dataset = EpicDatasetFullVideo(args.caption_data_test, args.bert_features,
                                            args.same_vid_sampling, args.epic_metadata, args.fps, args.feature_stride,
                                            DEVICE)

    # VALIDATION
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.worker_count,
        pin_memory=True,
    )

    visual_feature_orig_dim = args.visual_feature_orig_dim  # omnivore feature size
    cap_orig_dim = args.cap_orig_dim
    visual_feature_input_dim = args.shared_projection_dim
    cap_embed_dim = args.shared_projection_dim
    feature_embed_dim = args.feature_embed_dim
    linear_hidden_dim = args.linear_hidden_dim
    num_heads = args.num_heads

    if args.cross_attention:
        model = TemporalGroundingCross(visual_projection_switch=True, visual_feature_orig_dim=visual_feature_orig_dim,
                                       visual_feature_input_dim=visual_feature_input_dim,
                                       feature_embed_dim=feature_embed_dim,
                                       linear_hidden_dim=linear_hidden_dim, num_heads=num_heads,
                                       cap_input_dim=cap_orig_dim, cap_embed_dim=cap_embed_dim,
                                       device=DEVICE)
    else:
        model = TemporalGrounding(visual_projection_switch=True, visual_feature_orig_dim=visual_feature_orig_dim,
                                  visual_feature_input_dim=visual_feature_input_dim,
                                  feature_embed_dim=feature_embed_dim, linear_hidden_dim=linear_hidden_dim,
                                  num_heads=num_heads, cap_input_dim=cap_orig_dim, cap_embed_dim=cap_embed_dim,
                                  device=DEVICE)

    model = model.to(DEVICE)
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE), strict=False)
    model.eval()

    if args.random_baseline:
        iou_list, captions, preds_cap1, labels_cap1, feature_idxs, feature_times, clip_start_stop_times = \
            total_random_baseline(test_loader, args)
    else:
        iou_list, captions, preds_cap1, labels_cap1, feature_idxs, feature_times, clip_start_stop_times = \
            validate_full_videos(model, test_loader, args, DEVICE)


def validate_full_videos(model, test_loader, args, device):
    iou_list = []
    preds1_list = []
    labels1_list = []
    captions_list = []
    feature_idxs_list = []
    feature_times_list = []
    clip_start_stop_time_list = []
    iou_time_list = torch.tensor([])
    view_ranks = []

    if args.dataset == "egoexo4d":
        omnivore_dict = load_all_egovlpv2_features(args.caption_data_train, args.caption_data_val,
                                                            args.caption_data_test, args.video_feature_path)
    else:
        omnivore_dict = load_all_omnivore_features(args.caption_data_train, args.caption_data_val, args.caption_data_test,
                                               args.video_feature_path)

    model.eval()
    for feature_idxs, feature_times, bert_features, num_tokens, label_caption1, caption1, clip_start_stop_time, \
        video_id, view_rank in tqdm(test_loader):

        features = load_omnivore_clip_features(omnivore_dict, feature_idxs, video_id).to(device)

        num_features = feature_idxs.shape[1]

        #ground each keystep in current window separately, but compute statistics over window before appending to iou_list, etc.
        for bert_f, lab_cap, num_tok, cl_sta_sto in zip(bert_features, label_caption1, num_tokens, clip_start_stop_time):
            bert_features_cap1 = bert_f.to(device)
            label_caption1 = lab_cap.to(device)
            num_tokens_cap1 = num_tok.to(device)
        
            assert num_features == label_caption1.size()[1]
            # Calculate the output logits
            if num_features <= args.seg_size:
                preds_cap1 = model.predict(features, bert_features_cap1, num_tokens_cap1, args.val)
                preds_cap1 = preds_cap1.squeeze(2).detach()
            else:
                feature_segment_list, feature_idx_list = split_features(features, args.seg_size, args.overlap)
                full_pred_list = []
                for feature_segment in feature_segment_list:
                    preds_cap1 = model.predict(feature_segment, bert_features_cap1, num_tokens_cap1, args.val)
                    preds_cap1 = preds_cap1.squeeze(2).detach()
                    preds_cap1 = preds_cap1.tolist()
                    full_pred_list.append(preds_cap1)
                preds_cap1 = recombine_preds(full_pred_list, feature_idx_list, num_features)
                preds_cap1 = torch.tensor(preds_cap1).unsqueeze(0).to(device)
                preds_cap1 = preds_cap1.detach()

            # Now want to get hard predictions from these predictions:
            final_preds = get_hard_preds(preds_cap1, threshold=args.pred_threshold)
            iou_time, pred_start_time, pred_end_time = iou_time_based(final_preds, feature_times, cl_sta_sto)
            iou_time_list = torch.cat((iou_time_list, iou_time))
            view_ranks.append(view_rank[0])

            video_path = os.path.join(args.video_feature_path, video_id[0] + '.mp4')
            pred_start_end_time = (pred_start_time, pred_end_time)
            visualize_grounding(video_path, cl_sta_sto, pred_start_end_time)

            preds1_list.append(preds_cap1)
            labels1_list.append(label_caption1)
            captions_list.append(caption1)
            feature_idxs_list.append(feature_idxs)
            feature_times_list.append(feature_times)
            clip_start_stop_time_list.append(cl_sta_sto)

    # iou stats
    print("LENGTH IOU TIME LIST:")
    print(len(iou_time_list))
    print(len(view_ranks))
    print(view_ranks)

    iou_time_above_1, iou_time_above_3, iou_time_above_5, iou_time_above_7, \
        frac_time_above_1, frac_time_above_3, frac_time_above_5, frac_time_above_7 \
        = iou_stats(iou_time_list, view_ranks)

    mr = (frac_time_above_1 + frac_time_above_3 + frac_time_above_5 + frac_time_above_7) / 4

    print(f'Threshold {args.pred_threshold}  0.1: {frac_time_above_1},  0.3: {frac_time_above_3},  '
          f'0.5: {frac_time_above_5},  0.7: {frac_time_above_7},  mR: {mr}')

    with open(os.path.join(args.results_path, args.dataset), 'a') as file:
        file.write(f'Threshold {args.pred_threshold}  0.1: {frac_time_above_1},  0.3: {frac_time_above_3},  '
                   f'0.5: {frac_time_above_5},  0.7: {frac_time_above_7},  mR: {mr} \n')

    return iou_list, captions_list, preds1_list, labels1_list, feature_idxs_list, feature_times_list, \
        clip_start_stop_time_list

def visualize_grounding(video_path, clip_start_stop_time, pred_start_end_time):
    # Load the video
    cap = cv2.VideoCapture(video_path)
    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Calculate the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Convert the clip start and stop times to frame numbers
    clip_start_frame = int(clip_start_stop_time[0] * fps)
    clip_stop_frame = int(clip_start_stop_time[1] * fps)
    # Convert the predicted start and end times to frame numbers
    pred_start_frame = int(pred_start_end_time[0] * fps)
    pred_end_frame = int(pred_start_end_time[1] * fps)
    # Iterate over the frames in the video
    for i in range(total_frames):
        ret, frame = cap.read()
        # If we are within the clip, draw a green rectangle around the frame
        if clip_start_frame <= i <= clip_stop_frame:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 5)
        # If we are within the predicted segment, draw a red rectangle around the frame
        if pred_start_frame <= i <= pred_end_frame:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 5)
        # Display the frame
        cv2.imshow('Frame', frame)
        # Exit on key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release the video capture object
    cap.release()
    # Close all OpenCV windows
    cv2.destroyAllWindows()


def total_random_baseline(test_loader, args):

    iou_list = []
    preds1_list = []
    labels1_list = []
    captions_list = []
    feature_idxs_list = []
    feature_times_list = []
    clip_start_stop_time_list = []
    iou_time_list = torch.tensor([])
    view_ranks = []

    ### ORIGINAL CODE BELOW----------------------

    iou_time_list = torch.tensor([])
    labels_cap1 = []
    clip_start_stop_times = []
    captions = []
    for feature_idxs, feature_times, bert_features, num_tokens, label_caption1, caption1, clip_start_stop_time, \
        video_id, view_rank in tqdm(test_loader):

        #ground each keystep in current window separately, but compute statistics over window before appending to iou_list, etc.
        for bert_f, lab_cap, num_tok, cl_sta_sto in zip(bert_features, label_caption1, num_tokens, clip_start_stop_time):

            labels_cap1.append(lab_cap)
            clip_start_stop_time_list.append(cl_sta_sto)
            captions.append(caption1)
            duration = feature_times[0, -1]
            pred_times = np.random.uniform(0, duration, 2)
            pred_times = torch.tensor(pred_times)
            pred_start_time = pred_times[0]
            pred_end_time = pred_times[1]
            start_time = cl_sta_sto[0]
            end_time = cl_sta_sto[1]
            iou_time_value = iou_time(pred_start_time, pred_end_time, start_time, end_time)

            iou_time_list = torch.cat((iou_time_list, iou_time_value))
            view_ranks.append(view_rank[0])

    iou_time_above_1, iou_time_above_3, iou_time_above_5, iou_time_above_7, \
        frac_time_above_1, frac_time_above_3, frac_time_above_5, frac_time_above_7 \
        = iou_stats(iou_time_list, view_ranks)

    mr = (frac_time_above_1 + frac_time_above_3 + frac_time_above_5 + frac_time_above_7) / 4

    print(f'0.1: {frac_time_above_1},  0.3: {frac_time_above_3},  0.5: {frac_time_above_5},  0.7: {frac_time_above_7}'
          f'  mR:  {mr} \n')
    with open(args.results_file, 'a') as file:
        file.write(f'0.1: {frac_time_above_1},  0.3: {frac_time_above_3},  '
                   f'0.5: {frac_time_above_5},  0.7: {frac_time_above_7},  mR: {mr} \n')

    return iou_time_list, captions, pred_times, labels_cap1, feature_idxs, feature_times, clip_start_stop_times


if __name__ == "__main__":
    main(parser.parse_args())
