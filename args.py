import argparse
from pathlib import Path

def create_parser():
    parser = argparse.ArgumentParser(
        description="Train a temporal grounding model on a long-form fine-grained video dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="ego4d", type=str, help="which dataset to use, ego4d or epic")
    parser.add_argument("--val", default=False)
    parser.add_argument("--model-path", default="pretrained/ego4d_pretrained.pth")
    parser.add_argument("--video-feature-path", default="features/ego4d/omnivore_features_ego4d/")
    parser.add_argument("--caption-data-train", default="data/ego4d/ego4d_train.csv")
    parser.add_argument("--caption-data-val", default="data/ego4d/ego4d_val.csv")
    parser.add_argument("--caption-data-test", default="data/ego4d/ego4d_test.csv")
    parser.add_argument("--window-data-test", default="/private/home/arjunrs1/egoexo4d_features/grounding_test_exo_ks=True_ct=False_exos=all_windows.csv")
    parser.add_argument("--ego4d-metadata", default="data/ego4d/ego4d_metadata.csv")
    parser.add_argument("--epic-metadata", default="data/epic/epic_metadata.csv")
    parser.add_argument("--egoexo4d-train-metadata", default="/private/home/arjunrs1/exo_narration_grounding/splits/egoexo4d_splits/train.csv")
    parser.add_argument("--egoexo4d-val-metadata", default="/private/home/arjunrs1/exo_narration_grounding/splits/egoexo4d_splits/val.csv")
    parser.add_argument("--egoexo4d-test-metadata", default="/private/home/arjunrs1/exo_narration_grounding/splits/egoexo4d_splits/val.csv")
    parser.add_argument("--epic-all-data", default="data/epic/epic_all_data.csv")
    parser.add_argument("--egoexo4d-all-data", default="/private/home/arjunrs1/exo_narration_grounding/data_processing/time_interval_annotation_files/narration_annotations/train.csv")
    parser.add_argument("--bert-features", default="features/ego4d/BERT_features_ego4d.lmdb")
    parser.add_argument("--results-path", default="results/ego4d/")
    parser.add_argument("--egovlp-data", default="data/ego4d/egovlp_params_correct.json")
    parser.add_argument("--random-baseline", default=False, type=bool, help="Whether to use random baseline for evaluation")
    parser.add_argument("--log-dir", default=Path("/private/home/arjunrs1/CliMer/logs"), type=Path)
    parser.add_argument("--learning-rate", default=1e-5, type=float, help="Learning rate")
    parser.add_argument("--train-batch-size", default=32, type=int, help="Number of examples in each batch")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs to train for")
    parser.add_argument("--val-frequency", default=1, type=int, help="How frequently to test the model on the validation set in number of epochs")
    parser.add_argument("--log-frequency", default=10, type=int, help="How frequently to save logs to tensorboard in number of steps")
    parser.add_argument("--checkpoint-save-path", type=Path, default='/private/home/arjunrs1/CliMer/checkpoints')
    parser.add_argument("--checkpoint-frequency", type=int, default=1, help="Save a checkpoint every N epochs")
    parser.add_argument("--worker-count", default=6, type=int, help="Number of worker processes used to load data.")
    parser.add_argument("--same-vid-sampling", default=True, type=bool, help="whether or not combined videos should be sampled from the same video")
    parser.add_argument("--cross-attention", default=True, type=bool, help="Whether to use cross attention between words and frames as interaction rather than hadamard product")
    parser.add_argument("--balanced", default=True, type=bool, help="Whether to balance positive and negative frames in the loss")
    parser.add_argument("--combine", default=True, type=bool, help="Whether to use the video combination trick or to just use a single caption in its own sequence")
    parser.add_argument("--fixed-clip-length", default="None", type=str, help="Whether to use a fixed clip length - value is string of the length or 'None'")
    parser.add_argument("--clip-adjacent-timestamps", default="None", type=str, help="Whether to use adjacent timestamps as clip boundaries - value is 'half', 'full' or 'None'")
    parser.add_argument("--egovlp", default=False, type=bool, help="Whether to use egovlp method for generating clips")
    parser.add_argument("--fps", default=30, type=float, help="FPS of the videos for which features have been generated")
    parser.add_argument("--feature-stride", default=16, type=float, help="stride in number of frames of the features generated for each video")
    parser.add_argument("--seg-size", default=2000, type=float, help="Segment size for inference")
    parser.add_argument("--overlap", default=1000, type=float, help="Overlap between segments for inference")
    parser.add_argument("--pred-threshold", default=0.8, type=float, help="Prediction threshold during evaluation")
    parser.add_argument("--visual-feature-orig-dim", default=1536, type=int, help="size of input visual features")
    parser.add_argument("--cap-orig-dim", default=768, type=int, help="size of input caption features")
    parser.add_argument("--shared-projection-dim", default=2048, type=int, help="size of projection")
    parser.add_argument("--feature-embed-dim", default=3072, type=int, help="size of feature embedding dimension, split across heads")
    parser.add_argument("--linear-hidden-dim", default=1024, type=int, help="size of hidden dimension in linear layers")
    parser.add_argument("--num-heads", default=6, type=int, help="size of projection")
    parser.add_argument("--use-keysteps", default=False, type=bool, help="Whether to train on keysteps or not")
    
    return parser


