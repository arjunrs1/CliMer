from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple
import pandas as pd
import cv2
import torch
import requests
import numpy as np
from PIL import Image
import json
from tqdm import tqdm
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data.dataloader import default_collate

class VideoFrameDataset(Dataset):
    def __init__(self, annotations_df_path, transform=None):
        """
        Args:
            annotations_df (pd.DataFrame): DataFrame containing video annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        data = pd.read_csv(annotations_df_path)
        data['duration'] = pd.to_numeric(data['duration'], errors='coerce')
        data.sort_values(by='duration', ascending=True, inplace=True)
        self.annotations_df = data
        self.transform = transform

    def __len__(self):
        return len(self.annotations_df)

    def sample_frames(self, start_frame, end_frame, num_samples=1):
        return np.linspace(start_frame, end_frame, num=num_samples, dtype=int)

    @staticmethod
    def collate_fn(batch):
        images = [item['image'] for item in batch]
        nouns = [item['nouns'] for item in batch]
        narr_ids = [item['narr_id'] for item in batch]
        images = torch.stack(images, dim=0)  # Assuming images are already tensors
        return {'image': images, 'nouns': nouns, 'narr_ids': narr_ids}

    def __getitem__(self, idx):
        row = self.annotations_df.iloc[idx]
        video_name = row['video_id']
        start_frame = row['start_frame']
        end_frame = row['end_frame']
        narr_id = row['clip_id']
        nouns = eval(row['noun_class'])
        
        take_name = video_name.rsplit("_", 1)[0] if "aria" not in video_name else video_name.rsplit("_", 2)[0]
        cam_id = video_name.rsplit("_", 1)[1] if "aria" not in video_name else "_".join(video_name.rsplit("_", 2)[1:])

        video_path = f'/datasets01/egoexo4d/v2/takes/{take_name}/frame_aligned_videos/downscaled/448/{cam_id}.mp4'
        frame_number = self.sample_frames(start_frame, end_frame)[0] #TODO: Adjust to allow multiple frames
        nouns = eval(row['noun_class'])  # Assuming nouns are stored as a string representation of a list
        if "hand" not in nouns:
            if "hands" not in nouns:
                nouns.append("hand")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Video {video_path} cannot be opened")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise IOError(f"Failed to read frame {frame_number} from video {video_path}")
        
        # Convert frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)

        if self.transform:
            image = self.transform(image)

        return {"image": image, "nouns": nouns, "narr_id": narr_id}

@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))

def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon

def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert a polygon to a segmentation mask.

    Args:
    - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
    - image_shape (tuple): Shape of the image (height, width) for the mask.

    Returns:
    - np.ndarray: Segmentation mask with the polygon filled.
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask

def load_image(image_str: str) -> Image.Image:
    if image_str.startswith("http"):
        image = Image.open(requests.get(image_str, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_str).convert("RGB")

    return image

def get_boxes(results: DetectionResult) -> List[List[List[float]]]:
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)

    return [boxes]

def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks

def detect(images, labels_batch, threshold, object_detector):
    batch_results = []
    """ print("Batch sizes:")
    print(len(images))
    print(len(labels_batch)) """
    for image, labs in zip(images, labels_batch):
        labels = [label if label.endswith(".") else label + "." for label in labs]
        results = object_detector(image, candidate_labels=labels, threshold=threshold)
        detection_results = [DetectionResult.from_dict(result) for result in results]
        batch_results.append(detection_results)
        """ print("results")
        print(results)
        print("labels")
        print(labs)
        print("detection results")
        print(detection_results) """
    return batch_results

def segment(images, batch_detection_results, polygon_refinement, segmentator, processor, device):
    batch_segmented_results = []
    """ print("batch detections:")
    print(batch_detection_results)
    print("DONE") """
    for image, detection_results in zip(images, batch_detection_results):
        """ print(detection_results) """
        boxes = get_boxes(detection_results)
        """ print(image.shape)
        print(boxes) """
        inputs = processor(images=image, input_boxes=boxes, do_rescale=False, return_tensors="pt").to(device)
        outputs = segmentator(**inputs)
        masks = processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=inputs.original_sizes,
            reshaped_input_sizes=inputs.reshaped_input_sizes
        )[0]
        masks = refine_masks(masks, polygon_refinement)
        for detection_result, mask in zip(detection_results, masks):
            detection_result.mask = mask
        batch_segmented_results.append(detection_results)
    return batch_segmented_results

def compute_union_area(masks: List[np.ndarray]) -> int:
    if not masks:
        return 0
    # Stack masks to create a 3D array and compute the logical OR across all masks
    union_mask = np.logical_or.reduce(np.stack(masks, axis=0))
    return np.sum(union_mask)

def tensor_to_pil(tensor):
    # Convert a tensor to a PIL Image
    return transforms.ToPILImage()(tensor)

def main():
    csv_file = '/private/home/arjunrs1/CliMer/data/egoexo4d/egoexo4d_all_views_keysteps_test.csv'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector_id = "IDEA-Research/grounding-dino-tiny"
    object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)
    segmenter_id = "facebook/sam-vit-base"
    segmentor = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
    processor = AutoProcessor.from_pretrained(segmenter_id)

    # Define transformations if required (e.g., resizing, normalization)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    # Assuming 'annotations_df' is defined and loaded
    dataset = VideoFrameDataset(csv_file, transform=transform)
    # Create DataLoader
    data_loader = DataLoader(dataset, batch_size=4, collate_fn=dataset.collate_fn, shuffle=False, num_workers=1)

    results = []
    for data in tqdm(data_loader):
            images = data['image']
            nouns = data['nouns']

            pil_images = [tensor_to_pil(image) for image in images]
            # Perform detection
            detections = detect(pil_images, nouns, 0.2, object_detector)
            # Convert images to tensors if necessary for segmentation
            tensor_images = [transforms.ToTensor()(img) for img in pil_images]
            tensor_images = torch.stack(tensor_images).to(device)
            # Perform segmentation
            segmented_results = segment(tensor_images, detections, polygon_refinement=True, segmentator=segmentor, processor=processor, device=device)
            # Extract masks from detections and compute the union area
            all_masks = []
            for batch in segmented_results:
                all_masks.extend([detection.mask for detection in batch if detection.mask is not None])
            area = compute_union_area(all_masks)
            
            print(area)

if __name__ == "__main__":
    main()