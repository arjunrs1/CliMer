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

def detect(
    image,
    labels,
    threshold,
    object_detector
) -> List[Dict[str, Any]]:
    """
    Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
    """

    labels = [label if label.endswith(".") else label+"." for label in labels]

    results = object_detector(image,  candidate_labels=labels, threshold=threshold)
    results = [DetectionResult.from_dict(result) for result in results]

    return results

def segment(
    image,
    detection_results,
    polygon_refinement = False,
    segmentator=None,
    processor=None
) -> List[DetectionResult]:
    """
    Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
    """

    boxes = get_boxes(detection_results)
    inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)

    outputs = segmentator(**inputs)
    masks = processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes
    )[0]

    masks = refine_masks(masks, polygon_refinement)

    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask

    return detection_results

def compute_union_area(masks: List[np.ndarray]) -> int:
    if not masks:
        return 0
    # Stack masks to create a 3D array and compute the logical OR across all masks
    union_mask = np.logical_or.reduce(np.stack(masks, axis=0))
    return np.sum(union_mask)

def grounded_segmentation_area(
    image,
    labels,
    threshold,
    polygon_refinement = False,
    detector = None,
    segmentor=None,
    processor=None
) -> int:
    if isinstance(image, str):
        image = load_image(image)

    detections = detect(image, labels, threshold, detector)
    detections = segment(image, detections, polygon_refinement, segmentator=segmentor, processor=processor)

    # Extract masks from detections and compute the union area
    masks = [detection.mask for detection in detections if detection.mask is not None]
    return compute_union_area(masks)

def sample_frames(start_frame, end_frame, num_samples):
    return np.linspace(start_frame, end_frame, num=num_samples, dtype=int)

def process_narr(take_name, cam_id, start_frame, end_frame, nouns, num_samples=1, detector_model=None, segmentor=None, processor=None):
    frame_numbers = sample_frames(start_frame, end_frame, num_samples)
    union_areas = []

    video_path = f'/datasets01/egoexo4d/v2/takes/{take_name}/frame_aligned_videos/{cam_id}.mp4'
    cap = cv2.VideoCapture(video_path)
    for frame_number in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # Set the current frame position
        ret, frame_cv2 = cap.read()
        
        if not ret:
            print(f"Failed to read frame {frame_number} from video {video_path}")
            continue
        
        # Convert OpenCV frame to PIL Image
        frame = cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        try:
            area = grounded_segmentation_area(
                image=frame,
                labels=nouns,
                threshold=0.3,
                polygon_refinement=True,
                detector=detector_model,
                segmentor=segmentor,
                processor=processor
            )
        except:
            continue
        union_areas.append(area)
    cap.release()  # Release the video capture object
    # Compute the mean of the union areas
    mean_area = np.mean(union_areas) if union_areas else 0
    return mean_area

csv_file = '/private/home/arjunrs1/CliMer/data/egoexo4d/egoexo4d_all_views_keysteps_test.csv'
data = pd.read_csv(csv_file)
data['duration'] = pd.to_numeric(data['duration'], errors='coerce')
data.sort_values(by='duration', ascending=True, inplace=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
detector_id = "IDEA-Research/grounding-dino-tiny"
object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)
segmenter_id = "facebook/sam-vit-base"
segmentor = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
processor = AutoProcessor.from_pretrained(segmenter_id)
print(device)

results = {}
for index, row in tqdm(data.iterrows(), total=data.shape[0]):
    video_name = row['video_id']
    start_frame = row['start_frame']
    end_frame = row['end_frame']
    narr_id = row['clip_id']
    nouns = eval(row['noun_class'])
    if "hand" not in nouns:
        if "hands" not in nouns:
            nouns.append("hand")
    take_name = video_name.rsplit("_", 1)[0] if "aria" not in video_name else video_name.rsplit("_", 2)[0]
    cam_id = video_name.rsplit("_", 1)[1] if "aria" not in video_name else "_".join(video_name.rsplit("_", 2)[1:])

    mean_area = process_narr(take_name, cam_id, start_frame, end_frame, nouns, detector_model=object_detector, segmentor=segmentor, processor=processor)
    results[narr_id] = mean_area

    if index % 20 == 0:
        with open('gSAM_test_map.json', 'w') as json_file:
            json.dump(results, json_file, indent=4)

with open('gSAM_test_map.json', 'w') as json_file:
    json.dump(results, json_file, indent=4)