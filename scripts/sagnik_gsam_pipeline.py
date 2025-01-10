import os
import random
import numpy as np
import json, pickle
from tqdm import tqdm
import argparse
from PIL import Image, ImageDraw, ImageFont
import cv2 as cv

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

import decord
from decord import VideoReader

# from mmpose.apis import MMPoseInferencer

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

from segment_anything import build_sam, SamPredictor

from transformers import BlipProcessor, BlipForConditionalGeneration


def ospif(file):
	return os.path.isfile(file)

def ospid(dir_):
	return os.path.isdir(dir_)

def pkl_dmp(obj, fp):
	with open(fp, "wb") as fo:
		pickle.dump(obj, fo, protocol=pickle.HIGHEST_PROTOCOL)

def pkl_ld(fp):
	with open(fp, "rb") as fi:
		pkl_content = pickle.load(fi)
	return pkl_content

def json_ld(fp):
	with open(fp, "r") as fi:
		json_content = json.load(fi)
	return json_content

def json_dmp(obj, fp):
	with open(fp, "w") as fo:
		json.dump(obj, fo)

def load_video(video_path, n_frms=8, height=224, width=224, sampling="uniform", return_msg = False):
	decord.bridge.set_bridge("torch")

	cap = cv.VideoCapture(video_path)
	n_ttl_frms = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
	cap.release()
	if n_ttl_frms == 0:
		frms = torch.zeros((n_frms, height, width, 3)).float()
		if not return_msg:
			return frms
		else:
			return frms, ""

	vr = VideoReader(uri=video_path, height=height, width=width)
	vlen = len(vr)
	start, end = 0, vlen

	orig_n_frms = n_frms
	n_frms = min(n_frms, vlen)

	if sampling == "uniform":
		indices = np.arange(start, end, vlen / n_frms).astype(int).tolist()
	elif sampling == "headtail":
		indices_h = sorted(rnd.sample(range(vlen // 2), n_frms // 2))
		indices_t = sorted(rnd.sample(range(vlen // 2, vlen), n_frms // 2))
		indices = indices_h + indices_t
	else:
		raise NotImplementedError

	assert len(indices) >= 1
	assert len(indices) <= orig_n_frms
	if len(indices) < orig_n_frms:
		indices = indices + ([indices[-1]] * (orig_n_frms - len(indices)))
	assert len(indices) == orig_n_frms, print(len(indices), orig_n_frms, vlen, n_frms)

	# get_batch -> T, H, W, C
	temp_frms = vr.get_batch(indices)
	# print(type(temp_frms))
	tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
	frms = tensor_frms.permute(0, 1, 2, 3).float()  # (T, H, W, C)
	assert frms.shape[0] == orig_n_frms

	if not return_msg:
		return frms

	fps = float(vr.get_avg_fps())
	sec = ", ".join([str(round(f / fps, 1)) for f in indices])
	# " " should be added in the start and end
	msg = f"The video contains {len(indices)} frames sampled at {sec} seconds. "
	return frms, msg


class CaptionDataset(torch.utils.data.Dataset):
	def __init__(self,
	             tkNm2strtNendTmstmp2othrsNlstAtmcDscs_fp=None,
	             dtpntClps_dr=None):

		self.dtpntClps_dr = dtpntClps_dr

		self.tkNm2strtNendTmstmp2othrsNlstAtmcDscs = pkl_ld(tkNm2strtNendTmstmp2othrsNlstAtmcDscs_fp)

		self.lst_dtpnts = []
		for tk_nm, v in tqdm(self.tkNm2strtNendTmstmp2othrsNlstAtmcDscs.items()):
			for strtNendTmstmp, v1 in v.items():
				# for vw in ["aria", "1", "2", "3", "4"]:
				#     clp_fp = f"{dtpntClps_dr}/{vw}/{tk_nm}/{strtNendTmstmp[0]}_{strtNendTmstmp[1]}.mp4"
				#     assert ospif(clp_fp)

				#     frms = load_video(clp_fp)

				assert 'text' in v1
				lst_atmcDscs = v1['text']
				assert isinstance(lst_atmcDscs, list)

				self.lst_dtpnts.append({"take_name": tk_nm,
				                        "startNendTimestamp": strtNendTmstmp,
				                        "startNend_clipName": v1["startNend_clipName"],
				                        "startNend_frameIdx": v1["startNend_frameIdx"],
				                        "atomic_descriptions": lst_atmcDscs})

	def __len__(self):
		return len(self.lst_dtpnts)		# 6, len(self.lst_dtpnts)

	def __getitem__(self, idx):
		# print(idx)
		assert idx < len(self.lst_dtpnts)

		tk_nm = self.lst_dtpnts[idx]["take_name"]
		strtNendTmstmp = self.lst_dtpnts[idx]["startNendTimestamp"]
		strtNendClpNm = self.lst_dtpnts[idx]["startNend_clipName"]
		strtNendFrmIdx = self.lst_dtpnts[idx]["startNend_frameIdx"]
		lst_atmcDscs = self.lst_dtpnts[idx]["atomic_descriptions"]
		assert isinstance(lst_atmcDscs, list)
		assert isinstance(lst_atmcDscs[0], str)
		# print(idx, lst_atmcDscs)

		frms_alVws = []
		for vw in ["aria", "1", "2", "3", "4"]:
			# clp_fp = f"{self.dtpntClps_dr}/{vw}/{tk_nm}/{strtNendTmstmp[0]}_{strtNendTmstmp[1]}.mp4"
			clp_fp = f"{self.dtpntClps_dr}/{vw}/{tk_nm}/{strtNendClpNm[0]}_{strtNendClpNm[1]}__{strtNendFrmIdx[0]}_{strtNendFrmIdx[1]}__{strtNendTmstmp[0]}_{strtNendTmstmp[1]}.mp4"
			assert ospif(clp_fp)

			frms = load_video(clp_fp)
			# print(len(frms))
			# exit("here")
			frms = frms[len(frms) // 2: len(frms) // 2 + 1]

			frms_alVws.append(frms)

		frms_alVws = torch.cat(frms_alVws, dim=0)
		# print(idx, frms_alVws.shape)

		return idx, frms_alVws,\
		        {"all_gtCaptions": lst_atmcDscs, 
		         "take_name": tk_nm, 
		         "startNendTimestamp": f"{strtNendTmstmp[0]}_{strtNendTmstmp[1]}"}


def transform_image(image_pil):
	transform = T.Compose(
		[
			T.RandomResize([800], max_size=1333),
			T.ToTensor(),
			T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
		]
	)
	image, _ = transform(image_pil, None)  # 3, h, w
	return image


def load_model(model_config_path, model_checkpoint_path, device):
	args = SLConfig.fromfile(model_config_path)
	args.device = device
	model = build_model(args)
	checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
	load_res = model.load_state_dict(
		clean_state_dict(checkpoint["model"]), strict=False)
	print(load_res)
	_ = model.eval()
	return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True):
	caption = caption.lower()
	caption = caption.strip()
	if not caption.endswith("."):
		caption = caption + "."

	with torch.no_grad():
		outputs = model(image[None], captions=[caption])
	logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
	boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
	logits.shape[0]

	# filter output
	logits_filt = logits.clone()
	boxes_filt = boxes.clone()
	filt_mask = logits_filt.max(dim=1)[0] > box_threshold
	logits_filt = logits_filt[filt_mask]  # num_filt, 256
	boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
	logits_filt.shape[0]

	# get phrase
	tokenlizer = model.tokenizer
	tokenized = tokenlizer(caption)
	# build pred
	pred_phrases = []
	scores = []
	for logit, box in zip(logits_filt, boxes_filt):
		pred_phrase = get_phrases_from_posmap(
			logit > text_threshold, tokenized, tokenlizer)
		if with_logits:
			pred_phrases.append(
			    pred_phrase + f"({str(logit.max().item())[:4]})")
		else:
			pred_phrases.append(pred_phrase)
		scores.append(logit.max().item())

	return boxes_filt, torch.Tensor(scores), pred_phrases


def run_grounded_sam(input_image, 
                     text_prompt,  
                     box_threshold, 
                     text_threshold, 
                     iou_threshold, 
                     device=None,
                     groundingdino_model=None,
					 sam_predictor=None,
                     ):

	""" make dir """
	# os.makedirs(output_dir, exist_ok=True)
	""" load image """
	image_pil = input_image.convert("RGB")
	transformed_image = transform_image(image_pil)

	assert groundingdino_model is not None

	""" run grounding dino model """
	boxes_filt, scores, pred_phrases = get_grounding_output(
	    groundingdino_model, transformed_image, text_prompt, box_threshold, text_threshold
	)

	size = image_pil.size

	# process boxes
	H, W = size[1], size[0]
	for i in range(boxes_filt.size(0)):
		boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
		boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
		boxes_filt[i][2:] += boxes_filt[i][:2]

	boxes_filt = boxes_filt.cpu()

	# print(f"Before NMS: {boxes_filt.shape[0]} boxes")
	# # print(boxes_filt)
	# # print("-" * 80)
	# # print(scores)
	# # print("-" * 80)
	# # print(pred_phrases)

	""" nms """
	# nms_idx = torchvision.ops.nms(
	#     boxes_filt, scores, iou_threshold).numpy().tolist()
	nms_idx = list(range(len(boxes_filt)))
	boxes_filt = boxes_filt[nms_idx]

	pred_phrases = [pred_phrases[idx] for idx in nms_idx]
	# print(f"After NMS: {boxes_filt.shape[0]} boxes")

	assert sam_predictor is not None

	image = np.array(image_pil)
	sam_predictor.set_image(image)

	transformed_boxes = sam_predictor.transform.apply_boxes_torch(
	    boxes_filt, image.shape[:2]).to(device)
	# print(transformed_boxes.shape)
	if len(transformed_boxes) == 0:
		return 0.

	masks, _, __ = sam_predictor.predict_torch(
	    point_coords=None,
	    point_labels=None,
	    boxes=transformed_boxes,
	    multimask_output=False,
	)
	# print(type(masks), masks.shape, masks.dtype, torch.max(masks[0]), torch.min(masks[0]))

	""" iterating over masks """
	mask_union = None
	for mask in masks:
		if mask_union is None:
			mask_union = mask
		else:
			mask_union = mask_union + mask


	pxl_objctnss = torch.sum((mask_union.int() == 1).int()).item() /\
					(mask_union.shape[0] * mask_union.shape[1] * mask_union.shape[2])
	assert 0. <= pxl_objctnss <= 1.
	return pxl_objctnss


def main():
	# parser = argparse.ArgumentParser()
	# parser.add_argument("--thresh") 
	# parser.add_argument("--hand_thresh")
	# parser.add_argument("--first_obj_thresh")
	# parser.add_argument("--second_obj_thresh")
	# parser.add_argument("--model_weights", default=f"./final_on_blur_model_0399999.pth")
	# # parser.add_argument("--data_dir", default=f"./images")
	# args = parser.parse_args()

	torch.manual_seed(42)
	np.random.seed(42)
	random.seed(42)


	GROUNDING_SAM_ROOT = '/checkpoint/sagnikmjr2002/code'

	CONFIG_FP = f"{GROUNDING_SAM_ROOT}/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
	assert ospif(CONFIG_FP)

	CKPT_FP = f"{GROUNDING_SAM_ROOT}/GroundingDINO/ckpts/groundingdino_swint_ogc_hf.pth" # groundingdino_swint_ogc_github, groundingdino_swint_ogc_hf
	assert ospif(CKPT_FP)

	SAM_CKPT_FP = f"{GROUNDING_SAM_ROOT}/av_bvs/sam/ckpts/sam_vit_h_4b8939.pth"
	assert ospif(SAM_CKPT_FP)

	device = "cuda" if torch.cuda.is_available() else "cpu"
	# print(device)

	""" initialize SAM """
	groundingdino_model = load_model(
							CONFIG_FP, 
							CKPT_FP, 
							device=device, 
	    )

	""" initialize SAM """
	sam = build_sam(checkpoint=SAM_CKPT_FP)
	sam.to(device=device)
	sam_predictor = SamPredictor(sam)


	VERSION = 2 # 1, 2
	SPLIT = "val"
	BATCH_SIZE = 1 # 1, 2, 4
	NUM_WORKERS = 1 # 1, 4
	BOX_THRESHOLD = 0.3 # 0.35 = github, 0.3 = hf
	TEXT_THRESHOLD = 0.25
	IOU_THRESHOLD = 0.8 # 0.8 = hf

	ROOT_DR = f"/checkpoint/sagnikmjr2002/code/av_bvs/AV-bestView-selection-in-multiView-videos/data/ego_exo/v{VERSION}"
	assert ospid(ROOT_DR)

	TK_NM___2___STRT_N_END_TMSTMP___2___OTHRS__N__ATMC_DSCS___FP = f"{ROOT_DR}/misc/" +\
	                                                                f"take__2__startNendTimestamp__2__timestamp_n_startNendClipName_n_startNendFrameIdx_n_listAtomicDescriptions__obeyingTakeLenConstraint__val__20takesPercent.pkl"	# __20takesPercent
	assert ospif(TK_NM___2___STRT_N_END_TMSTMP___2___OTHRS__N__ATMC_DSCS___FP)

	ALL_UNQ_NNS_TRN_FP = f"{ROOT_DR}/misc/allUniqueNouns_train.pkl"
	assert ospif(ALL_UNQ_NNS_TRN_FP)
	alUnqNns_trn = pkl_ld(ALL_UNQ_NNS_TRN_FP)
	TEXT_PROMPT = ""
	for nn in list(alUnqNns_trn):
	    TEXT_PROMPT += f"{nn} . "
	TEXT_PROMPT = TEXT_PROMPT.strip()

	DTPNT_CLPS_DR = f"{ROOT_DR}/datapoint_clips"
	assert ospid(DTPNT_CLPS_DR)


	cptn_dtst = CaptionDataset(tkNm2strtNendTmstmp2othrsNlstAtmcDscs_fp=TK_NM___2___STRT_N_END_TMSTMP___2___OTHRS__N__ATMC_DSCS___FP,
	                           dtpntClps_dr=DTPNT_CLPS_DR)
	cptn_dtLdr = torch.utils.data.DataLoader(
	                            cptn_dtst,
	                            batch_size=BATCH_SIZE,
	                            shuffle=False,
	                            num_workers=NUM_WORKERS,
	                            pin_memory=True,
	                            drop_last=False,
	                        )

	DMP_FP = f"{ROOT_DR}/misc/" +\
	         f"take__2__startNendTimestamp__2__pixelObjectnessPerView_forListAtomicDescriptionsObeyingTakeLenConstraint__val__20takesPercent.pkl"	# __20takesPercent

	# TMP_IMG_DMP_FP = f"{os.path.dirname(os.path.abspath(__file__))}/tmp.jpg"


	tkNm2strtNendTmstmp2allNrmlzdObjctPxlAr = {}
	cnt = 0
	for i, (dtpnt_idx, alVw_frms, lstGtCptns_n_metadata) in enumerate(tqdm(cptn_dtLdr)):
		# print(alVw_frms.dtype, alVw_frms.shape)
		# print(i, dtpnt_idx, lstGtCptns_n_metadata['all_gtCaptions'])

		assert isinstance(lstGtCptns_n_metadata['all_gtCaptions'][0], list), print(type(lstGtCptns_n_metadata['all_gtCaptions'][0]), lstGtCptns_n_metadata['all_gtCaptions'][0])
		assert isinstance(lstGtCptns_n_metadata['all_gtCaptions'][0][0], str), print(lstGtCptns_n_metadata['all_gtCaptions'][0][0])
		if len(lstGtCptns_n_metadata['all_gtCaptions']) != len(alVw_frms):
			assert len(lstGtCptns_n_metadata['all_gtCaptions'][0]) == 1
			tmp_lst = []
			for ele in lstGtCptns_n_metadata['all_gtCaptions']:
			    tmp_lst += ele
			lstGtCptns_n_metadata['all_gtCaptions'] = [tmp_lst] 
			# print("h1")

		assert len(lstGtCptns_n_metadata['all_gtCaptions']) ==\
		        len(alVw_frms) ==\
		         1, print(len(lstGtCptns_n_metadata['all_gtCaptions']), 
		                  len(alVw_frms), 
		                  lstGtCptns_n_metadata['all_gtCaptions'],
		                  lstGtCptns_n_metadata['take_name'],
		                  lstGtCptns_n_metadata['startNendTimestamp'])
		# print(lstGtCptns_n_metadata['all_gtCaptions'][0])

		al_nrmlzdObjctPxlAr = []
		# print(alVw_frms.shape)
		for frm in alVw_frms[0]:
			# assert not ospif(TMP_IMG_DMP_FP)
			frm_npy = frm.detach().cpu().numpy().astype("uint8")
			# print("1: ", frm_npy.shape, frm_npy.dtype, frm_npy.max(), frm_npy.min())

			frm_pil = Image.fromarray(frm_npy)
			# frm_pil.save(TMP_IMG_DMP_FP)
			# assert ospif(TMP_IMG_DMP_FP)

			nrmlzdObjctPxlAr = run_grounded_sam(
									frm_pil,
									TEXT_PROMPT,
									BOX_THRESHOLD,
									TEXT_THRESHOLD,
									IOU_THRESHOLD,
									device=device,
									groundingdino_model=groundingdino_model,
									sam_predictor=sam_predictor)

			# al_nrmlzdObjctPxlAr.append(nrmlzdObjctPxlAr)
			al_nrmlzdObjctPxlAr += [nrmlzdObjctPxlAr] * 8

			# _ = os.system(f"rm {TMP_IMG_DMP_FP}")
			# exit()

		""" al_nrmlzdObjctPxlAr -> nm_vws * nm_frms """
		assert isinstance(lstGtCptns_n_metadata['take_name'][0], str)
		assert isinstance(lstGtCptns_n_metadata['startNendTimestamp'][0], str), print(lstGtCptns_n_metadata['startNendTimestamp'][0])

		if lstGtCptns_n_metadata['take_name'][0] not in tkNm2strtNendTmstmp2allNrmlzdObjctPxlAr:
		    tkNm2strtNendTmstmp2allNrmlzdObjctPxlAr[lstGtCptns_n_metadata['take_name'][0]] = {}
		strtNendTmstmp = (float(lstGtCptns_n_metadata['startNendTimestamp'][0].split('_')[0]), 
		                  float(lstGtCptns_n_metadata['startNendTimestamp'][0].split('_')[1]))
		assert strtNendTmstmp not in tkNm2strtNendTmstmp2allNrmlzdObjctPxlAr[lstGtCptns_n_metadata['take_name'][0]]
		tkNm2strtNendTmstmp2allNrmlzdObjctPxlAr[lstGtCptns_n_metadata['take_name'][0]][strtNendTmstmp] = al_nrmlzdObjctPxlAr

		cnt += 1
		# print(cnt)
		# if cnt == 2:	# 2, 5 
		#     break

	pkl_dmp(tkNm2strtNendTmstmp2allNrmlzdObjctPxlAr, DMP_FP)
    

if __name__ == '__main__':
    main()