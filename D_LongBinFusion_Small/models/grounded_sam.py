import argparse
import os
import sys

import numpy as np
import json
import torch
from PIL import Image

gsam_path = '/workspace/Grounded-Segment-Anything'
sys.path.append(os.path.join(gsam_path, "GroundingDINO"))
sys.path.append(os.path.join(gsam_path, "segment_anything"))

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt


import torchvision.transforms as TF

def load_image(image_input):
    """
    image_input: str (파일 경로) 또는 torch.Tensor (3,H,W or H,W,3)
    return: image_pil, image_tensor
    """
    # 1. 입력 타입에 따라 PIL.Image 준비
    if isinstance(image_input, str):   # 파일 경로일 경우
        image_pil = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, torch.Tensor):
        if image_input.ndim == 3:
            if image_input.shape[0] == 3:       # (3,H,W)
                image_pil = TF.ToPILImage()(image_input)
            elif image_input.shape[2] == 3:     # (H,W,3)
                image_pil = TF.ToPILImage()(image_input.permute(2,0,1))
            else:
                raise ValueError("Tensor shape must be (3,H,W) or (H,W,3)")
        else:
            raise ValueError("Tensor must be 3D (CHW or HWC)")
    else:
        raise TypeError("image_input must be str path or torch.Tensor")

    # 2. transform 정의
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])

    # 3. transform 적용
    image, _ = transform(image_pil, None)  # (3,H,W)
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, bert_base_uncased_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    args.bert_base_uncased_path = bert_base_uncased_path
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    #print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
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
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask-150.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask-150.json'), 'w') as f:
        json.dump(json_data, f)


def save_mask_data_npz(output_dir, mask_list, box_list, label_list):
    os.makedirs(output_dir, exist_ok=True)

    value = 0  # 0 for background

    # 마스크 전체를 인덱스 맵으로 합치기
    mask_img = torch.zeros(mask_list.shape[-2:], dtype=torch.int32)
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1

    # 시각화용 jpg 저장 (선택)
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask-150.jpg'),
                bbox_inches="tight", dpi=300, pad_inches=0.0)
    plt.close()

    # 메타데이터 준비
    meta_data = [{
        'value': 0,
        'label': 'background'
    }]

    values, names, logits, boxes = [], [], [], []

    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1]  # ')' 제거

        values.append(value)
        names.append(name)
        logits.append(float(logit))
        boxes.append(box.cpu().numpy())

        meta_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.cpu().numpy().tolist(),
        })

    # npz 저장 (mask, box, label, logit 등 모두 포함)
    np.savez_compressed(
        os.path.join(output_dir, 'mask-150.npz'),
        mask=mask_img.cpu().numpy(),
        values=np.array(values),
        labels=np.array(names),
        logits=np.array(logits),
        boxes=np.array(boxes),
        meta=np.array(meta_data, dtype=object)  # 메타데이터 딕셔너리 리스트도 함께 저장
    )


class GroundedSegmentAnything:
    def __init__(self, config_file, grounded_checkpoint, sam_checkpoint,
                 sam_version="vit_h", device="cuda",
                 sam_hq_checkpoint=None, use_sam_hq=False,
                 box_threshold=0.3, text_threshold=0.25,
                 bert_base_uncased_path=None):

        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        # GroundingDINO 로드
        self.dino_model = load_model(
            config_file, grounded_checkpoint, bert_base_uncased_path, device=device
        )

        # SAM 로드
        if use_sam_hq and sam_hq_checkpoint is not None:
            sam_model = sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint)
        else:
            sam_model = sam_model_registry[sam_version](checkpoint=sam_checkpoint)

        self.sam_predictor = SamPredictor(sam_model.to(device))

    def predict(self, input_image, text_prompt):
        """
        input_image: torch.Tensor (3,H,W) or np.ndarray (H,W,3, uint8)
        text_prompt: str
        """
        # 1. GroundingDINO 실행
        if isinstance(input_image, torch.Tensor):
            _, image = load_image(input_image)  # normalize, tensor (3,H,W)
        else:
            _, image = load_image(input_image)

        boxes_filt, pred_phrases = get_grounding_output(
            self.dino_model, image, text_prompt,
            self.box_threshold, self.text_threshold,
            device=self.device
        )
        
        # 2. predictor에 이미지 세팅
        if isinstance(input_image, torch.Tensor):
            image_np = input_image.permute(1, 2, 0).detach().cpu().numpy()
            if image_np.dtype != np.uint8:
                image_np = (np.clip(image_np, 0, 1) * 255).astype(np.uint8)
        else:
            image_np = np.array(input_image)


        # 박스 없으면 안전하게 종료
        if boxes_filt is None or boxes_filt.numel() == 0 or boxes_filt.shape[0] == 0:
            H, W = image_np.shape[:2]
            empty_masks = torch.zeros((0, 1, H, W), device=self.device, dtype=torch.float32)
            return empty_masks, boxes_filt, pred_phrases  # 호출부에서 스킵하도록

        self.sam_predictor.set_image(image_np)

        # 3. 박스 스케일 보정
        H, W = image_np.shape[:2]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
            boxes_filt, image_np.shape[:2]
        ).to(self.device)

        # 4. SAM 마스크 예측
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.device),
            multimask_output=False,
        )

        return masks, boxes_filt, pred_phrases


    
