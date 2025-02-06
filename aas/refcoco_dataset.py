# modified from https://github.com/mbzuai-oryx/groundingLMM/blob/main/dataset/segm_datasets/RefCOCO_Segm_ds.py

import json
import numpy as np
import os
import torch

from pycocotools import mask as mask_utils

from .refcoco_refer import REFER


class ReferSegmDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir,
                 num_classes_per_sample=3,
                 refer_segm_data="refcoco||refcoco+||refcocog||refclef",
                 split="train",
                 sample=None):
        self.num_classes_per_sample = num_classes_per_sample
        self.dataset_dir = dataset_dir
        self.split = split
        self.initialize_refer_segm_data(refer_segm_data)
        if sample is not None:
            sample_idx = json.load(open(sample))
            self.idx_to_ds = [self.idx_to_ds[i] for i in sample_idx]
            self.idx_to_dsidx = [self.idx_to_dsidx[i] for i in sample_idx]

    def initialize_refer_segm_data(self, refer_segm_data):
        dataset_dir = os.path.join(self.dataset_dir, "Refer_Segm")
        self.refer_seg_ds_list = refer_segm_data.split("||")
        # ['refclef', 'refcoco', 'refcoco+', 'refcocog']
        self.refer_segm_data = {}
        self.idx_to_ds = []
        self.idx_to_dsidx = []

        for dataset_name in self.refer_seg_ds_list:
            splitBy = "umd" if dataset_name == "refcocog" else "unc"
            refer_api = REFER(dataset_dir, dataset_name, splitBy)
            ref_ids_split = refer_api.getRefIds(split=self.split)
            images_ids_split = refer_api.getImgIds(ref_ids=ref_ids_split)
            refs_split = refer_api.loadRefs(ref_ids=ref_ids_split)
            refer_seg_ds = {
                "images": self.load_images(refer_api, images_ids_split, dataset_dir, dataset_name),
                "annotations": refer_api.Anns,
                "img2refs": self.create_img_to_refs_mapping(refs_split)
            }

            print(f"dataset {dataset_name} (refs {splitBy}) ({self.split} split) has {len(refer_seg_ds['images'])} "
                  f"images and {len(refer_seg_ds['annotations'])} annotations.")
            print(f"\033[92m----SEG-{self.split}:"
                  f" Loaded ReferSeg - {dataset_name} dataset ----\033[0m")

            self.refer_segm_data[dataset_name] = refer_seg_ds
            self.idx_to_ds.extend([dataset_name] * len(refer_seg_ds["images"]))
            self.idx_to_dsidx.extend(list(range(len(refer_seg_ds["images"]))))

    def load_images(self, refer_api, images_ids_split, dataset_dir, dataset_name):
        images = []
        loaded_images = refer_api.loadImgs(image_ids=images_ids_split)
        for item in loaded_images:
            item = item.copy()
            if dataset_name == "refclef":
                item["file_name"] = os.path.join(dataset_dir, "images", "saiapr_tc-12", item["file_name"])
            else:
                item["file_name"] = os.path.join(dataset_dir.replace("/Refer_Segm", ""), "coco_2014/train2014",
                                                 item["file_name"])
            images.append(item)
        return images

    def create_img_to_refs_mapping(self, refs_split):
        img2refs = {}
        for ref in refs_split:
            img2refs[ref["image_id"]] = img2refs.get(ref["image_id"], []) + [ref, ]
        return img2refs

    def __len__(self):
        return len(self.idx_to_dsidx)

    def __getitem__(self, idx):
        dataset_name = self.idx_to_ds[idx]
        refer_seg_ds = self.refer_segm_data[dataset_name]
        images = refer_seg_ds["images"]
        annotations = refer_seg_ds["annotations"]
        img2refs = refer_seg_ds["img2refs"]
        idx = self.idx_to_dsidx[idx]
        image_info = images[idx]
        image_id = image_info["id"]
        refs = img2refs[image_id]
        assert len(refs) > 0
        # if len(refs) == 0:
        #     return self.__getitem__(0)

        sents = []
        ann_ids = []
        for ref in refs:
            for sent in ref["sentences"]:
                text = sent["sent"]
                sents.append(text)
                ann_ids.append(ref["ann_id"])
        if len(sents) >= self.num_classes_per_sample:
            sampled_inds = np.random.choice(
                list(range(len(sents))), size=self.num_classes_per_sample, replace=False
            )
        else:
            sampled_inds = list(range(len(sents)))
        sampled_sents = [sents[ind] for ind in sampled_inds]
        sampled_ann_ids = [ann_ids[ind] for ind in sampled_inds]
        selected_labels = sampled_sents

        image_path = image_info["file_name"]

        masks = []
        for ann_id in sampled_ann_ids:
            if not isinstance(ann_id, list):
                ann_id = [ann_id]

            if -1 in ann_id:
                assert len(ann_id) == 1
                m = np.zeros((image_info["height"], image_info["width"])).astype(np.uint8)
                masks.append(m)
                continue

            m_final = np.zeros((image_info["height"], image_info["width"])).astype(np.uint8)
            for ann_id_i in ann_id:
                ann = annotations[ann_id_i]

                if len(ann["segmentation"]) == 0:
                    m = np.zeros((image_info["height"], image_info["width"])).astype(np.uint8)
                else:
                    if type(ann["segmentation"][0]) == list:  # polygon
                        rle = mask_utils.frPyObjects(ann["segmentation"], image_info["height"], image_info["width"])
                    else:
                        rle = ann["segmentation"]
                        for i in range(len(rle)):
                            if not isinstance(rle[i]["counts"], bytes):
                                rle[i]["counts"] = rle[i]["counts"].encode()
                    m = mask_utils.decode(rle)
                    m = np.sum(m, axis=2)  # sometimes there are multiple binary map (corresponding to multiple segs)
                    m = m.astype(np.uint8)  # convert to np.uint8
                m_final = m_final | m
            m = m_final
            masks.append(m)

        return image_path, masks, selected_labels
