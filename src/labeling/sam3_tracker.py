"""
SAM 3.1 Multiplex Video Tracking Engine for Football.
"""

import os
import gc
from typing import Optional, Dict, Any, List, Tuple
import cv2
import numpy as np
import torch
from safetensors.torch import load_file

from sam3.model_builder import (
    build_sam3_multiplex_video_model,
    _create_multiplex_tri_backbone,
    _create_text_encoder,
    _create_sam3_transformer,
    _create_segmentation_head,
    _create_geometry_encoder,
    _create_dot_product_scoring,
)
from sam3.model.sam3_multiplex_base import Sam3MultiplexPredictorWrapper
from sam3.model.sam3_multiplex_detector import Sam3MultiplexDetector
from sam3.model.sam3_multiplex_tracking import Sam3MultiplexTrackingWithInteractivity
from sam3.model.sam3_multiplex_video_predictor import Sam3MultiplexVideoPredictor
from sam3.model.vl_combiner import SAM3VLBackboneTri


def build_sam3_multiplex_from_safetensors(
    safetensors_path: str,
    bpe_path: Optional[str] = None,
    max_num_objects: int = 16,
    multiplex_count: int = 16,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Sam3MultiplexVideoPredictor:
    """
    Build and initialize the SAM 3.1 Multiplex video predictor from a .safetensors checkpoint.
    """
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"Safetensors checkpoint not found at: {safetensors_path}")

    if bpe_path is None:
        # Check standard location inside cloned sam3
        candidate = os.path.join(
            os.path.dirname(__file__), "..", "..", "sam3", "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz"
        )
        bpe_path = os.path.abspath(candidate)
        if not os.path.exists(bpe_path):
            raise FileNotFoundError(f"BPE vocabulary not found at {bpe_path}")

    print(f"[SAM 3.1] Initializing model architecture (device={device})...")
    
    # 1. Build tracker backbone
    tracker_model = build_sam3_multiplex_video_model(
        checkpoint_path=None,
        load_from_HF=False,
        multiplex_count=multiplex_count,
        use_fa3=False,
        use_rope_real=True,
        compile=False,
        strict_state_dict_loading=False,
    )
    del tracker_model.backbone
    tracker_model.backbone = None

    sam2_predictor = Sam3MultiplexPredictorWrapper(
        model=tracker_model,
        per_obj_inference=False,
        fill_hole_area=0,
        is_multiplex=True,
        is_multiplex_dynamic=True,
    )

    # 2. Build detector backbone
    tri_neck = _create_multiplex_tri_backbone(compile_mode=None, use_fa3=False, use_rope_real=True)
    text_encoder = _create_text_encoder(bpe_path)
    backbone = SAM3VLBackboneTri(scalp=0, visual=tri_neck, text=text_encoder)
    transformer = _create_sam3_transformer(use_fa3=False)
    segmentation_head = _create_segmentation_head(use_fa3=False)
    geometry_encoder = _create_geometry_encoder()
    dot_prod_scoring = _create_dot_product_scoring()

    detector = Sam3MultiplexDetector(
        num_feature_levels=1,
        backbone=backbone,
        transformer=transformer,
        segmentation_head=segmentation_head,
        semantic_segmentation_head=None,
        input_geometry_encoder=geometry_encoder,
        use_early_fusion=True,
        use_dot_prod_scoring=True,
        dot_prod_scoring=dot_prod_scoring,
        supervise_joint_box_scores=True,
        is_multiplex=True,
    )

    # 3. Assemble full tracking model
    demo_model = Sam3MultiplexTrackingWithInteractivity(
        tracker=sam2_predictor,
        detector=detector,
        score_threshold_detection=0.35,
        det_nms_thresh=0.1,
        det_nms_use_iom=True,
        assoc_iou_thresh=0.1,
        new_det_thresh=0.6,
        hotstart_delay=15,
        hotstart_unmatch_thresh=8,
        hotstart_dup_thresh=8,
        suppress_unmatched_only_within_hotstart=False,
        suppress_overlapping_based_on_recent_occlusion_threshold=0.7,
        suppress_det_close_to_boundary=True,
        fill_hole_area=0,
        recondition_every_nth_frame=16,
        use_iom_recondition=True,
        iom_thresh_recondition=0.5,
        masklet_confirmation_enable=True,
        reconstruction_bbox_iou_thresh=-1,
        reconstruction_bbox_det_score=0.8,
        max_num_objects=max_num_objects,
        postprocess_batch_size=1,
        use_batched_grounding=False,
        batched_grounding_batch_size=1,
        max_num_kboxes=0,
        sprinkle_removal_area=0,
        is_multiplex=True,
        image_size=1008,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        compile_model=False,
    )

    print(f"[SAM 3.1] Loading weights from {safetensors_path}...")
    ckpt = load_file(safetensors_path)

    # Check key remapping if necessary
    needs_remap = any(k.startswith("sam3_model.") or k.startswith("sam2_predictor.") for k in ckpt)
    if needs_remap:
        remapped_ckpt = {}
        for k, v in ckpt.items():
            new_k = k
            if k.startswith("sam3_model."):
                new_k = "detector." + k[len("sam3_model.") :]
            elif k.startswith("sam2_predictor."):
                new_k = "tracker." + k[len("sam2_predictor.") :]
            remapped_ckpt[new_k] = v
        ckpt = remapped_ckpt

    missing, unexpected = demo_model.load_state_dict(ckpt, strict=False)
    if unexpected:
        print(f"[SAM 3.1] Warning: unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    
    demo_model.to(device).eval()

    predictor = Sam3MultiplexVideoPredictor(
        model=demo_model,
        session_expiration_sec=3600,
        default_output_prob_thresh=0.4,
        async_loading_frames=False,
        warm_up=False,
    )
    print("[SAM 3.1] Predictor initialized successfully!")
    return predictor


class FootballSAMTracker:
    """
    High-level tracker for football videos using SAM 3.1.
    """
    def __init__(self, safetensors_path: str, bpe_path: Optional[str] = None):
        self.safetensors_path = safetensors_path
        self.bpe_path = bpe_path
        self.predictor = build_sam3_multiplex_from_safetensors(safetensors_path, bpe_path)

    def track_video(
        self,
        frames_dir: str,
        text_prompt: str = "small moving soccer ball on the pitch",
        point_prompt: Optional[Tuple[int, int]] = None,
        box_prompt: Optional[Tuple[int, int, int, int]] = None,
        prompt_frame_idx: int = 0,
        output_prob_thresh: float = 0.30,
    ) -> Dict[int, np.ndarray]:
        """
        Runs tracking across frames located in `frames_dir`.
        Returns a dict mapping frame_idx -> binary_mask (H, W) or None.
        """
        # Start session on frames directory with video frames offloaded to CPU
        start_res = self.predictor.handle_request(dict(
            type="start_session",
            resource_path=frames_dir,
            offload_video_to_cpu=True,
        ))
        session_id = start_res["session_id"]
        print(f"[SAM 3.1] Session started: {session_id}")

        masks_per_frame: Dict[int, np.ndarray] = {}

        try:
            # Add prompt
            prompt_req: Dict[str, Any] = dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=prompt_frame_idx,
                output_prob_thresh=output_prob_thresh,
            )

            if point_prompt is not None:
                # Convert point (x, y) to relative coordinates if needed
                # Find image dimensions from frame
                sample_img_path = os.path.join(frames_dir, "0000.jpg")
                if not os.path.exists(sample_img_path):
                    sample_img_path = os.path.join(frames_dir, sorted(os.listdir(frames_dir))[0])
                sample_img = cv2.imread(sample_img_path)
                h, w = sample_img.shape[:2]
                
                # Rel coords in [0, 1]
                rel_pt = [point_prompt[0] / float(w), point_prompt[1] / float(h)]
                prompt_req["points"] = torch.tensor([[rel_pt]], dtype=torch.float32)
                prompt_req["point_labels"] = torch.tensor([[1]], dtype=torch.int32)
                prompt_req["obj_id"] = 1
                prompt_req["rel_coordinates"] = True
                print(f"[SAM 3.1] Adding point prompt: abs={point_prompt}, rel={rel_pt}")
            elif box_prompt is not None:
                sample_img_path = os.path.join(frames_dir, sorted(os.listdir(frames_dir))[0])
                sample_img = cv2.imread(sample_img_path)
                h, w = sample_img.shape[:2]
                rel_box = [box_prompt[0] / float(w), box_prompt[1] / float(h), box_prompt[2] / float(w), box_prompt[3] / float(h)]
                prompt_req["bounding_boxes"] = torch.tensor([[rel_box]], dtype=torch.float32)
                prompt_req["bounding_box_labels"] = torch.tensor([[1]], dtype=torch.int32)
                prompt_req["obj_id"] = 1
                prompt_req["rel_coordinates"] = True
                print(f"[SAM 3.1] Adding box prompt: abs={box_prompt}, rel={rel_box}")
            else:
                prompt_req["text"] = text_prompt
                print(f"[SAM 3.1] Adding text prompt: '{text_prompt}' on frame {prompt_frame_idx}")

            prompt_res = self.predictor.handle_request(prompt_req)
            initial_out = prompt_res.get("outputs", {})
            if "out_binary_masks" in initial_out and len(initial_out["out_binary_masks"]) > 0:
                m = initial_out["out_binary_masks"][0]
                if isinstance(m, torch.Tensor):
                    m = m.detach().cpu().numpy().squeeze()
                masks_per_frame[prompt_frame_idx] = m

            # Propagate across video
            print("[SAM 3.1] Propagating throughout the video...")
            stream = self.predictor.handle_stream_request(dict(
                type="propagate_in_video",
                session_id=session_id,
                propagation_direction="both",
                start_frame_index=prompt_frame_idx,
                output_prob_thresh=output_prob_thresh,
            ))

            target_obj_idx = 0
            try:
                for item in stream:
                    f_idx = item["frame_index"]
                    out = item["outputs"]
                    masks = out.get("out_binary_masks")
                    if masks is not None and len(masks) > 0:
                        m = masks[target_obj_idx]
                        if isinstance(m, torch.Tensor):
                            m = m.detach().cpu().numpy().squeeze()
                        masks_per_frame[f_idx] = m
                    else:
                        masks_per_frame[f_idx] = None
            except Exception as e:
                print(f"[SAM 3.1] Warning during video propagation: {e}. Keeping frames collected so far.")

        finally:
            # Always close session and clean GPU memory
            try:
                self.predictor.handle_request(dict(
                    type="close_session",
                    session_id=session_id,
                ))
            except Exception as e:
                print(f"[SAM 3.1] Warning closing session: {e}")

            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[SAM 3.1] Session closed and GPU cache freed.")

        return masks_per_frame
