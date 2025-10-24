import json
from enum import auto, Enum
from typing import Union

import cv2
import numpy as np
from imgui_bundle import imgui as gui, implot, ImVec2, ImVec4

from core import Run
from imguio import IGUI, IPlottable
from TexturePool import TexturePool

TensorLike = Union[np.ndarray, "torch.Tensor"]  # type: ignore


class TensorModality(Enum):
    SCALAR = auto()  # float
    RGB_IMAGE = auto()  # [H,W,3]
    CHW_IMAGE_GRAYSCALE = auto()  # [1,H,W]
    BCHW_IMAGE_GRAYSCALE = auto()  # [B,1,H,W]
    HW_SEGMENTATION_MASK = auto()  # [H,W] class ids
    HWC_FLOW_FIELD = auto()  # [H,W,2] flow vectors
    HWC_DEPTH_MAP = auto()  # [H,W] depth values
    VQGAN_LATENTS = auto()  # [num_tokens, latent_dim]
    ATTENTION_HEATMAP = auto()  # [H,W] attention weights
    CLASS_PROBABILITIES = auto()  # [num_classes]
    EMBEDDINGS = auto()  # [num_points, 2]
    LOGITS = auto()  # [vocab_size]
    TEXT = auto()  # generated string
    TOKEN_PROBABILITIES = auto()  # [num_tokens, vocab_size]
    ATTENTION_PATTERNS = auto()  # [num_heads, seq_len, seq_len]
    POINT_CLOUD = auto()  # [num_points, 3]
    NORMAL_MAP = auto()  # [H,W,3]
    SEMANTIC_MAP = auto()  # [H,W,C]
    INSTANCE_MAP = auto()  # [H,W] instance ids
    KEYPOINTS = auto()  # [num_keypoints, 2]
    POSE = auto()  # [num_joints, 3]
    BOUNDING_BOXES = auto()  # [num_boxes, 4]
    AUDIO_WAVEFORM = auto()  # [samples, channels]
    SPECTROGRAM = auto()  # [freq_bins, time_steps]
    FEATURE_VECTORS = auto()  # [batch, dim]
    GRAPH = auto()  # graph (nodes/edges)
    VOXELS = auto()  # [D,H,W]
    MESH = auto()  # vertices/faces
    CAMERA_PARAMS = auto()  # intrinsics/extrinsics
    OPTICAL_FLOW = auto()  # [H,W,2]
    DISPARITY = auto()  # [H,W]
    SURFACE_NORMALS = auto()  # [H,W,3]
    MATERIAL_PARAMS = auto()  # [H,W,C]
    LANGUAGE_EMBEDDINGS = auto()  # [seq_len, embed_dim]
    LATENT_CODES = auto()  # [batch, latent_dim]
    STYLE_VECTORS = auto()  # [batch, style_dim]
    MASKS = auto()  # [H,W] binary
    CONFIDENCE_SCORES = auto()  # [batch]
    JOINT_ANGLES = auto()  # [num_joints]
    TACTILE_READINGS = auto()  # [H,W,C]
    IMU_DATA = auto()  # [timesteps, 6]
    LIDAR_SCAN = auto()  # [num_points, 3]
    RADAR_DATA = auto()  # [H,W]
    THERMAL_IMAGE = auto()  # [H,W]
    EVENT_STREAM = auto()  # [num_events, 4]
    FORCE_TORQUE = auto()  # [6]
    OCCUPANCY_GRID = auto()  # [H,W]
    SIGNED_DISTANCE = auto()  # [H,W]
    TRAJECTORY = auto()  # [timesteps, dims]
    ACTION_MASK = auto()  # [num_actions]
    REWARD_SIGNAL = auto()  # scalar RL reward
    STATE_VECTOR = auto()  # [state_dim]


class TensorView(IGUI, IPlottable):
    """Renders tensor-like data using a modality-aware strategy."""

    def __init__(self, id: str, tensor: TensorLike, modality: TensorModality):
        self.id = id
        self.tensor: TensorLike = tensor
        self.modality: TensorModality = modality

        # Display controls
        self.scale: float = 3.0
        self.vector_spacing: int = 20  # sampling stride for vector fields
        self.vector_scale: float = 1.0

        # Internal state
        self._initialized = False
        self._texture_pool: TexturePool = TexturePool()
        # support multi-slice tensors decomposed into multiple textures
        self._texids: list[int] = []
        self._texture_sizes: list[ImVec2] = []

    def __del__(self):
        """Return any leased textures when the view is destroyed."""
        for texid in self._texids:
            self._texture_pool.release(texid)

    def _filter_config_keys(self) -> dict:
        """Collect only user-tunable fields for serialization."""
        return {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_") and k not in ["tensor", "modality", "id"]
        }

    def write(self, run: Run):
        """
        Persist the current tensor and its view settings to:
        runs/<run_id>/tensors/<id>/{i}.bin/.json
        """
        tensor_dir = run.run_dir / "tensors" / self.id
        tensor_dir.mkdir(parents=True, exist_ok=True)

        tensor_path = tensor_dir / f"{run.i}.bin"
        with open(tensor_path, "wb") as f:
            np.save(f, self.tensor)

        meta_path = tensor_dir / f"{run.i}.json"
        meta = {"modality": self.modality.name, "config": self._filter_config_keys()}
        with open(meta_path, "w") as f:
            json.dump(meta, f)

    def read(self, run: Run):
        """Load a previously recorded tensor + settings for this view."""
        tensor_path = run.run_dir / "tensors" / self.id / f"{run.i}.bin"
        meta_path = run.run_dir / "tensors" / self.id / f"{run.i}.json"

        if tensor_path.exists() and meta_path.exists():
            with open(tensor_path, "rb") as f:
                self.tensor = np.load(f)

            with open(meta_path, "r") as f:
                meta = json.load(f)
                self.modality = TensorModality[meta["modality"]]
                for k, v in meta["config"].items():
                    setattr(self, k, v)

    def _get_rgb_visualization(self) -> np.ndarray:
        """
        Convert the underlying tensor to an RGB HWC uint8 image
        based on the selected modality.
        """
        import torch  # local import to avoid hard dependency if unused

        tensor = self.tensor
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()

        match self.modality:
            case TensorModality.RGB_IMAGE:
                return tensor

            case TensorModality.BCHW_IMAGE_GRAYSCALE:
                # take first item in batch, expand 1->3 channels, transpose CHW->HWC
                return np.repeat(tensor[0], 3, axis=0).transpose(1, 2, 0)

            case TensorModality.CHW_IMAGE_GRAYSCALE:
                # accept [1,H,W] -> [H,W,3]
                if tensor.shape[0] == 1:
                    tensor = np.ascontiguousarray(tensor[0])
                denom = (tensor.max() - tensor.min()) or 1.0
                normalized = (tensor - tensor.min()) / denom
                return np.ascontiguousarray(np.stack([normalized] * 3, axis=-1))

            case TensorModality.HW_SEGMENTATION_MASK:
                img = (tensor * 255 / (tensor.max() or 1)).astype(np.uint8)
                return cv2.applyColorMap(img, cv2.COLORMAP_JET)

            case TensorModality.HWC_FLOW_FIELD:
                mag, ang = cv2.cartToPolar(tensor[..., 0], tensor[..., 1])
                hsv = np.zeros((*tensor.shape[:2], 3), dtype=np.uint8)
                hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
                hsv[..., 1] = 255
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

            case TensorModality.HWC_DEPTH_MAP | TensorModality.DISPARITY:
                img = (tensor * 255 / (tensor.max() or 1)).astype(np.uint8)
                return cv2.applyColorMap(img, cv2.COLORMAP_TURBO)

            case TensorModality.NORMAL_MAP | TensorModality.SURFACE_NORMALS:
                return np.clip(((tensor + 1.0) * 127.5), 0, 255).astype(np.uint8)

            case TensorModality.SEMANTIC_MAP:
                num_classes = tensor.shape[-1]
                colors = np.random.randint(0, 255, (num_classes, 3))
                rgb = np.zeros((*tensor.shape[:2], 3), dtype=np.uint8)
                for i in range(num_classes):
                    rgb[tensor[..., i] > 0.5] = colors[i]
                return rgb

            case TensorModality.INSTANCE_MAP:
                instances = np.unique(tensor)
                colors = np.random.randint(0, 255, (len(instances), 3))
                rgb = np.zeros((*tensor.shape, 3), dtype=np.uint8)
                for i, inst in enumerate(instances):
                    rgb[tensor == inst] = colors[i]
                return rgb

            case _:
                return np.zeros((64, 64, 3), dtype=np.uint8)

    def initialize(self, texture_pool: TexturePool):
        """Create texture(s) for the current tensor using the provided pool."""
        if self._initialized:
            return
        self._initialized = True
        self._texture_pool = texture_pool

        rgb_data = self._get_rgb_visualization()
        texid = texture_pool.rent()
        self._texids.append(texid)
        self._texture_sizes.append(ImVec2(rgb_data.shape[1], rgb_data.shape[0]))
        texture_pool.update_texture(texid, rgb_data)

    def gui(self):
        """Draw the view within an ImGui context."""
        if not self._initialized:
            return

        tensor = self.tensor
        modality = self.modality
        _ = gui.get_content_region_avail()  # reserved for future layout use

        if modality == TensorModality.HWC_FLOW_FIELD:
            if implot.begin_plot(f"{self.id}##flow_plot"):
                gui.same_line()
                gui.begin_group()
                _, self.vector_spacing = gui.slider_int("Grid Spacing", self.vector_spacing, 5, 50)
                _, self.vector_scale = gui.slider_float("Vector Scale", self.vector_scale, 0.1, 5.0)
                gui.end_group()

                h, w = tensor.shape[:2]
                draw_list = implot.get_plot_draw_list()
                implot.push_plot_clip_rect()

                for y in range(0, h, self.vector_spacing):
                    for x in range(0, w, self.vector_spacing):
                        vx, vy = tensor[y, x]
                        start = implot.plot_to_pixels(implot.Point(x, y))
                        end = implot.plot_to_pixels(implot.Point(x + vx * self.vector_scale, y + vy * self.vector_scale))
                        draw_list.add_line(
                            ImVec2(start.x, start.y),
                            ImVec2(end.x, end.y),
                            gui.get_color_u32(ImVec4(1, 1, 1, 0.8)),
                            1.0,
                        )

                implot.pop_plot_clip_rect()
                implot.end_plot()

        elif modality == TensorModality.TEXT:
            gui.text_wrapped(str(tensor))

        else:
            for i, texid in enumerate(self._texids):
                gui.image(
                    texid,
                    ImVec2(self._texture_sizes[i].x * self.scale, self._texture_sizes[i].y * self.scale),
                )

    def plot(self, label: str, tensor, modality: TensorModality):
        """
        Overlay plot primitives for the provided tensor inside an active ImPlot plot.
        """
        if not self._initialized or tensor is None:
            return

        if modality == TensorModality.SCALAR:
            if len(tensor.shape) == 1:
                implot.plot_line(f"{label}##plot", tensor)
            elif len(tensor.shape) == 2:
                for i in range(tensor.shape[0]):
                    implot.plot_line(f"{label}_{i}##plot", tensor[i, :])

        elif modality == TensorModality.HWC_FLOW_FIELD:
            h, w = tensor.shape[:2]
            draw_list = implot.get_plot_draw_list()
            implot.push_plot_clip_rect()

            for y in range(0, h, self.vector_spacing):
                for x in range(0, w, self.vector_spacing):
                    vx, vy = tensor[y, x]
                    start = implot.plot_to_pixels(implot.Point(x, y))
                    end = implot.plot_to_pixels(implot.Point(x + vx * self.vector_scale, y + vy * self.vector_scale))
                    draw_list.add_line(
                        ImVec2(start.x, start.y),
                        ImVec2(end.x, end.y),
                        gui.get_color_u32(ImVec4(1, 1, 1, 0.8)),
                        1.0,
                    )

            implot.pop_plot_clip_rect()

        elif modality == TensorModality.TEXT:
            # textual content is rendered in gui(), not as a plot primitive
            pass

        else:
            if len(tensor.shape) == 2:
                implot.plot_heatmap(f"{label}##heatmap", tensor)
            else:
                gui.text_wrapped(f"Unsupported tensor shape for plotting: {tensor.shape}")
