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
    SCALAR = auto()              # float
    RGB_IMAGE = auto()           # [H,W,3]
    CHW_IMAGE_GRAYSCALE = auto() # [1,H,W]
    BCHW_IMAGE_GRAYSCALE = auto()# [B,1,H,W]
    HW_SEGMENTATION_MASK = auto()# [H,W] class ids
    HWC_FLOW_FIELD = auto()      # [H,W,2] vector field
    HWC_DEPTH_MAP = auto()       # [H,W] depth
    VQGAN_LATENTS = auto()       # [tokens, dim]
    ATTENTION_HEATMAP = auto()   # [H,W]
    CLASS_PROBABILITIES = auto() # [C]
    EMBEDDINGS = auto()          # [N,2]
    LOGITS = auto()              # [V]
    TEXT = auto()                # string payload
    TOKEN_PROBABILITIES = auto() # [T,V]
    ATTENTION_PATTERNS = auto()  # [H,T,T]
    POINT_CLOUD = auto()         # [N,3]
    NORMAL_MAP = auto()          # [H,W,3]
    SEMANTIC_MAP = auto()        # [H,W,C]
    INSTANCE_MAP = auto()        # [H,W] instance ids
    KEYPOINTS = auto()           # [K,2]
    POSE = auto()                # [J,3]
    BOUNDING_BOXES = auto()      # [B,4]
    AUDIO_WAVEFORM = auto()      # [samples, channels]
    SPECTROGRAM = auto()         # [F,T]
    FEATURE_VECTORS = auto()     # [B,D]
    GRAPH = auto()               # nodes/edges
    VOXELS = auto()              # [D,H,W]
    MESH = auto()                # vertices/faces
    CAMERA_PARAMS = auto()       # intrinsics/extrinsics
    OPTICAL_FLOW = auto()        # [H,W,2]
    DISPARITY = auto()           # [H,W]
    SURFACE_NORMALS = auto()     # [H,W,3]
    MATERIAL_PARAMS = auto()     # [H,W,C]
    LANGUAGE_EMBEDDINGS = auto() # [T,E]
    LATENT_CODES = auto()        # [B,Z]
    STYLE_VECTORS = auto()       # [B,S]
    MASKS = auto()               # [H,W] binary
    CONFIDENCE_SCORES = auto()   # [B]
    JOINT_ANGLES = auto()        # [J]
    TACTILE_READINGS = auto()    # [H,W,C]
    IMU_DATA = auto()            # [T,6]
    LIDAR_SCAN = auto()          # [N,3]
    RADAR_DATA = auto()          # [H,W]
    THERMAL_IMAGE = auto()       # [H,W]
    EVENT_STREAM = auto()        # [N,4]
    FORCE_TORQUE = auto()        # [6]
    OCCUPANCY_GRID = auto()      # [H,W]
    SIGNED_DISTANCE = auto()     # [H,W]
    TRAJECTORY = auto()          # [T,D]
    ACTION_MASK = auto()         # [A]
    REWARD_SIGNAL = auto()       # scalar RL reward
    STATE_VECTOR = auto()        # [state_dim]


class TensorView(IGUI, IPlottable):
    """Visualizes tensor-like outputs using the specified modality."""

    def __init__(self, id: str, tensor: TensorLike, modality: TensorModality):
        self.id = id
        self.tensor: TensorLike = tensor
        self.modality: TensorModality = modality

        # UI controls
        self.scale: float = 3.0
        self.vector_spacing: int = 20  # stride for vector visualization
        self.vector_scale: float = 1.0

        # Internal resources
        self._initialized = False
        self._texture_pool: TexturePool = TexturePool()
        self._texids: list[int] = []        # textures backing the view
        self._texture_sizes: list[ImVec2] = []  # per-texture sizes

    def __del__(self):
        """Return all leased textures back to the pool."""
        for texid in self._texids:
            self._texture_pool.release(texid)

    def _filter_config_keys(self) -> dict:
        """Gather serializable view options (exclude tensor/state)."""
        return {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_") and k not in ["tensor", "modality", "id"]
        }

    def write(self, run: Run):
        """
        Persist this view's tensor and metadata under:
        runs/<run_id>/tensors/<id>/{i}.bin and {i}.json
        """
        tensor_dir = run.run_dir / "tensors" / self.id
        tensor_dir.mkdir(parents=True, exist_ok=True)

        with open(tensor_dir / f"{run.i}.bin", "wb") as f:
            np.save(f, self.tensor)

        meta = {"modality": self.modality.name, "config": self._filter_config_keys()}
        with open(tensor_dir / f"{run.i}.json", "w") as f:
            json.dump(meta, f)

    def read(self, run: Run):
        """Load previously saved tensor and view configuration for this step."""
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
        Convert the internal tensor to an HWC uint8 RGB image
        according to the current modality.
        """
        import torch  # optional dependency; imported lazily

        array = self.tensor
        if isinstance(array, torch.Tensor):
            array = array.detach().cpu().numpy()

        match self.modality:
            case TensorModality.RGB_IMAGE:
                return array

            case TensorModality.BCHW_IMAGE_GRAYSCALE:
                # First batch, CHW with C=1 -> HWC (C=3)
                chw = np.repeat(array[0], 3, axis=0)      # [3,H,W]
                return chw.transpose(1, 2, 0)             # [H,W,3]

            case TensorModality.CHW_IMAGE_GRAYSCALE:
                # [1,H,W] -> normalize -> [H,W,3]
                if array.shape[0] == 1:
                    array = np.ascontiguousarray(array[0])  # [H,W]
                rng = float(array.max() - array.min()) or 1.0
                norm = (array - array.min()) / rng
                return np.ascontiguousarray(np.stack([norm] * 3, axis=-1))

            case TensorModality.HW_SEGMENTATION_MASK:
                img = (array * 255 / (array.max() or 1)).astype(np.uint8)
                return cv2.applyColorMap(img, cv2.COLORMAP_JET)

            case TensorModality.HWC_FLOW_FIELD:
                mag, ang = cv2.cartToPolar(array[..., 0], array[..., 1])
                hsv = np.zeros((*array.shape[:2], 3), dtype=np.uint8)
                hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
                hsv[..., 1] = 255
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

            case TensorModality.HWC_DEPTH_MAP | TensorModality.DISPARITY:
                img = (array * 255 / (array.max() or 1)).astype(np.uint8)
                return cv2.applyColorMap(img, cv2.COLORMAP_TURBO)

            case TensorModality.NORMAL_MAP | TensorModality.SURFACE_NORMALS:
                return np.clip(((array + 1.0) * 127.5), 0, 255).astype(np.uint8)

            case TensorModality.SEMANTIC_MAP:
                classes = array.shape[-1]
                colors = np.random.randint(0, 255, (classes, 3))
                rgb = np.zeros((*array.shape[:2], 3), dtype=np.uint8)
                for i in range(classes):
                    rgb[array[..., i] > 0.5] = colors[i]
                return rgb

            case TensorModality.INSTANCE_MAP:
                ids = np.unique(array)
                colors = np.random.randint(0, 255, (len(ids), 3))
                rgb = np.zeros((*array.shape, 3), dtype=np.uint8)
                for i, inst in enumerate(ids):
                    rgb[array == inst] = colors[i]
                return rgb

            case _:
                return np.zeros((64, 64, 3), dtype=np.uint8)

    def initialize(self, texture_pool: TexturePool):
        """Create texture(s) for the current visualization via the pool."""
        if self._initialized:
            return
        self._initialized = True
        self._texture_pool = texture_pool

        rgb = self._get_rgb_visualization()
        texid = texture_pool.rent()
        self._texids.append(texid)
        self._texture_sizes.append(ImVec2(rgb.shape[1], rgb.shape[0]))
        texture_pool.update_texture(texid, rgb)

    def gui(self):
        """Render this view within an ImGui window."""
        if not self._initialized:
            return

        modality = self.modality
        _ = gui.get_content_region_avail()  # reserved for future layout logic

        if modality == TensorModality.HWC_FLOW_FIELD:
            if implot.begin_plot(f"{self.id}##flow_plot"):
                gui.same_line()
                gui.begin_group()
                _, self.vector_spacing = gui.slider_int("Grid Spacing", self.vector_spacing, 5, 50)
                _, self.vector_scale = gui.slider_float("Vector Scale", self.vector_scale, 0.1, 5.0)
                gui.end_group()

                field = self.tensor
                h, w = field.shape[:2]
                draw = implot.get_plot_draw_list()
                implot.push_plot_clip_rect()

                for y in range(0, h, self.vector_spacing):
                    for x in range(0, w, self.vector_spacing):
                        vx, vy = field[y, x]
                        p0 = implot.plot_to_pixels(implot.Point(x, y))
                        p1 = implot.plot_to_pixels(implot.Point(x + vx * self.vector_scale, y + vy * self.vector_scale))
                        draw.add_line(
                            ImVec2(p0.x, p0.y),
                            ImVec2(p1.x, p1.y),
                            gui.get_color_u32(ImVec4(1, 1, 1, 0.8)),
                            1.0,
                        )

                implot.pop_plot_clip_rect()
                implot.end_plot()

        elif modality == TensorModality.TEXT:
            gui.text_wrapped(str(self.tensor))

        else:
            for i, texid in enumerate(self._texids):
                gui.image(
                    texid,
                    ImVec2(self._texture_sizes[i].x * self.scale, self._texture_sizes[i].y * self.scale),
                )

    def plot(self, label: str, tensor, modality: TensorModality):
        """
        Draw plot overlays for a tensor inside an active ImPlot context.
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
            draw = implot.get_plot_draw_list()
            implot.push_plot_clip_rect()

            for y in range(0, h, self.vector_spacing):
                for x in range(0, w, self.vector_spacing):
                    vx, vy = tensor[y, x]
                    p0 = implot.plot_to_pixels(implot.Point(x, y))
                    p1 = implot.plot_to_pixels(implot.Point(x + vx * self.vector_scale, y + vy * self.vector_scale))
                    draw.add_line(
                        ImVec2(p0.x, p0.y),
                        ImVec2(p1.x, p1.y),
                        gui.get_color_u32(ImVec4(1, 1, 1, 0.8)),
                        1.0,
                    )

            implot.pop_plot_clip_rect()

        elif modality == TensorModality.TEXT:
            # textual content is not plotted in ImPlot
            pass

        else:
            if len(tensor.shape) == 2:
                implot.plot_heatmap(f"{label}##heatmap", tensor)
            else:
                gui.text_wrapped(f"Unsupported tensor shape for plotting: {tensor.shape}")
``
