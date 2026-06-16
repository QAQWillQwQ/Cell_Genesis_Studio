from __future__ import annotations

import csv
import math
import os
import re
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from .geometry import make_cell_rings

os.environ.setdefault("XDG_CACHE_HOME", "/tmp/celluniverse_local_python_app_cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/celluniverse_local_python_app_numba_cache")


def _redirect_napari_appdirs_to_tmp() -> None:
    try:
        import appdirs
    except Exception:
        return
    if getattr(appdirs, "_celluniverse_redirected", False):
        return

    base = Path("/tmp/celluniverse_local_python_app_napari")
    base.mkdir(parents=True, exist_ok=True)
    original = {
        "user_cache_dir": appdirs.user_cache_dir,
        "user_config_dir": appdirs.user_config_dir,
        "user_data_dir": appdirs.user_data_dir,
        "user_state_dir": appdirs.user_state_dir,
        "user_log_dir": appdirs.user_log_dir,
    }

    def redirected(kind: str, fallback):
        def inner(appname=None, appauthor=None, version=None, *args, **kwargs):
            if appname == "napari":
                marker = str(version or "default")
                path = base / kind / marker
                path.mkdir(parents=True, exist_ok=True)
                return str(path)
            return fallback(appname, appauthor, version, *args, **kwargs)

        return inner

    for key, fallback in original.items():
        setattr(appdirs, key, redirected(key.replace("user_", "").replace("_dir", ""), fallback))
    appdirs._celluniverse_redirected = True


_redirect_napari_appdirs_to_tmp()

try:
    import cv2
except ImportError:
    cv2 = None


CATEGORY_LABELS = {
    "mature": "mature cell",
    "split": "newly divided daughter pair",
    "pre_split": "pre division cell",
}


CATEGORY_DISPLAY = {
    "none": "Non Selected",
    "mature": "Mature cell",
    "split": "Newly divided daughter pair",
    "pre_split": "Pre division cell",
}


COLORMAP_OPTIONS = ["viridis", "gray", "magma", "inferno", "plasma", "cyan", "green", "red"]


CATEGORY_COLORS_BGR = {
    "mature": (40, 210, 70),
    "split": (220, 70, 255),
    "pre_split": (30, 210, 255),
}


AUTO_DETECT_NOTE = "auto detected by Cell Lumen and PCA"
CPP_CELLUNIVERSE_BINARY = Path("/Users/wangyiding/CellUniverse/C++/build/celluniverse")
DEFAULT_CELL_LUMEN_CONFIG = Path(
    "/Users/wangyiding/CellUniverse/C++/config/C.elegans developing embryo/Concentrated/"
    "C_elegans_DensityAuto_Best.yaml"
)
AUTO_LUMEN_CONFIG_FALLBACK = Path(
    "/Users/wangyiding/CellUniverse/C++/config/C.elegans developing embryo/Concentrated/"
    "CONCENTRATED_SINGLE_FILE_DENSITY_SWITCH_CANDIDATE_NOT_RUN_VERIFIED_20260609.yaml"
)
AUTO_DETECT_CONFIG_PRESETS = [
    DEFAULT_CELL_LUMEN_CONFIG,
    AUTO_LUMEN_CONFIG_FALLBACK,
]
DEFAULT_ELLIPSOID_OPACITY = 1.0
BACKGROUND_RING_SEGMENTS = 24
BACKGROUND_RINGS_PER_AXIS = 1
SELECTED_RING_SEGMENTS = 48
SELECTED_RINGS_PER_AXIS = 3


@dataclass
class ManualCell:
    name: str
    frame: int
    category: str
    x: float
    y: float
    z: float
    a: float = 13.0
    b: float = 13.0
    c: float = 8.0
    theta_x: float = 0.0
    theta_y: float = 0.0
    theta_z: float = 0.0
    pair_id: str = ""
    pair_role: str = ""
    finalized: bool = False
    notes: str = ""
    order: int = 0

    @property
    def cell_type(self) -> str:
        if self.category == "split":
            return "Cell type 2"
        if self.category == "pre_split":
            return "Cell type 3"
        return "Cell type 1"


def read_volume(path: Path) -> np.ndarray:
    if cv2 is not None:
        ok, pages = cv2.imreadmulti(str(path), [], cv2.IMREAD_UNCHANGED)
    else:
        ok, pages = False, []
    if ok and pages and cv2 is not None:
        gray_pages: list[np.ndarray] = []
        for page in pages:
            if page.ndim == 3:
                if page.shape[-1] == 4:
                    page = page[..., :3]
                page = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
            gray_pages.append(page.astype(np.float32))
        if len(gray_pages) == 1:
            return gray_pages[0][None, :, :]
        return np.stack(gray_pages, axis=0)

    import tifffile

    data = np.asarray(tifffile.imread(path))
    if data.ndim == 2:
        return data.astype(np.float32)[None, :, :]
    if data.ndim == 3 and data.shape[-1] in (3, 4):
        if cv2 is not None:
            gray = cv2.cvtColor(data[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = np.mean(data[..., :3], axis=-1)
        return gray.astype(np.float32)[None, :, :]
    if data.ndim == 4 and data.shape[-1] in (3, 4):
        return np.mean(data[..., :3], axis=-1).astype(np.float32)
    return data.astype(np.float32)


def normalize_u8(image: np.ndarray) -> np.ndarray:
    data = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(data, [1.0, 99.7])
    if not math.isfinite(float(low)) or not math.isfinite(float(high)) or high <= low:
        low = float(data.min())
        high = float(data.max())
    if high <= low:
        high = low + 1.0
    return np.clip((data - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)


def qimage_from_bgr(image: np.ndarray) -> QtGui.QImage:
    if cv2 is not None:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        rgb = image[..., ::-1].copy()
    height, width = rgb.shape[:2]
    return QtGui.QImage(rgb.data, width, height, width * 3, QtGui.QImage.Format_RGB888).copy()


def gray_to_bgr(gray: np.ndarray) -> np.ndarray:
    if cv2 is not None:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return np.repeat(gray[..., None], 3, axis=2)


def resize_gray_height(gray: np.ndarray, height: int) -> np.ndarray:
    height = max(1, int(height))
    if cv2 is not None:
        return cv2.resize(gray, (gray.shape[1], height), interpolation=cv2.INTER_LINEAR)
    if height == gray.shape[0]:
        return gray
    source_y = np.linspace(0, gray.shape[0] - 1, height)
    low = np.floor(source_y).astype(int)
    high = np.clip(low + 1, 0, gray.shape[0] - 1)
    weight = (source_y - low)[:, None]
    return ((1.0 - weight) * gray[low, :] + weight * gray[high, :]).astype(gray.dtype)


def draw_line(image: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], color: tuple[int, int, int], thickness: int = 1) -> None:
    if cv2 is not None:
        cv2.line(image, p1, p2, color, thickness, cv2.LINE_AA)
        return
    x1, y1 = p1
    x2, y2 = p2
    steps = max(1, int(math.hypot(x2 - x1, y2 - y1)))
    for t in np.linspace(0.0, 1.0, steps + 1):
        x = int(round(x1 + (x2 - x1) * t))
        y = int(round(y1 + (y2 - y1) * t))
        draw_circle(image, (x, y), max(1, thickness), color, filled=True)


def draw_circle(
    image: np.ndarray,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    *,
    filled: bool,
) -> None:
    if cv2 is not None:
        cv2.circle(image, center, radius, color, -1 if filled else 1, cv2.LINE_AA)
        return
    cx, cy = center
    radius = max(1, int(radius))
    y0 = max(0, cy - radius)
    y1 = min(image.shape[0], cy + radius + 1)
    x0 = max(0, cx - radius)
    x1 = min(image.shape[1], cx + radius + 1)
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    dist = (xx - cx) ** 2 + (yy - cy) ** 2
    if filled:
        mask = dist <= radius * radius
    else:
        inner = max(0, radius - 1)
        mask = (dist <= radius * radius) & (dist >= inner * inner)
    image[y0:y1, x0:x1][mask] = color


def draw_ellipse_outline(
    image: np.ndarray,
    center: tuple[int, int],
    axes: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    if cv2 is not None:
        cv2.ellipse(image, center, axes, 0, 0, 360, color, thickness, cv2.LINE_AA)
        return
    cx, cy = center
    rx, ry = max(1, axes[0]), max(1, axes[1])
    previous: tuple[int, int] | None = None
    for angle in np.linspace(0.0, 2.0 * math.pi, 181):
        point = (int(round(cx + rx * math.cos(angle))), int(round(cy + ry * math.sin(angle))))
        if previous is not None:
            draw_line(image, previous, point, color, thickness)
        previous = point


def blend_overlay(base: np.ndarray, overlay: np.ndarray, alpha: float) -> None:
    if cv2 is not None:
        cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0, base)
        return
    base[:] = np.clip(overlay.astype(np.float32) * alpha + base.astype(np.float32) * (1.0 - alpha), 0, 255).astype(
        np.uint8
    )


class ProjectionCanvas(QtWidgets.QLabel):
    coordinatePicked = QtCore.pyqtSignal(str, float, float, str)

    def __init__(self, plane: str, title: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.plane = plane
        self.title = title
        self._image_shape: tuple[int, int] = (1, 1)
        self._source_pixmap: QtGui.QPixmap | None = None
        self._dragging = False
        self.setMinimumSize(260, 190)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet("QLabel { background: #090909; border: 1px solid #333; }")
        self.setMouseTracking(True)

    def set_bgr(self, image: np.ndarray) -> None:
        self._image_shape = image.shape[:2]
        self._source_pixmap = QtGui.QPixmap.fromImage(qimage_from_bgr(image))
        self._set_scaled_pixmap()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        self._set_scaled_pixmap()
        super().resizeEvent(event)

    def _set_scaled_pixmap(self) -> None:
        if self._source_pixmap is None:
            return
        self.setPixmap(
            self._source_pixmap.scaled(
                self.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        )

    def _event_to_image_xy(self, event: QtGui.QMouseEvent) -> tuple[float, float] | None:
        pixmap = self.pixmap()
        if pixmap is None:
            return None
        label_w, label_h = max(1, self.width()), max(1, self.height())
        image_h, image_w = self._image_shape
        scale = min(label_w / image_w, label_h / image_h)
        draw_w = image_w * scale
        draw_h = image_h * scale
        offset_x = (label_w - draw_w) * 0.5
        offset_y = (label_h - draw_h) * 0.5
        x = (event.x() - offset_x) / max(1e-6, scale)
        y = (event.y() - offset_y) / max(1e-6, scale)
        if x < 0 or y < 0 or x >= image_w or y >= image_h:
            return None
        return x, y

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            self._dragging = True
            point = self._event_to_image_xy(event)
            if point is not None:
                self.coordinatePicked.emit(self.plane, point[0], point[1], "press")

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._dragging:
            point = self._event_to_image_xy(event)
            if point is not None:
                self.coordinatePicked.emit(self.plane, point[0], point[1], "drag")

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            point = self._event_to_image_xy(event)
            if point is not None:
                self.coordinatePicked.emit(self.plane, point[0], point[1], "release")
            self._dragging = False


class InitialStateBuilder(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.volume: np.ndarray | None = None
        self.volume_path: Path | None = None
        self.cells: list[ManualCell] = []
        self.selected_index: int = -1
        self._pair_counter = 1
        self._cell_counter = 1
        self._updating_controls = False
        self._cached_base: dict[str, np.ndarray] = {}
        self._drag_mode: tuple[str, str] | None = None
        self.napari_viewer = None
        self._napari_volume_path: Path | None = None
        self._syncing_napari = False
        self._napari_keys_bound = False
        self._last_control_data: np.ndarray | None = None
        self._embedded_napari_widget: QtWidgets.QWidget | None = None
        self._creating_candidate = False
        self._drop_event_filter_installed = False
        self._interaction_mode = "view"
        self._napari_auto_mouse_bound = False
        self._background_rings_dirty = True
        self._center_layer_cell_indices: list[int] = []
        self._busy_dialog: QtWidgets.QDialog | None = None
        self._busy_log_widget: QtWidgets.QPlainTextEdit | None = None
        self._busy_status_label: QtWidgets.QLabel | None = None
        self._auto_detect_log_lines: list[str] = []
        self._auto_detect_preset_config: Path = DEFAULT_CELL_LUMEN_CONFIG
        self.auto_process: QtCore.QProcess | None = None
        self._auto_detect_csv: Path | None = None
        self._build_ui()
        self._napari_sync_timer = QtCore.QTimer(self)
        self._napari_sync_timer.setSingleShot(True)
        self._napari_sync_timer.timeout.connect(self.sync_napari)
        self._background_ring_refresh_timer = QtCore.QTimer(self)
        self._background_ring_refresh_timer.setSingleShot(True)
        self._background_ring_refresh_timer.timeout.connect(self._refresh_background_ring_layer)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setAcceptDrops(True)
        self._install_drop_event_filter()

    def _build_ui(self) -> None:
        main = QtWidgets.QHBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)

        controls_container = QtWidgets.QWidget()
        controls_container.setObjectName("ControlPanel")
        controls_container.setMinimumWidth(440)
        controls = QtWidgets.QVBoxLayout()
        controls_container.setLayout(controls)
        controls.setContentsMargins(16, 14, 16, 14)
        controls.setSpacing(8)

        controls_scroll = QtWidgets.QScrollArea()
        controls_scroll.setObjectName("ControlsScroll")
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        controls_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        controls_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        controls_scroll.setMinimumWidth(460)
        controls_scroll.setMaximumWidth(540)
        controls_scroll.setWidget(controls_container)
        main.addWidget(controls_scroll, 0)

        workflow = QtWidgets.QLabel(
            "Workflow: 1. Open one frame real TIFF stack.  "
            "2. Use the napari 3D editor to place and resize cells.  "
            "3. Finish each cell.  4. Save Initial_frame.csv."
        )
        workflow.setWordWrap(True)
        workflow.setObjectName("WorkflowLabel")
        controls.addWidget(workflow)

        load_group = QtWidgets.QGroupBox("Step 1: Frame input")
        load_form = QtWidgets.QFormLayout(load_group)
        self.frame_spin = QtWidgets.QSpinBox()
        self.frame_spin.setRange(0, 9999)
        self.frame_spin.setValue(0)
        self.tif_path = QtWidgets.QLineEdit()
        self.tif_path.setPlaceholderText("Choose one real TIFF stack first")
        choose_tif = QtWidgets.QPushButton("Open frame TIFF first")
        choose_tif.clicked.connect(self.choose_tif)
        load_initial = QtWidgets.QPushButton("Load initial CSV")
        load_initial.clicked.connect(self.choose_initial_csv)
        open_napari = QtWidgets.QPushButton("Open 3D Napari Adjuster")
        open_napari.clicked.connect(self.open_napari_adjuster)
        load_form.addRow("frame", self.frame_spin)
        load_form.addRow(self.tif_path, choose_tif)
        load_form.addRow(load_initial)
        load_form.addRow(open_napari)
        controls.addWidget(load_group)

        voxel_group = QtWidgets.QGroupBox("Step 2: Voxel size override")
        voxel_form = QtWidgets.QFormLayout(voxel_group)
        voxel_form.setVerticalSpacing(10)
        voxel_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.voxel_x = self._double_spin(0.01, 100.0, 1.0, 0.1)
        self.voxel_y = self._double_spin(0.01, 100.0, 1.0, 0.1)
        self.voxel_z = self._double_spin(0.01, 100.0, 7.0, 0.1)
        self.colormap_combo = QtWidgets.QComboBox()
        self.colormap_combo.addItems(COLORMAP_OPTIONS)
        self.colormap_combo.setCurrentText("viridis")
        self.colormap_combo.currentTextChanged.connect(self.apply_colormap_to_volume)
        self.ellipsoid_opacity = self._double_spin(0.0, 1.0, DEFAULT_ELLIPSOID_OPACITY, 0.02)
        self.ellipsoid_opacity.valueChanged.connect(self.set_ellipsoid_opacity_from_spin)
        self.ellipsoid_opacity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.ellipsoid_opacity_slider.setRange(0, 100)
        self.ellipsoid_opacity_slider.setValue(int(round(DEFAULT_ELLIPSOID_OPACITY * 100)))
        self.ellipsoid_opacity_slider.valueChanged.connect(self.set_ellipsoid_opacity_from_slider)
        opacity_widget = QtWidgets.QWidget()
        opacity_widget.setObjectName("FullWidthControl")
        opacity_layout = QtWidgets.QVBoxLayout(opacity_widget)
        opacity_layout.setContentsMargins(0, 0, 0, 0)
        opacity_layout.setSpacing(8)
        opacity_value_row = QtWidgets.QHBoxLayout()
        opacity_value_row.setContentsMargins(0, 0, 0, 0)
        opacity_value_row.setSpacing(10)
        opacity_title = QtWidgets.QLabel("shell opacity")
        opacity_title.setObjectName("MutedLabel")
        self.ellipsoid_opacity.setMinimumWidth(96)
        self.ellipsoid_opacity.setMaximumWidth(120)
        opacity_value_row.addWidget(opacity_title)
        opacity_value_row.addStretch(1)
        opacity_value_row.addWidget(self.ellipsoid_opacity)
        opacity_layout.addLayout(opacity_value_row)
        opacity_layout.addWidget(self.ellipsoid_opacity_slider)
        self.ellipsoid_visibility_button = QtWidgets.QPushButton("Hide Ellipsoid Shells")
        self.ellipsoid_visibility_button.setCheckable(True)
        self.ellipsoid_visibility_button.setChecked(True)
        self.ellipsoid_visibility_button.clicked.connect(self.toggle_ellipsoid_shells)
        self.voxel_status = QtWidgets.QLabel("TIFF metadata not loaded yet.")
        self.voxel_status.setWordWrap(True)
        self.voxel_status.setObjectName("MutedLabel")
        for spin in (self.voxel_x, self.voxel_y, self.voxel_z):
            spin.valueChanged.connect(self.refresh_views)
        voxel_form.addRow("x voxel", self.voxel_x)
        voxel_form.addRow("y voxel", self.voxel_y)
        voxel_form.addRow("z voxel", self.voxel_z)
        voxel_form.addRow("colormap", self.colormap_combo)
        voxel_form.addRow(opacity_widget)
        voxel_form.addRow("shell display", self.ellipsoid_visibility_button)
        voxel_form.addRow("metadata", self.voxel_status)
        controls.addWidget(voxel_group)

        auto_group = QtWidgets.QGroupBox("Auto detection")
        auto_layout = QtWidgets.QVBoxLayout(auto_group)
        auto_config_row = QtWidgets.QHBoxLayout()
        self.auto_config_path = QtWidgets.QLineEdit(str(DEFAULT_CELL_LUMEN_CONFIG))
        choose_auto_config = QtWidgets.QPushButton("Choose config")
        choose_auto_config.clicked.connect(self.choose_auto_config)
        auto_config_row.addWidget(self.auto_config_path, 1)
        auto_config_row.addWidget(choose_auto_config)
        self.auto_detect_button = QtWidgets.QPushButton("Auto Detect Cells (Cell Lumen & PCA)")
        self.auto_detect_button.setObjectName("AutoDetectButton")
        self.auto_detect_button.setMinimumHeight(34)
        self.auto_detect_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.auto_detect_button.clicked.connect(self.run_auto_detect_cells)
        self.clean_auto_button = QtWidgets.QPushButton("Clean Auto Detect")
        self.clean_auto_button.setObjectName("CleanAutoButton")
        self.clean_auto_button.setMinimumHeight(34)
        self.clean_auto_button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.clean_auto_button.clicked.connect(self.clean_auto_detected_cells)
        self.auto_detect_config_button = QtWidgets.QToolButton()
        self.auto_detect_config_button.setObjectName("AutoDetectConfigMenu")
        self.auto_detect_config_button.setText("▼")
        self.auto_detect_config_button.setToolTip("Pick an auto-detect config")
        self.auto_detect_config_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.auto_detect_config_button.setMinimumHeight(34)
        self.auto_detect_config_button.setFixedWidth(32)
        self.auto_detect_config_button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self._refresh_auto_detect_config_menu()
        auto_button_row = QtWidgets.QHBoxLayout()
        auto_button_row.addWidget(self.auto_detect_button, 1)
        auto_button_row.addWidget(self.auto_detect_config_button)
        auto_button_row.addWidget(self.clean_auto_button)
        auto_layout.addLayout(auto_config_row)
        auto_layout.addLayout(auto_button_row)
        controls.addWidget(auto_group)

        add_group = QtWidgets.QGroupBox("Step 3: Manual cell")
        add_form = QtWidgets.QFormLayout(add_group)
        self.category_combo = QtWidgets.QComboBox()
        for key in ("none", "mature", "split", "pre_split"):
            self.category_combo.addItem(CATEGORY_DISPLAY[key], key)
        self.category_combo.setCurrentIndex(0)
        self.category_combo.currentIndexChanged.connect(self.prepare_candidate_for_selected_category)
        new_cell = QtWidgets.QPushButton("New cell / pair")
        new_cell.clicked.connect(self.add_cell_or_pair)
        finish = QtWidgets.QPushButton("Finish Adjust / Enter")
        finish.clicked.connect(self.finish_selected)
        mode_widget = QtWidgets.QWidget()
        mode_layout = QtWidgets.QHBoxLayout(mode_widget)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(6)
        self.view_mode_button = QtWidgets.QPushButton("View / Rotate")
        self.edit_mode_button = QtWidgets.QPushButton("Edit Ellipsoid")
        self.view_mode_button.setCheckable(True)
        self.edit_mode_button.setCheckable(True)
        self.view_mode_button.clicked.connect(lambda: self.set_interaction_mode("view"))
        self.edit_mode_button.clicked.connect(lambda: self.set_interaction_mode("edit"))
        mode_layout.addWidget(self.view_mode_button)
        mode_layout.addWidget(self.edit_mode_button)
        add_form.addRow("cell kind", self.category_combo)
        add_form.addRow("mouse mode", mode_widget)
        add_form.addRow(new_cell)
        add_form.addRow(finish)
        controls.addWidget(add_group)

        shape_group = QtWidgets.QGroupBox("Selected ellipsoid")
        shape_form = QtWidgets.QFormLayout(shape_group)
        self.x_spin = self._double_spin(0, 10000, 100, 0.5)
        self.y_spin = self._double_spin(0, 10000, 100, 0.5)
        self.z_spin = self._double_spin(0, 10000, 10, 0.5)
        self.a_spin = self._double_spin(1, 300, 13, 0.5)
        self.b_spin = self._double_spin(1, 300, 13, 0.5)
        self.c_spin = self._double_spin(1, 300, 8, 0.5)
        for spin in (self.x_spin, self.y_spin, self.z_spin, self.a_spin, self.b_spin, self.c_spin):
            spin.valueChanged.connect(self._controls_to_cell)
        shape_form.addRow("x center", self.x_spin)
        shape_form.addRow("y center", self.y_spin)
        shape_form.addRow("z center", self.z_spin)
        shape_form.addRow("x radius a", self.a_spin)
        shape_form.addRow("y radius b", self.b_spin)
        shape_form.addRow("z radius c", self.c_spin)
        controls.addWidget(shape_group)

        actions = QtWidgets.QHBoxLayout()
        delete_btn = QtWidgets.QPushButton("Delete")
        delete_btn.clicked.connect(self.delete_selected)
        prev_btn = QtWidgets.QPushButton("Prev")
        prev_btn.clicked.connect(lambda: self.select_relative(-1))
        next_btn = QtWidgets.QPushButton("Next")
        next_btn.clicked.connect(lambda: self.select_relative(1))
        actions.addWidget(prev_btn)
        actions.addWidget(next_btn)
        actions.addWidget(delete_btn)
        controls.addLayout(actions)

        self.cell_list = QtWidgets.QListWidget()
        self.cell_list.currentRowChanged.connect(self.select_cell)
        controls.addWidget(self.cell_list, 1)

        save_group = QtWidgets.QGroupBox("Step 4: Save")
        save_form = QtWidgets.QFormLayout(save_group)
        self.output_dir = QtWidgets.QLineEdit(
            "/Users/wangyiding/CellUniverse/Cell_Genesis_Studio/output/manual_initials"
        )
        choose_output = QtWidgets.QPushButton("Choose folder")
        choose_output.clicked.connect(self.choose_output_dir)
        save_btn = QtWidgets.QPushButton("Save Initial_frame.csv")
        save_btn.clicked.connect(self.save_csv)
        save_form.addRow(self.output_dir, choose_output)
        save_form.addRow(save_btn)
        controls.addWidget(save_group)

        hint = QtWidgets.QLabel("3D controls: use View / Rotate for the napari camera. Use Edit Ellipsoid to drag the red center or the green X, cyan Y, and orange Z handles. Keyboard fine adjustment only works in Edit Ellipsoid mode.")
        hint.setWordWrap(True)
        hint.setObjectName("MutedLabel")
        controls.addWidget(hint)

        workspace = QtWidgets.QGroupBox("3D Manual Cell Workspace")
        workspace_layout = QtWidgets.QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(22, 22, 22, 22)
        self.workspace_status = QtWidgets.QLabel(
            "Open a frame TIFF stack to launch the napari 3D editor. "
            "After the volume is loaded, choose a cell kind or press New cell / pair to create a draggable 3D ellipsoid."
        )
        self.workspace_status.setWordWrap(True)
        self.workspace_status.setAlignment(QtCore.Qt.AlignCenter)
        self.workspace_status.setObjectName("WorkspaceStatus")
        self.viewer_host = QtWidgets.QFrame()
        self.viewer_host.setObjectName("ViewerHost")
        self.viewer_host.setAcceptDrops(True)
        self.viewer_host.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.viewer_host_layout = QtWidgets.QVBoxLayout(self.viewer_host)
        self.viewer_host_layout.setContentsMargins(0, 0, 0, 0)
        self.viewer_host_layout.addWidget(self.workspace_status, 1)
        focus_napari = QtWidgets.QPushButton("Focus 3D Napari Editor")
        focus_napari.clicked.connect(self.open_napari_adjuster)
        workspace_layout.addWidget(self.viewer_host, 1)
        workspace_layout.addWidget(focus_napari)
        main.addWidget(workspace, 1)

        self._install_shortcuts()
        self.set_interaction_mode("view")

    def _view_box(self, title: str, canvas: ProjectionCanvas) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(box)
        label = QtWidgets.QLabel(title)
        label.setObjectName("MutedLabel")
        layout.addWidget(label)
        layout.addWidget(canvas, 1)
        return box

    def _double_spin(self, low: float, high: float, value: float, step: float) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setDecimals(3)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    @property
    def selected_cell(self) -> ManualCell | None:
        if 0 <= self.selected_index < len(self.cells):
            return self.cells[self.selected_index]
        return None

    def choose_tif(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open frame tif", "", "TIFF files (*.tif *.tiff);;All files (*)")
        if not path:
            return
        self.load_tif(Path(path))

    def _install_drop_event_filter(self) -> None:
        if self._drop_event_filter_installed:
            return
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
            self._drop_event_filter_installed = True

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.KeyPress and self._handle_edit_key(event):
            return True
        if event.type() in (QtCore.QEvent.DragEnter, QtCore.QEvent.DragMove):
            tif_path = self._first_tif_from_mime(event.mimeData())
            if tif_path is not None:
                event.acceptProposedAction()
                return True
        if event.type() == QtCore.QEvent.Drop:
            tif_path = self._first_tif_from_mime(event.mimeData())
            if tif_path is not None:
                event.acceptProposedAction()
                self.load_tif(tif_path)
                return True
        return super().eventFilter(watched, event)

    def _handle_edit_key(self, event: QtGui.QKeyEvent) -> bool:
        if self._interaction_mode != "edit":
            return False
        if self.selected_cell is None:
            return False
        focus = QtWidgets.QApplication.focusWidget()
        editable_types = (
            QtWidgets.QLineEdit,
            QtWidgets.QTextEdit,
            QtWidgets.QPlainTextEdit,
            QtWidgets.QSpinBox,
            QtWidgets.QDoubleSpinBox,
            QtWidgets.QComboBox,
        )
        if isinstance(focus, editable_types):
            return False
        step = 5.0 if event.modifiers() & QtCore.Qt.ShiftModifier else 1.0
        key = event.key()
        if key in (QtCore.Qt.Key_A, QtCore.Qt.Key_Left):
            self.nudge_selected(dx=-step)
            return True
        if key in (QtCore.Qt.Key_D, QtCore.Qt.Key_Right):
            self.nudge_selected(dx=step)
            return True
        if key in (QtCore.Qt.Key_W, QtCore.Qt.Key_Up):
            self.nudge_selected(dy=-step)
            return True
        if key in (QtCore.Qt.Key_S, QtCore.Qt.Key_Down):
            self.nudge_selected(dy=step)
            return True
        if key == QtCore.Qt.Key_Q:
            self.nudge_selected(dz=-step)
            return True
        if key == QtCore.Qt.Key_E:
            self.nudge_selected(dz=step)
            return True
        if key in (QtCore.Qt.Key_Plus, QtCore.Qt.Key_Equal):
            self.scale_selected(step)
            return True
        if key in (QtCore.Qt.Key_Minus, QtCore.Qt.Key_Underscore):
            self.scale_selected(-step)
            return True
        if key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            self.finish_selected()
            return True
        return False

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        tif_path = self._first_tif_from_mime(event.mimeData())
        if tif_path is not None:
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent) -> None:
        tif_path = self._first_tif_from_mime(event.mimeData())
        if tif_path is not None:
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        tif_path = self._first_tif_from_mime(event.mimeData())
        if tif_path is not None:
            event.acceptProposedAction()
            self.load_tif(tif_path)
        else:
            super().dropEvent(event)

    def _first_tif_from_mime(self, mime: QtCore.QMimeData) -> Path | None:
        candidates: list[str] = []
        if mime.hasUrls():
            candidates.extend(url.toLocalFile() for url in mime.urls() if url.isLocalFile())
        if mime.hasText():
            candidates.extend(line.strip().strip("'\"") for line in mime.text().splitlines())
        for candidate in candidates:
            if not candidate:
                continue
            path = Path(candidate).expanduser()
            if path.suffix.lower() in {".tif", ".tiff"} and path.is_file():
                return path
        return None

    def load_tif(self, path: Path) -> None:
        self.volume = read_volume(path)
        self.volume_path = path
        self.tif_path.setText(str(path))
        self._apply_tif_voxel_size_if_available(path)
        match = "".join(ch if ch.isdigit() else " " for ch in path.stem).split()
        if match:
            self.frame_spin.setValue(int(match[-1]))
        self._cached_base.clear()
        self.cells.clear()
        self._mark_background_rings_dirty()
        self.selected_index = -1
        self.cell_list.clear()
        self.category_combo.setCurrentIndex(0)
        z, y, x = self.volume.shape
        for spin, maximum in ((self.x_spin, x - 1), (self.y_spin, y - 1), (self.z_spin, z - 1)):
            spin.setMaximum(max(1, maximum))
        if not self.cells:
            self.x_spin.setValue(x / 2)
            self.y_spin.setValue(y / 2)
            self.z_spin.setValue(z / 2)
        self.refresh_views()
        self.open_napari_adjuster()

    def _apply_tif_voxel_size_if_available(self, path: Path) -> None:
        found: list[str] = []
        try:
            import tifffile

            with tifffile.TiffFile(path) as tif:
                imagej = tif.imagej_metadata or {}
                if isinstance(imagej, dict) and imagej.get("spacing"):
                    self.voxel_z.setValue(float(imagej["spacing"]))
                    found.append(f"z={float(imagej['spacing']):.4g} from ImageJ spacing")

                ome = tif.ome_metadata or ""
                for axis, spin in (("X", self.voxel_x), ("Y", self.voxel_y), ("Z", self.voxel_z)):
                    match = re.search(rf'PhysicalSize{axis}="([^"]+)"', ome)
                    if match:
                        spin.setValue(float(match.group(1)))
                        found.append(f"{axis.lower()}={float(match.group(1)):.4g} from OME")
        except Exception:
            found = []
        if hasattr(self, "voxel_status"):
            if found:
                self.voxel_status.setText("Auto detected: " + ", ".join(found))
            else:
                self.voxel_status.setText("No reliable TIFF voxel metadata found. Using manual override values.")

    def choose_output_dir(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose output folder", self.output_dir.text())
        if path:
            self.output_dir.setText(path)

    def choose_initial_csv(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load initial CSV", "", "CSV files (*.csv);;All files (*)")
        if not path:
            return
        self.load_initial_csv(Path(path))

    def choose_auto_config(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose Cell Lumen config",
            str(DEFAULT_CELL_LUMEN_CONFIG.parent),
            "YAML files (*.yaml *.yml);;All files (*)",
        )
        if path:
            self._set_auto_detect_config_path(Path(path))

    def _refresh_auto_detect_config_menu(self) -> None:
        candidates = self._auto_detect_config_candidates()
        menu = QtWidgets.QMenu(self.auto_detect_config_button)
        if not candidates:
            no_action = QtWidgets.QAction("No config files found", menu)
            no_action.setEnabled(False)
            menu.addAction(no_action)
        else:
            for config_path in candidates:
                action = QtWidgets.QAction(config_path.name, menu)
                action.setToolTip(str(config_path))
                action.triggered.connect(lambda checked=False, p=config_path: self._set_auto_detect_config_path(p))
                menu.addAction(action)
        menu.addSeparator()
        menu.addAction("Browse custom config...", self.choose_auto_config)
        self.auto_detect_config_button.setMenu(menu)

    def _auto_detect_config_candidates(self) -> list[Path]:
        seen = set[str]()
        ordered: list[Path] = []
        for config_path in AUTO_DETECT_CONFIG_PRESETS:
            resolved = config_path.expanduser().resolve()
            if resolved.exists() and str(resolved) not in seen:
                seen.add(str(resolved))
                ordered.append(resolved)
        for fallback in sorted(
            list(DEFAULT_CELL_LUMEN_CONFIG.parent.glob("*.yml")) + list(DEFAULT_CELL_LUMEN_CONFIG.parent.glob("*.yaml")),
            key=lambda p: p.name.lower(),
        ):
            resolved = fallback.resolve()
            if str(resolved) not in seen:
                seen.add(str(resolved))
                ordered.append(resolved)
        return ordered

    def _set_auto_detect_config_path(self, path: Path) -> None:
        self._auto_detect_preset_config = path
        self.auto_config_path.setText(str(path))
        self.auto_config_path.setToolTip(str(path))
        self._refresh_auto_detect_config_menu()
        if path.exists():
            self.auto_config_path.setStyleSheet("")
        else:
            self.auto_config_path.setStyleSheet("border-color: #e56b6b;")

    def run_auto_detect_cells(self) -> None:
        if self.volume is None or self.volume_path is None:
            QtWidgets.QMessageBox.warning(self, "No frame", "Open one TIFF frame before running auto detection.")
            return
        if self.auto_process is not None and self.auto_process.state() != QtCore.QProcess.NotRunning:
            QtWidgets.QMessageBox.information(self, "Auto detection running", "Cell Lumen is already running.")
            return

        binary = CPP_CELLUNIVERSE_BINARY
        if not binary.exists():
            QtWidgets.QMessageBox.critical(
                self,
                "Cell Lumen binary missing",
                f"Cannot find the C++ CellUniverse binary:\n{binary}\n\nBuild the C++ project first.",
            )
            return

        config_path = Path(self.auto_config_path.text()).expanduser()
        if not config_path.exists():
            QtWidgets.QMessageBox.critical(self, "Config missing", f"Cannot find the Cell Lumen config:\n{config_path}")
            return

        output_dir = self._auto_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        frame = self.frame_spin.value()
        stem = re.sub(r"[^A-Za-z0-9_]+", "_", self.volume_path.stem).strip("_") or "frame"
        csv_output = output_dir / f"CellLumen_auto_detect_{stem}_{frame:03d}.csv"
        self._auto_detect_csv = csv_output

        self._clear_auto_log()
        self._show_busy_dialog(
            "Cell Lumen Detecting",
            "Auto detection is running. Keep this app active and we will show live output here.",
        )
        self._append_auto_log("Starting Cell Lumen auto detection.")
        self._append_auto_log(f"Input TIFF: {self.volume_path}")
        self._append_auto_log(f"Config: {config_path}")
        self._append_auto_log(f"Output CSV: {csv_output}")
        self._append_auto_log("TIFF preview writing is skipped for this App pass so the UI gets the CSV faster.")

        process = QtCore.QProcess(self)
        process.setProgram(str(binary))
        process.setArguments(["--cell-lumen", str(self.volume_path), str(output_dir), str(config_path), str(csv_output)])
        process.setWorkingDirectory(str(CPP_CELLUNIVERSE_BINARY.parent.parent))
        process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        env = QtCore.QProcessEnvironment.systemEnvironment()
        env.insert("CELLUNIVERSE_CELL_LUMEN_SKIP_TIFF", "1")
        process.setProcessEnvironment(env)
        process.readyReadStandardOutput.connect(self._read_auto_detect_output)
        process.finished.connect(self._auto_process_finished)
        self.auto_process = process
        self._set_auto_detect_running(True)
        process.start()
        if not process.waitForStarted(3000):
            self._hide_busy_dialog()
            self._set_auto_detect_running(False)
            self.auto_process = None
            QtWidgets.QMessageBox.critical(self, "Auto detection failed", "Cell Lumen process could not start.")

    def _auto_output_dir(self) -> Path:
        return Path(self.output_dir.text()).expanduser() / "auto_detect"

    def _append_auto_log(self, text: str) -> None:
        cleaned = text.rstrip()
        if not cleaned:
            return
        self._auto_detect_log_lines.append(cleaned)
        if len(self._auto_detect_log_lines) > 500:
            del self._auto_detect_log_lines[:-500]
        if hasattr(self, "auto_log"):
            self.auto_log.moveCursor(QtGui.QTextCursor.End)
            self.auto_log.insertPlainText(cleaned + "\n")
            self.auto_log.moveCursor(QtGui.QTextCursor.End)
        if self._busy_log_widget is not None:
            self._busy_log_widget.moveCursor(QtGui.QTextCursor.End)
            self._busy_log_widget.insertPlainText(cleaned + "\n")
            self._busy_log_widget.moveCursor(QtGui.QTextCursor.End)

    def _show_busy_dialog(self, title: str, message: str) -> None:
        self._hide_busy_dialog()
        dialog = QtWidgets.QDialog(self)
        dialog.setObjectName("AutoDetectBusyDialog")
        dialog.setWindowTitle(title)
        dialog.setMinimumWidth(620)
        dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
        dialog.setWindowModality(QtCore.Qt.ApplicationModal)
        dialog.setWindowFlags(dialog.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)
        title_label = QtWidgets.QLabel("Cell Lumen Auto Detection")
        title_label.setObjectName("AutoDetectDialogTitle")
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title_label)
        self._busy_status_label = QtWidgets.QLabel(message)
        self._busy_status_label.setWordWrap(True)
        layout.addWidget(self._busy_status_label)
        self._busy_log_widget = QtWidgets.QPlainTextEdit()
        self._busy_log_widget.setObjectName("AutoDetectBusyLog")
        self._busy_log_widget.setReadOnly(True)
        self._busy_log_widget.setMinimumHeight(220)
        self._busy_log_widget.setMaximumHeight(280)
        layout.addWidget(self._busy_log_widget)
        if self._auto_detect_log_lines:
            self._busy_log_widget.setPlainText("\n".join(self._auto_detect_log_lines))
            self._busy_log_widget.moveCursor(QtGui.QTextCursor.End)
        self._busy_dialog = dialog
        self._append_auto_log(message)
        dialog.show()
        QtWidgets.QApplication.processEvents()

    def _update_busy_dialog(self, message: str) -> None:
        self._append_auto_log(message)
        if self._busy_dialog is not None:
            if self._busy_status_label is not None:
                self._busy_status_label.setText(message)
            QtWidgets.QApplication.processEvents()

    def _hide_busy_dialog(self) -> None:
        if self._busy_dialog is not None:
            self._busy_dialog.close()
            self._busy_dialog.deleteLater()
            self._busy_dialog = None
        self._busy_log_widget = None
        self._busy_status_label = None
        QtWidgets.QApplication.processEvents()

    def _set_auto_detect_running(self, running: bool) -> None:
        if hasattr(self, "auto_detect_button"):
            self.auto_detect_button.setEnabled(not running)
            self.auto_detect_button.setText("Running Cell Lumen..." if running else "Auto Detect Cells (Cell Lumen & PCA)")
        if hasattr(self, "clean_auto_button"):
            self.clean_auto_button.setEnabled(not running)
        if hasattr(self, "auto_detect_config_button"):
            self.auto_detect_config_button.setEnabled(not running)

    def _read_auto_detect_output(self) -> None:
        if self.auto_process is None:
            return
        data = bytes(self.auto_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in data.splitlines():
            self._append_auto_log(line)

    def _auto_process_finished(self, exit_code: int, status: QtCore.QProcess.ExitStatus) -> None:
        process = self.auto_process
        self._read_auto_detect_output()
        self._set_auto_detect_running(False)
        self.auto_process = None
        self._hide_busy_dialog()
        if process is not None:
            process.deleteLater()

        csv_path = self._auto_detect_csv
        if status != QtCore.QProcess.NormalExit or exit_code != 0:
            self._append_auto_log(f"Cell Lumen failed with exit code {exit_code}.")
            QtWidgets.QMessageBox.critical(
                self,
                "Auto detection failed",
                "Cell Lumen did not finish successfully. Check the auto detection log for the exact C++ message.",
            )
            return
        if csv_path is None or not csv_path.exists():
            self._append_auto_log("Cell Lumen finished but did not create the expected CSV.")
            QtWidgets.QMessageBox.critical(self, "Auto detection failed", "Cell Lumen finished but no output CSV was found.")
            return

        try:
            self._show_busy_dialog("Preparing Editable Cells", "Cell Lumen finished. Importing detected ellipsoids...")
            imported = self._import_auto_detect_csv(csv_path)
            self._update_busy_dialog(f"Imported {imported} detected cells. Preparing napari editable layers...")
            if imported > 0:
                self.open_napari_adjuster()
                self._update_busy_dialog("Drawing low opacity editable ellipsoid shells and selected cell handles...")
                self._refresh_background_ring_layer()
                self.set_interaction_mode("edit")
            self._append_auto_log(f"Imported {imported} auto detected cells into the manual editor.")
        except Exception as exc:
            self._hide_busy_dialog()
            self._append_auto_log(f"Could not import auto detection CSV: {exc}")
            QtWidgets.QMessageBox.critical(self, "Auto CSV import failed", f"Cannot import the Cell Lumen output:\n{exc}")
            return
        finally:
            self._hide_busy_dialog()

        QtWidgets.QMessageBox.information(
            self,
            "Auto detection finished",
            f"Imported {imported} Cell Lumen cells.\n\nThe result is an editable first pass. Please delete or adjust wrong cells before saving.",
        )

    def _import_auto_detect_csv(self, path: Path) -> int:
        frame = self.frame_spin.value()
        imported: list[ManualCell] = []
        existing_names = {cell.name for cell in self.cells if not self._is_auto_detected_cell(cell)}
        self._append_auto_log("Parsing Cell Lumen CSV and converting rows into editable ellipsoids.")
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError("CSV has no header row.")
            required = {"x", "y", "z"}
            if not required.issubset(set(reader.fieldnames)):
                raise ValueError("Cell Lumen CSV must contain x, y, and z columns.")
            for row_index, row in enumerate(reader, start=1):
                name = (row.get("name") or "").strip() or f"CL{row_index:03d}"
                if name in existing_names:
                    name = f"CL{row_index:03d}"
                existing_names.add(name)
                major = self._float_from_row(row, ("majorRadius", "aRadius", "radiusA", "radius"), 13.0)
                middle = self._float_from_row(row, ("bRadius", "middleRadius", "radiusB"), major)
                minor_from_cell_lumen = row.get("minorRadius") not in (None, "") and row.get("cRadius") in (None, "")
                minor = self._float_from_row(row, ("minorRadius", "cRadius", "radiusC"), min(major, middle, 8.0))
                if minor_from_cell_lumen:
                    minor = minor / self.z_display_scale()
                cell = ManualCell(
                    name=name,
                    frame=frame,
                    category="mature",
                    x=self._float_from_row(row, ("x",), 0.0),
                    y=self._float_from_row(row, ("y",), 0.0),
                    z=self._float_from_row(row, ("z",), 0.0),
                    a=max(1.0, major),
                    b=max(1.0, middle),
                    c=max(1.0, minor),
                    finalized=True,
                    notes=f"{AUTO_DETECT_NOTE}; source={path.name}; z radius converted for napari display",
                    order=len(imported) + 1,
                )
                self._clamp_cell_to_volume(cell)
                imported.append(cell)
                if row_index % 100 == 0:
                    self._update_busy_dialog(f"Converted {row_index} detected cells into editable ellipsoids...")

        self.cells = [cell for cell in self.cells if not self._is_auto_detected_cell(cell)]
        start_index = len(self.cells)
        for offset, cell in enumerate(imported):
            cell.order = start_index + offset + 1
        self.cells.extend(imported)
        self._mark_background_rings_dirty()
        self._reload_list(select=start_index if imported else min(self.selected_index, len(self.cells) - 1), refresh=False)
        self._append_auto_log(f"Converted {len(imported)} Cell Lumen rows into editable cells.")
        return len(imported)

    def _float_from_row(self, row: dict[str, str], keys: tuple[str, ...], default: float) -> float:
        for key in keys:
            value = row.get(key)
            if value is None or str(value).strip() == "":
                continue
            try:
                return float(value)
            except ValueError:
                continue
        return float(default)

    def _is_auto_detected_cell(self, cell: ManualCell) -> bool:
        return cell.notes.startswith(AUTO_DETECT_NOTE)

    def clean_auto_detected_cells(self) -> None:
        if self.auto_process is not None and self.auto_process.state() != QtCore.QProcess.NotRunning:
            QtWidgets.QMessageBox.information(self, "Auto detection running", "Wait for Cell Lumen to finish before cleaning results.")
            return
        before = len(self.cells)
        self.cells = [cell for cell in self.cells if not self._is_auto_detected_cell(cell)]
        removed = before - len(self.cells)
        self._mark_background_rings_dirty()
        self._reload_list(select=min(self.selected_index, len(self.cells) - 1))
        self._append_auto_log(f"Cleaned {removed} auto detected cells.")
        if removed:
            self.update_napari_edit_layers()

    def load_initial_csv(self, path: Path) -> None:
        loaded: list[ManualCell] = []
        frame = self.frame_spin.value()
        match = "".join(ch if ch.isdigit() else " " for ch in path.stem).split()
        if match:
            frame = int(match[-1])
            self.frame_spin.setValue(frame)
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"cellType", "z", "y", "x"}
            if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
                QtWidgets.QMessageBox.warning(
                    self,
                    "Invalid CSV",
                    "Initial CSV must contain cellType,z,y,x columns.",
                )
                return
            for row_index, row in enumerate(reader, start=1):
                cell_type = (row.get("cellType") or "").strip()
                lowered = cell_type.lower()
                if "type 2" in lowered:
                    category = "split"
                    name = f"S{row_index:03d}"
                elif "type 3" in lowered:
                    category = "pre_split"
                    name = f"P{row_index:03d}"
                else:
                    category = "mature"
                    name = f"M{row_index:03d}"
                loaded.append(
                    ManualCell(
                        name=(row.get("name") or row.get("cellName") or name).strip(),
                        frame=frame,
                        category=category,
                        z=float(row["z"]),
                        y=float(row["y"]),
                        x=float(row["x"]),
                        a=float(row.get("aRadius") or row.get("majorRadius") or row.get("radiusA") or 13.0),
                        b=float(row.get("bRadius") or row.get("radiusB") or row.get("middleRadius") or row.get("aRadius") or 13.0),
                        c=float(row.get("cRadius") or row.get("minorRadius") or row.get("radiusC") or 8.0),
                        pair_id=(row.get("pairId") or "").strip(),
                        pair_role=(row.get("pairRole") or "").strip(),
                        order=row_index,
                        finalized=True,
                        notes=f"loaded from {path.name}",
                    )
                )
        self.cells = loaded
        self._cell_counter = len([cell for cell in loaded if cell.category != "split"]) + 1
        self._pair_counter = max(1, len([cell for cell in loaded if cell.category == "split"]) // 2 + 1)
        self._mark_background_rings_dirty()
        self._reload_list(select=0 if self.cells else -1)

    def add_cell_or_pair(self) -> None:
        if self.volume is None:
            QtWidgets.QMessageBox.warning(self, "No frame", "Open a frame tif first.")
            return
        if self.category_combo.currentData() == "none":
            QtWidgets.QMessageBox.information(self, "No cell kind", "Choose a cell kind before creating a 3D ellipsoid.")
            return
        if self._creating_candidate:
            return
        self._creating_candidate = True
        try:
            self._add_cell_or_pair_impl()
        finally:
            self._creating_candidate = False

    def _add_cell_or_pair_impl(self) -> None:
        self.open_napari_adjuster()
        category = self.category_combo.currentData()
        frame = self.frame_spin.value()
        x, y, z = self._candidate_center()
        a, b, c = self.a_spin.value(), self.b_spin.value(), self.c_spin.value()
        if category == "split":
            pair_id = f"split_pair_{self._pair_counter:03d}"
            self._pair_counter += 1
            left = ManualCell(
                name=f"S{self._pair_counter - 1:03d}A",
                frame=frame,
                category=category,
                x=max(0.0, x - a * 0.75),
                y=y,
                z=z,
                a=a,
                b=b,
                c=c,
                pair_id=pair_id,
                pair_role="left",
                order=len(self.cells) + 1,
            )
            right = ManualCell(
                name=f"S{self._pair_counter - 1:03d}B",
                frame=frame,
                category=category,
                x=x + a * 0.75,
                y=y,
                z=z,
                a=a,
                b=b,
                c=c,
                pair_id=pair_id,
                pair_role="right",
                order=len(self.cells) + 2,
            )
            self.cells.extend([left, right])
            self._mark_background_rings_dirty()
            self._reload_list(select=len(self.cells) - 2)
        else:
            prefix = "M" if category == "mature" else "P"
            cell = ManualCell(
                name=f"{prefix}{self._cell_counter:03d}",
                frame=frame,
                category=category,
                x=x,
                y=y,
                z=z,
                a=a,
                b=b,
                c=c,
                order=len(self.cells) + 1,
            )
            self._cell_counter += 1
            self.cells.append(cell)
            self._mark_background_rings_dirty()
            self._reload_list(select=len(self.cells) - 1)
        self.set_interaction_mode("edit")

    def prepare_candidate_for_selected_category(self) -> None:
        if self.volume is None:
            return
        current = self.selected_cell
        category = self.category_combo.currentData()
        if category == "none":
            if current is not None and not current.finalized:
                self._remove_active_unfinalized_group()
                self._reload_list(select=-1)
            return
        if current is not None and not current.finalized and current.category == category:
            return
        if current is not None and not current.finalized:
            self._remove_active_unfinalized_group()
        self.add_cell_or_pair()

    def _remove_active_unfinalized_group(self) -> None:
        current = self.selected_cell
        if current is None or current.finalized:
            return
        if current.pair_id:
            self.cells = [cell for cell in self.cells if cell.pair_id != current.pair_id]
        else:
            del self.cells[self.selected_index]
        self.selected_index = -1
        self._mark_background_rings_dirty()

    def _candidate_center(self) -> tuple[float, float, float]:
        if self.volume is None:
            return self.x_spin.value(), self.y_spin.value(), self.z_spin.value()
        z_size, y_size, x_size = self.volume.shape
        return x_size * 0.5, y_size * 0.5, z_size * 0.5

    def finish_selected(self) -> None:
        cell = self.selected_cell
        if cell is None:
            return
        cell.finalized = True
        if cell.category == "split" and cell.pair_id:
            for index, other in enumerate(self.cells):
                if other.pair_id == cell.pair_id and other.name != cell.name and not other.finalized:
                    self._reload_list(select=index)
                    return
        self._reload_list(select=self.selected_index)

    def delete_selected(self) -> None:
        if self.selected_cell is None:
            return
        del self.cells[self.selected_index]
        self._mark_background_rings_dirty()
        self._reload_list(select=min(self.selected_index, len(self.cells) - 1))

    def select_relative(self, delta: int) -> None:
        if not self.cells:
            return
        self._reload_list(select=max(0, min(len(self.cells) - 1, self.selected_index + delta)), refresh=False)

    def select_cell(self, index: int) -> None:
        self.selected_index = index
        self._cell_to_controls()
        self.update_napari_edit_layers()

    def _reload_list(self, select: int = -1, *, refresh: bool = True) -> None:
        self.cell_list.blockSignals(True)
        self.cell_list.clear()
        for cell in self.cells:
            done = "✓" if cell.finalized else "…"
            pair = f" {cell.pair_role}" if cell.pair_role else ""
            self.cell_list.addItem(
                f"{done} {cell.name}{pair}  {CATEGORY_DISPLAY[cell.category]}  "
                f"x={cell.x:.1f} y={cell.y:.1f} z={cell.z:.1f}  "
                f"a/b/c={cell.a:.1f}/{cell.b:.1f}/{cell.c:.1f}"
            )
        self.cell_list.blockSignals(False)
        if self.cells:
            select = max(0, min(len(self.cells) - 1, select))
            self.cell_list.setCurrentRow(select)
            self.selected_index = select
        else:
            self.selected_index = -1
        self._cell_to_controls()
        if refresh:
            self.refresh_views()
        else:
            self.update_napari_edit_layers()

    def _cell_to_controls(self) -> None:
        cell = self.selected_cell
        if cell is None:
            return
        self._updating_controls = True
        self.x_spin.setValue(cell.x)
        self.y_spin.setValue(cell.y)
        self.z_spin.setValue(cell.z)
        self.a_spin.setValue(cell.a)
        self.b_spin.setValue(cell.b)
        self.c_spin.setValue(cell.c)
        self._updating_controls = False

    def _clear_shape_controls(self) -> None:
        self._updating_controls = True
        self.x_spin.setValue(0.0)
        self.y_spin.setValue(0.0)
        self.z_spin.setValue(0.0)
        self.a_spin.setValue(1.0)
        self.b_spin.setValue(1.0)
        self.c_spin.setValue(1.0)
        self._updating_controls = False

    def _reset_workspace_after_successful_save(self) -> None:
        self.volume = None
        self.volume_path = None
        self._napari_volume_path = None
        self._cached_base.clear()
        self.cells.clear()
        self.selected_index = -1
        self._center_layer_cell_indices.clear()
        self._last_control_data = None
        self._drag_mode = None
        self._auto_detect_csv = None
        self._background_ring_refresh_timer.stop()
        self._napari_sync_timer.stop()
        self._background_rings_dirty = True

        self.tif_path.clear()
        self.frame_spin.setValue(0)
        self.category_combo.blockSignals(True)
        self.category_combo.setCurrentIndex(0)
        self.category_combo.blockSignals(False)
        self.cell_list.blockSignals(True)
        self.cell_list.clear()
        self.cell_list.blockSignals(False)
        self._clear_shape_controls()
        self.set_interaction_mode("view")
        self._clear_auto_log()
        self._append_auto_log("Saved CSV files. Workspace was cleared for the next TIFF frame.")

        if self.napari_viewer is not None:
            self._syncing_napari = True
            try:
                for layer in list(self.napari_viewer.layers):
                    self.napari_viewer.layers.remove(layer)
                self.napari_viewer.dims.ndisplay = 3
            except Exception as exc:
                self._log_napari_error("_reset_workspace_after_successful_save", exc)
            finally:
                self._syncing_napari = False

        self.refresh_views()

    def _clear_auto_log(self) -> None:
        self._auto_detect_log_lines.clear()
        if hasattr(self, "auto_log"):
            self.auto_log.clear()
        if self._busy_log_widget is not None:
            self._busy_log_widget.clear()

    def _controls_to_cell(self) -> None:
        if self._updating_controls:
            return
        cell = self.selected_cell
        if cell is None:
            return
        cell.x = self.x_spin.value()
        cell.y = self.y_spin.value()
        cell.z = self.z_spin.value()
        cell.a = self.a_spin.value()
        cell.b = self.b_spin.value()
        cell.c = self.c_spin.value()
        self._clamp_cell_to_volume(cell)
        self._update_selected_list_item()
        self._mark_background_rings_dirty(schedule=True)
        self.update_napari_edit_layers()

    def pick_coordinate(self, plane: str, u: float, v: float, phase: str) -> None:
        cell = self.selected_cell
        if cell is None:
            return
        z_scale = self.z_display_scale()
        if phase == "press" or self._drag_mode is None or self._drag_mode[0] != plane:
            self._drag_mode = (plane, self._pick_drag_mode(cell, plane, u, v, z_scale))

        mode = self._drag_mode[1]
        if plane == "xy":
            if mode == "radius_a":
                cell.a = max(1.0, abs(u - cell.x))
            elif mode == "radius_b":
                cell.b = max(1.0, abs(v - cell.y))
            else:
                cell.x = u
                cell.y = v
        elif plane == "xz":
            z_value = v / z_scale
            if mode == "radius_a":
                cell.a = max(1.0, abs(u - cell.x))
            elif mode == "radius_c":
                cell.c = max(1.0, abs(z_value - cell.z))
            else:
                cell.x = u
                cell.z = z_value
        elif plane == "yz":
            z_value = v / z_scale
            if mode == "radius_b":
                cell.b = max(1.0, abs(u - cell.y))
            elif mode == "radius_c":
                cell.c = max(1.0, abs(z_value - cell.z))
            else:
                cell.y = u
                cell.z = z_value

        self._clamp_cell_to_volume(cell)
        self._cell_to_controls()
        self._update_selected_list_item()
        self._mark_background_rings_dirty(schedule=True)
        self.update_napari_edit_layers()
        if phase == "release":
            self._drag_mode = None

    def _pick_drag_mode(self, cell: ManualCell, plane: str, u: float, v: float, z_scale: float) -> str:
        threshold = 14.0

        def near(point_u: float, point_v: float) -> bool:
            return math.hypot(u - point_u, v - point_v) <= threshold

        if plane == "xy":
            if near(cell.x - cell.a, cell.y) or near(cell.x + cell.a, cell.y):
                return "radius_a"
            if near(cell.x, cell.y - cell.b) or near(cell.x, cell.y + cell.b):
                return "radius_b"
        elif plane == "xz":
            z_view = cell.z * z_scale
            if near(cell.x - cell.a, z_view) or near(cell.x + cell.a, z_view):
                return "radius_a"
            if near(cell.x, z_view - cell.c * z_scale) or near(cell.x, z_view + cell.c * z_scale):
                return "radius_c"
        elif plane == "yz":
            z_view = cell.z * z_scale
            if near(cell.y - cell.b, z_view) or near(cell.y + cell.b, z_view):
                return "radius_b"
            if near(cell.y, z_view - cell.c * z_scale) or near(cell.y, z_view + cell.c * z_scale):
                return "radius_c"
        return "center"

    def z_display_scale(self) -> float:
        xy = max(1e-6, 0.5 * (self.voxel_x.value() + self.voxel_y.value()))
        return max(0.2, min(20.0, self.voxel_z.value() / xy))

    def refresh_views(self) -> None:
        if hasattr(self, "workspace_status"):
            if self.volume is None:
                self.workspace_status.setText("Open a frame TIFF stack to launch the napari 3D editor.")
            else:
                count = len(self.cells)
                self.workspace_status.setText(
                    f"Embedded napari 3D workspace is active.\n"
                    f"Frame {self.frame_spin.value():03d} | manual cells {count}\n"
                    f"Create a cell, then drag the red center and XYZ handles directly in 3D."
                )
        if not hasattr(self, "xy_canvas"):
            self.sync_napari()
            return
        if self.volume is None:
            blank = np.zeros((360, 480, 3), dtype=np.uint8)
            for canvas in (self.xy_canvas, self.xz_canvas, self.yz_canvas):
                canvas.set_bgr(blank)
            return
        self.xy_canvas.set_bgr(self._render_plane("xy"))
        self.xz_canvas.set_bgr(self._render_plane("xz"))
        self.yz_canvas.set_bgr(self._render_plane("yz"))
        self.sync_napari()

    def _base_plane(self, plane: str) -> np.ndarray:
        assert self.volume is not None
        z_scale = self.z_display_scale()
        selected_x = self.selected_cell.x if self.selected_cell is not None else -1.0
        selected_y = self.selected_cell.y if self.selected_cell is not None else -1.0
        key = f"{plane}:{z_scale:.3f}:{selected_x:.1f}:{selected_y:.1f}"
        if key in self._cached_base:
            return self._cached_base[key].copy()
        if plane == "xy":
            gray = normalize_u8(np.max(self.volume, axis=0))
        elif plane == "xz":
            center_y = self.selected_cell.y if self.selected_cell is not None else self.volume.shape[1] * 0.5
            slab = self._slab_indices(center_y, self.volume.shape[1], 4)
            gray = normalize_u8(np.max(self.volume[:, slab, :], axis=1))
            gray = resize_gray_height(gray, max(1, int(round(gray.shape[0] * z_scale))))
        else:
            center_x = self.selected_cell.x if self.selected_cell is not None else self.volume.shape[2] * 0.5
            slab = self._slab_indices(center_x, self.volume.shape[2], 4)
            gray = normalize_u8(np.max(self.volume[:, :, slab], axis=2))
            gray = resize_gray_height(gray, max(1, int(round(gray.shape[0] * z_scale))))
        bgr = gray_to_bgr(gray)
        self._cached_base[key] = bgr
        return bgr.copy()

    def _slab_indices(self, center: float, limit: int, half_width: int) -> slice:
        middle = int(round(center))
        start = max(0, middle - half_width)
        stop = min(limit, middle + half_width + 1)
        if stop <= start:
            stop = min(limit, start + 1)
        return slice(start, stop)

    def _render_plane(self, plane: str) -> np.ndarray:
        image = self._base_plane(plane)
        overlay = image.copy()
        z_scale = self.z_display_scale()
        for index, cell in enumerate(self.cells):
            color = CATEGORY_COLORS_BGR[cell.category]
            selected = index == self.selected_index
            thickness = 2 if selected else 1
            alpha = 0.35 if selected else 0.22
            if plane == "xy":
                center = (int(round(cell.x)), int(round(cell.y)))
                axes = (max(1, int(round(cell.a))), max(1, int(round(cell.b))))
                x_axis = ((center[0] - axes[0], center[1]), (center[0] + axes[0], center[1]))
                y_axis = ((center[0], center[1] - axes[1]), (center[0], center[1] + axes[1]))
            elif plane == "xz":
                center = (int(round(cell.x)), int(round(cell.z * z_scale)))
                axes = (max(1, int(round(cell.a))), max(1, int(round(cell.c * z_scale))))
                x_axis = ((center[0] - axes[0], center[1]), (center[0] + axes[0], center[1]))
                y_axis = ((center[0], center[1] - axes[1]), (center[0], center[1] + axes[1]))
            else:
                center = (int(round(cell.y)), int(round(cell.z * z_scale)))
                axes = (max(1, int(round(cell.b))), max(1, int(round(cell.c * z_scale))))
                x_axis = ((center[0] - axes[0], center[1]), (center[0] + axes[0], center[1]))
                y_axis = ((center[0], center[1] - axes[1]), (center[0], center[1] + axes[1]))
            draw_ellipse_outline(overlay, center, axes, color, thickness)
            draw_ellipse_outline(overlay, center, (max(1, axes[0] // 2), max(1, axes[1] // 2)), color, 1)
            draw_line(overlay, x_axis[0], x_axis[1], (255, 80, 80), 1)
            draw_line(overlay, y_axis[0], y_axis[1], (80, 210, 255), 1)
            for handle in (x_axis[0], x_axis[1], y_axis[0], y_axis[1]):
                draw_circle(overlay, handle, 5 if selected else 3, (245, 245, 245), filled=False)
            dot_color = (0, 0, 255) if selected else (0, 130, 255)
            draw_circle(overlay, center, 4 if selected else 3, dot_color, filled=True)
            blend_overlay(image, overlay, alpha)
        return image

    def _install_shortcuts(self) -> None:
        # Key handling is centralized in eventFilter(). Qt shortcuts and napari
        # key bindings can both receive the same physical key press, which made a
        # single movement request run more than once and could leave the viewer
        # doing repeated layer updates. Keeping this method as a no-op preserves
        # the old call site while making keyboard edits deterministic.
        return

    def nudge_selected(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> None:
        cell = self.selected_cell
        if cell is None:
            return
        cell.x += dx
        cell.y += dy
        cell.z += dz
        self._clamp_cell_to_volume(cell)
        self._cell_to_controls()
        self._update_selected_list_item()
        self._mark_background_rings_dirty(schedule=True)
        self.update_napari_edit_layers()

    def scale_selected(self, delta: float) -> None:
        cell = self.selected_cell
        if cell is None:
            return
        cell.a = max(1.0, cell.a + delta)
        cell.b = max(1.0, cell.b + delta)
        cell.c = max(1.0, cell.c + delta * 0.5)
        self._cell_to_controls()
        self._update_selected_list_item()
        self._mark_background_rings_dirty(schedule=True)
        self.update_napari_edit_layers()

    def _clamp_cell_to_volume(self, cell: ManualCell) -> None:
        if self.volume is None:
            return
        z, y, x = self.volume.shape
        cell.x = float(np.clip(cell.x, 0, x - 1))
        cell.y = float(np.clip(cell.y, 0, y - 1))
        cell.z = float(np.clip(cell.z, 0, z - 1))

    def _layer_scale(self) -> tuple[float, float, float]:
        return (self.voxel_z.value(), self.voxel_y.value(), self.voxel_x.value())

    def open_napari_adjuster(self) -> None:
        if self.volume is None:
            QtWidgets.QMessageBox.warning(self, "No frame", "Open a frame TIFF stack before opening Napari.")
            return
        try:
            import napari
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Napari unavailable", f"Cannot import napari:\n{exc}")
            return

        try:
            if self.napari_viewer is None:
                try:
                    self.napari_viewer = napari.Viewer(
                        title="CellUniverse Manual Initial 3D Adjuster",
                        ndisplay=3,
                        show=False,
                    )
                except TypeError:
                    self.napari_viewer = napari.Viewer(
                        title="CellUniverse Manual Initial 3D Adjuster",
                        ndisplay=3,
                    )
                self._install_napari_key_bindings()
                self._install_napari_auto_mouse_gate()
                self._embed_napari_window()
            self._sync_napari_volume(force=True)
            self.sync_napari()
            self._embed_napari_window()
            if self.napari_viewer is not None and getattr(self.napari_viewer, "dims", None) is not None:
                self.napari_viewer.dims.ndisplay = 3
            if self.napari_viewer is not None:
                self.napari_viewer.window.qt_viewer.canvas.native.setFocus()
        except Exception as exc:
            self._log_napari_error("open_napari_adjuster", exc)
            QtWidgets.QMessageBox.critical(self, "Napari launch failed", f"Cannot open the 3D adjuster:\n{exc}")

    def _embed_napari_window(self) -> None:
        if self.napari_viewer is None:
            return
        qt_window = getattr(self.napari_viewer.window, "qt_viewer", None)
        if qt_window is None:
            qt_window = getattr(self.napari_viewer.window, "_qt_window", None)
        if qt_window is None:
            return
        if self._embedded_napari_widget is not None and self._embedded_napari_widget is not qt_window:
            self.viewer_host_layout.removeWidget(self._embedded_napari_widget)
            self._embedded_napari_widget.setParent(None)
            self._embedded_napari_widget.deleteLater()
            self._embedded_napari_widget = None
        if self._embedded_napari_widget is not qt_window:
            self.workspace_status.hide()
            qt_window.setParent(self.viewer_host)
            qt_window.setWindowFlags(QtCore.Qt.Widget)
            if hasattr(qt_window, "menuBar"):
                qt_window.menuBar().hide()
            self.viewer_host_layout.addWidget(qt_window, 1)
            self._embedded_napari_widget = qt_window
        qt_window.show()

    def sync_napari(self) -> bool:
        if self.napari_viewer is None or self.volume is None:
            return False
        try:
            self._sync_napari_volume(force=False)
            self._sync_napari_cells()
            return True
        except Exception:
            self._log_napari_error("sync_napari", traceback.format_exc())
            if hasattr(self, "workspace_status"):
                self.workspace_status.setText(
                    "Napari layer sync failed. The volume viewer is still kept alive.\n"
                    "Check /tmp/celluniverse_local_python_app.log for the full traceback."
                )
            return False

    def _sync_napari_volume(self, *, force: bool) -> None:
        if self.napari_viewer is None or self.volume is None:
            return
        layer_name = "real frame volume"
        scale = self._layer_scale()
        if force or self._napari_volume_path != self.volume_path or layer_name not in self.napari_viewer.layers:
            if layer_name in self.napari_viewer.layers:
                self._remove_napari_layer(layer_name)
            low, high = np.percentile(self.volume, [1.0, 99.7])
            self.napari_viewer.add_image(
                self.volume,
                name=layer_name,
                colormap=self.colormap_combo.currentText() if hasattr(self, "colormap_combo") else "viridis",
                opacity=0.72,
                scale=scale,
                rendering="mip",
                contrast_limits=(float(low), float(high)),
            )
            self._napari_volume_path = self.volume_path
        else:
            self.napari_viewer.layers[layer_name].scale = scale
            self.apply_colormap_to_volume()

    def apply_colormap_to_volume(self) -> None:
        if self.napari_viewer is None:
            return
        layer_name = "real frame volume"
        if layer_name not in self.napari_viewer.layers:
            return
        try:
            self.napari_viewer.layers[layer_name].colormap = self.colormap_combo.currentText()
        except Exception as exc:
            self._log_napari_error("apply_colormap_to_volume", exc)

    def set_ellipsoid_opacity_from_slider(self, value: int) -> None:
        if hasattr(self, "ellipsoid_opacity"):
            self.ellipsoid_opacity.blockSignals(True)
            self.ellipsoid_opacity.setValue(float(value) / 100.0)
            self.ellipsoid_opacity.blockSignals(False)
        self.update_napari_edit_layers()

    def set_ellipsoid_opacity_from_spin(self, value: float) -> None:
        if hasattr(self, "ellipsoid_opacity_slider"):
            self.ellipsoid_opacity_slider.blockSignals(True)
            self.ellipsoid_opacity_slider.setValue(int(round(float(value) * 100)))
            self.ellipsoid_opacity_slider.blockSignals(False)
        self.update_napari_edit_layers()

    def _ellipsoid_shells_visible(self) -> bool:
        return not hasattr(self, "ellipsoid_visibility_button") or self.ellipsoid_visibility_button.isChecked()

    def toggle_ellipsoid_shells(self) -> None:
        visible = self._ellipsoid_shells_visible()
        self.ellipsoid_visibility_button.setText("Hide Ellipsoid Shells" if visible else "Show Ellipsoid Shells")
        self._background_rings_dirty = True
        if self.napari_viewer is not None:
            if visible:
                self.update_napari_edit_layers()
                self._refresh_background_ring_layer()
            else:
                self._remove_napari_layer("manual ellipsoid rings")
                self._remove_napari_layer("selected ellipsoid rings")
                self._remove_napari_layer("selected XYZ axes")

    def _sync_napari_cells(self) -> None:
        assert self.napari_viewer is not None
        self._syncing_napari = True
        scale = self._layer_scale()
        try:
            finalized_items = [(index, cell) for index, cell in enumerate(self.cells) if cell.finalized]
            finalized_cells = [cell for _, cell in finalized_items]
            self._center_layer_cell_indices = [index for index, _ in finalized_items]
            centers = np.asarray([[cell.z, cell.y, cell.x] for cell in finalized_cells], dtype=float)
            colors = [(1.0, 0.04, 0.02, 0.92) for _ in finalized_cells]
            sizes = [6.5 for _ in finalized_cells]
            centers_layer = self._replace_points_layer("finalized cell centers", centers, colors, sizes, scale)
            if centers_layer is not None:
                self._set_layer_mode(centers_layer, "select")
                self._connect_napari_event(centers_layer, "selected_data", self._napari_cell_selection_changed)

            control_points, control_colors, control_sizes = self._selected_control_points()
            control_layer = self._replace_points_layer(
                "selected 3D center and XYZ handles",
                np.asarray(control_points, dtype=float) if control_points else np.empty((0, 3)),
                control_colors,
                control_sizes,
                scale,
            )
            if control_layer is not None:
                self._set_layer_mode(control_layer, "select")
                self._connect_napari_event(control_layer, "data", self._napari_controls_changed)
                try:
                    self.napari_viewer.layers.selection.active = control_layer
                except Exception:
                    pass
                self._last_control_data = np.asarray(control_layer.data, dtype=float).copy()

            self._sync_background_ring_layer(scale, force=self._background_rings_dirty)
            self._sync_selected_ring_layer(scale)
            self._remove_napari_layer("selected XYZ axes")
            self._apply_napari_interaction_mode()
        finally:
            self._syncing_napari = False

    def _replace_points_layer(
        self,
        name: str,
        points: np.ndarray,
        colors: list[tuple[float, float, float, float]],
        sizes: list[float],
        scale: tuple[float, float, float],
    ):
        assert self.napari_viewer is not None
        if name in self.napari_viewer.layers:
            self._remove_napari_layer(name)
        if len(points) == 0:
            return None
        size = max(sizes) if sizes else 8.0
        kwargs = {
            "name": name,
            "size": size,
            "face_color": self._napari_color_argument(colors),
            "border_color": np.asarray((0.0, 0.0, 0.0, 0.65), dtype=float),
            "border_width": 0.18,
            "scale": scale,
            "out_of_slice_display": True,
        }
        try:
            return self.napari_viewer.add_points(points, **kwargs)
        except TypeError:
            kwargs.pop("out_of_slice_display", None)
            return self.napari_viewer.add_points(points, **kwargs)

    def _napari_color_argument(self, colors: list[tuple[float, float, float, float]]):
        if not colors:
            return np.asarray((1.0, 0.0, 0.0, 1.0), dtype=float)
        if len(colors) == 1:
            return np.asarray(colors[0], dtype=float)
        return np.asarray(colors, dtype=float)

    def _connect_napari_event(self, layer, event_name: str, callback) -> None:
        try:
            getattr(layer.events, event_name).connect(callback)
        except Exception:
            return

    def _set_layer_mode(self, layer, mode: str) -> None:
        try:
            layer.mode = mode
        except Exception:
            pass

    def set_interaction_mode(self, mode: str) -> None:
        self._interaction_mode = "edit" if mode == "edit" else "view"
        if hasattr(self, "view_mode_button") and hasattr(self, "edit_mode_button"):
            self.view_mode_button.blockSignals(True)
            self.edit_mode_button.blockSignals(True)
            self.view_mode_button.setChecked(self._interaction_mode == "view")
            self.edit_mode_button.setChecked(self._interaction_mode == "edit")
            self.view_mode_button.blockSignals(False)
            self.edit_mode_button.blockSignals(False)
        self._apply_napari_interaction_mode()

    def _apply_napari_interaction_mode(self) -> None:
        if self.napari_viewer is None:
            return
        if self._interaction_mode == "edit":
            self._activate_control_layer()
        else:
            self._activate_view_layer()

    def _activate_control_layer(self) -> None:
        if self.napari_viewer is None:
            return
        layer_name = "selected 3D center and XYZ handles"
        if layer_name not in self.napari_viewer.layers:
            return
        try:
            layer = self.napari_viewer.layers[layer_name]
            self._set_layer_mode(layer, "select")
            self.napari_viewer.layers.selection.active = layer
        except Exception:
            pass

    def _activate_view_layer(self) -> None:
        if self.napari_viewer is None:
            return
        for layer_name in (
            "selected 3D center and XYZ handles",
            "finalized cell centers",
            "manual ellipsoid rings",
            "selected XYZ axes",
        ):
            if layer_name in self.napari_viewer.layers:
                self._set_layer_mode(self.napari_viewer.layers[layer_name], "pan_zoom")
        if "real frame volume" in self.napari_viewer.layers:
            try:
                self.napari_viewer.layers.selection.active = self.napari_viewer.layers["real frame volume"]
            except Exception:
                pass

    def _replace_shapes_layer(
        self,
        name: str,
        paths: list[np.ndarray],
        colors: list[tuple[float, float, float, float]],
        scale: tuple[float, float, float],
        *,
        edge_width: float,
        opacity: float,
    ) -> None:
        assert self.napari_viewer is not None
        if name in self.napari_viewer.layers:
            self._remove_napari_layer(name)
        if not paths:
            return
        try:
            self.napari_viewer.add_shapes(
                paths,
                shape_type="path",
                name=name,
                edge_color=self._napari_color_argument(colors),
                edge_width=max(edge_width, 0.1),
                opacity=opacity,
                scale=scale,
            )
        except TypeError:
            layer = self.napari_viewer.add_shapes(
                paths,
                shape_type="path",
                name=name,
                edge_color=self._napari_color_argument(colors),
                edge_width=max(edge_width, 0.1),
                opacity=opacity,
            )
            try:
                layer.scale = scale
            except Exception:
                pass

    def _sync_background_ring_layer(self, scale: tuple[float, float, float], *, force: bool) -> None:
        if self.napari_viewer is None:
            return
        if not self._ellipsoid_shells_visible():
            self._remove_napari_layer("manual ellipsoid rings")
            self._background_rings_dirty = True
            return
        if not force and not self._background_rings_dirty and "manual ellipsoid rings" in self.napari_viewer.layers:
            layer = self.napari_viewer.layers["manual ellipsoid rings"]
            layer.scale = scale
            layer.opacity = self.ellipsoid_opacity.value()
            return

        rings: list[np.ndarray] = []
        ring_colors: list[tuple[float, float, float, float]] = []
        if self.cells:
            self._append_auto_log(
                f"Drawing {len(self.cells)} editable ellipsoid shells with low detail background rings."
            )
        for index, cell in enumerate(self.cells, start=1):
            cell_rings = make_cell_rings(
                cell,
                segments=BACKGROUND_RING_SEGMENTS,
                rings_per_axis=BACKGROUND_RINGS_PER_AXIS,
                radius_scale=1.0,
            )
            rings.extend(cell_rings)
            ring_colors.extend([self._rgba_for_cell(cell, alpha=1.0)] * len(cell_rings))
            if index % 80 == 0:
                QtWidgets.QApplication.processEvents()
        self._replace_shapes_layer(
            "manual ellipsoid rings",
            rings,
            ring_colors,
            scale,
            edge_width=0.16,
            opacity=self.ellipsoid_opacity.value(),
        )
        self._background_rings_dirty = False

    def _sync_selected_ring_layer(self, scale: tuple[float, float, float]) -> None:
        if self.napari_viewer is None:
            return
        layer_name = "selected ellipsoid rings"
        cell = self.selected_cell
        if cell is None or not self._ellipsoid_shells_visible():
            self._remove_napari_layer(layer_name)
            return
        paths = make_cell_rings(
            cell,
            segments=SELECTED_RING_SEGMENTS,
            rings_per_axis=SELECTED_RINGS_PER_AXIS,
            radius_scale=1.0,
        )
        colors = [self._rgba_for_cell(cell, alpha=1.0)] * len(paths)
        if layer_name in self.napari_viewer.layers:
            layer = self.napari_viewer.layers[layer_name]
            layer.data = paths
            layer.scale = scale
            layer.opacity = self.ellipsoid_opacity.value()
            try:
                layer.edge_color = self._napari_color_argument(colors)
                layer.edge_width = 0.32
            except Exception:
                pass
            return
        self._replace_shapes_layer(
            layer_name,
            paths,
            colors,
            scale,
            edge_width=0.32,
            opacity=self.ellipsoid_opacity.value(),
        )

    def _mark_background_rings_dirty(self, *, schedule: bool = False) -> None:
        self._background_rings_dirty = True
        if schedule and self.napari_viewer is not None:
            self._background_ring_refresh_timer.start(700)

    def _refresh_background_ring_layer(self) -> None:
        if self._syncing_napari or self.napari_viewer is None or self.volume is None:
            return
        self._syncing_napari = True
        try:
            self._sync_background_ring_layer(self._layer_scale(), force=True)
        except Exception as exc:
            self._log_napari_error("_refresh_background_ring_layer", traceback.format_exc() if not isinstance(exc, str) else exc)
        finally:
            self._syncing_napari = False

    def update_napari_edit_layers(self) -> None:
        if self._syncing_napari or self.napari_viewer is None or self.volume is None:
            return
        self._syncing_napari = True
        try:
            scale = self._layer_scale()
            if "finalized cell centers" in self.napari_viewer.layers and self._center_layer_cell_indices:
                center_cells = [
                    self.cells[index]
                    for index in self._center_layer_cell_indices
                    if 0 <= index < len(self.cells) and self.cells[index].finalized
                ]
                self.napari_viewer.layers["finalized cell centers"].data = np.asarray(
                    [[cell.z, cell.y, cell.x] for cell in center_cells],
                    dtype=float,
                )
                self.napari_viewer.layers["finalized cell centers"].scale = scale
            control_points, _, _ = self._selected_control_points()
            if "selected 3D center and XYZ handles" in self.napari_viewer.layers and control_points:
                control_layer = self.napari_viewer.layers["selected 3D center and XYZ handles"]
                # Programmatic layer.data changes fire napari's layer data
                # event. _syncing_napari stays true for the whole block so the
                # data callback treats this as an internal redraw instead of
                # another user drag, which prevents recursive refresh freezes.
                control_layer.data = np.asarray(control_points, dtype=float)
                control_layer.scale = scale
                self._last_control_data = np.asarray(control_layer.data, dtype=float).copy()

            self._sync_selected_ring_layer(scale)
            if "manual ellipsoid rings" in self.napari_viewer.layers:
                ring_layer = self.napari_viewer.layers["manual ellipsoid rings"]
                ring_layer.scale = scale
                ring_layer.opacity = self.ellipsoid_opacity.value()

            self._remove_napari_layer("selected XYZ axes")
            self._apply_napari_interaction_mode()
        except Exception as exc:
            self._log_napari_error("update_napari_edit_layers", traceback.format_exc() if not isinstance(exc, str) else exc)
        finally:
            self._syncing_napari = False

    def _selected_axis_paths(self) -> tuple[list[np.ndarray], list[tuple[float, float, float, float]]]:
        cell = self.selected_cell
        if cell is None:
            return [], []
        z, y, x = cell.z, cell.y, cell.x
        paths = [
            np.asarray([[z, y, x - cell.a], [z, y, x + cell.a]], dtype=float),
            np.asarray([[z, y - cell.b, x], [z, y + cell.b, x]], dtype=float),
            np.asarray([[z - cell.c, y, x], [z + cell.c, y, x]], dtype=float),
        ]
        colors = [
            (0.16, 1.0, 0.08, 1.0),
            (0.0, 0.95, 0.9, 1.0),
            (1.0, 0.55, 0.08, 1.0),
        ]
        return paths, colors

    def _selected_handle_points(self) -> tuple[list[np.ndarray], list[tuple[float, float, float, float]]]:
        cell = self.selected_cell
        if cell is None:
            return [], []
        z, y, x = cell.z, cell.y, cell.x
        points = [
            np.asarray([z, y, x - cell.a], dtype=float),
            np.asarray([z, y, x + cell.a], dtype=float),
            np.asarray([z, y - cell.b, x], dtype=float),
            np.asarray([z, y + cell.b, x], dtype=float),
            np.asarray([z - cell.c, y, x], dtype=float),
            np.asarray([z + cell.c, y, x], dtype=float),
        ]
        colors = [
            (0.16, 1.0, 0.08, 1.0),
            (0.16, 1.0, 0.08, 1.0),
            (0.0, 0.95, 0.9, 1.0),
            (0.0, 0.95, 0.9, 1.0),
            (1.0, 0.55, 0.08, 1.0),
            (1.0, 0.55, 0.08, 1.0),
        ]
        return points, colors

    def _selected_control_points(
        self,
    ) -> tuple[list[np.ndarray], list[tuple[float, float, float, float]], list[float]]:
        cell = self.selected_cell
        if cell is None:
            return [], [], []
        handle_points, handle_colors = self._selected_handle_points()
        center = np.asarray([cell.z, cell.y, cell.x], dtype=float)
        center_size = max(7.0, 0.36 * (cell.a + cell.b + cell.c))
        return (
            [center] + handle_points,
            [(1.0, 0.02, 0.02, 1.0)] + handle_colors,
            [max(center_size, 10.0)] + [12.0] * len(handle_points),
        )

    def _remove_napari_layer(self, name: str) -> None:
        assert self.napari_viewer is not None
        try:
            layer = self.napari_viewer.layers[name]
            self.napari_viewer.layers.remove(layer)
        except Exception:
            pass

    def _log_napari_error(self, context: str, error: object) -> None:
        message = str(error)
        try:
            with Path("/tmp/celluniverse_local_python_app.log").open("a", encoding="utf-8") as handle:
                handle.write(f"\n[NAPARI ERROR] {context}\n{message}\n")
        except Exception:
            pass

    def _napari_controls_changed(self, event=None) -> None:
        if self._syncing_napari or self.napari_viewer is None:
            return
        cell = self.selected_cell
        layer_name = "selected 3D center and XYZ handles"
        if cell is None or layer_name not in self.napari_viewer.layers:
            return
        data = np.asarray(self.napari_viewer.layers[layer_name].data, dtype=float)
        if data.shape[0] < 7 or data.shape[1] < 3:
            return
        moved_index = -1
        if self._last_control_data is not None and self._last_control_data.shape == data.shape:
            deltas = np.linalg.norm(data - self._last_control_data, axis=1)
            moved_index = int(np.argmax(deltas))
        if moved_index <= 0:
            old_a, old_b, old_c = cell.a, cell.b, cell.c
            cell.z, cell.y, cell.x = map(float, data[0, :3])
            cell.a, cell.b, cell.c = old_a, old_b, old_c
        elif moved_index in (1, 2):
            cell.a = max(1.0, float(abs(data[moved_index, 2] - cell.x)))
        elif moved_index in (3, 4):
            cell.b = max(1.0, float(abs(data[moved_index, 1] - cell.y)))
        elif moved_index in (5, 6):
            cell.c = max(1.0, float(abs(data[moved_index, 0] - cell.z)))
        self._clamp_cell_to_volume(cell)
        self._last_control_data = data.copy()
        self._cell_to_controls()
        self._update_selected_list_item()
        self._mark_background_rings_dirty(schedule=True)
        self.update_napari_edit_layers()

    def _napari_cell_selection_changed(self, event=None) -> None:
        if self._syncing_napari or self.napari_viewer is None:
            return
        layer_name = "finalized cell centers"
        if layer_name not in self.napari_viewer.layers:
            return
        selected_data = getattr(self.napari_viewer.layers[layer_name], "selected_data", set())
        if not selected_data:
            return
        point_index = min(selected_data)
        if point_index >= len(self._center_layer_cell_indices):
            return
        index = self._center_layer_cell_indices[point_index]
        if 0 <= index < len(self.cells) and index != self.selected_index:
            self._reload_list(select=index, refresh=False)

    def _update_selected_list_item(self) -> None:
        cell = self.selected_cell
        if cell is None or not (0 <= self.selected_index < self.cell_list.count()):
            return
        done = "✓" if cell.finalized else "…"
        pair = f" {cell.pair_role}" if cell.pair_role else ""
        self.cell_list.item(self.selected_index).setText(
            f"{done} {cell.name}{pair}  {CATEGORY_DISPLAY[cell.category]}  "
            f"x={cell.x:.1f} y={cell.y:.1f} z={cell.z:.1f}  "
            f"a/b/c={cell.a:.1f}/{cell.b:.1f}/{cell.c:.1f}"
        )

    def _schedule_napari_sync(self) -> None:
        if self.napari_viewer is not None:
            self._napari_sync_timer.start(220)

    def _install_napari_key_bindings(self) -> None:
        if self.napari_viewer is None or self._napari_keys_bound:
            return
        # The application-wide event filter already handles W/A/S/D, Q/E, +,
        # -, and Enter even when the embedded napari canvas has focus. Binding
        # the same keys inside napari gives two independent paths for one key
        # press, so we intentionally leave napari's own key map untouched.
        self._napari_keys_bound = True

    def _install_napari_auto_mouse_gate(self) -> None:
        if self.napari_viewer is None or self._napari_auto_mouse_bound:
            return
        try:
            self.napari_viewer.mouse_drag_callbacks.insert(0, self._napari_auto_mouse_gate)
            self._napari_auto_mouse_bound = True
        except Exception as exc:
            self._log_napari_error("_install_napari_auto_mouse_gate", exc)

    def _napari_auto_mouse_gate(self, viewer, event) -> None:
        if self.napari_viewer is None:
            return
        if getattr(event, "type", None) != "mouse_press":
            return
        button = getattr(event, "button", None)
        if button not in (1, None):
            return
        hit_index = self._control_point_hit_index(event)
        if hit_index is not None:
            self.set_interaction_mode("edit")
            self._select_control_point(hit_index)
            return
        center_index = self._center_point_hit_index(event)
        if center_index is not None:
            self._select_cell_from_center_point(center_index)
            return
        else:
            self.set_interaction_mode("view")

    def _center_point_hit_index(self, event) -> int | None:
        if self.napari_viewer is None:
            return None
        layer_name = "finalized cell centers"
        if layer_name not in self.napari_viewer.layers:
            return None
        layer = self.napari_viewer.layers[layer_name]
        if len(getattr(layer, "data", [])) == 0:
            return None
        try:
            value = layer._get_value_(
                position=event.position,
                view_direction=event.view_direction,
                dims_displayed=event.dims_displayed,
                world=True,
            )
        except Exception as exc:
            self._log_napari_error("_center_point_hit_index", exc)
            return None
        if value is None:
            return None
        return int(value)

    def _select_cell_from_center_point(self, point_index: int) -> None:
        if point_index >= len(self._center_layer_cell_indices):
            return
        cell_index = self._center_layer_cell_indices[point_index]
        if not (0 <= cell_index < len(self.cells)):
            return
        self._reload_list(select=cell_index, refresh=False)
        self.set_interaction_mode("edit")
        self._select_control_point(0)

    def _control_point_hit_index(self, event) -> int | None:
        if self.napari_viewer is None:
            return None
        layer_name = "selected 3D center and XYZ handles"
        if layer_name not in self.napari_viewer.layers:
            return None
        layer = self.napari_viewer.layers[layer_name]
        if len(getattr(layer, "data", [])) == 0:
            return None
        try:
            value = layer._get_value_(
                position=event.position,
                view_direction=event.view_direction,
                dims_displayed=event.dims_displayed,
                world=True,
            )
        except Exception as exc:
            self._log_napari_error("_control_point_hit_index", exc)
            return None
        if value is None:
            return None
        return int(value)

    def _select_control_point(self, point_index: int) -> None:
        if self.napari_viewer is None:
            return
        layer_name = "selected 3D center and XYZ handles"
        if layer_name not in self.napari_viewer.layers:
            return
        try:
            self.napari_viewer.layers[layer_name].selected_data = {point_index}
        except Exception as exc:
            self._log_napari_error("_select_control_point", exc)

    def _rgba_for_cell(self, cell: ManualCell, *, alpha: float) -> tuple[float, float, float, float]:
        b, g, r = CATEGORY_COLORS_BGR[cell.category]
        return (r / 255.0, g / 255.0, b / 255.0, alpha)

    def save_csv(self) -> None:
        if not self.cells:
            QtWidgets.QMessageBox.information(self, "No cells", "No manual cells to save.")
            return
        output_dir = Path(self.output_dir.text()).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        frame = self.frame_spin.value()
        z_scale = self.z_display_scale()
        initial_output = output_dir / f"Initial_{frame:03d}.csv"
        shape_output = output_dir / f"Initial_{frame:03d}_manual_shapes.csv"
        pair_output = output_dir / f"Initial_{frame:03d}_daughter_pairs.csv"
        initial_fieldnames = [
            "file",
            "name",
            "cellType",
            "z",
            "y",
            "x",
            "aRadius",
            "bRadius",
            "cRadius",
            "pairId",
            "pairRole",
            "manualCategory",
            "isTrash",
        ]
        with initial_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=initial_fieldnames)
            writer.writeheader()
            for cell in self.cells:
                writer.writerow(
                    {
                        "file": f"t{frame:03d}.tif",
                        "name": cell.name,
                        "cellType": cell.cell_type,
                        "z": f"{cell.z:.6f}",
                        "y": f"{cell.y:.6f}",
                        "x": f"{cell.x:.6f}",
                        "aRadius": f"{cell.a:.6f}",
                        "bRadius": f"{cell.b:.6f}",
                        "cRadius": f"{cell.c * z_scale:.6f}",
                        "pairId": cell.pair_id,
                        "pairRole": cell.pair_role,
                        "manualCategory": CATEGORY_LABELS[cell.category],
                        "isTrash": "0",
                    }
                )

        pair_rows: list[dict[str, str]] = []
        for pair_id in sorted({cell.pair_id for cell in self.cells if cell.pair_id}):
            pair_cells = [cell for cell in self.cells if cell.pair_id == pair_id]
            left = next((cell for cell in pair_cells if cell.pair_role == "left"), None)
            right = next((cell for cell in pair_cells if cell.pair_role == "right"), None)
            if left is None or right is None:
                continue
            distance = math.sqrt((left.x - right.x) ** 2 + (left.y - right.y) ** 2 + (left.z - right.z) ** 2)
            pair_rows.append(
                {
                    "pairId": pair_id,
                    "leftName": left.name,
                    "rightName": right.name,
                    "leftCellType": left.cell_type,
                    "rightCellType": right.cell_type,
                    "leftX": f"{left.x:.6f}",
                    "leftY": f"{left.y:.6f}",
                    "leftZ": f"{left.z:.6f}",
                    "rightX": f"{right.x:.6f}",
                    "rightY": f"{right.y:.6f}",
                    "rightZ": f"{right.z:.6f}",
                    "centerDistance": f"{distance:.6f}",
                }
            )
        if pair_rows:
            with pair_output.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0].keys()))
                writer.writeheader()
                writer.writerows(pair_rows)
        else:
            try:
                if pair_output.exists():
                    pair_output.unlink()
            except Exception as exc:
                self._log_napari_error("remove_stale_pair_csv", exc)

        fieldnames = [
            "cellType",
            "z",
            "y",
            "x",
            "name",
            "file",
            "manualCategory",
            "pairId",
            "pairRole",
            "aRadius",
            "bRadius",
            "cRadius",
            "theta_x",
            "theta_y",
            "theta_z",
            "voxel_size_x",
            "voxel_size_y",
            "voxel_size_z",
            "finalized",
            "notes",
        ]
        with shape_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for cell in self.cells:
                writer.writerow(
                    {
                        "cellType": cell.cell_type,
                        "z": f"{cell.z:.6f}",
                        "y": f"{cell.y:.6f}",
                        "x": f"{cell.x:.6f}",
                        "name": cell.name,
                        "file": f"t{frame:03d}.tif",
                        "manualCategory": CATEGORY_LABELS[cell.category],
                        "pairId": cell.pair_id,
                        "pairRole": cell.pair_role,
                        "aRadius": f"{cell.a:.6f}",
                        "bRadius": f"{cell.b:.6f}",
                        "cRadius": f"{cell.c:.6f}",
                        "theta_x": f"{cell.theta_x:.6f}",
                        "theta_y": f"{cell.theta_y:.6f}",
                        "theta_z": f"{cell.theta_z:.6f}",
                        "voxel_size_x": f"{self.voxel_x.value():.6f}",
                        "voxel_size_y": f"{self.voxel_y.value():.6f}",
                        "voxel_size_z": f"{self.voxel_z.value():.6f}",
                        "finalized": "1" if cell.finalized else "0",
                        "notes": cell.notes,
                    }
                )
        message = (
            f"Saved CellUniverse initial CSV with center and C++ ready ellipsoid radii:\n{initial_output}\n\n"
            f"Saved manual editing metadata with raw napari display radii for review and reload:\n{shape_output}"
        )
        if pair_rows:
            message += f"\n\nSaved daughter pair matching table:\n{pair_output}"
        message += "\n\nThe current 3D workspace will be cleared after you close this message."
        QtWidgets.QMessageBox.information(self, "Saved", message)
        self._reset_workspace_after_successful_save()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        cell = self.selected_cell
        if cell is None:
            super().keyPressEvent(event)
            return
        step = 5.0 if event.modifiers() & QtCore.Qt.ShiftModifier else 1.0
        key = event.key()
        if key == QtCore.Qt.Key_A:
            self.nudge_selected(dx=-step)
            return
        elif key == QtCore.Qt.Key_D:
            self.nudge_selected(dx=step)
            return
        elif key == QtCore.Qt.Key_W:
            self.nudge_selected(dy=-step)
            return
        elif key == QtCore.Qt.Key_S:
            self.nudge_selected(dy=step)
            return
        elif key == QtCore.Qt.Key_Q:
            self.nudge_selected(dz=-step)
            return
        elif key == QtCore.Qt.Key_E:
            self.nudge_selected(dz=step)
            return
        elif key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            self.finish_selected()
            return
        elif key in (QtCore.Qt.Key_Plus, QtCore.Qt.Key_Equal):
            self.scale_selected(step)
            return
        elif key in (QtCore.Qt.Key_Minus, QtCore.Qt.Key_Underscore):
            self.scale_selected(-step)
            return
        else:
            super().keyPressEvent(event)
            return
