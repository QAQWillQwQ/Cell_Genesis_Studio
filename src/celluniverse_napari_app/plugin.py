from __future__ import annotations

from pathlib import Path

from qtpy.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .cli import DEFAULT_FIXED_GT
from .overlays import bake_overlay_tif
from .viewer import open_review_viewer


class ReviewWidget(QWidget):
    def __init__(self, viewer=None):
        super().__init__()
        self.viewer = viewer
        self.setWindowTitle("CellUniverse Review")

        self.frame = QSpinBox()
        self.frame.setRange(0, 9999)
        self.frame.setValue(170)
        self.cells = QLineEdit()
        self.gt = QLineEdit(str(DEFAULT_FIXED_GT))
        self.run_dir = QLineEdit()
        self.base_tif = QLineEdit()
        self.output_tif = QLineEdit()

        form = QFormLayout()
        form.addRow("Frame", self.frame)
        form.addRow("cells.csv", self._path_row(self.cells, file_mode=True))
        form.addRow("Fixed GT", self._path_row(self.gt, file_mode=True))
        form.addRow("Run folder", self._path_row(self.run_dir, file_mode=False))
        form.addRow("Base tif", self._path_row(self.base_tif, file_mode=True))
        form.addRow("Overlay output", self._path_row(self.output_tif, save_mode=True))

        open_button = QPushButton("Open review")
        open_button.clicked.connect(self.open_review)
        overlay_button = QPushButton("Export overlay tif")
        overlay_button.clicked.connect(self.export_overlay)

        buttons = QHBoxLayout()
        buttons.addWidget(open_button)
        buttons.addWidget(overlay_button)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addLayout(buttons)
        self.setLayout(layout)

    def _path_row(self, edit: QLineEdit, file_mode: bool = True, save_mode: bool = False) -> QWidget:
        box = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        button = QPushButton("Browse")

        def choose() -> None:
            if save_mode:
                path, _ = QFileDialog.getSaveFileName(self, "Choose output tif", edit.text(), "TIFF (*.tif *.tiff)")
            elif file_mode:
                path, _ = QFileDialog.getOpenFileName(self, "Choose file", edit.text())
            else:
                path = QFileDialog.getExistingDirectory(self, "Choose folder", edit.text())
            if path:
                edit.setText(path)

        button.clicked.connect(choose)
        layout.addWidget(edit)
        layout.addWidget(button)
        box.setLayout(layout)
        return box

    def open_review(self) -> None:
        run_dir = Path(self.run_dir.text()) if self.run_dir.text().strip() else None
        open_review_viewer(
            frame=int(self.frame.value()),
            cells_csv=Path(self.cells.text()),
            gt_csv=Path(self.gt.text()) if self.gt.text().strip() else None,
            run_dir=run_dir,
        )

    def export_overlay(self) -> None:
        bake_overlay_tif(
            frame=int(self.frame.value()),
            base_tif=Path(self.base_tif.text()),
            cells_csv=Path(self.cells.text()),
            output_tif=Path(self.output_tif.text()),
            gt_csv=Path(self.gt.text()) if self.gt.text().strip() else None,
        )
