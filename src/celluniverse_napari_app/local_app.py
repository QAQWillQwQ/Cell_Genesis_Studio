from __future__ import annotations

import sys
from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from .initial_state_tool import InitialStateBuilder


PROJECT_ROOT = Path("/Users/wangyiding/CellUniverse")
SHOWCASE_SCRIPT = PROJECT_ROOT / "C++/scripts/build_embryo_showcase_mp4.py"
DEFAULT_TIF_DIR = Path("/Volumes/T9/🦠Cell Universe/🟣Output/Visualization/ALL_1~171_VISUAL_TIF")
DEFAULT_LINEAGE_CSV = Path(
    "/Volumes/T9/🦠Cell Universe/🟣Output/✅C.elegans_developing embryo_CorrectOutput 11.76G/"
    "Yiding_Embryo_1~171_FinalLineageTree.csv"
)
DEFAULT_GT_CSV = PROJECT_ROOT / "C++/config/embryo/ground_truth/embryo_FixedGroundTruth.csv"
DEFAULT_VIDEO_OUTPUT = Path(
    "/Volumes/T9/🦠Cell Universe/🟣Output/Visualization/demo_videos/embryo_showcase_app_export.mp4"
)


class VideoExportPanel(QtWidgets.QWidget):
    """Small GUI wrapper around the existing OpenCV showcase script."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.process: QtCore.QProcess | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        main = QtWidgets.QVBoxLayout(self)

        form_box = QtWidgets.QGroupBox("Showcase MP4 export")
        form = QtWidgets.QFormLayout(form_box)
        self.tif_dir = self._path_line(DEFAULT_TIF_DIR)
        self.lineage_csv = self._path_line(DEFAULT_LINEAGE_CSV)
        self.gt_csv = self._path_line(DEFAULT_GT_CSV)
        self.output_mp4 = self._path_line(DEFAULT_VIDEO_OUTPUT)
        form.addRow("tif folder", self._path_picker(self.tif_dir, folder=True))
        form.addRow("lineage csv", self._path_picker(self.lineage_csv))
        form.addRow("fixed GT csv", self._path_picker(self.gt_csv))
        form.addRow("output mp4", self._path_picker(self.output_mp4, save=True))

        frame_row = QtWidgets.QHBoxLayout()
        self.first_frame = self._spin(0, 9999, 1)
        self.last_frame = self._spin(0, 9999, 171)
        self.display_last_frame = self._spin(0, 9999, 171)
        frame_row.addWidget(QtWidgets.QLabel("first"))
        frame_row.addWidget(self.first_frame)
        frame_row.addWidget(QtWidgets.QLabel("last"))
        frame_row.addWidget(self.last_frame)
        frame_row.addWidget(QtWidgets.QLabel("display last"))
        frame_row.addWidget(self.display_last_frame)
        form.addRow("frames", frame_row)

        render_row = QtWidgets.QHBoxLayout()
        self.fps = self._double_spin(0.1, 120.0, 4.0, 1.0)
        self.width = self._spin(320, 8192, 3840)
        self.height = self._spin(240, 8192, 2160)
        self.png_only = QtWidgets.QCheckBox("PNG only")
        self.keep_frames = QtWidgets.QCheckBox("keep PNG frames")
        self.keep_frames.setChecked(True)
        render_row.addWidget(QtWidgets.QLabel("fps"))
        render_row.addWidget(self.fps)
        render_row.addWidget(QtWidgets.QLabel("width"))
        render_row.addWidget(self.width)
        render_row.addWidget(QtWidgets.QLabel("height"))
        render_row.addWidget(self.height)
        render_row.addWidget(self.png_only)
        render_row.addWidget(self.keep_frames)
        form.addRow("render", render_row)

        tuning_row = QtWidgets.QHBoxLayout()
        self.review_scale = self._double_spin(0.5, 6.0, 2.0, 0.25)
        self.ring_opacity = self._double_spin(0.05, 1.0, 0.82, 0.05)
        self.disable_rotation = QtWidgets.QCheckBox("disable 3D rotation")
        tuning_row.addWidget(QtWidgets.QLabel("review scale"))
        tuning_row.addWidget(self.review_scale)
        tuning_row.addWidget(QtWidgets.QLabel("ring opacity"))
        tuning_row.addWidget(self.ring_opacity)
        tuning_row.addWidget(self.disable_rotation)
        form.addRow("visual style", tuning_row)

        main.addWidget(form_box)

        buttons = QtWidgets.QHBoxLayout()
        self.run_btn = QtWidgets.QPushButton("Run export")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.run_btn.clicked.connect(self.run_export)
        self.stop_btn.clicked.connect(self.stop_export)
        buttons.addWidget(self.run_btn)
        buttons.addWidget(self.stop_btn)
        buttons.addStretch(1)
        main.addLayout(buttons)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(4000)
        main.addWidget(self.log, 1)

    def _path_line(self, path: Path) -> QtWidgets.QLineEdit:
        line = QtWidgets.QLineEdit(str(path))
        line.setMinimumWidth(520)
        return line

    def _path_picker(
        self,
        line: QtWidgets.QLineEdit,
        *,
        folder: bool = False,
        save: bool = False,
    ) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        button = QtWidgets.QPushButton("Browse")

        def choose() -> None:
            current = line.text()
            if folder:
                path = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose folder", current)
            elif save:
                path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Choose output", current, "MP4 files (*.mp4)")
            else:
                path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose file", current, "CSV files (*.csv);;All files (*)")
            if path:
                line.setText(path)

        button.clicked.connect(choose)
        layout.addWidget(line, 1)
        layout.addWidget(button)
        return box

    def _spin(self, low: int, high: int, value: int) -> QtWidgets.QSpinBox:
        spin = QtWidgets.QSpinBox()
        spin.setRange(low, high)
        spin.setValue(value)
        return spin

    def _double_spin(self, low: float, high: float, value: float, step: float) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setDecimals(3)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    def build_command(self) -> list[str]:
        command = [
            sys.executable,
            "-u",
            str(SHOWCASE_SCRIPT),
            "--tif-dir",
            self.tif_dir.text(),
            "--lineage-csv",
            self.lineage_csv.text(),
            "--gt",
            self.gt_csv.text(),
            "--output",
            self.output_mp4.text(),
            "--first-frame",
            str(self.first_frame.value()),
            "--last-frame",
            str(self.last_frame.value()),
            "--display-last-frame",
            str(self.display_last_frame.value()),
            "--fps",
            str(self.fps.value()),
            "--width",
            str(self.width.value()),
            "--height",
            str(self.height.value()),
            "--review-render-scale",
            str(self.review_scale.value()),
            "--review-ring-opacity",
            str(self.ring_opacity.value()),
        ]
        if self.png_only.isChecked():
            command.append("--png-only")
        if self.keep_frames.isChecked():
            command.append("--keep-frames")
        if self.disable_rotation.isChecked():
            command.append("--disable-3d-rotation")
        return command

    def run_export(self) -> None:
        if self.process is not None:
            return
        if not SHOWCASE_SCRIPT.is_file():
            QtWidgets.QMessageBox.critical(self, "Missing script", f"Cannot find:\n{SHOWCASE_SCRIPT}")
            return
        command = self.build_command()
        self.log.appendPlainText("$ " + " ".join(command))
        self.process = QtCore.QProcess(self)
        self.process.setProgram(command[0])
        self.process.setArguments(command[1:])
        self.process.setWorkingDirectory(str(PROJECT_ROOT))
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_process_output)
        self.process.finished.connect(self._process_finished)
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.process.start()

    def stop_export(self) -> None:
        if self.process is None:
            return
        self.log.appendPlainText("[APP] stopping export process")
        self.process.terminate()
        QtCore.QTimer.singleShot(3000, self._force_kill_if_running)

    def _force_kill_if_running(self) -> None:
        if self.process is not None and self.process.state() != QtCore.QProcess.NotRunning:
            self.process.kill()

    def _read_process_output(self) -> None:
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if text:
            self.log.appendPlainText(text.rstrip())
            self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def _process_finished(self, exit_code: int, status: QtCore.QProcess.ExitStatus) -> None:
        if self.process is not None:
            self._read_process_output()
        self.log.appendPlainText(f"[APP] export finished exit={exit_code} status={status}")
        self.process = None
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)


class CellUniverseLocalApp(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Cell Genesis Studio")
        self.resize(1680, 1040)
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(InitialStateBuilder(), "Manual Initial Builder")
        tabs.addTab(VideoExportPanel(), "Showcase Video Export")
        self.setCentralWidget(tabs)


def run_app() -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Cell Genesis Studio")
    window = CellUniverseLocalApp()
    window.show()
    return int(app.exec_())


def main() -> int:
    return run_app()


if __name__ == "__main__":
    raise SystemExit(main())
