from __future__ import annotations

from dataclasses import dataclass

from PyQt5 import QtGui, QtWidgets


@dataclass(frozen=True)
class Theme:
    name: str
    window: str
    panel: str
    panel_alt: str
    input_bg: str
    input_alt: str
    text: str
    muted: str
    faint: str
    border: str
    border_strong: str
    accent: str
    accent_hover: str
    accent_soft: str
    button: str
    button_hover: str
    danger: str
    log_bg: str
    canvas_bg: str


DARK_THEME = Theme(
    name="dark",
    window="#0b0b0c",
    panel="#171718",
    panel_alt="#202123",
    input_bg="#09090a",
    input_alt="#111113",
    text="#f2f2f3",
    muted="#b8b9bd",
    faint="#777a80",
    border="#2d2e31",
    border_strong="#494b50",
    accent="#73767d",
    accent_hover="#8c8f96",
    accent_soft="rgba(255, 255, 255, 0.12)",
    button="#242529",
    button_hover="#313238",
    danger="#ff6b6b",
    log_bg="#050506",
    canvas_bg="#030304",
)


LIGHT_THEME = Theme(
    name="light",
    window="#eef2f8",
    panel="#ffffff",
    panel_alt="#f6f8fc",
    input_bg="#ffffff",
    input_alt="#f3f6fb",
    text="#182033",
    muted="#5c6678",
    faint="#8993a5",
    border="#d3dbea",
    border_strong="#b8c4d8",
    accent="#1778ff",
    accent_hover="#005ee6",
    accent_soft="rgba(23, 120, 255, 0.13)",
    button="#f7f9fd",
    button_hover="#edf3ff",
    danger="#c93535",
    log_bg="#f7f9fd",
    canvas_bg="#05070d",
)


def _palette(theme: Theme) -> QtGui.QPalette:
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(theme.window))
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(theme.text))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(theme.input_bg))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(theme.input_alt))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(theme.panel))
    palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(theme.text))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor(theme.text))
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(theme.button))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(theme.text))
    palette.setColor(QtGui.QPalette.BrightText, QtGui.QColor(theme.danger))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(theme.accent))
    palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#ffffff"))
    return palette


def stylesheet(theme: Theme) -> str:
    return f"""
    QMainWindow, QDialog {{
        background: {theme.window};
        color: {theme.text};
    }}

    QWidget {{
        color: {theme.text};
        font-family: "Helvetica Neue", Arial, sans-serif;
        font-size: 13px;
        selection-background-color: {theme.accent};
        selection-color: #ffffff;
    }}

    QScrollArea#ControlsScroll {{
        background: {theme.window};
        border: 0;
    }}

    QScrollArea#ControlsScroll > QWidget > QWidget {{
        background: {theme.window};
    }}

    QWidget#ControlPanel {{
        background: {theme.window};
    }}

    QWidget#FullWidthControl {{
        background: transparent;
    }}

    QToolBar#AppearanceBar {{
        background: {theme.panel};
        border: 0;
        border-bottom: 1px solid {theme.border};
        spacing: 10px;
        padding: 8px 12px;
    }}

    QToolBar#AppearanceBar QComboBox {{
        min-width: 120px;
    }}

    QMenuBar {{
        background: {theme.window};
        color: {theme.text};
        border-bottom: 1px solid {theme.border};
    }}

    QMenuBar::item:selected {{
        background: {theme.accent_soft};
        border-radius: 5px;
    }}

    QMenu {{
        background: {theme.panel};
        color: {theme.text};
        border: 1px solid {theme.border};
        border-radius: 8px;
        padding: 6px;
    }}

    QMenu::item {{
        padding: 6px 20px;
        border-radius: 6px;
    }}

    QMenu::item:selected {{
        background: {theme.accent_soft};
        color: {theme.text};
    }}

    QLabel#AppTitle {{
        color: {theme.text};
        font-size: 15px;
        font-weight: 700;
        letter-spacing: 0px;
    }}

    QLabel#MutedLabel, QLabel#WorkflowLabel {{
        color: {theme.muted};
        font-weight: 600;
    }}

    QTabWidget::pane {{
        background: {theme.window};
        border: 0;
        padding: 8px;
    }}

    QTabBar::tab {{
        background: {theme.panel};
        color: {theme.muted};
        border: 1px solid {theme.border};
        border-bottom-color: {theme.border};
        padding: 7px 18px;
        min-width: 140px;
    }}

    QTabBar::tab:first {{
        border-top-left-radius: 9px;
        border-bottom-left-radius: 9px;
    }}

    QTabBar::tab:last {{
        border-top-right-radius: 9px;
        border-bottom-right-radius: 9px;
    }}

    QTabBar::tab:selected {{
        background: {theme.accent};
        color: #ffffff;
        border-color: {theme.accent};
        font-weight: 700;
    }}

    QGroupBox {{
        background: {theme.panel};
        border: 1px solid {theme.border};
        border-radius: 12px;
        margin-top: 18px;
        padding: 12px;
        font-weight: 700;
    }}

    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 14px;
        padding: 0 7px;
        color: {theme.muted};
        background: {theme.window};
    }}

    QPushButton {{
        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                    stop: 0 {theme.button_hover}, stop: 1 {theme.button});
        color: {theme.text};
        border: 1px solid {theme.border_strong};
        border-radius: 8px;
        padding: 6px 12px;
        min-height: 24px;
        font-weight: 600;
    }}

    QPushButton:hover {{
        background: {theme.button_hover};
        border-color: {theme.accent};
    }}

    QPushButton:pressed {{
        background: {theme.accent_soft};
        border-color: {theme.accent};
    }}

    QPushButton:checked {{
        background: {theme.accent};
        color: #ffffff;
        border-color: {theme.accent};
    }}

    QPushButton:disabled {{
        color: {theme.faint};
        background: {theme.input_alt};
        border-color: {theme.border};
    }}

    QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background: {theme.input_bg};
        color: {theme.text};
        border: 1px solid {theme.border};
        border-radius: 8px;
        padding: 5px 8px;
        min-height: 24px;
    }}

    QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus,
    QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
        border-color: {theme.accent};
        background: {theme.input_alt};
    }}

    QPlainTextEdit {{
        background: {theme.log_bg};
        font-family: "SF Mono", Menlo, Consolas, monospace;
        font-size: 12px;
        line-height: 1.35em;
    }}

    QListWidget {{
        background: {theme.input_bg};
        border: 1px solid {theme.border};
        border-radius: 8px;
        padding: 4px;
        outline: 0;
    }}

    QListWidget::item {{
        min-height: 22px;
        padding: 3px 6px;
        border-radius: 6px;
    }}

    QListWidget::item:selected {{
        background: {theme.accent_soft};
        color: {theme.text};
        border: 1px solid {theme.accent};
    }}

    QComboBox::drop-down {{
        border: 0;
        width: 24px;
    }}

    QComboBox QAbstractItemView {{
        background: {theme.panel};
        color: {theme.text};
        border: 1px solid {theme.border};
        border-radius: 8px;
        padding: 4px;
        selection-background-color: {theme.accent_soft};
        selection-color: {theme.text};
    }}

    QCheckBox {{
        color: {theme.text};
        spacing: 8px;
    }}

    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border-radius: 5px;
        border: 1px solid {theme.border_strong};
        background: {theme.input_bg};
    }}

    QCheckBox::indicator:hover {{
        border-color: {theme.accent};
    }}

    QCheckBox::indicator:checked {{
        background: {theme.accent};
        border-color: {theme.accent};
    }}

    QSlider::groove:horizontal {{
        height: 5px;
        background: {theme.border};
        border-radius: 3px;
    }}

    QSlider::sub-page:horizontal {{
        background: {theme.accent};
        border-radius: 3px;
    }}

    QSlider::handle:horizontal {{
        background: #ffffff;
        border: 2px solid {theme.accent};
        width: 16px;
        height: 16px;
        margin: -6px 0;
        border-radius: 8px;
    }}

    QProgressBar {{
        background: {theme.input_bg};
        color: {theme.text};
        border: 1px solid {theme.border};
        border-radius: 8px;
        text-align: center;
        min-height: 18px;
    }}

    QProgressBar::chunk {{
        background: {theme.accent};
        border-radius: 7px;
    }}

    QScrollBar:vertical, QScrollBar:horizontal {{
        background: transparent;
        border: 0;
        margin: 2px;
    }}

    QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
        background: {theme.border_strong};
        border-radius: 5px;
        min-height: 28px;
        min-width: 28px;
    }}

    QFrame#ViewerHost {{
        background: {theme.canvas_bg};
        border: 1px solid {theme.border_strong};
        border-radius: 10px;
    }}

    QLabel#WorkspaceStatus {{
        background: {theme.log_bg};
        border: 1px solid {theme.border_strong};
        border-radius: 10px;
        color: {theme.muted};
        font-size: 18px;
        font-weight: 700;
        padding: 36px;
    }}

    QMessageBox {{
        background: {theme.panel};
        color: {theme.text};
    }}

    QToolTip {{
        background: {theme.panel};
        color: {theme.text};
        border: 1px solid {theme.border};
        border-radius: 6px;
        padding: 6px;
    }}
    """


def apply_theme(app: QtWidgets.QApplication, theme_name: str) -> None:
    theme = DARK_THEME if theme_name.lower() == "dark" else LIGHT_THEME
    app.setStyle("Fusion")
    app.setPalette(_palette(theme))
    app.setStyleSheet(stylesheet(theme))
