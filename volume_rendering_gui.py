import os
import sys
import json
import numpy as np
import nrrd
from PIL import Image

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QFont, QMouseEvent, QWheelEvent, QPainter, QPen, QBrush, QColor, QPainterPath, QImage
from PyQt6.QtSvgWidgets import QSvgWidget
from PyQt6.QtWidgets import (
    QApplication,
    QColorDialog,
    QDialog,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QGridLayout,
    QLineEdit,
    QMenu,
)

import ProjectiveGeometry23.utils as pgu
from ProjectiveGeometry23.central_projection import ProjectionMatrix
from ProjectiveGeometry23.source_detector_geometry import SourceDetectorGeometry
from ProjectiveGeometry23.homography import rotation_x, rotation_z, scale
try:
    from ct_recon_fdk_astra.fileformats.ompl import load_ompl
except ImportError:
    from fileformats.ompl import load_ompl
from xray_epipolar_consistency import VolumeRenderer

# -----------------------------------------------------------------------------
# SVG Snip & Composer setup
# -----------------------------------------------------------------------------
import ProjectiveGeometry23.svg_utils
from svg_snip.Composer import Group, Composer
import svg_snip.Elements3D as e3d
import svg_snip.Elements as e2d

import svg_snip.Elements as _elements
import svg_snip.Elements3D as _elements3d

def fix_8digit_hex_svg(svg_code):
    import re
    pattern = r'(fill|stroke)="#([0-9a-fA-F]{6})([0-9a-fA-F]{2})"'
    def repl(match):
        attr = match.group(1)
        color = match.group(2)
        alpha_hex = match.group(3)
        alpha_val = int(alpha_hex, 16) / 255.0
        return f'{attr}="#{color}" {attr}-opacity="{alpha_val:.3f}"'
    return re.sub(pattern, repl, svg_code)

def get_model_matrix_from_nrrd_header(header):
    # Default values: Identity 3x3 directions and zero origin
    directions = np.eye(3)
    origin = np.zeros(3)
    
    # 1. Parse space directions / spacings
    space_dirs = header.get('space directions')
    if space_dirs is not None:
        try:
            valid_dirs = []
            for i in range(3):
                d = space_dirs[i]
                if d is not None and not np.isnan(d).any():
                    valid_dirs.append(np.array(d, dtype=float))
                else:
                    # Fallback to spacings for this axis
                    fallback_d = np.zeros(3)
                    fallback_d[i] = 1.0
                    spacings = header.get('spacings')
                    if spacings is not None and len(spacings) > i:
                        sp = spacings[i]
                        if sp is not None and not np.isnan(sp):
                            fallback_d[i] = float(sp)
                    valid_dirs.append(fallback_d)
            directions = np.column_stack(valid_dirs)
        except Exception as e:
            print(f"Warning: Failed to parse space directions: {e}")
    else:
        # Fallback to spacings
        spacings = header.get('spacings')
        if spacings is not None:
            for i in range(min(len(spacings), 3)):
                sp = spacings[i]
                if sp is not None and not np.isnan(sp):
                    directions[i, i] = float(sp)
                    
    # 2. Parse space origin
    space_origin = header.get('space origin')
    if space_origin is not None:
        try:
            if not np.isnan(space_origin).any():
                origin = np.array(space_origin, dtype=float)
        except Exception as e:
            print(f"Warning: Failed to parse space origin: {e}")
            
    M = np.eye(4)
    M[:3, :3] = directions
    M[:3, 3] = origin
    return M

def safe_point(P, X, **kwargs):
    x = P @ pgu.cvec(X)
    if abs(x[2][0]) <= 1e-5:
        return ""
    return e3d.point(P, X, **kwargs)

def safe_line(P, X1, X2, **kwargs):
    x1 = P @ pgu.cvec(X1)
    x2 = P @ pgu.cvec(X2)
    if abs(x1[2][0]) <= 1e-5 or abs(x2[2][0]) <= 1e-5:
        return ""
    return e3d.line(P, X1, X2, **kwargs)

def safe_arrow(P, X1, X2, **kwargs):
    x1 = P @ pgu.cvec(X1)
    x2 = P @ pgu.cvec(X2)
    if abs(x1[2][0]) <= 1e-5 or abs(x2[2][0]) <= 1e-5:
        return ""
    return e3d.arrow(P, X1, X2, **kwargs)

def safe_text(P, X, **kwargs):
    x = P @ pgu.cvec(X)
    if abs(x[2][0]) <= 1e-5:
        return ""
    return e3d.text(P, X, **kwargs)

Composer.declared_shapes[safe_arrow] = Composer.declared_shapes[_elements.arrow]

def patched_svg_source_detector(P, projection: ProjectionMatrix, draw_on_detector=None, composer=None, **kwargs):
    sdg = SourceDetectorGeometry(projection)
    C = pgu.cvec(sdg.source_position)
    O = sdg.detector_origin
    U = pgu.cvec(sdg.axis_direction_Upx) * projection.image_size[0]
    V = pgu.cvec(sdg.axis_direction_Vpx) * projection.image_size[1]

    # Check if dark theme is active
    is_dark = False
    from PyQt6.QtWidgets import QApplication
    app_inst = QApplication.instance()
    if app_inst:
        for widget in app_inst.topLevelWidgets():
            if hasattr(widget, 'raycast_pass_type'):
                is_dark = widget.raycast_pass_type in ("EmissionAbsorption", "MIP")
                break

    stroke_color = "gray"
    fill_color = "white" if is_dark else "black"
    det_fill = "#ffffff20" if is_dark else "#00000020"

    group = Group("Source Detector Geometry")

    if draw_on_detector is not None:
        T_detector = sdg.central_projection_3d
        group.add(draw_on_detector, P=P@T_detector, **kwargs)

    x_c = P @ C
    show_source_and_frustum = abs(x_c[2][0]) > 1e-5

    if show_source_and_frustum:
        group.add(e3d.point, P=P, X=C, r=1, fill=fill_color)

    group.add(e3d.polygon, P=P, Xs=[O, O+U, O+V+U, O+V], fill=det_fill, stroke=stroke_color, stroke_back=stroke_color)
    if kwargs.get('show_axis_labels', True):
        group.add(e3d.arrow, P=P, X1=O, X2=O + U, stroke="magenta")
        group.add(e3d.arrow, P=P, X1=O, X2=O + V, stroke="cyan")
        group.add(e3d.text, P=P, X=O + U * 1.05, content="U", fill="magenta", font_size="12px", font_family="sans-serif")
        group.add(e3d.text, P=P, X=O + V * 1.05, content="V", fill="cyan", font_size="12px", font_family="sans-serif")
        det_label = "virtual detector" if kwargs.get('is_virtual', False) else "detector"
        group.add(e3d.text, P=P, X=O, content=det_label, fill=fill_color, font_size="12px", font_family="sans-serif")

    if show_source_and_frustum:
        group.add(e3d.line, P=P, X1=C, X2=O, stroke=stroke_color, stroke_width=1.5)
        group.add(e3d.line, P=P, X1=C, X2=O+V, stroke=stroke_color, stroke_width=1.5)
        group.add(e3d.line, P=P, X1=C, X2=O+U, stroke=stroke_color, stroke_width=1.5)
        group.add(e3d.line, P=P, X1=C, X2=O+V+U, stroke=stroke_color, stroke_width=1.5)

    if kwargs.get('show_axis_labels', True):
        if show_source_and_frustum and 'label_source' in kwargs:
            group.add(e3d.text, P=P, X=C, content=kwargs['label_source'], fill=fill_color)
        if 'label_detector' in kwargs:
            group.add(e3d.text, P=P, X=O, content=kwargs['label_detector'], fill=fill_color)
        
    svg_code, used_funcs = group(composer=composer, **kwargs)
    return fix_8digit_hex_svg(svg_code), used_funcs

def patched_volume(P, shape, model_matrix=np.eye(4), color_axes=True, lighting=True, composer=None, **kwargs):
    group = e3d.volume(P, shape=shape, model_matrix=model_matrix, color_axes=False, lighting=lighting, **kwargs)
    
    # Check if dark theme is active
    is_dark = False
    from PyQt6.QtWidgets import QApplication
    app_inst = QApplication.instance()
    if app_inst:
        for widget in app_inst.topLevelWidgets():
            if hasattr(widget, 'raycast_pass_type'):
                is_dark = widget.raycast_pass_type in ("EmissionAbsorption", "MIP")
                break

    fill_color = "white" if is_dark else "black"

    if color_axes:
        X, Y, Z = shape[2], shape[1], shape[0]
        corners_voxel = [
            np.array([0, 0, 0, 1]),
            np.array([X, 0, 0, 1]),
            np.array([0, Y, 0, 1]),
            np.array([0, 0, Z, 1]),
        ]
        corners = [model_matrix @ c for c in corners_voxel]
        group.add(safe_arrow, P=P, X1=corners[0], X2=corners[1], stroke='red', stroke_width=2)
        group.add(safe_arrow, P=P, X1=corners[0], X2=corners[2], stroke='green', stroke_width=2)
        group.add(safe_arrow, P=P, X1=corners[0], X2=corners[3], stroke='blue', stroke_width=2)
        text_kwargs = kwargs.copy()
        text_kwargs.pop("stroke", None)
        text_kwargs.pop("stroke_width", None)
        text_kwargs.pop("fill", None)
        group.add(e3d.text, P=P, X=corners[0], content="volume", fill=fill_color, stroke="none", stroke_width=0, font_size="12px", font_family="sans-serif", **text_kwargs)

    group_kwargs = {k: v for k, v in kwargs.items() if k not in ["fill", "stroke", "stroke_width"]}
    svg_code, used_funcs = group(composer=composer, **group_kwargs)
    return fix_8digit_hex_svg(svg_code), used_funcs

def patched_trajectory(P, disp_src, active_idx, num_views, composer=None, **kwargs):
    group = Group("Source Trajectory")
    
    part1_idxs = np.arange(0, active_idx)
    part2_idxs = np.arange(active_idx + 1, num_views)
    
    def draw_part_trajectory(g, idxs):
        if len(idxs) < 2:
            return
        if len(idxs) > 90:
            sampled_idxs = idxs[np.round(np.linspace(0, len(idxs) - 1, 90)).astype(int)]
        else:
            sampled_idxs = idxs
        for i in range(len(sampled_idxs) - 1):
            g.add(safe_line, X1=disp_src[sampled_idxs[i]], X2=disp_src[sampled_idxs[i + 1]], stroke="#00adb5", stroke_width=1.2)
            
    draw_part_trajectory(group, part1_idxs)
    draw_part_trajectory(group, part2_idxs)
    
    svg_code, used_funcs = group(composer=composer, P=P)
    return fix_8digit_hex_svg(svg_code), used_funcs
# Apply the patches to the library at runtime
ProjectiveGeometry23.svg_utils.svg_source_detector = patched_svg_source_detector

# -----------------------------------------------------------------------------
# QWidget helper classes
# -----------------------------------------------------------------------------
class ViewportWidget(QSvgWidget):
    """3D SVG Orbit Trajectory Viewport with Mouse rotation."""
    viewChanged = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.app = parent
        self.default_R = rotation_x(-0.7)[:3, :3] @ rotation_z(0.0)[:3, :3]
        self.R_view = self.default_R.copy()
        
        self.yaw = 0.0
        self.pitch = -0.7
        self.s = 0.2
        
        self.last_mouse_pos = None
        self.P_display = None
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setStyleSheet("background: white; border: 1px solid #ccc;")
        
        # Fast rendering during active interaction
        self.is_interacting = False
        self.preview_qimage = None
        self.bg_color = QColor(255, 255, 255)
        
        self.wheel_timer = QTimer(self)
        self.wheel_timer.setSingleShot(True)
        self.wheel_timer.timeout.connect(self.end_wheel_interaction)
        
    def end_wheel_interaction(self):
        self.is_interacting = False
        self.viewChanged.emit()
        
    def paintEvent(self, event):
        use_fast_path = self.is_interacting or (self.app and not self.app.chk_svg_overlay.isChecked())
        if use_fast_path and self.preview_qimage is not None:
            painter = QPainter(self)
            painter.fillRect(self.rect(), self.bg_color)
            pad_x, pad_y, target_w, target_h = getattr(self.app, 'curr_pad', (0, 0, self.width(), self.height()))
            painter.drawImage(int(pad_x), int(pad_y), self.preview_qimage)
            painter.end()
        else:
            super().paintEvent(event)
            
    def resizeEvent(self, event):
        super().resizeEvent(event)
        w, h = self.width(), self.height()
        if w > 0 and h > 0:
            self.P_display = ProjectionMatrix.perspective_look_at(
                eye=np.array([0, 0, 900]),
                center=np.array([0, 0, 0]),
                image_size=(w, h),
                fovy_rad=0.2
            )
        self.viewChanged.emit()

    def mousePressEvent(self, event: QMouseEvent):
        if self.app and self.app.chk_selected_view.isChecked():
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_mouse_pos = event.position()
            self.is_interacting = True

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.app and self.app.chk_selected_view.isChecked():
            return
        if self.last_mouse_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
            pos = event.position()
            diff = pos - self.last_mouse_pos
            self.last_mouse_pos = pos
            
            self.yaw -= diff.x() * 0.01
            self.pitch -= diff.y() * 0.01
            
            self.R_view = rotation_x(self.pitch)[:3, :3] @ rotation_z(self.yaw)[:3, :3]
            self.is_interacting = True
            self.viewChanged.emit()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.app and self.app.chk_selected_view.isChecked():
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_mouse_pos = None
            self.is_interacting = False
            self.viewChanged.emit()

    def wheelEvent(self, event: QWheelEvent):
        if self.app and self.app.chk_selected_view.isChecked():
            return
        delta = event.angleDelta().y()
        if delta > 0:
            self.s *= 1.1
        elif delta < 0:
            self.s /= 1.1
        self.s = max(0.001, min(100.0, self.s))
        self.is_interacting = True
        self.wheel_timer.start(200)
        self.viewChanged.emit()


class TransferFunctionEditor(QWidget):
    """Interactive 2D transfer function widget with background histogram."""
    changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = [
            {"t": 0.0, "alpha": 0.0, "color": (0, 0, 0)},
            {"t": 1.0, "alpha": 1.0, "color": (255, 255, 255)}
        ]
        self.selected_idx = None
        self.histogram = None
        self.setMinimumHeight(140)
        self.setMouseTracking(True)
        self.setStyleSheet("background-color: #1e1e1e; border: 1px solid #3c3c3c;")
        
    def set_histogram(self, hist):
        self.histogram = hist
        self.update()
        
    def get_interpolated(self, t):
        sorted_pts = sorted(self.points, key=lambda p: p["t"])
        if t <= sorted_pts[0]["t"]:
            pt = sorted_pts[0]
            return pt["color"], pt["alpha"]
        if t >= sorted_pts[-1]["t"]:
            pt = sorted_pts[-1]
            return pt["color"], pt["alpha"]
            
        for i in range(len(sorted_pts) - 1):
            p0 = sorted_pts[i]
            p1 = sorted_pts[i+1]
            if p0["t"] <= t <= p1["t"]:
                denom = p1["t"] - p0["t"]
                f = (t - p0["t"]) / denom if denom > 1e-6 else 0.0
                r = p0["color"][0] * (1.0 - f) + p1["color"][0] * f
                g = p0["color"][1] * (1.0 - f) + p1["color"][1] * f
                b = p0["color"][2] * (1.0 - f) + p1["color"][2] * f
                alpha = p0["alpha"] * (1.0 - f) + p1["alpha"] * f
                return (int(r), int(g), int(b)), alpha
        return (255, 255, 255), 0.0

    def generate_tf_array(self, volume_min, volume_max, tf_min, tf_max):
        tf = np.zeros((256, 4), dtype=np.float32)
        v_range = volume_max - volume_min
        if v_range <= 1e-6:
            v_range = 1.0
            
        tf_range = tf_max - tf_min
        if tf_range <= 1e-6:
            tf_range = 1.0
            
        for i in range(256):
            sample = i / 255.0
            v = volume_min + sample * v_range
            t = (v - tf_min) / tf_range
            t = max(0.0, min(1.0, t))
            color, alpha = self.get_interpolated(t)
            tf[i, 0] = color[0] / 255.0
            tf[i, 1] = color[1] / 255.0
            tf[i, 2] = color[2] / 255.0
            tf[i, 3] = alpha
        return tf

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        band_h = 20
        band_y = h - band_h
        plot_h = h - band_h - 5
        
        # 1. Background histogram
        if self.histogram is not None and len(self.histogram) == 256:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(80, 80, 80, 80)))
            bar_w = w / 256.0
            for i in range(256):
                val = self.histogram[i]
                bar_h = val * (plot_h - 10)
                painter.drawRect(int(i * bar_w), int(plot_h - bar_h), int(bar_w + 1), int(bar_h))
                
        # 2. Grid lines
        painter.setPen(QPen(QColor(80, 80, 80, 80), 1, Qt.PenStyle.DashLine))
        for y_frac in [0.25, 0.5, 0.75]:
            painter.drawLine(0, int(plot_h * y_frac), w, int(plot_h * y_frac))
            
        # 2.5. Draw Checkerboard background in color band
        sq_size = 5
        for bx in range(0, w, sq_size):
            for by in range(band_y, h, sq_size):
                if ((bx // sq_size) + (by // sq_size)) % 2 == 0:
                    painter.fillRect(bx, by, sq_size, sq_size, QColor(255, 255, 255))
                else:
                    painter.fillRect(bx, by, sq_size, sq_size, QColor(200, 200, 200))
                    
        # Draw TF Color Band overlay
        for x in range(w):
            t = x / max(1.0, float(w))
            color, alpha = self.get_interpolated(t)
            painter.fillRect(x, band_y, 1, band_h, QColor(int(color[0]), int(color[1]), int(color[2]), int(alpha * 255)))
            
        # 3. Draw linear segments connecting points
        sorted_pts = sorted(self.points, key=lambda p: p["t"])
        path = QPainterPath()
        pt0 = sorted_pts[0]
        path.moveTo(int(pt0["t"] * w), int((1.0 - pt0["alpha"]) * (plot_h - 10) + 5))
        for p in sorted_pts[1:]:
            path.lineTo(int(p["t"] * w), int((1.0 - p["alpha"]) * (plot_h - 10) + 5))
            
        painter.setPen(QPen(QColor(220, 220, 220), 2, Qt.PenStyle.SolidLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)
        
        # 4. Draw points as handles (colored by their RGB values)
        for idx, p in enumerate(self.points):
            x = int(p["t"] * w)
            y = int((1.0 - p["alpha"]) * (plot_h - 10) + 5)
            
            r, g, b = p["color"]
            painter.setBrush(QBrush(QColor(r, g, b)))
            
            if idx == self.selected_idx:
                painter.setPen(QPen(QColor(255, 165, 0), 2))
                r_dot = 6
            else:
                painter.setPen(QPen(QColor(240, 240, 240), 1))
                r_dot = 4
                
            painter.drawEllipse(x - r_dot, y - r_dot, r_dot * 2, r_dot * 2)

    def _find_point_near(self, pos):
        w = self.width()
        h = self.height()
        plot_h = h - 25
        for idx, p in enumerate(self.points):
            px = p["t"] * w
            py = (1.0 - p["alpha"]) * (plot_h - 10) + 5
            dist = np.hypot(pos.x() - px, pos.y() - py)
            if dist < 12.0:
                return idx
        return None

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            idx = self._find_point_near(event.position())
            self.selected_idx = idx
            self.update()
            self.changed.emit()
            self.update_color_dialog_from_selection()
        elif event.button() == Qt.MouseButton.RightButton:
            w = self.width()
            h = self.height()
            plot_h = h - 25
            t = max(0.0, min(1.0, event.position().x() / w))
            alpha = max(0.0, min(1.0, 1.0 - (event.position().y() - 5) / (plot_h - 10)))
            
            color, _ = self.get_interpolated(t)
            new_pt = {"t": t, "alpha": alpha, "color": list(color)}
            self.points.append(new_pt)
            self.selected_idx = len(self.points) - 1
            self.update()
            self.changed.emit()
            self.update_color_dialog_from_selection()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.selected_idx is not None and event.buttons() & Qt.MouseButton.LeftButton:
            w = self.width()
            h = self.height()
            plot_h = h - 25
            t = max(0.0, min(1.0, event.position().x() / w))
            alpha = max(0.0, min(1.0, 1.0 - (event.position().y() - 5) / (plot_h - 10)))
            
            pt = self.points[self.selected_idx]
            
            if pt["t"] == 0.0:
                pt["alpha"] = alpha
            elif pt["t"] == 1.0:
                pt["alpha"] = alpha
            else:
                sorted_pts = sorted(self.points, key=lambda p: p["t"])
                s_idx = sorted_pts.index(pt)
                t_min = sorted_pts[s_idx - 1]["t"] + 0.01
                t_max = sorted_pts[s_idx + 1]["t"] - 0.01
                pt["t"] = max(t_min, min(t_max, t))
                pt["alpha"] = alpha
                
            self.update()
            self.changed.emit()

    def update_color_dialog_from_selection(self):
        if hasattr(self, 'color_dialog') and self.color_dialog is not None and self.color_dialog.isVisible():
            if self.selected_idx is not None:
                pt = self.points[self.selected_idx]
                r, g, b = pt["color"]
                self.color_dialog.blockSignals(True)
                self.color_dialog.setCurrentColor(QColor(r, g, b))
                self.color_dialog.blockSignals(False)

    def pick_color(self):
        if self.selected_idx is not None:
            pt = self.points[self.selected_idx]
            r, g, b = pt["color"]
            
            if hasattr(self, 'color_dialog') and self.color_dialog is not None:
                if self.color_dialog.isVisible():
                    self.color_dialog.setCurrentColor(QColor(r, g, b))
                    self.color_dialog.raise_()
                    self.color_dialog.activateWindow()
                    return
            
            dialog = QColorDialog(QColor(r, g, b), self)
            dialog.setWindowTitle("Select Control Point Color")
            dialog.setOption(QColorDialog.ColorDialogOption.NoButtons, True)
            dialog.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog, True)
            
            def on_color_changed(color):
                if color.isValid() and self.selected_idx is not None:
                    pt = self.points[self.selected_idx]
                    pt["color"] = [color.red(), color.green(), color.blue()]
                    self.update()
                    self.changed.emit()
            
            dialog.currentColorChanged.connect(on_color_changed)
            
            self.color_dialog = dialog
            dialog.show()

    def delete_selected_point(self):
        if self.selected_idx is not None:
            pt = self.points[self.selected_idx]
            if pt["t"] == 0.0 or pt["t"] == 1.0:
                QMessageBox.warning(self, "Cannot Delete", "Endpoints at 0.0 and 1.0 cannot be deleted.")
                return
            self.points.pop(self.selected_idx)
            self.selected_idx = None
            self.update()
            self.changed.emit()


class VolumeRenderingGUIApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Volume Rendering GUI")
        self.resize(1100, 850)
        
        # Selectable status bar setup
        from PyQt6.QtWidgets import QLabel
        self.status_label = QLabel()
        self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.statusBar().addWidget(self.status_label)
        self.statusBar().showMessage = lambda text, timeout=0: self.status_label.setText(text)
        
        self.volume_data = None
        self.current_volume_path = ""
        self.P_list = []
        self.model_matrix = np.eye(4)
        self.renderer = None
        self.T = np.eye(4)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.last_dir = os.path.join(base_dir, "transfer_functions")
        
        # 3D trajectory visualization state
        self.isocenter = np.array([0.0, 0.0, 0.0])
        self.rotation_axis = np.array([0.0, 0.0, 1.0])
        self.source_positions_hom = []
        self.T_align = np.eye(4)
        self.mean_sid = 0.0
        self.mean_sdd = 0.0
        self.unit = "mm"
        
        self.cached_disp_src = []
        self.cached_disp_det = []
        self._cached_sdg_maps = {}
        self.voxel_dimensions = None
        self.clip_max_extent = 0.0
        
        # Rendering physical & normalization bounds
        self.vol_min = 0.0
        self.vol_max = 255.0
        self.orig_vol_min = 0.0
        self.orig_vol_max = 255.0
        
        # Rendering state - Default is DRR
        self.raycast_pass_type = "DRR"
        self.samples_per_voxel_val = 1.5
        self.iso_value_val = 0.5
        
        # Create transfer functions directory and default presets programmatically
        self.create_default_presets()
        
        self.init_ui()
        
        self._play_timer = QTimer()
        self._play_timer.setInterval(100)
        self._play_timer.timeout.connect(self._advance_view)
        
    def create_default_presets(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        tf_dir = os.path.join(base_dir, "transfer_functions")
        os.makedirs(tf_dir, exist_ok=True)
        
        presets = {
            "default": {
                "points": [
                    {"t": 0.0, "alpha": 0.0, "color": [0, 0, 0]},
                    {"t": 1.0, "alpha": 1.0, "color": [255, 255, 255]}
                ]
            },
            "Fire Ramp": {
                "points": [
                    {"t": 0.0, "alpha": 0.0, "color": [0, 0, 0]},
                    {"t": 0.5, "alpha": 0.5, "color": [255, 128, 0]},
                    {"t": 1.0, "alpha": 1.0, "color": [255, 255, 255]}
                ]
            }
        }
        for name, data in presets.items():
            p_path = os.path.join(tf_dir, f"{name}.json")
            if not os.path.exists(p_path):
                try:
                    with open(p_path, "w") as f:
                        json.dump(data, f, indent=2)
                except Exception as e:
                    print(f"Warning: Failed to write default preset {name}: {e}")

    def init_ui(self):
        self._setup_menu_bar()
        
        # Main Splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(main_splitter)
        
        # Center container widget with gray padding layout
        self.viewport_container = QWidget()
        self.viewport_container.setStyleSheet("background-color: #444444;")
        container_layout = QHBoxLayout(self.viewport_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Central SVG Viewport
        self.viewport = ViewportWidget(self)
        self.viewport.viewChanged.connect(self.render_viewport)
        container_layout.addWidget(self.viewport)
        main_splitter.addWidget(self.viewport_container)
        
        # 2. Right Sidebar Panel
        self.right_scroll = QScrollArea()
        self.right_scroll.setWidgetResizable(True)
        self.right_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.right_scroll.setFixedWidth(355)
        
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(10)
        
        # File Box redesign
        grp_file = QGroupBox("File")
        file_layout = QGridLayout(grp_file)
        file_layout.setSpacing(6)
        
        file_layout.addWidget(QLabel("Voxel Data:"), 0, 0)
        self.lbl_volume_dim = QLabel("Not loaded")
        self.lbl_volume_dim.setStyleSheet("font-weight: bold;")
        file_layout.addWidget(self.lbl_volume_dim, 0, 1, 1, 2)
        
        vol_btn_layout = QHBoxLayout()
        self.btn_open_vol = QPushButton("Open...")
        self.btn_open_vol.clicked.connect(self.load_volume_action)
        self.btn_view_vol = QPushButton("View...")
        self.btn_view_vol.clicked.connect(self.view_volume_action)
        vol_btn_layout.addWidget(self.btn_open_vol)
        vol_btn_layout.addWidget(self.btn_view_vol)
        file_layout.addLayout(vol_btn_layout, 1, 0, 1, 3)
        
        file_layout.addWidget(QLabel("Trajectory:"), 2, 0)
        self.lbl_trajectory_info = QLabel("Not loaded")
        self.lbl_trajectory_info.setStyleSheet("font-weight: bold;")
        file_layout.addWidget(self.lbl_trajectory_info, 2, 1, 1, 2)
        
        self.btn_open_traj = QPushButton("Open...")
        self.btn_open_traj.clicked.connect(self.load_trajectory_action)
        file_layout.addWidget(self.btn_open_traj, 3, 0, 1, 3)
        
        file_layout.addWidget(QLabel("Filename Prefix:"), 4, 0)
        self.txt_prefix = QLineEdit("image_")
        file_layout.addWidget(self.txt_prefix, 4, 1, 1, 2)
        
        self.btn_export_sino = QPushButton("Export Sinogram...")
        self.btn_export_sino.clicked.connect(self.export_sinogram_action)
        self.btn_export_sino.setStyleSheet("font-weight: bold; background-color: #2e7d32; color: white;")
        file_layout.addWidget(self.btn_export_sino, 5, 0, 1, 3)
        
        right_layout.addWidget(grp_file)
        
        # Active View settings
        grp_active = QGroupBox("Active View Slider")
        active_layout = QVBoxLayout(grp_active)
        active_layout.setSpacing(8)
        chk_layout = QHBoxLayout()
        self.chk_selected_view = QCheckBox("Selected Projection View")
        self.chk_selected_view.setChecked(False)
        self.chk_selected_view.toggled.connect(self.render_viewport)
        chk_layout.addWidget(self.chk_selected_view)
        
        self.chk_svg_overlay = QCheckBox("SVG Overlay")
        self.chk_svg_overlay.setChecked(True)
        self.chk_svg_overlay.toggled.connect(self.render_viewport)
        chk_layout.addWidget(self.chk_svg_overlay)

        
        active_layout.addLayout(chk_layout)
        
        slider_layout = QHBoxLayout()
        self.btn_play = QPushButton("\u25b6")
        self.btn_play.setFixedWidth(28)
        self.btn_play.setCheckable(True)
        self.btn_play.clicked.connect(self._toggle_play)
        slider_layout.addWidget(self.btn_play)
        
        self.active_view_slider = QSlider(Qt.Orientation.Horizontal)
        self.active_view_slider.setRange(0, 0)
        self.active_view_spin = QSpinBox()
        self.active_view_spin.setRange(0, 0)
        self.active_view_spin.setFixedWidth(60)
        
        self.active_view_slider.valueChanged.connect(self.active_view_spin.setValue)
        self.active_view_spin.valueChanged.connect(self.active_view_slider.setValue)
        self.active_view_slider.valueChanged.connect(self.render_viewport)
        
        slider_layout.addWidget(self.active_view_slider)
        slider_layout.addWidget(self.active_view_spin)
        active_layout.addLayout(slider_layout)
        right_layout.addWidget(grp_active)
        
        # Volume Rendering Settings
        grp_render = QGroupBox("Volume Rendering Settings")
        render_layout = QVBoxLayout(grp_render)
        render_layout.setSpacing(8)
        
        pass_layout = QHBoxLayout()
        pass_layout.addWidget(QLabel("Raycast Pass:"))
        self.combo_pass = QComboBox()
        self.combo_pass.addItems(["DRR", "MIP", "IsoSurface", "EmissionAbsorption", "EmissionAbsorptionShaded"])
        self.combo_pass.currentTextChanged.connect(self.on_pass_changed)
        self.combo_pass.setCurrentText(self.raycast_pass_type)
        pass_layout.addWidget(self.combo_pass)
        render_layout.addLayout(pass_layout)
        
        spv_layout = QHBoxLayout()
        spv_layout.addWidget(QLabel("Samples/Voxel:"))
        self.spv_spin = QDoubleSpinBox()
        self.spv_spin.setRange(0.1, 2.0)
        self.spv_spin.setValue(self.samples_per_voxel_val)
        self.spv_spin.setSingleStep(0.1)
        self.spv_spin.valueChanged.connect(self.on_spv_changed)
        spv_layout.addWidget(self.spv_spin)
        render_layout.addLayout(spv_layout)

        # Clip Plane controls (single row)
        clip_layout = QHBoxLayout()
        self.lbl_clip = QLabel("Clip Plane:")
        self.combo_clip = QComboBox()
        self.combo_clip.addItems(["-", "X", "Y", "Z", "-X", "-Y", "-Z"])
        self.combo_clip.currentTextChanged.connect(self.on_clip_axis_changed)
        self.clip_slider = QSlider(Qt.Orientation.Horizontal)
        self.clip_slider.setRange(-1000, 1000)
        self.clip_slider.setValue(0)
        self.clip_slider.setEnabled(False)
        self.clip_slider.valueChanged.connect(self.on_clip_slider_changed)
        
        clip_layout.addWidget(self.lbl_clip)
        clip_layout.addWidget(self.combo_clip)
        clip_layout.addWidget(self.clip_slider)
        render_layout.addLayout(clip_layout)
        
        # Iso-value slider container
        self.iso_container = QWidget()
        iso_layout_sub = QVBoxLayout(self.iso_container)
        iso_layout_sub.setContentsMargins(0, 0, 0, 0)
        
        self.iso_label = QLabel("Iso Value: 0.50")
        self.iso_slider = QSlider(Qt.Orientation.Horizontal)
        self.iso_slider.setRange(0, 1000)
        self.iso_slider.setValue(int(self.iso_value_val * 1000.0))
        self.iso_slider.valueChanged.connect(self.on_iso_slider_changed)
        
        iso_layout_sub.addWidget(self.iso_label)
        iso_layout_sub.addWidget(self.iso_slider)
        render_layout.addWidget(self.iso_container)
        self.iso_container.setVisible(False)
        
        right_layout.addWidget(grp_render)
        
        # Interactive Transfer Function Editor Panel
        grp_tf = QGroupBox("Interactive Transfer Function (E/A)")
        tf_layout = QVBoxLayout(grp_tf)
        tf_layout.setSpacing(6)
        
        # Preset selector
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Preset:"))
        self.combo_tf = QComboBox()
        self.combo_tf.addItems(["default", "Fire Ramp"])
        self.combo_tf.setCurrentText("default")
        self.combo_tf.currentTextChanged.connect(self.on_tf_preset_changed)
        preset_layout.addWidget(self.combo_tf)
        tf_layout.addLayout(preset_layout)
        
        # Intensity Range: two line edits and an Auto button
        intensity_layout = QHBoxLayout()
        intensity_layout.addWidget(QLabel("Intensity Range:"))
        
        self.txt_tf_min = QLineEdit("0.0")
        self.txt_tf_min.setFixedWidth(80)
        self.txt_tf_min.editingFinished.connect(self.on_tf_intensity_text_changed)
        intensity_layout.addWidget(self.txt_tf_min)
        
        self.txt_tf_max = QLineEdit("255.0")
        self.txt_tf_max.setFixedWidth(80)
        self.txt_tf_max.editingFinished.connect(self.on_tf_intensity_text_changed)
        intensity_layout.addWidget(self.txt_tf_max)
        
        self.btn_tf_auto = QPushButton("Auto")
        self.btn_tf_auto.setFixedWidth(50)
        self.btn_tf_auto.clicked.connect(self.on_tf_auto_clicked)
        intensity_layout.addWidget(self.btn_tf_auto)
        
        tf_layout.addLayout(intensity_layout)
        
        # 2D Spline Editor Widget
        self.tf_editor = TransferFunctionEditor(self)
        self.tf_editor.changed.connect(self.on_tf_editor_changed)
        tf_layout.addWidget(self.tf_editor)
        
        # Dropdown button and color picking layout
        btn_layout = QHBoxLayout()
        self.btn_tf_color = QPushButton("Change Color...")
        self.btn_tf_color.clicked.connect(self.tf_editor.pick_color)
        btn_layout.addWidget(self.btn_tf_color)
        
        self.btn_tf_menu = QPushButton("...")
        self.btn_tf_menu.setFixedWidth(40)
        tf_menu = QMenu(self)
        
        a_del = QAction("Delete Selected Point", self)
        a_del.triggered.connect(self.tf_editor.delete_selected_point)
        tf_menu.addAction(a_del)
        
        tf_menu.addSeparator()
        
        a_save = QAction("Save TF to File...", self)
        a_save.triggered.connect(self.save_tf_to_file)
        tf_menu.addAction(a_save)
        
        a_load = QAction("Load TF from File...", self)
        a_load.triggered.connect(self.load_tf_from_file)
        tf_menu.addAction(a_load)
        
        self.btn_tf_menu.setMenu(tf_menu)
        btn_layout.addWidget(self.btn_tf_menu)
        
        tf_layout.addLayout(btn_layout)
        right_layout.addWidget(grp_tf)
        right_layout.addStretch(1)
        
        self.right_scroll.setWidget(right_widget)
        main_splitter.addWidget(self.right_scroll)
        
        main_splitter.setSizes([780, 320])
        
    def _setup_menu_bar(self):
        mb = self.menuBar()
        m_file = mb.addMenu("File")
        
        a_load_json = QAction("Load Config (JSON)...", self)
        a_load_json.triggered.connect(self.load_config_action)
        m_file.addAction(a_load_json)
        
        a_load_vol = QAction("Load 3D Volume (NRRD)...", self)
        a_load_vol.triggered.connect(self.load_volume_action)
        m_file.addAction(a_load_vol)
        
        a_load_traj = QAction("Load Trajectory (OMPL)...", self)
        a_load_traj.triggered.connect(self.load_trajectory_action)
        m_file.addAction(a_load_traj)
        
        m_file.addSeparator()
        a_exit = QAction("Exit", self)
        a_exit.triggered.connect(self.close)
        m_file.addAction(a_exit)

        # Axis Menu
        m_axis = mb.addMenu("Axis")
        
        a_center = QAction("Center", self)
        a_center.triggered.connect(self.axis_center)
        m_axis.addAction(a_center)
        
        a_reset = QAction("Reset", self)
        a_reset.triggered.connect(self.axis_reset)
        m_axis.addAction(a_reset)
        
        m_axis.addSeparator()
        
        a_swap_xy = QAction("Swap X <-> Y", self)
        a_swap_xy.triggered.connect(self.axis_swap_xy)
        m_axis.addAction(a_swap_xy)
        
        a_swap_xz = QAction("Swap X <-> Z", self)
        a_swap_xz.triggered.connect(self.axis_swap_xz)
        m_axis.addAction(a_swap_xz)
        
        a_swap_yz = QAction("Swap Y <-> Z", self)
        a_swap_yz.triggered.connect(self.axis_swap_yz)
        m_axis.addAction(a_swap_yz)
        
        m_axis.addSeparator()
        
        a_flip_x = QAction("Flip X", self)
        a_flip_x.triggered.connect(self.axis_flip_x)
        m_axis.addAction(a_flip_x)
        
        a_flip_y = QAction("Flip Y", self)
        a_flip_y.triggered.connect(self.axis_flip_y)
        m_axis.addAction(a_flip_y)
        
        a_flip_z = QAction("Flip Z", self)
        a_flip_z.triggered.connect(self.axis_flip_z)
        m_axis.addAction(a_flip_z)
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.render_viewport()

    def load_config_action(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Config JSON", self.last_dir, "JSON Configs (*.json);;All Files (*)"
        )
        if not path:
            return
        self.last_dir = os.path.dirname(path)
        self.load_config(path)

    def load_config(self, config_path):
        try:
            self.statusBar().showMessage(f"Loading config JSON: {os.path.basename(config_path)}...")
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            
            config_dir = os.path.dirname(os.path.abspath(config_path))
            
            def resolve(p):
                if not p:
                    return ""
                if os.path.isabs(p):
                    return p
                cfg_rel = os.path.normpath(os.path.join(config_dir, p))
                if os.path.exists(cfg_rel):
                    return cfg_rel
                data_dir = cfg.get("data_dir", "")
                if data_dir:
                    resolved_data_dir = data_dir
                    if not os.path.isabs(data_dir):
                        resolved_data_dir = os.path.normpath(os.path.join(config_dir, data_dir))
                    data_rel = os.path.normpath(os.path.join(resolved_data_dir, p))
                    if os.path.exists(data_rel):
                        return data_rel
                return cfg_rel

            # 1. Parse dimensions and matrix
            self.voxel_dimensions = cfg.get("voxel_dimensions", None)
            self.model_matrix = np.array(cfg.get("model_matrix", np.eye(4).tolist()), dtype=np.float64)
            self.unit = cfg.get("unit", "mm")
            
            # 2. Load Trajectory (OMPL)
            ompl_file = cfg.get("ompl_file", "")
            if ompl_file:
                ompl_path = resolve(ompl_file)
                if os.path.exists(ompl_path):
                    self._load_trajectory(ompl_path)
                else:
                    self.statusBar().showMessage(f"Warning: OMPL file not found at: {ompl_path}")
            
            # 3. Load Volume (NRRD output_file / input_volume)
            vol_path = None
            for key in ["output_file", "output", "volume", "volume_file", "input_volume"]:
                if key in cfg and cfg[key]:
                    path_to_try = resolve(cfg[key])
                    if os.path.exists(path_to_try):
                        vol_path = path_to_try
                        break
            
            if vol_path:
                self._load_volume(vol_path, use_header_matrix=("model_matrix" not in cfg))
            else:
                self.statusBar().showMessage("Warning: No matching 3D volume file found/resolved in config JSON.")
            
            self._update_file_info()
            self.statusBar().showMessage(f"Config loaded successfully: {os.path.basename(config_path)}")
            self.render_viewport()
        except Exception as e:
            QMessageBox.critical(self, "Error Loading Config", f"Could not load config JSON:\n{str(e)}")
            self.statusBar().showMessage("Error loading config JSON.")

    def load_volume_action(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load 3D Volume", self.last_dir, "NRRD Volumes (*.nrrd);;All Files (*)"
        )
        if not path:
            return
        self.last_dir = os.path.dirname(path)
        try:
            self._load_volume(path)
            self._update_file_info()
            self.render_viewport()
        except Exception as e:
            QMessageBox.critical(self, "Error Loading Volume", f"Could not load NRRD volume:\n{str(e)}")
            self.statusBar().showMessage("Error loading volume.")

    def _load_volume(self, path, use_header_matrix=True):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        from PyQt6.QtWidgets import QProgressDialog
        progress = QProgressDialog("Initializing volume load...", None, 0, 4, self)
        progress.setWindowTitle("Loading Volume")
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()
        
        try:
            progress.setLabelText("Reading NRRD file from disk...")
            progress.setValue(1)
            QApplication.processEvents()
            
            vol, header = nrrd.read(path)
            
            progress.setLabelText("Normalizing volume data...")
            progress.setValue(2)
            QApplication.processEvents()
            
            vol = np.transpose(vol, (2, 1, 0))
            self.orig_vol_min = float(np.min(vol))
            self.orig_vol_max = float(np.max(vol))
            
            # Scale volume to 0-255 range for GPU compatibility
            v_range = self.orig_vol_max - self.orig_vol_min
            if v_range > 1e-6:
                self.volume_data = ((vol - self.orig_vol_min) / v_range * 255.0).astype(np.float32)
            else:
                self.volume_data = vol.astype(np.float32)
                
            self.vol_min = 0.0
            self.vol_max = 255.0
            self.current_volume_path = path
            
            if use_header_matrix:
                self.model_matrix = get_model_matrix_from_nrrd_header(header)
                
            self.txt_tf_min.blockSignals(True)
            self.txt_tf_max.blockSignals(True)
            self.txt_tf_min.setText(f"{self.orig_vol_min:.4g}")
            self.txt_tf_max.setText(f"{self.orig_vol_max:.4g}")
            self.txt_tf_min.blockSignals(False)
            self.txt_tf_max.blockSignals(False)
            
            progress.setLabelText("Computing volume histogram...")
            progress.setValue(3)
            QApplication.processEvents()
            
            # Compute histogram on the scaled volume range [0, 255]
            hist, bins = np.histogram(self.volume_data, bins=256, range=(0.0, 255.0))
            hist_log = np.log1p(hist)
            m_hist = np.max(hist_log)
            if m_hist > 0:
                hist_log /= m_hist
            self.tf_editor.set_histogram(hist_log.tolist())
            
            progress.setLabelText("Initializing GPU renderer...")
            progress.setValue(4)
            QApplication.processEvents()
            
            if self.voxel_dimensions is None or use_header_matrix:
                self.voxel_dimensions = list(self.volume_data.shape[::-1])
                
            self.renderer = VolumeRenderer(self.volume_data, model_matrix=self.model_matrix, use_ess=True)
            self.renderer.raycast_pass = self.raycast_pass_type
            self.renderer.samples_per_voxel = self.samples_per_voxel_val
            
            # Map default normalized iso_value_val to 0-255 GPU coordinate
            v_iso_phys = self.orig_vol_min + self.iso_value_val * (self.orig_vol_max - self.orig_vol_min)
            v_iso_gpu = (v_iso_phys - self.orig_vol_min) / v_range * 255.0 if v_range > 1e-6 else v_iso_phys
            self.renderer.iso_value = float(v_iso_gpu)
            
            self.renderer.ray_length_weighted = False
            self.apply_tf_from_editor()
            self.update_clip_plane_range()
            self.statusBar().showMessage(f"Volume loaded: {os.path.basename(path)}")
        finally:
            progress.close()
            QApplication.restoreOverrideCursor()

    def view_volume_action(self):
        if not self.current_volume_path:
            QMessageBox.warning(self, "No Volume", "No 3D volume is currently loaded.")
            return
        import subprocess
        try:
            cmd = ["/home/aaichert/.virtualenvs/xray/bin/NrrdView3D", self.current_volume_path]
            subprocess.Popen(cmd)
            self.statusBar().showMessage("Started NrrdView3D in a separate process.")
        except Exception as e:
            QMessageBox.critical(self, "Error Running NrrdView3D", f"Could not launch NrrdView3D:\n{e}")

    def load_trajectory_action(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Trajectory", self.last_dir, "OMPL Trajectories (*.ompl);;All Files (*)"
        )
        if not path:
            return
        self.last_dir = os.path.dirname(path)
        try:
            self._load_trajectory(path)
            self._update_file_info()
            self.render_viewport()
        except Exception as e:
            QMessageBox.critical(self, "Error Loading Trajectory", f"Could not load OMPL trajectory:\n{str(e)}")
            self.statusBar().showMessage("Error loading trajectory.")

    def _load_trajectory(self, path):
        self.P_list = load_ompl(path)
        n = len(self.P_list)
        self.active_view_slider.setRange(0, max(0, n - 1))
        self.active_view_spin.setRange(0, max(0, n - 1))
        self.active_view_slider.setValue(0)
        
        try:
            self.source_positions_hom = [p.getCenterOfProjection().reshape(-1, 1) for p in self.P_list]
            sources = np.array([C.flatten()[:3] for C in self.source_positions_hom])
            self.isocenter = self.estimate_isocenter()
            
            P_first = self.P_list[0]
            sdg_first = SourceDetectorGeometry(P_first)
            s_first = sdg_first.source_position.flatten()[:3]
            O_first = sdg_first.detector_origin.flatten()[:3]
            U_first = sdg_first.axis_direction_Upx.flatten()[:3] * P_first.image_size[0]
            V_first = sdg_first.axis_direction_Vpx.flatten()[:3] * P_first.image_size[1]
            d_first = O_first + 0.5 * U_first + 0.5 * V_first
            
            v_sd = d_first - s_first
            v_si = self.isocenter - s_first
            
            if np.dot(v_si, v_sd) <= 0:
                for p in self.P_list:
                    p.pixel_spacing = -p.pixel_spacing
            
            self.mean_sid = np.mean([np.linalg.norm(src - self.isocenter) for src in sources])
            self.mean_sdd = np.mean([abs(SourceDetectorGeometry(p).source_detector_distance) for p in self.P_list])
            
            if n >= 3:
                centered = sources - np.mean(sources, axis=0)
                _, _, Vt = np.linalg.svd(centered)
                self.rotation_axis = Vt[-1, :]
                if (self.rotation_axis[2] < 0 or 
                    (abs(self.rotation_axis[2]) < 1e-7 and self.rotation_axis[1] < 0) or
                    (abs(self.rotation_axis[2]) < 1e-7 and abs(self.rotation_axis[1]) < 1e-7 and self.rotation_axis[0] < 0)):
                    self.rotation_axis = -self.rotation_axis
                
                a = self.rotation_axis / np.linalg.norm(self.rotation_axis)
                v = np.array([1.0, 0.0, 0.0])
                if abs(np.dot(a, v)) > 0.9:
                    v = np.array([0.0, 0.0, 1.0])
                r1 = np.cross(v, a)
                r1 = r1 / np.linalg.norm(r1)
                r2 = np.cross(a, r1)
                r2 = r2 / np.linalg.norm(r2)
                R = np.vstack([r1, r2, a])
                
                self.T_align = np.eye(4)
                self.T_align[:3, :3] = R
            else:
                self.rotation_axis = np.array([0.0, 0.0, 1.0])
                self.T_align = np.eye(4)
                
            r_mean = self.mean_sid
            D = 2.0 * r_mean
            s_scale = 250.0 / D if D > 1e-6 else 1.0
            self.viewport.s = 0.2 * s_scale
            
        except Exception as ex:
            self.isocenter = np.array([0.0, 0.0, 0.0])
            self.rotation_axis = np.array([0.0, 0.0, 1.0])
            self.source_positions_hom = []
            self.T_align = np.eye(4)
            self.mean_sid = 0.0
            self.mean_sdd = 0.0
            print(f"Warning: Geometry extraction failed: {ex}")
            
        self.update_trajectory_coordinates()
        self.statusBar().showMessage(f"Trajectory loaded: {os.path.basename(path)}")

    def update_trajectory_coordinates(self):
        if not self.P_list:
            self.cached_disp_src = []
            self.cached_disp_det = []
            return
        self.cached_disp_src = [pm.getCenterOfProjection().reshape(-1, 1) for pm in self.P_list]
        self.cached_disp_det = []
        for pm, sdg in zip(self.P_list, self.get_sdg_list(self.P_list)):
            O_det = sdg.detector_origin.flatten()[:3]
            U_det = sdg.axis_direction_Upx.flatten()[:3] * pm.image_size[0]
            V_det = sdg.axis_direction_Vpx.flatten()[:3] * pm.image_size[1]
            C_det = O_det + 0.5 * U_det + 0.5 * V_det
            self.cached_disp_det.append(np.array([C_det[0], C_det[1], C_det[2], 1.0]).reshape(-1, 1))

    def get_sdg_list(self, p_list):
        if not p_list:
            return []
        state_key = tuple((id(p), p.pixel_spacing, tuple(p.P.flat)) for p in p_list)
        if not hasattr(self, '_cached_sdg_maps') or self._cached_sdg_maps is None:
            self._cached_sdg_maps = {}
        if state_key not in self._cached_sdg_maps:
            self._cached_sdg_maps[state_key] = [SourceDetectorGeometry(p) for p in p_list]
        return self._cached_sdg_maps[state_key]

    def estimate_isocenter(self):
        if not self.P_list:
            return np.array([0.0, 0.0, 0.0])
        A_mat = np.zeros((3, 3))
        b_vec = np.zeros(3)
        for p in self.P_list:
            C = p.getCenterOfProjection().flatten()
            if abs(C[3]) > 1e-12:
                C = C[:3] / C[3]
            else:
                C = C[:3]
            sdg = SourceDetectorGeometry(p)
            O_det = sdg.detector_origin.flatten()[:3]
            U_det = sdg.axis_direction_Upx.flatten()[:3] * p.image_size[0]
            V_det = sdg.axis_direction_Vpx.flatten()[:3] * p.image_size[1]
            C_det = O_det + 0.5 * U_det + 0.5 * V_det
            
            r = C_det - C
            r_norm = np.linalg.norm(r)
            if r_norm > 1e-12:
                r /= r_norm
            M_proj = np.eye(3) - np.outer(r, r)
            A_mat += M_proj
            b_vec += M_proj @ C
        return np.linalg.pinv(A_mat) @ b_vec

    def _update_file_info(self):
        dim_info = "Not loaded"
        if self.volume_data is not None:
            dim_info = " x ".join(map(str, self.volume_data.shape[::-1])) + " (W x H x D)"
        self.lbl_volume_dim.setText(dim_info)
            
        traj_info = "Not loaded"
        if self.P_list:
            traj_info = f"({len(self.P_list)} views)"
        self.lbl_trajectory_info.setText(traj_info)
        
    def _toggle_play(self, checked):
        if checked:
            self.btn_play.setText("\u23f8")
            self._play_timer.start()
        else:
            self.btn_play.setText("\u25b6")
            self._play_timer.stop()
            
    def _advance_view(self):
        max_val = self.active_view_slider.maximum()
        if max_val <= 0:
            return
        cur = self.active_view_slider.value()
        self.active_view_slider.setValue(0 if cur >= max_val else cur + 1)
        
    def on_pass_changed(self, text):
        self.raycast_pass_type = text
        if self.renderer:
            self.renderer.raycast_pass = text
        self.iso_container.setVisible(text == "IsoSurface")
        
        # Dynamically change viewport background stylesheet
        is_dark = text in ("EmissionAbsorption", "EmissionAbsorptionShaded", "MIP")
        bg_style = "background: black; border: 1px solid #555;" if is_dark else "background: white; border: 1px solid #ccc;"
        self.viewport.setStyleSheet(bg_style)
        
        self.render_viewport()
        
    def on_spv_changed(self, val):
        self.samples_per_voxel_val = val
        if self.renderer:
            self.renderer.samples_per_voxel = val
        self.render_viewport()
        
    def on_iso_slider_changed(self, val):
        s = val / 1000.0
        self.iso_value_val = s
        self.iso_label.setText(f"Iso Value: {s:.2f}")
        if self.renderer:
            try:
                tf_min = float(self.txt_tf_min.text().strip())
                tf_max = float(self.txt_tf_max.text().strip())
            except ValueError:
                tf_min = self.orig_vol_min
                tf_max = self.orig_vol_max
            
            # Find the physical value mapped inside the intensity range
            v_iso_phys = tf_min + s * (tf_max - tf_min)
            
            # Map physical value to 0-255 GPU coordinate
            v_range = self.orig_vol_max - self.orig_vol_min
            v_iso_gpu = (v_iso_phys - self.orig_vol_min) / v_range * 255.0 if v_range > 1e-6 else v_iso_phys
            self.renderer.iso_value = float(v_iso_gpu)
        self.render_viewport()
        
    def on_tf_preset_changed(self, text):
        self.tf_preset_name = text
        self.load_preset_points()
        self.apply_tf_from_editor()
        self.render_viewport()
        
    def load_preset_points(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        p_path = os.path.join(base_dir, "transfer_functions", f"{self.tf_preset_name}.json")
        
        if os.path.exists(p_path):
            try:
                with open(p_path, "r") as f:
                    data = json.load(f)
                points = []
                for pt in data["points"]:
                    points.append({
                        "t": float(pt["t"]),
                        "alpha": float(pt["alpha"]),
                        "color": tuple(pt["color"])
                    })
                self.tf_editor.points = points
                self.tf_editor.selected_idx = None
                self.tf_editor.update()
                return
            except Exception as e:
                print(f"Warning: Failed to load preset from JSON: {e}")
                
        if self.tf_preset_name == "default":
            self.tf_editor.points = [
                {"t": 0.0, "alpha": 0.0, "color": (0, 0, 0)},
                {"t": 1.0, "alpha": 1.0, "color": (255, 255, 255)}
            ]
        elif self.tf_preset_name == "Fire Ramp":
            self.tf_editor.points = [
                {"t": 0.0, "alpha": 0.0, "color": (0, 0, 0)},
                {"t": 0.5, "alpha": 0.5, "color": (255, 128, 0)},
                {"t": 1.0, "alpha": 1.0, "color": (255, 255, 255)}
            ]
        self.tf_editor.selected_idx = None
        self.tf_editor.update()

    def on_clip_axis_changed(self, text):
        if text == "None" or text == "-":
            self.clip_slider.setEnabled(False)
            self.clip_slider.setValue(0)
            self.clip_slider.setToolTip("")
            if self.renderer:
                self.renderer.clip_planes = None
        else:
            self.clip_slider.setEnabled(True)
            self.clip_slider.setValue(0)
            self.update_clip_plane()
        self.render_viewport()
        
    def on_clip_slider_changed(self, val):
        self.update_clip_plane()
        self.render_viewport()

    def update_clip_plane(self):
        if not self.renderer:
            return
            
        axis = self.combo_clip.currentText()
        if axis == "None" or axis == "-" or self.clip_max_extent <= 1e-6:
            self.renderer.clip_planes = None
            self.clip_slider.setEnabled(False)
            self.clip_slider.setToolTip("")
        else:
            self.clip_slider.setEnabled(True)
            val = self.clip_slider.value()
            d_phys = (val / 1000.0) * self.clip_max_extent
            tooltip_text = f"Clip Distance: {d_phys:.2f} {self.unit}"
            self.clip_slider.setToolTip(tooltip_text)
            self.statusBar().showMessage(tooltip_text)
            
            # Compute plane equation in voxel coordinates
            M = self.model_matrix
            R = M[:3, :3]
            W, H, D = self.voxel_dimensions
            C_vox = np.array([W / 2.0, H / 2.0, D / 2.0])
            
            is_negative = axis.startswith("-")
            axis_name = axis.replace("-", "")
            
            if axis_name == "X":
                e_a = np.array([1.0, 0.0, 0.0])
            elif axis_name == "Y":
                e_a = np.array([0.0, 1.0, 0.0])
            elif axis_name == "Z":
                e_a = np.array([0.0, 0.0, 1.0])
            else:
                e_a = np.array([1.0, 0.0, 0.0])
                
            v_a = R @ e_a
            v_a_norm = np.linalg.norm(v_a)
            if v_a_norm > 1e-12:
                n_phys = v_a / v_a_norm
            else:
                n_phys = e_a
                
            n_vox = R.T @ n_phys
            E_vox = np.zeros(4)
            E_vox[:3] = n_vox
            E_vox[3] = -np.dot(n_vox, C_vox) - d_phys
            
            if is_negative:
                E_vox = -E_vox
            
            self.renderer.clip_planes = [E_vox]

    def update_clip_plane_range(self):
        if self.volume_data is None:
            self.clip_max_extent = 0.0
            self.clip_slider.setEnabled(False)
            self.clip_slider.setToolTip("")
            return
            
        W, H, D = self.voxel_dimensions
        M = self.model_matrix
        col_x = M[:3, 0]
        col_y = M[:3, 1]
        col_z = M[:3, 2]

        s_x = np.linalg.norm(col_x)
        s_y = np.linalg.norm(col_y)
        s_z = np.linalg.norm(col_z)

        L_x = W * s_x
        L_y = H * s_y
        L_z = D * s_z

        self.clip_max_extent = float(max(L_x, L_y, L_z))
        self.update_clip_plane()

    def on_tf_editor_changed(self):
        self.apply_tf_from_editor()
        self.render_viewport()
        
    def on_tf_intensity_text_changed(self):
        try:
            # Support scientific notation like 1.0e5
            min_val = float(self.txt_tf_min.text().strip())
            max_val = float(self.txt_tf_max.text().strip())
            if min_val > max_val:
                min_val, max_val = max_val, min_val
                self.txt_tf_min.setText(f"{min_val:.4g}")
                self.txt_tf_max.setText(f"{max_val:.4g}")
            
            # Since the physical Iso Value depends on TF Intensity Range limits, we update it too
            self.on_iso_slider_changed(self.iso_slider.value())
            self.apply_tf_from_editor()
            self.render_viewport()
        except ValueError:
            pass

    def on_tf_auto_clicked(self):
        self.txt_tf_min.setText(f"{self.orig_vol_min:.4g}")
        self.txt_tf_max.setText(f"{self.orig_vol_max:.4g}")
        # Update physical Iso Value
        self.on_iso_slider_changed(self.iso_slider.value())
        self.apply_tf_from_editor()
        self.render_viewport()

    def apply_tf_from_editor(self):
        if not self.renderer:
            return
        try:
            tf_min = float(self.txt_tf_min.text().strip())
            tf_max = float(self.txt_tf_max.text().strip())
        except ValueError:
            tf_min = self.orig_vol_min
            tf_max = self.orig_vol_max
            
        tf_data = self.tf_editor.generate_tf_array(
            self.orig_vol_min, self.orig_vol_max,
            tf_min, tf_max
        )
        self.renderer.transfer_function = tf_data

    def load_tf_from_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Transfer Function", self.last_dir, "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return
        self.last_dir = os.path.dirname(path)
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if "points" in data:
                points = []
                for pt in data["points"]:
                    points.append({
                        "t": float(pt["t"]),
                        "alpha": float(pt["alpha"]),
                        "color": tuple(pt["color"])
                    })
                self.tf_editor.points = points
                self.tf_editor.selected_idx = None
                self.tf_editor.update()
                self.apply_tf_from_editor()
                self.render_viewport()
                self.statusBar().showMessage(f"Transfer function loaded from: {os.path.basename(path)}")
            else:
                QMessageBox.warning(self, "Invalid File", "JSON file does not contain a 'points' field.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load transfer function:\n{e}")

    def save_tf_to_file(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Transfer Function", self.last_dir, "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return
        self.last_dir = os.path.dirname(path)
        try:
            data = {"points": self.tf_editor.points}
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            self.statusBar().showMessage(f"Transfer function saved to: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save transfer function:\n{e}")

    def export_sinogram_action(self):
        if self.volume_data is None:
            QMessageBox.warning(self, "No Volume", "No 3D volume is loaded.")
            return
        if not self.P_list:
            QMessageBox.warning(self, "No Trajectory", "No trajectory is loaded.")
            return
            
        output_dir = QFileDialog.getExistingDirectory(self, "Select Empty Output Directory", self.last_dir)
        if not output_dir:
            return
        self.last_dir = output_dir
        
        if os.listdir(output_dir):
            res = QMessageBox.question(
                self, "Directory Not Empty",
                "The selected directory is not empty. Do you want to proceed and overwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if res == QMessageBox.StandardButton.No:
                return
                
        # Temporarily switch to DRR pass for sinogram export
        orig_pass = self.raycast_pass_type
        if self.renderer:
            self.renderer.raycast_pass = "DRR"
            
        from PyQt6.QtWidgets import QProgressDialog
        import tifffile
        
        n_views = len(self.P_list)
        progress = QProgressDialog("Forward projecting views...", "Cancel", 0, n_views, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        
        prefix = self.txt_prefix.text().strip()
        
        try:
            for idx, P_active in enumerate(self.P_list):
                if progress.wasCanceled():
                    break
                    
                progress.setValue(idx)
                progress.setLabelText(f"Forward projecting view {idx+1}/{n_views}...")
                QApplication.processEvents()
                
                # Render using the transformed ProjectionMatrix
                P_opt = ProjectionMatrix(P_active.P @ self.T, image_size=P_active.image_size, pixel_spacing=P_active.pixel_spacing)
                img = self.renderer.render(P_opt)
                
                filename = f"{prefix}{idx+1:04d}.tif"
                filepath = os.path.join(output_dir, filename)
                
                tifffile.imwrite(filepath, img, compression='zlib')
                
            if not progress.wasCanceled():
                progress.setValue(n_views)
                
                # Save trajectory.ompl
                ompl_path = os.path.join(output_dir, "trajectory.ompl")
                self.save_ompl(ompl_path, self.P_list)
                
                # Save reconstruction.json
                image_files = [f"{prefix}{i+1:04d}.tif" for i in range(n_views)]
                recon_cfg = {
                    "data_dir": ".",
                    "ompl_file": "trajectory.ompl",
                    "image_files": image_files,
                    "voxel_dimensions": self.voxel_dimensions,
                    "model_matrix": self.model_matrix.tolist(),
                    "output_file": "reconstruction.nrrd",
                    "filter_type": "hann",
                    "convert_to_line_integral": False
                }
                with open(os.path.join(output_dir, "reconstruction.json"), "w") as f:
                    json.dump(recon_cfg, f, indent=2)
                    
                QMessageBox.information(self, "Export Successful", f"Sinogram exported successfully to:\n{output_dir}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"An error occurred during export:\n{str(e)}")
        finally:
            if self.renderer:
                self.renderer.raycast_pass = orig_pass
            progress.close()

    def save_ompl(self, path, P_list):
        with open(path, "w") as f:
            for p in P_list:
                P_mat = p.P
                f.write(f"[{P_mat[0,0]} {P_mat[0,1]} {P_mat[0,2]} {P_mat[0,3]}; "
                        f"{P_mat[1,0]} {P_mat[1,1]} {P_mat[1,2]} {P_mat[1,3]}; "
                        f"{P_mat[2,0]} {P_mat[2,1]} {P_mat[2,2]} {P_mat[2,3]}]\n")

    def _build_scene_svg(self, w_w, w_h, P, preview_pil_image=None):
        svg = Composer((w_w, w_h))
        pad_x, pad_y, target_w, target_h = getattr(self, 'curr_pad', (0, 0, w_w, w_h))
        
        is_dark = self.raycast_pass_type in ("EmissionAbsorption", "EmissionAbsorptionShaded", "MIP")
        bg_fill = "#000000" if is_dark else "#ffffff"
        lbl_color = "white" if is_dark else "black"
        traj_color = "white" if is_dark else "black"
        text_color = "#bbbbbb" if is_dark else "#333333"

        if preview_pil_image is not None:
            if self.chk_selected_view.isChecked() and self.P_list:
                proj_bg = "#000000" if is_dark else "#444444"
                svg.add(e2d.rect, x=0, y=0, width=w_w, height=w_h, fill=proj_bg)
            else:
                svg.add(e2d.rect, x=0, y=0, width=w_w, height=w_h, fill=bg_fill)
            svg.add(e2d.image, data=preview_pil_image, x=pad_x, y=pad_y, width=target_w, height=target_h, sparse=1)
        else:
            svg.add(e2d.rect, x=0, y=0, width=w_w, height=w_h, fill=bg_fill)

        if not self.chk_svg_overlay.isChecked():
            return svg

        half_sid = self.mean_sid / 2.0 if self.mean_sid > 0.0 else 80.0
        try:
            k = int(np.round(np.log10(half_sid)))
            axis_length = 10.0 ** k
        except:
            axis_length = 10.0
        len_str = f"{int(axis_length)} {self.unit}"

        show_lbls = True
        svg.add(safe_arrow, X1=[0,0,0,1], X2=[axis_length,0,0,1], stroke='red', stroke_width=2)
        svg.add(safe_arrow, X1=[0,0,0,1], X2=[0,axis_length,0,1], stroke='green', stroke_width=2)
        svg.add(safe_arrow, X1=[0,0,0,1], X2=[0,0,axis_length,1], stroke='blue', stroke_width=2)
        if show_lbls:
            svg.add(safe_text, X=[0,0,0,1], content="world", fill=lbl_color, font_size='11px', font_family='sans-serif')
            svg.add(safe_text, X=[axis_length*1.1,0,0,1], content=f"X ({len_str})", fill='red', font_size='11px', font_family='sans-serif')
            svg.add(safe_text, X=[0,axis_length*1.1,0,1], content=f"Y ({len_str})", fill='green', font_size='11px', font_family='sans-serif')
            svg.add(safe_text, X=[0,0,axis_length*1.1,1], content=f"Z ({len_str})", fill='blue', font_size='11px', font_family='sans-serif')

        if self.voxel_dimensions is not None:
            try:
                mean_dim = np.mean(self.voxel_dimensions)
                p = 1
                while p < mean_dim:
                    p *= 10
                p = max(p // 10, 1)
                if p * 5 < mean_dim:
                    p = p * 5
                elif p * 2 < mean_dim:
                    p = p * 2
                
                M = self.model_matrix
                vox_orig = M @ np.array([0, 0, 0, 1])
                vox_x = M @ np.array([p, 0, 0, 1])
                vox_y = M @ np.array([0, p, 0, 1])
                vox_z = M @ np.array([0, 0, p, 1])
                
                txt_x = M @ np.array([p * 1.1, 0, 0, 1])
                txt_y = M @ np.array([0, p * 1.1, 0, 1])
                txt_z = M @ np.array([0, 0, p * 1.1, 1])
                
                svg.add(safe_arrow, X1=vox_orig, X2=vox_x, stroke='red', stroke_width=2)
                svg.add(safe_arrow, X1=vox_orig, X2=vox_y, stroke='green', stroke_width=2)
                svg.add(safe_arrow, X1=vox_orig, X2=vox_z, stroke='blue', stroke_width=2)
                if show_lbls:
                    svg.add(safe_text, X=vox_orig, content="volume", fill=lbl_color, font_size='11px', font_family='sans-serif')
                    svg.add(safe_text, X=txt_x, content=f"X ({p} voxels)", fill='red', font_size='11px', font_family='sans-serif')
                    svg.add(safe_text, X=txt_y, content=f"Y ({p} voxels)", fill='green', font_size='11px', font_family='sans-serif')
                    svg.add(safe_text, X=txt_z, content=f"Z ({p} voxels)", fill='blue', font_size='11px', font_family='sans-serif')
            except Exception as e:
                print(f"Warning: Failed to render voxel coordinate system: {e}")

        import ProjectiveGeometry23.pluecker as pluecker
        from ProjectiveGeometry23.svg_utils import svg_pluecker_line
        L_rot = pluecker.join_points(pgu.homogenize(self.isocenter), pgu.infinite(self.rotation_axis))
        svg.add(svg_pluecker_line, L=L_rot, stroke="yellow", stroke_dasharray="4,4", stroke_width=1.5)

        show_trajectory = True
        if self.P_list and show_trajectory:
            disp_det = getattr(self, 'cached_disp_det', [])
            num_views = len(disp_det)
            if num_views > 180:
                all_idx = np.round(np.linspace(0, num_views - 1, 180)).astype(int)
                for i in range(len(all_idx) - 1):
                    a, b = all_idx[i], all_idx[i + 1]
                    if a < num_views and b < num_views:
                        svg.add(safe_line, X1=disp_det[a], X2=disp_det[b], stroke=traj_color, stroke_width=1.2)
            else:
                for i in range(num_views - 1):
                    svg.add(safe_line, X1=disp_det[i], X2=disp_det[i + 1], stroke=traj_color, stroke_width=1.2)

        svg.add(e2d.text, x=15, y=25, content=f"Views: {len(self.P_list)}", fill=text_color, font_size="13px", font_family="monospace")
        return svg

    def _add_dynamic_elements(self, svg_obj, P_view, active_idx, show_pyramid, show_current_source):
        if not self.P_list:
            svg_obj.add(e2d.text, x=15, y=50, content="No Trajectory Loaded", fill="#d00000", font_size="13px", font_family="monospace")
            return

        if active_idx >= len(self.P_list):
            active_idx = 0

        disp_src = getattr(self, 'cached_disp_src', [])
        disp_det = getattr(self, 'cached_disp_det', [])
        num_views = len(disp_src)

        if self.voxel_dimensions is not None:
            shape = (self.voxel_dimensions[2], self.voxel_dimensions[1], self.voxel_dimensions[0])
            svg_obj.add(
                patched_volume,
                shape=shape,
                model_matrix=self.model_matrix,
                color_axes=False,
                lighting=True,
                fill="#00ff4015",
                stroke="#00ff4080",
                stroke_width=1.0,
            )

        show_traj = True
        if num_views > 0 and show_traj:
            svg_obj.add(
                patched_trajectory,
                disp_src=disp_src,
                active_idx=active_idx,
                num_views=num_views
            )

            if show_current_source and active_idx < num_views:
                svg_obj.add(safe_point, X=disp_src[active_idx], r=6, fill="#ff2d55")

        if show_traj:
            closest_indices = {}
            try:
                a = self.rotation_axis / np.linalg.norm(self.rotation_axis)
                v = np.array([1.0, 0.0, 0.0])
                if abs(np.dot(a, v)) > 0.9:
                    v = np.array([0.0, 0.0, 1.0])
                r3 = np.cross(a, v)
                r3 /= np.linalg.norm(r3)
                r1 = np.cross(a, r3)
                r1 /= np.linalg.norm(r1)
                angles_deg = []
                for C_hom in self.source_positions_hom:
                    C = C_hom.flatten()[:3] - self.isocenter
                    angles_deg.append(np.arctan2(np.dot(C, r3), np.dot(C, r1)) * 180.0 / np.pi)
                ref_angle = angles_deg[0]
                angles_deg = np.array([(d - ref_angle) % 360.0 for d in angles_deg])
                for target in [45, 90, 135, 180, 225, 270, 315]:
                    idx = int(np.argmin(np.abs(angles_deg - target)))
                    closest_indices[idx] = angles_deg[idx]
            except:
                pass

            for idx, actual_angle in closest_indices.items():
                if idx != active_idx and idx < num_views:
                    C = disp_src[idx]
                    val = round(actual_angle, 1)
                    label_str = f".  {idx} ({int(val) if val == int(val) else val}°)"
                    if num_views > 180:
                        svg_obj.add(safe_point, X=C, r=3, fill="#00adb5")
                    svg_obj.add(safe_text, X=C, content=label_str, fill="#00adb5", font_size="10px", font_family="monospace")
                    
                    is_dark = self.raycast_pass_type in ("EmissionAbsorption", "MIP")
                    line_stroke = "#ffffff" if is_dark else "#000000"
                    svg_obj.add(safe_line, X1=disp_src[idx], X2=disp_det[idx], stroke=line_stroke, stroke_opacity=0.251 if not is_dark else 0.4)

        if show_pyramid and active_idx < len(self.P_list):
            P_active = self.P_list[active_idx]
            show_lbls = True
            
            if self.voxel_dimensions is not None:
                def draw_voxel_wireframe(P, **kwargs):
                    X, Y, Z = self.voxel_dimensions
                    P_local = P @ self.model_matrix
                    group = Group("Voxel Wireframe and Axes")
                    group.add(e3d.wire_cube, P=P_local, min=[0,0,0], max=[X,Y,Z],
                              stroke="#ff2d55", stroke_width=1.0)
                    
                    mean_dim = np.mean(self.voxel_dimensions)
                    p = 1
                    while p < mean_dim:
                        p *= 10
                    p = max(p // 10, 1)
                    if p * 5 < mean_dim:
                        p = p * 5
                    elif p * 2 < mean_dim:
                        p = p * 2
                    
                    vox_orig = np.array([0, 0, 0, 1])
                    vox_x = np.array([p, 0, 0, 1])
                    vox_y = np.array([0, p, 0, 1])
                    vox_z = np.array([0, 0, p, 1])
                    
                    group.add(safe_arrow, P=P_local, X1=vox_orig, X2=vox_x, stroke='red', stroke_width=2)
                    group.add(safe_arrow, P=P_local, X1=vox_orig, X2=vox_y, stroke='green', stroke_width=2)
                    group.add(safe_arrow, P=P_local, X1=vox_orig, X2=vox_z, stroke='blue', stroke_width=2)
                    return group
                
                svg_obj.add(ProjectiveGeometry23.svg_utils.svg_source_detector,
                            projection=P_active,
                            draw_on_detector=draw_voxel_wireframe,
                            label_source=f".  C{active_idx}",
                            is_virtual=False,
                            show_axis_labels=show_lbls)
        
    def render_viewport(self):
        if self.viewport.P_display is None:
            return
            
        try:
            w_w = self.viewport_container.width()
            w_h = self.viewport_container.height()
            if w_w <= 0 or w_h <= 0:
                return
            
            active_idx = self.active_view_slider.value() if self.P_list else 0
            
            is_proj_view = self.chk_selected_view.isChecked() and self.P_list
            
            # Reset layout stretch constraints so viewport stretches to fill container
            self.viewport.setMinimumSize(0, 0)
            self.viewport.setMaximumSize(16777215, 16777215)
            
            if is_proj_view:
                orig_W, orig_H = self.P_list[0].image_size
                aspect_target = orig_W / orig_H
                aspect_viewport = w_w / w_h
                if aspect_viewport > aspect_target:
                    target_h = w_h
                    target_w = int(w_h * aspect_target)
                    pad_x = (w_w - target_w) / 2
                    pad_y = 0
                else:
                    target_w = w_w
                    target_h = int(w_w / aspect_target)
                    pad_x = 0
                    pad_y = (w_h - target_h) / 2
                
                P_active = self.P_list[active_idx]
                P_active_P_T = P_active.P @ self.T
                scale_factor = target_w / orig_W
                H_scale = np.array([
                    [scale_factor, 0, 0],
                    [0, scale_factor, 0],
                    [0, 0, 1]
                ], dtype=float)
                
                H_shift = np.array([
                    [1, 0, pad_x],
                    [0, 1, pad_y],
                    [0, 0, 1]
                ], dtype=float)
                
                P_proj = ProjectionMatrix(H_scale @ P_active_P_T, image_size=(target_w, target_h), pixel_spacing=P_active.pixel_spacing / scale_factor)
                self.curr_pad = (pad_x, pad_y, target_w, target_h)
            else:
                self.viewport_container.layout().setAlignment(Qt.AlignmentFlag(0))
                
                T_rot = np.eye(4)
                T_rot[:3, :3] = self.viewport.R_view
                T_view = scale(self.viewport.s) @ T_rot
                
                s_scale = 1.0
                if self.mean_sid > 0:
                    s_scale = 250.0 / (2.0 * self.mean_sid)
                
                T_display = np.eye(4)
                if len(self.P_list) > 0:
                    C_active = self.source_positions_hom[active_idx].flatten()[:3]
                    r = C_active - self.isocenter
                    u_x = r / np.linalg.norm(r)
                    a = self.rotation_axis / np.linalg.norm(self.rotation_axis)
                    u_y = np.cross(a, u_x)
                    u_y /= np.linalg.norm(u_y)
                    u_z = np.cross(u_x, u_y)
                    
                    R_align = np.vstack([u_x, u_y, u_z])
                    T_display[:3, :3] = s_scale * R_align
                    T_display[:3, 3] = -s_scale * (R_align @ self.isocenter)
                    
                P_view = self.viewport.P_display.P @ T_view @ T_display @ self.T
                P_proj = ProjectionMatrix(P_view, image_size=(w_w, w_h), pixel_spacing=1.0)
                self.curr_pad = (0, 0, w_w, w_h)
            
            # 2. Render CUDA Volume Projection
            use_fast_path = self.viewport.is_interacting or not self.chk_svg_overlay.isChecked()
            preview_pil_image = None
            preview_qimage = None
            if self.renderer is not None:
                # Update status bar message with active range of used data
                try:
                    tf_min = float(self.txt_tf_min.text().strip())
                    tf_max = float(self.txt_tf_max.text().strip())
                except ValueError:
                    tf_min = self.orig_vol_min
                    tf_max = self.orig_vol_max
                
                if self.raycast_pass_type == "IsoSurface":
                    s = self.iso_slider.value() / 1000.0
                    v_iso_phys = tf_min + s * (tf_max - tf_min)
                    self.statusBar().showMessage(f"Pass: IsoSurface | Active data range: >= {v_iso_phys:.2f}")
                elif self.raycast_pass_type in ("EmissionAbsorption", "EmissionAbsorptionShaded"):
                    if not np.isnan(self.renderer._tf_min_val) and not np.isnan(self.renderer._tf_max_val):
                        used_min = tf_min + (self.renderer._tf_min_val / 255.0) * (tf_max - tf_min)
                        used_max = tf_min + (self.renderer._tf_max_val / 255.0) * (tf_max - tf_min)
                        self.statusBar().showMessage(f"Pass: {self.raycast_pass_type} | Active data range: {used_min:.2f} to {used_max:.2f}")
                    else:
                        self.statusBar().showMessage(f"Pass: {self.raycast_pass_type} | Active data range: [full range]")
                else:
                    self.statusBar().showMessage(f"Pass: {self.raycast_pass_type}")
                
                img = self.renderer.render(P_proj)
                
                if img.ndim == 3 and img.shape[2] == 4:
                    if self.raycast_pass_type == "IsoSurface":
                        # In C++ vr_raycast_iso, channels 0-2 contain raw voxel coordinates, and channel 3 contains lambertian shading.
                        # Since coordinates can be large index values, we map them to grayscale using channel 3 (shading).
                        lambertian = np.clip(img[..., 3], 0.0, 1.0)
                        disp_img = np.zeros_like(img, dtype=np.uint8)
                        val_8bit = (lambertian * 255.0).astype(np.uint8)
                        disp_img[..., 0] = val_8bit
                        disp_img[..., 1] = val_8bit
                        disp_img[..., 2] = val_8bit
                        disp_img[..., 3] = np.where(lambertian > 1e-5, 255, 0).astype(np.uint8)
                    else:
                        disp_img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
                    
                    height, width, channels = disp_img.shape
                    bytesPerLine = channels * width
                    preview_qimage = QImage(disp_img.data, width, height, bytesPerLine, QImage.Format.Format_RGBA8888).copy()
                    
                    if not use_fast_path:
                        preview_pil_image = Image.fromarray(disp_img, mode='RGBA')
                else:
                    max_val = np.max(img)
                    if max_val > 0:
                        disp_img = (img / max_val * 255.0).astype(np.uint8)
                    else:
                        disp_img = np.zeros_like(img, dtype=np.uint8)
                    
                    height, width = disp_img.shape
                    bytesPerLine = width
                    preview_qimage = QImage(disp_img.data, width, height, bytesPerLine, QImage.Format.Format_Grayscale8).copy()
                    
                    if not use_fast_path:
                        preview_pil_image = Image.fromarray(disp_img, mode='L')
            
            if use_fast_path:
                self.viewport.preview_qimage = preview_qimage
                is_dark = self.raycast_pass_type in ("EmissionAbsorption", "EmissionAbsorptionShaded", "MIP")
                self.viewport.bg_color = QColor(0, 0, 0) if is_dark else QColor(255, 255, 255)
                self.viewport.update()
            else:
                self.viewport.preview_qimage = None
                # 3. Assemble and Render SVG
                svg_obj = self._build_scene_svg(w_w, w_h, P=P_view, preview_pil_image=preview_pil_image)
                if self.chk_svg_overlay.isChecked():
                    self._add_dynamic_elements(
                        svg_obj,
                        P_view=P_view,
                        active_idx=active_idx,
                        show_pyramid=not is_proj_view,
                        show_current_source=not is_proj_view
                    )
                raw_svg = svg_obj.render(P=P_view)
                fixed_svg = fix_8digit_hex_svg(raw_svg)
                
                self.viewport.load(fixed_svg.encode('utf-8'))
            
        except Exception as e:
            self.statusBar().showMessage(f"Viewport rendering error: {str(e)}")

    def apply_local_transform(self, M_local):
        if self.voxel_dimensions is None:
            return
        X, Y, Z = self.voxel_dimensions
        C_voxel = np.array([X / 2.0, Y / 2.0, Z / 2.0, 1.0])
        C_world = self.model_matrix @ C_voxel
        cx, cy, cz = C_world[:3]
        
        T_to_origin = np.array([
            [1.0, 0.0, 0.0, -cx],
            [0.0, 1.0, 0.0, -cy],
            [0.0, 0.0, 1.0, -cz],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        T_from_origin = np.array([
            [1.0, 0.0, 0.0, cx],
            [0.0, 1.0, 0.0, cy],
            [0.0, 0.0, 1.0, cz],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        M_rel = T_from_origin @ M_local @ T_to_origin
        self.T = self.T @ M_rel
        self.render_viewport()

    def axis_center(self):
        if self.voxel_dimensions is None:
            return
        X, Y, Z = self.voxel_dimensions
        C_voxel = np.array([X / 2.0, Y / 2.0, Z / 2.0, 1.0])
        C_world = self.model_matrix @ C_voxel
        
        R = self.T[:3, :3]
        t = -R @ C_world[:3]
        self.T[:3, 3] = t
        self.render_viewport()

    def axis_reset(self):
        self.T = np.eye(4)
        self.render_viewport()

    def axis_swap_xy(self):
        M_local = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.apply_local_transform(M_local)

    def axis_swap_xz(self):
        M_local = np.array([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.apply_local_transform(M_local)

    def axis_swap_yz(self):
        M_local = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.apply_local_transform(M_local)

    def axis_flip_x(self):
        M_local = np.array([
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.apply_local_transform(M_local)

    def axis_flip_y(self):
        M_local = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.apply_local_transform(M_local)

    def axis_flip_z(self):
        M_local = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.apply_local_transform(M_local)

def main():
    app = QApplication(sys.argv)
    
    config_file = None
    nrrd_file = None
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg.endswith(".json"):
            config_file = arg
        elif arg.endswith(".nrrd"):
            nrrd_file = arg
        
    win = VolumeRenderingGUIApp()
    
    if config_file is None and nrrd_file is None:
        config_file, _ = QFileDialog.getOpenFileName(
            win, "Open Reconstruction Config JSON", "", "JSON Configs (*.json);;All Files (*)"
        )
        if not config_file:
            sys.exit(0)
            
    if config_file:
        win.load_config(config_file)
    elif nrrd_file:
        win._load_volume(nrrd_file, use_header_matrix=True)
        win._update_file_info()
        
    win.show()
    QApplication.processEvents()
    win.resize(1101, 850)
    QApplication.processEvents()
    win.resize(1100, 850)
    QApplication.processEvents()
    win.render_viewport()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
