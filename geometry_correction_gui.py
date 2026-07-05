import os
import sys
import json
import numpy as np
import base64
import io
import re
import traceback
import webbrowser
import shutil
import datetime
import subprocess
from copy import deepcopy
from PIL import Image
from PIL import Image as PILImage
import nrrd
from tifffile import imread
import reconstruct



from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QTextEdit, QPushButton, QLabel, QCheckBox, QSlider,
    QSpinBox, QDoubleSpinBox, QStatusBar, QFileDialog, QGroupBox, QScrollArea, QFrame,
    QMessageBox, QListWidget, QComboBox, QTableWidget, QTableWidgetItem, QTabWidget,
    QHeaderView, QListWidgetItem, QInputDialog, QLineEdit, QProgressDialog, QDialog, QMenu
)
from PyQt6.QtSvgWidgets import QSvgWidget
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer, QProcess, QUrl
from PyQt6.QtGui import QFont, QAction, QDesktopServices

# Matplotlib integration
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import NullFormatter

# Import ProjectiveGeometry23 components
from ProjectiveGeometry23.central_projection import ProjectionMatrix
from ProjectiveGeometry23.source_detector_geometry import SourceDetectorGeometry
from ProjectiveGeometry23.svg_utils import svg_homogeneous_line
from ProjectiveGeometry23.utils import hessianNormalForm
from ProjectiveGeometry23 import pluecker
import svg_snip.Elements3D as e3d
from svg_snip.Composer import Composer
from svg_snip.Composer import image as svg_image
from svg_snip.Elements import rect, polyline

# Import ct_recon_fdk_astra components
from ct_recon_fdk_astra.fileformats.ompl import load_ompl, save_ompl
from ct_recon_fdk_astra.gui.reconstruction_gui import ReconstructionGUIApp
from ct_recon_fdk_astra.gui.process_console import AnsiHtmlParser, ProcessConsoleWindow
from ct_recon_fdk_astra.gui.nrrd_view_3d import NrrdView3DWindow
import ct_recon_fdk_astra.gui.nrrd_view_3d as nrrd_view_3d

# Import Epipolar Consistency components
from xray_epipolar_consistency.scan import Scan
import xray_epipolar_consistency as ecc
from xray_epipolar_consistency import ProgressBar
from xray_epipolar_consistency.parameterization import from_dict, ParameterizationChain
from xray_epipolar_consistency.parameterization.detector_shift import DetectorShift
from xray_epipolar_consistency.parameterization.detector_orientation import DetectorOrientation
from xray_epipolar_consistency.parameterization.object_pose import ObjectPose
from xray_epipolar_consistency.parameterization.rotation_axis import RotationAxis
from xray_epipolar_consistency.parameterization.distance import Distance
from xray_epipolar_consistency.parameterization.gantry_angle import GantryAngle
from xray_epipolar_consistency.parameterization.time_variant import LinearDrift, ContinuousMotion, TimeVariant
from xray_epipolar_consistency.parameterization.turntable import Turntable
from xray_epipolar_consistency.parameterization.source_shift_agc import SourceShiftAGC

# Available Parameterizations mapping
AVAILABLE_PARAMS = {
    "Detector Shift": DetectorShift,
    "Detector Orientation": DetectorOrientation,
    "Object Pose": ObjectPose,
    "Rotation Axis": RotationAxis,
    "Distance": Distance,
    "Gantry Angle": GantryAngle,
    "Linear Drift": LinearDrift,
    "Continuous Motion": ContinuousMotion,
    "Turntable": Turntable,
    "Source Shift AGC": SourceShiftAGC,
}


def resolve_relative(config_path, relative_path):
    if not relative_path:
        return ""
    if os.path.isabs(relative_path):
        return relative_path
    if config_path:
        config_dir = os.path.dirname(os.path.abspath(config_path))
    else:
        config_dir = os.getcwd()
    return os.path.normpath(os.path.join(config_dir, relative_path))


def safe_relpath(path, start=None):
    if not path:
        return ""
    path_abs = os.path.abspath(path)
    start_abs = os.path.abspath(start) if start else os.getcwd()
    try:
        rel = os.path.relpath(path_abs, start_abs)
        # Count parent directory traversals
        rel_norm = rel.replace('\\', '/')
        parts = rel_norm.split('/')
        if parts.count('..') > 3:
            return path_abs
        return rel
    except ValueError:
        return path_abs


class QtProgressBarContext:
    """
    Context manager to route xray_epipolar_consistency progress updates to QProgressDialog.
    """
    def __init__(self, parent, title="Progress"):
        self.parent = parent
        self.title = title
        self.dialog = None

    def __enter__(self):
        ProgressBar.set_callback(self.on_progress)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        ProgressBar.set_callback(None)
        if self.dialog is not None:
            self.dialog.close()
            self.dialog = None

    def on_progress(self, current, total, desc):
        if self.dialog is None:
            self.dialog = QProgressDialog(desc, "Cancel", 0, total if total else 100, self.parent)
            self.dialog.setWindowTitle(self.title)
            self.dialog.setWindowModality(Qt.WindowModality.WindowModal)
            self.dialog.setMinimumDuration(300)
            self.dialog.show()
        else:
            if desc:
                self.dialog.setLabelText(desc)
            if total:
                self.dialog.setMaximum(total)
            self.dialog.setValue(current)
        
        QApplication.processEvents()
        if self.dialog.wasCanceled():
            raise KeyboardInterrupt("User canceled operation")


# Thread & Buffer helper classes removed. ProcessConsoleWindow is used instead.


from ct_recon_fdk_astra.gui.reconstruction_gui import ReconstructionGUIApp

class InteractiveReconstructionGUI(ReconstructionGUIApp):
    def __init__(self, config_path, parent_window, edit_config_only=False):
        self.parent_window = parent_window
        super().__init__(config_path, edit_config_only=edit_config_only)
        
    def closeEvent(self, event):
        super().closeEvent(event)
        try:
            if self.current_config_path and os.path.exists(self.current_config_path):
                self.parent_window._skip_load_dialog = True
                self.parent_window.load_config_file(self.current_config_path)
        except Exception as e:
            print(f"Error reloading config in parent window: {e}")
        self.parent_window.show()


class ReapplyParameterizationDialog(QDialog):
    """Dialog showing loaded parameters with non-zero values, with Cancel and Apply buttons."""
    def __init__(self, parent, non_zero_params):
        super().__init__(parent)
        self.setWindowTitle("Review Loaded Parameters")
        self.setMinimumSize(450, 300)
        
        layout = QVBoxLayout(self)
        
        layout.addWidget(QLabel("<b>The following non-zero parameters will be applied to the loaded trajectory:</b>"))
        
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Parameter Name", "Value"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        self.table.setRowCount(len(non_zero_params))
        for idx, (name, val) in enumerate(non_zero_params):
            item_name = QTableWidgetItem(name)
            if isinstance(val, (list, np.ndarray, tuple)):
                formatted_val = ", ".join(f"{v:.4f}" for v in val)
                item_val = QTableWidgetItem(f"[{formatted_val}]")
            else:
                item_val = QTableWidgetItem(f"{val:.6f}")
                
            for item in (item_name, item_val):
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                
            self.table.setItem(idx, 0, item_name)
            self.table.setItem(idx, 1, item_val)
            
        layout.addWidget(self.table)
        
        btn_layout = QHBoxLayout()
        
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_cancel)
        
        btn_apply = QPushButton("Apply")
        btn_apply.setDefault(True)
        btn_apply.clicked.connect(self.accept)
        btn_layout.addWidget(btn_apply)
        
        layout.addLayout(btn_layout)


class CompareReconstructionsDialog(QDialog):
    def __init__(self, parent, nrrd_filenames):
        super().__init__(parent)
        self.setWindowTitle("Compare Reconstructions")
        self.setMinimumWidth(400)
        layout = QVBoxLayout(self)
        
        layout.addWidget(QLabel("<b>Select .nrrd files to compare in NrrdView3D:</b>"))
        
        # Scroll area in case there are many files
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(5, 5, 5, 5)
        
        self.checkboxes = {}
        for name in nrrd_filenames:
            chk = QCheckBox(name)
            chk.setChecked(False)
            scroll_layout.addWidget(chk)
            self.checkboxes[name] = chk
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll, 1)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_cancel)
        
        btn_ok = QPushButton("OK")
        btn_ok.setDefault(True)
        btn_ok.clicked.connect(self.accept)
        btn_layout.addWidget(btn_ok)
        
        layout.addLayout(btn_layout)
        
    def get_selected_files(self):
        return [name for name, chk in self.checkboxes.items() if chk.isChecked()]


from PyQt6.QtWidgets import QProgressBar
import io

class StdoutRedirector(io.TextIOBase):
    def __init__(self, signal):
        super().__init__()
        self.signal = signal
    def write(self, text):
        if text.strip():
            self.signal.emit(text)
        return len(text)
    def flush(self):
        pass

class OptimizationThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(object) # passes the result dict or exception
    
    def __init__(self, scan, stage_dict, metric_config):
        super().__init__()
        self.scan = scan
        self.stage_dict = stage_dict
        self.metric_config = metric_config
        self.calib = None
        
    def run(self):
        # Redirect stdout to capture prints from CalibrationAndMotionCorrection
        old_stdout = sys.stdout
        sys.stdout = StdoutRedirector(self.log_signal)
        
        try:
            # We copy the scan's images and projection matrices to make sure we don't mess with them until confirmed
            # Let's clone the projection matrices
            Ps_copy = [ProjectionMatrix(P.P.copy(), P.image_size, P.pixel_spacing) for P in self.scan.Ps]
            # Images are read-only numpy arrays, so reference is fine
            scan_copy = Scan(self.scan.Is, Ps_copy)
            
            calib = ecc.CalibrationAndMotionCorrection(
                Is=scan_copy.Is,
                Ps=scan_copy.Ps,
                stages=[self.stage_dict],
                metric_config=self.metric_config
            )
            self.calib = calib
            result = calib.optimize()
            self.finished_signal.emit(result)
        except Exception as e:
            traceback.print_exc()
            # If the user cancelled, we can build the best results dictionary
            if self.calib and hasattr(self.calib, "current_problem") and self.calib.current_problem:
                prob = self.calib.current_problem
                if prob.is_cancelled and prob.best_parameters is not None:
                    # Construct a result dictionary with the best parameters found so far
                    best_chain = deepcopy(prob.parameterization)
                    best_chain.set_parameter_vector(prob.best_parameters)
                    
                    image_sizes = [P.image_size.copy() for P in Ps_copy]
                    pixel_spacings = [P.pixel_spacing for P in Ps_copy]
                    Ps_initial_pm = [
                        ProjectionMatrix(P_arr.P.copy(), img_sz, px_sp)
                        for P_arr, img_sz, px_sp in zip(Ps_copy, image_sizes, pixel_spacings)
                    ]
                    Ps_optimized = best_chain.apply_to_trajectory(Ps_initial_pm)
                    
                    result = {
                        "is_cancelled": True,
                        "optimized_parameterization": best_chain.to_dict(),
                        "cost_history": prob.cost_function_values,
                        "optimization_time_sec": 0.0,
                        "Ps_optimized": [P.P.tolist() for P in Ps_optimized],
                        "stages": [{
                            "name": self.stage_dict["name"],
                            "optimizer": self.stage_dict["classname"],
                            "final_cost": prob.best_cost,
                            "parameter_vector": list(prob.best_parameters),
                            "parameters": {k: best_chain[k]["value"] for k in best_chain},
                        }]
                    }
                    self.finished_signal.emit(result)
                    return
            self.finished_signal.emit(e)
        finally:
            sys.stdout = old_stdout

class OptimizationProgressDialog(QDialog):
    def __init__(self, parent, scan, stage, metric_config):
        super().__init__(parent)
        self.setWindowTitle("Optimizing Stage...")
        self.setMinimumSize(600, 400)
        self.setModal(True)
        
        self.stage = stage
        self.scan = scan
        self.optimization_result = None
        
        # Import and instantiate ANSI HTML Parser
        from ct_recon_fdk_astra.gui.process_console import AnsiHtmlParser
        self.ansi_parser = AnsiHtmlParser()
        
        # Layout
        layout = QVBoxLayout(self)
        
        self.lbl_status = QLabel(f"Running optimization for stage '{stage.name}'...")
        layout.addWidget(self.lbl_status)
        
        # Busy progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        layout.addWidget(self.progress_bar)
        
        # Console output TextEdit
        self.txt_console = QTextEdit()
        self.txt_console.setReadOnly(True)
        self.txt_console.setFont(QFont("Courier New", 10))
        layout.addWidget(self.txt_console)
        
        # Buttons (Cancel/Discard and Apply)
        btn_layout = QHBoxLayout()
        
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.cancel_optimization)
        btn_layout.addWidget(self.btn_cancel)
        
        self.btn_apply = QPushButton("Apply")
        self.btn_apply.setEnabled(False)
        self.btn_apply.clicked.connect(self.accept)
        btn_layout.addWidget(self.btn_apply)
        
        layout.addLayout(btn_layout)
        
        # Prepare Stage Dict
        chain = ParameterizationChain(stage.parameterizations)
        self.stage_dict = {
            "name": stage.name,
            "module": "xray_epipolar_consistency.optimizer",
            "classname": stage.optimizer,
            "parameterization": chain.to_dict(),
            "kwargs": {
                "options": {
                    "maxiter": stage.maxiter,
                    "ftol": stage.ftol
                }
            }
        }
        
        # Thread setup
        self.thread = OptimizationThread(scan, self.stage_dict, metric_config)
        self.thread.log_signal.connect(self.append_log)
        self.thread.finished_signal.connect(self.on_finished)
        
    def start(self):
        self.thread.start()
        
    def append_log(self, text):
        self.txt_console.moveCursor(self.txt_console.textCursor().MoveOperation.End)
        html_text = self.ansi_parser.parse(text)
        self.txt_console.insertHtml(html_text)
        self.txt_console.moveCursor(self.txt_console.textCursor().MoveOperation.End)
        
    def closeEvent(self, event):
        if self.thread.isRunning():
            self.cancel_optimization()
            event.ignore()
        else:
            super().closeEvent(event)

    def cancel_optimization(self):
        if self.thread.isRunning():
            self.lbl_status.setText("Cancelling optimization...")
            if self.thread.calib and hasattr(self.thread.calib, "current_problem") and self.thread.calib.current_problem:
                self.thread.calib.current_problem.is_cancelled = True
        else:
            self.reject()
        
    def on_finished(self, result):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        
        if isinstance(result, Exception):
            self.lbl_status.setText("Optimization failed.")
            self.btn_cancel.setText("Close")
            # If user clicked Cancel, the exception is likely the "cancelled by user" RuntimeError
            if self.thread.calib and hasattr(self.thread.calib, "current_problem") and self.thread.calib.current_problem:
                if self.thread.calib.current_problem.is_cancelled:
                    self.reject()
                    return
            QMessageBox.critical(self, "Optimization Error", f"An error occurred during optimization:\n{result}")
            return
            
        if result.get("is_cancelled", False):
            self.lbl_status.setText("Optimization cancelled.")
            initial_cost = result["cost_history"][0] if result["cost_history"] else 0.0
            best_cost = result["stages"][-1]["final_cost"]
            
            reply = QMessageBox.question(
                self,
                "Apply Best Parameters?",
                f"Optimization was cancelled by the user.\n\n"
                f"Initial Cost: {initial_cost:.6f}\n"
                f"Best Cost Found: {best_cost:.6f}\n\n"
                f"Would you like to apply the best parameters found so far to the current state?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.optimization_result = result
                self.accept()
            else:
                self.reject()
            return
            
        self.lbl_status.setText("Optimization finished.")
        self.btn_cancel.setText("Discard")
        self.btn_apply.setEnabled(True)
        self.btn_apply.setDefault(True)
        self.optimization_result = result
        
        # Report the result of optimization
        initial_cost = result["cost_history"][0]
        final_cost = result["cost_history"][-1]
        time_taken = result.get("optimization_time_sec", 0.0)
        
        # Extract and group optimized parameter values
        param_lines = []
        if "stages" in result and len(result["stages"]) > 0:
            last_stage = result["stages"][-1]
            opt_params = last_stage.get("parameters", {})
            
            # Parse parameters to group _cpN suffixes
            grouped_params = {} # prefix -> {index: value}
            single_params = {}  # name -> value
            
            for p_name, p_val in opt_params.items():
                match = re.match(r'^(.*?)(?:_cp(\d+))$', p_name)
                if match:
                    prefix = match.group(1)
                    idx = int(match.group(2))
                    if prefix not in grouped_params:
                        grouped_params[prefix] = {}
                    grouped_params[prefix][idx] = p_val
                else:
                    single_params[p_name] = p_val
            
            # Assemble entries
            display_list = []
            for p_name, p_val in single_params.items():
                if isinstance(p_val, (list, np.ndarray, tuple)):
                    formatted_val = ", ".join(f"{v:.4f}" for v in p_val)
                    formatted_val = f"[{formatted_val}]"
                else:
                    formatted_val = f"{p_val:.6f}"
                display_list.append((p_name, formatted_val))
                
            for prefix, idx_map in grouped_params.items():
                sorted_indices = sorted(idx_map.keys())
                vector_vals = [idx_map[i] for i in sorted_indices]
                formatted_val = ", ".join(f"{v:.4f}" for v in vector_vals)
                display_list.append((prefix, f"[{formatted_val}]"))
                
            # Sort alphabetically by parameter/prefix name
            display_list.sort(key=lambda item: item[0])
            for name, formatted_val in display_list:
                param_lines.append(f"  • {name}: {formatted_val}")
                
        params_str = "\n".join(param_lines) if param_lines else "  (No parameters)"
        
        summary = (
            f"\n\n========================================\n"
            f"Optimization of Stage '{self.stage.name}' completed successfully!\n\n"
            f"Optimized Parameters:\n{params_str}\n\n"
            f"ECC Cost: {initial_cost:.6e} → {final_cost:.6e}\n"
            f"Time Taken: {time_taken:.2f} seconds\n"
            f"========================================\n"
        )
        self.append_log(summary)


class AutoConfigureResultDialog(QDialog):
    """Dialog showing the sorted table of enabled parameters with relative distances to minimum,
    and a button to open the HTML report."""
    def __init__(self, parent, sorted_enabled_results, html_path):
        super().__init__(parent)
        self.setWindowTitle("Auto Configure Results")
        self.setMinimumSize(550, 400)
        self.html_path = html_path
        
        layout = QVBoxLayout(self)
        
        layout.addWidget(QLabel("<b>Auto Configure completed. Sorted enabled parameters:</b>"))
        
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Parameter", "Optimal Value", "Rel. Distance from 0"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        self.table.setRowCount(len(sorted_enabled_results))
        for idx, r in enumerate(sorted_enabled_results):
            item_name = QTableWidgetItem(r["display_name"])
            item_val = QTableWidgetItem(f"{r['min_val']:.6f}")
            item_dist = QTableWidgetItem(f"{r['rel_dist']:.6f}")
            
            for item in (item_name, item_val, item_dist):
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                
            self.table.setItem(idx, 0, item_name)
            self.table.setItem(idx, 1, item_val)
            self.table.setItem(idx, 2, item_dist)
            
        layout.addWidget(self.table)
        
        btn_layout = QHBoxLayout()
        
        btn_open = QPushButton("Open HTML report")
        btn_open.clicked.connect(self.open_report)
        btn_layout.addWidget(btn_open)
        
        btn_ok = QPushButton("OK")
        btn_ok.setDefault(True)
        btn_ok.clicked.connect(self.accept)
        btn_layout.addWidget(btn_ok)
        
        layout.addLayout(btn_layout)
        
    def open_report(self):
        webbrowser.open("file://" + os.path.abspath(self.html_path))


class AnalysisParametersDialog(QDialog):
    """Dialog to query user for analysis parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Analysis Parameters")
        self.setMinimumWidth(420)
        
        # Local import of QFormLayout to be safe
        from PyQt6.QtWidgets import QFormLayout
        
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        
        # d_max
        self.spin_d_max = QDoubleSpinBox()
        self.spin_d_max.setRange(0.1, 1000.0)
        self.spin_d_max.setValue(10.0)
        self.spin_d_max.setDecimals(2)
        form_layout.addRow("Expected Max Detector Motion (d_max, px):", self.spin_d_max)
        
        # num_samples
        self.spin_num_samples = QSpinBox()
        self.spin_num_samples.setRange(10, 1000000)
        self.spin_num_samples.setValue(1000)
        self.spin_num_samples.setSingleStep(100)
        form_layout.addRow("Monte-Carlo Samples (num_samples):", self.spin_num_samples)
        
        # seed
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 999999)
        self.spin_seed.setValue(42)
        form_layout.addRow("Random Seed (seed):", self.spin_seed)
        
        # delta_p
        self.txt_delta_p = QLineEdit("1e-5")
        form_layout.addRow("Derivative Perturbation (delta_p):", self.txt_delta_p)
        
        # max_epipolar_views
        self.spin_max_views = QSpinBox()
        self.spin_max_views.setRange(2, 10000)
        self.spin_max_views.setValue(100)
        form_layout.addRow("Max Epipolar Views (max_epipolar_views):", self.spin_max_views)
        
        # gap_threshold
        self.spin_gap_threshold = QDoubleSpinBox()
        self.spin_gap_threshold.setRange(1.0, 100000.0)
        self.spin_gap_threshold.setValue(100.0)
        self.spin_gap_threshold.setDecimals(1)
        form_layout.addRow("Sloppy Gap Threshold (gap_threshold):", self.spin_gap_threshold)
        
        layout.addLayout(form_layout)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_ok = QPushButton("OK")
        btn_ok.clicked.connect(self.validate_and_accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_ok)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)
        
    def validate_and_accept(self):
        # Validate that delta_p is a valid float
        val_str = self.txt_delta_p.text().strip()
        try:
            float(val_str)
        except ValueError:
            QMessageBox.critical(self, "Invalid Value", "Please enter a valid numeric value (e.g. 1e-5 or 0.00001) for delta_p.")
            return
        self.accept()
        
    def get_values(self):
        return {
            "d_max": self.spin_d_max.value(),
            "num_samples": self.spin_num_samples.value(),
            "seed": self.spin_seed.value(),
            "delta_p": self.txt_delta_p.text().strip(),
            "max_epipolar_views": self.spin_max_views.value(),
            "gap_threshold": self.spin_gap_threshold.value()
        }


class AnalysisResultsDialog(QDialog):
    """Dialog showing the outputs of geometric identifiability analysis,
    with options to open directories/reports and apply recommendations."""
    def __init__(self, parent, out_dir, html_path1, html_path2, suggested_json, abs_path):
        super().__init__(parent)
        self.setWindowTitle("Analysis Results")
        self.setMinimumWidth(500)
        self.out_dir = out_dir
        self.html_path1 = html_path1
        self.html_path2 = html_path2
        self.suggested_json = suggested_json
        self.abs_path = abs_path
        
        layout = QVBoxLayout(self)
        
        label_title = QLabel("<h3>Analysis Completed Successfully</h3>")
        layout.addWidget(label_title)
        
        label_dir = QLabel(f"Output Directory:<br><code>{out_dir}</code>")
        label_dir.setWordWrap(True)
        label_dir.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(label_dir)
        
        layout.addSpacing(10)
        
        # Clickable buttons to open
        btn_open_dir = QPushButton("Open Output Directory")
        btn_open_dir.setStyleSheet("font-weight: bold; padding: 6px;")
        btn_open_dir.clicked.connect(self.open_dir_action)
        layout.addWidget(btn_open_dir)
        
        btn_open_html1 = QPushButton("Open Identifiability Report in Browser")
        btn_open_html1.setStyleSheet("padding: 6px;")
        btn_open_html1.clicked.connect(self.open_html1_action)
        layout.addWidget(btn_open_html1)
        
        btn_open_html2 = QPushButton("Open Optimization Advisor Report in Browser")
        btn_open_html2.setStyleSheet("padding: 6px;")
        btn_open_html2.clicked.connect(self.open_html2_action)
        layout.addWidget(btn_open_html2)
        
        layout.addSpacing(15)
        
        # Apply suggested config button
        btn_apply = QPushButton("Apply Suggested Configuration to Stage")
        btn_apply.setStyleSheet("font-weight: bold; background-color: #0284c7; color: white; padding: 8px;")
        btn_apply.clicked.connect(self.apply_config_action)
        layout.addWidget(btn_apply)
        
        layout.addSpacing(10)
        
        # Close button
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)

    def open_dir_action(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.out_dir))

    def open_html1_action(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.html_path1))

    def open_html2_action(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.html_path2))

    def apply_config_action(self):
        try:
            if not os.path.exists(self.suggested_json):
                QMessageBox.warning(self, "Apply Config Error", "Suggested configuration JSON file not found.")
                return
            
            # Read suggested config
            with open(self.suggested_json, 'r') as f:
                suggested_data = json.load(f)
                
            # Overwrite original stage config file
            with open(self.abs_path, 'w') as f:
                json.dump(suggested_data, f, indent=2)
                
            # Reload stage in GUI
            if self.parent():
                stage_obj = self.parent().load_stage_file(self.abs_path)
                if stage_obj:
                    self.parent().stages_cache[self.abs_path] = stage_obj
                    self.parent().select_stage(self.parent().list_stages.currentRow())
                    self.parent().update_json_preview()
                    
            QMessageBox.information(self, "Success", "Suggested configuration successfully applied and reloaded!")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Apply Config Error", f"Failed to apply suggested config:\n{e}")


class SweepWindow(QMainWindow):
    def __init__(self, parent=None, title="1D Consistency Sweep Plot"):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Window)
        self.setWindowTitle(title)
        self.resize(600, 450)
        
        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        layout.addWidget(self.canvas)
        
        # Menu Bar
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        
        save_svg_action = file_menu.addAction("Save as SVG...")
        save_svg_action.triggered.connect(self.save_as_svg)
        
        save_png_action = file_menu.addAction("Save as PNG...")
        save_png_action.triggered.connect(self.save_as_png)
        
        save_pdf_action = file_menu.addAction("Save as PDF...")
        save_pdf_action.triggered.connect(self.save_as_pdf)
        
    def save_as_svg(self):
        self.save_plot("SVG Files (*.svg)", "svg")
        
    def save_as_png(self):
        self.save_plot("PNG Files (*.png)", "png")
        
    def save_as_pdf(self):
        self.save_plot("PDF Files (*.pdf)", "pdf")
        
    def save_plot(self, file_filter, ext):
        file_path, _ = QFileDialog.getSaveFileName(self, f"Save Plot as {ext.upper()}", f"sweep_plot_1d.{ext}", file_filter)
        if file_path:
            try:
                self.fig.savefig(file_path, format=ext, bbox_inches='tight')
                self.statusBar().showMessage(f"Plot saved successfully to {file_path}", 5000)
            except Exception as e:
                QMessageBox.critical(self, "Error Saving Plot", f"Failed to save plot:\n{e}")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)


class Sweep2DWindow(QMainWindow):
    def __init__(self, parent=None, param_name1="", param_name2=""):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Window)
        self.setWindowTitle(f"2D Consistency Sweep Plot: {param_name1} vs {param_name2}")
        self.resize(700, 600)
        
        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection='3d')
        layout.addWidget(self.canvas)
        
        # Menu Bar
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        
        save_svg_action = file_menu.addAction("Save as SVG...")
        save_svg_action.triggered.connect(self.save_as_svg)
        
        save_png_action = file_menu.addAction("Save as PNG...")
        save_png_action.triggered.connect(self.save_as_png)
        
        save_pdf_action = file_menu.addAction("Save as PDF...")
        save_pdf_action.triggered.connect(self.save_as_pdf)
        
    def plot_surface(self, X, Y, Z, name1, name2, original_val1, original_val2):
        self.ax.clear()
        # Draw 3D surface
        self.ax.plot_surface(X, Y, Z, cmap='coolwarm', edgecolor='none', alpha=0.9)
        
        # Find current cost to plot optimized point
        current_cost = Z[np.abs(Y[:, 0] - original_val2).argmin(), np.abs(X[0, :] - original_val1).argmin()]
        self.ax.scatter([original_val1], [original_val2], [current_cost], color='red', s=100, marker='*', zorder=10, label='Optimized')
        
        self.ax.zaxis.set_major_formatter(NullFormatter())
        self.ax.set_xlabel(name1)
        self.ax.set_ylabel(name2)
        self.ax.set_zlabel('consistency [a.u.]')
        self.ax.set_title(f"ECC Dependency: {name1} vs {name2}")
        self.ax.legend()
        self.canvas.draw()
        
    def save_as_svg(self):
        self.save_plot("SVG Files (*.svg)", "svg")
        
    def save_as_png(self):
        self.save_plot("PNG Files (*.png)", "png")
        
    def save_as_pdf(self):
        self.save_plot("PDF Files (*.pdf)", "pdf")
        
    def save_plot(self, file_filter, ext):
        file_path, _ = QFileDialog.getSaveFileName(self, f"Save Plot as {ext.upper()}", f"sweep_plot_2d.{ext}", file_filter)
        if file_path:
            try:
                self.fig.savefig(file_path, format=ext, bbox_inches='tight')
                self.statusBar().showMessage(f"Plot saved successfully to {file_path}", 5000)
            except Exception as e:
                QMessageBox.critical(self, "Error Saving Plot", f"Failed to save plot:\n{e}")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)



class Stage:
    def __init__(self, name="Optimization Stage", optimizer="OptimizerPowell", maxiter=200, ftol=1e-6, eps=1e-3):
        self.name = name
        self.optimizer = optimizer
        self.maxiter = maxiter
        self.ftol = ftol
        self.eps = eps
        self.parameterizations = [] # List of instantiated parameterization objects


class InteractiveSvgWidget(QSvgWidget):
    hover_signal = pyqtSignal(float, float)  # (u, v) — mouse move
    click_signal = pyqtSignal(float, float)  # (u, v) — mouse press
    leave_signal = pyqtSignal()              # mouse left the widget

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_size = (1, 1)
        self.setMouseTracking(True)

    def setImageSize(self, w, h):
        self.image_size = (w, h)

    def _img_coords(self, event):
        w_width, w_height = self.width(), self.height()
        img_w, img_h = self.image_size
        if w_width > 0 and w_height > 0:
            pos = event.position()
            return pos.x() * (img_w / w_width), pos.y() * (img_h / w_height)
        return None, None

    def mouseMoveEvent(self, event):
        ix, iy = self._img_coords(event)
        if ix is not None:
            self.hover_signal.emit(ix, iy)
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            ix, iy = self._img_coords(event)
            if ix is not None:
                self.click_signal.emit(ix, iy)
        super().mousePressEvent(event)

    def leaveEvent(self, event):
        self.leave_signal.emit()
        super().leaveEvent(event)



class AspectWrapper(QWidget):
    def __init__(self, child_widget, aspect_ratio=1.0, parent=None):
        super().__init__(parent)
        self.child = child_widget
        self.child.setParent(self)
        self.aspect_ratio = aspect_ratio
        
    def setAspectRatio(self, aspect_ratio):
        self.aspect_ratio = aspect_ratio
        self.update_geometry()
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_geometry()
        
    def update_geometry(self):
        if not self.child:
            return
        w = self.width()
        h = self.height()
        if w <= 0 or h <= 0:
            return
        current_aspect = w / h
        if current_aspect > self.aspect_ratio:
            new_h = h
            new_w = int(h * self.aspect_ratio)
        else:
            new_w = w
            new_h = int(w / self.aspect_ratio)
        x = (w - new_w) // 2
        y = (h - new_h) // 2
        self.child.setGeometry(x, y, new_w, new_h)


def get_line_endpoints(l, W, H):
    # l is [a, b, c] such that a*u + b*v + c = 0
    a, b, c = l
    points = []
    
    # 1. Left boundary: u = 0
    if abs(b) > 1e-9:
        v = -c / b
        if 0 <= v <= H:
            points.append((0, v))
            
    # 2. Right boundary: u = W
    if abs(b) > 1e-9:
        v = -(c + a * W) / b
        if 0 <= v <= H:
            points.append((W, v))
            
    # 3. Top boundary: v = 0
    if abs(a) > 1e-9:
        u = -c / a
        if 0 <= u <= W:
            points.append((u, 0))
            
    # 4. Bottom boundary: v = H
    if abs(a) > 1e-9:
        u = -(c + b * H) / a
        if 0 <= u <= W:
            points.append((u, H))
            
    # Remove duplicate points if any (e.g. at corners)
    unique_points = []
    for p in points:
        if not any(np.allclose(p, up, atol=1e-5) for up in unique_points):
            unique_points.append(p)
            
    if len(unique_points) >= 2:
        return unique_points[0], unique_points[1]
    return None


def _peek_image_size(image_path):
    """Return (W, H) by reading only the file header — no full pixel data loaded."""
    path = str(image_path)
    try:
        if path.lower().endswith('.nrrd'):
            _, hdr = nrrd.read(path, index_order='C')
            # NRRD sizes field: (W, H, [slices])
            sizes = hdr.get('sizes', [0, 0])
            return int(sizes[0]), int(sizes[1])
        else:
            with PILImage.open(path) as img:
                return img.size  # (W, H)
    except Exception:
        return None


class ConfigLoadDialog(QDialog):
    """Dialog shown when a reconstruction config is opened.

    Reads only image *count* and first-image *dimensions* via a cheap header
    peek — no actual pixel data is loaded.  The user can set:
      - undersampling factor      (live "N → M views" hint)
      - Radon dtr size factor     (live size_t / size_α + memory estimate)
      - Gaussian sigma            (pre-smoothing applied before Radon transform)
    A warning appears next to the dtr spinbox when the estimated GPU memory
    for all RadonIntermediate buffers exceeds 4 GB.
    """

    _WARN_GB = 4.0

    def __init__(self, parent, *, recon_config, recon_config_path,
                 current_undersample=10, current_dtr_factor=1.0,
                 current_gaussian_sigma=None):
        super().__init__(parent)
        self.setWindowTitle("Dataset Load Settings")
        self.setModal(True)
        self.setMinimumWidth(500)

        # ---- resolve image list ----
        config_dir = os.path.dirname(os.path.abspath(recon_config_path))
        data_dir = recon_config.get("data_dir", "./")
        if not os.path.isabs(data_dir):
            data_dir = os.path.normpath(os.path.join(config_dir, data_dir))

        image_files = recon_config.get("image_files", [])
        self._total_views = len(image_files)

        # Resolve first image path for size peek
        self._img_w = self._img_h = None
        is_stack = (self._total_views == 1
                    and image_files
                    and image_files[0].lower().endswith('.nrrd'))

        if image_files:
            first = image_files[0]
            first_abs = first if os.path.isabs(first) else os.path.join(data_dir, first)
            if is_stack:
                try:
                    _, hdr = nrrd.read(first_abs, index_order='C')
                    sizes = hdr.get('sizes', [0, 0, 0])
                    self._img_w  = int(sizes[0])
                    self._img_h  = int(sizes[1])
                    self._total_views = int(sizes[2]) if len(sizes) > 2 else 1
                except Exception:
                    pass
            else:
                result = _peek_image_size(first_abs)
                if result:
                    self._img_w, self._img_h = result

        # ---- build UI ----
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(18, 14, 18, 14)

        # Dataset info (static, read-only)
        info_grp = QGroupBox("Dataset Info")
        info_lay = QVBoxLayout(info_grp)
        views_txt = str(self._total_views) if self._total_views else "unknown"
        size_txt  = (f"{self._img_w} × {self._img_h} px"
                     if (self._img_w and self._img_h) else "unknown")
        info_lay.addWidget(QLabel(f"<b>Total views:</b> {views_txt}"))
        info_lay.addWidget(QLabel(f"<b>Image size:</b>  {size_txt}"))
        layout.addWidget(info_grp)

        # Undersampling
        under_grp = QGroupBox("Undersampling")
        under_lay = QVBoxLayout(under_grp)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Use every N-th frame  (N =)"))
        self._spin_us = QSpinBox()
        self._spin_us.setRange(1, max(10000, self._total_views or 10000))
        self._spin_us.setValue(current_undersample)
        row1.addWidget(self._spin_us)
        self._lbl_us_hint = QLabel()
        row1.addWidget(self._lbl_us_hint)
        row1.addStretch()
        under_lay.addLayout(row1)
        layout.addWidget(under_grp)

        # DTR settings
        dtr_grp = QGroupBox("Radon Intermediate (dtr) Settings")
        dtr_lay = QVBoxLayout(dtr_grp)

        # dtr size factor + inline warning
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("dtr size factor:"))
        self._spin_dtr = QDoubleSpinBox()
        self._spin_dtr.setRange(0.01, 5.0)
        self._spin_dtr.setSingleStep(0.1)
        self._spin_dtr.setDecimals(2)
        self._spin_dtr.setValue(current_dtr_factor)
        row2.addWidget(self._spin_dtr)
        self._lbl_mem_warn = QLabel()   # ⚠️ warning, shown only when > 4 GB
        row2.addWidget(self._lbl_mem_warn)
        row2.addStretch()
        dtr_lay.addLayout(row2)

        # Gaussian sigma
        self._initializing = True
        self._user_toggled_gaussian = False

        if current_gaussian_sigma is not None:
            if current_gaussian_sigma > 0.0:
                is_checked = True
                sigma_val = current_gaussian_sigma
            else:
                is_checked = False
                sigma_val = 1.2
        else:
            if abs(current_dtr_factor - 1.0) < 1e-5:
                is_checked = False
                sigma_val = 1.2
            else:
                is_checked = True
                sigma_val = 1.2

        # If current_dtr_factor is 1.0, override is_checked to False
        if abs(current_dtr_factor - 1.0) < 1e-5:
            is_checked = False

        row3 = QHBoxLayout()
        self._chk_gaussian = QCheckBox("Include Gaussian pre-filter")
        self._chk_gaussian.setChecked(is_checked)
        self._chk_gaussian.setEnabled(abs(current_dtr_factor - 1.0) >= 1e-5)
        self._chk_gaussian.clicked.connect(self._on_gaussian_clicked)
        row3.addWidget(self._chk_gaussian)

        self._lbl_sigma = QLabel("σ:")
        row3.addWidget(self._lbl_sigma)

        self._spin_sigma = QDoubleSpinBox()
        self._spin_sigma.setRange(0.1, 20.0)
        self._spin_sigma.setSingleStep(0.1)
        self._spin_sigma.setDecimals(2)
        self._spin_sigma.setValue(sigma_val)
        self._spin_sigma.setEnabled(is_checked)
        self._spin_sigma.setToolTip(
            "Gaussian smoothing applied to each projection image\n"
            "before computing the Radon transform.\n"
            "Larger σ → smoother dtr, no undersampling, but blurs detail.\n"
            "Typical range: 0.8 – 2.4."
        )
        row3.addWidget(self._spin_sigma)
        row3.addStretch()
        dtr_lay.addLayout(row3)

        self._chk_gaussian.toggled.connect(self._spin_sigma.setEnabled)

        # Live size + memory estimate
        self._lbl_dtr_size = QLabel()
        dtr_lay.addWidget(self._lbl_dtr_size)
        layout.addWidget(dtr_grp)

        # OK / Cancel
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_ok = QPushButton("Load")
        btn_ok.setDefault(True)
        btn_cancel = QPushButton("Cancel")
        btn_row.addWidget(btn_ok)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)

        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)

        # Connect live updates
        self._spin_us.valueChanged.connect(self._update_hints)
        self._spin_dtr.valueChanged.connect(self._update_hints)
        self._update_hints()
        self._initializing = False

    def _on_gaussian_clicked(self):
        self._user_toggled_gaussian = True

    def _update_hints(self):
        U = self._spin_us.value()
        n_views = 0
        if self._total_views:
            n_views = (self._total_views + U - 1) // U
            self._lbl_us_hint.setText(
                f"→ <b>{n_views}</b> of <b>{self._total_views}</b> views"
                + (f"  (every {U}th)" if U > 1 else "  (all frames)")
            )
        else:
            self._lbl_us_hint.setText("")

        f = self._spin_dtr.value()
        is_one = abs(f - 1.0) < 1e-5
        self._chk_gaussian.setEnabled(not is_one)
        if is_one:
            self._chk_gaussian.setChecked(False)
        else:
            if not getattr(self, '_initializing', False) and not getattr(self, '_user_toggled_gaussian', False):
                self._chk_gaussian.setChecked(True)

        if self._img_w and self._img_h:
            diag       = (self._img_w ** 2 + self._img_h ** 2) ** 0.5
            size_t     = int(diag * f)
            size_alpha = int((3.14159265 / 2.0 * size_t + 1) // 2)
            floats_per_view = size_t * size_alpha
            total_bytes = floats_per_view * 4 * max(1, n_views)  # float32
            total_gb    = total_bytes / 1024**3

            mem_str = (f"{total_gb:.2f} GB" if total_gb >= 0.1
                       else f"{total_bytes / 1024**2:.1f} MB")
            self._lbl_dtr_size.setText(
                f"dtr buffer:  size_t = <b>{size_t}</b>,  "
                f"size_α = <b>{size_alpha}</b>  ·  "
                f"<b>{floats_per_view // 1024} K</b> floats/view  ·  "
                f"total GPU memory: <b>{mem_str}</b>"
            )

            if total_gb > self._WARN_GB:
                self._lbl_mem_warn.setText(
                    f"<span style='color:#e05000;font-weight:bold;'>"
                    f"⚠ {total_gb:.1f} GB — consider increasing undersampling "
                    f"or reducing the size factor</span>"
                )
            else:
                self._lbl_mem_warn.clear()
        else:
            self._lbl_dtr_size.setText(
                "Image dimensions unknown — dtr size cannot be estimated."
            )
            self._lbl_mem_warn.clear()

    def chosen_undersample(self):
        return self._spin_us.value()

    def chosen_dtr_factor(self):
        return self._spin_dtr.value()

    def chosen_gaussian_sigma(self):
        if self._chk_gaussian.isChecked():
            return self._spin_sigma.value()
        else:
            return 0.0


class GeometryCorrectionGUI(QMainWindow):
    def __init__(self, parent_window=None):
        self.parent_window = parent_window
        super().__init__()
        self.setWindowTitle("X-Ray Geometry Correction GUI")
        self.resize(1300, 850)
        
        self.P_list = []           # List of original matrices
        self.P_list_original = []  # Backup of full trajectory
        self.images_undersampled = [] # Undersampled images
        self.voxel_dimensions = [100, 100, 100]
        self.model_matrix = np.eye(4)
        
        self.current_recon_json = None
        self.original_recon_config_path = None
        self.current_config_path = None
        self._gaussian_sigma = None
        self.dirty = False
        
        self.scan = None
        self.stages_cache = {}      # mapping abs_path -> Stage
        self.initial_stages_to_check = []
        self.sweep_windows = []
        self.raw_images = None
        self.console_win = None
        self.recon_console_win = None
        self.sweep_param_instances = {}
        self.cost_matrix = None
        
        self.clicked_point_viewport = None
        self.clicked_point_coords   = None
        self.hover_point_viewport   = None
        self.hover_point_coords     = None
        self._base_svg     = {}  # {view_idx: (svg_str, P_invT, K_pencil, E0, E90, W, H)}
        self._dtr_base_svg = {}  # {view_idx: (svg_str, P_invT, K_pencil, E0, E90, w, h, size_alpha, size_t, range_t)}
        self._redundancy_artists = None  # (line0, line1, vline_hover, vline_click) persistent artists
        self._loaded_num_planes = None
        self._loaded_object_radius = None
        
        self.init_ui()
        self.create_actions_and_menus()
        
        # Load default stage configuration to start with
        self.load_default_stages()

    @property
    def P_list_undersampled(self):
        if not getattr(self, 'P_list', None):
            return []
        U = self.spin_undersample.value() if hasattr(self, 'spin_undersample') else 1
        Ps = self.P_list[::U]
        imgs = getattr(self, 'images_undersampled', None)
        if imgs is not None:
            Ps = Ps[:len(imgs)]
        return Ps

    def init_ui(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # --- TAB 1: CONFIG & STAGES ---
        tab_config = QWidget()
        tab_config_layout = QHBoxLayout(tab_config)
        
        # Left Panel: General Settings
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        grp_general = QGroupBox("General Settings")
        gen_layout = QVBoxLayout(grp_general)
        
        # Input Reconstruction Config
        gen_layout.addWidget(QLabel("Input Reconstruction JSON:"))
        recon_path_lay = QHBoxLayout()
        self.txt_recon_path = QLineEdit("No file loaded")
        self.txt_recon_path.textChanged.connect(self.update_json_preview)
        self.txt_recon_path.editingFinished.connect(self.on_recon_path_edited)
        btn_recon_browse = QPushButton("Browse...")
        btn_recon_browse.clicked.connect(self.browse_recon_json)
        recon_path_lay.addWidget(self.txt_recon_path, 1)
        recon_path_lay.addWidget(btn_recon_browse)
        gen_layout.addLayout(recon_path_lay)
        

        

        # spin_undersample is kept as a non-visible widget so all downstream
        # code (on_undersample_changed, save_geometry_config, etc.) continues
        # to work unchanged.  The value is set by ConfigLoadDialog.
        self.spin_undersample = QSpinBox()
        self.spin_undersample.setRange(1, 10000)
        self.spin_undersample.setValue(10)
        self.spin_undersample.valueChanged.connect(self.on_undersample_changed)


        # Output Directory
        gen_layout.addWidget(QLabel("Output Directory:"))
        out_path_lay = QHBoxLayout()
        self.txt_out_dir = QLineEdit("Select Output Folder")
        self.txt_out_dir.textChanged.connect(self.update_json_preview)
        btn_out_browse = QPushButton("Browse...")
        btn_out_browse.clicked.connect(self.browse_out_dir)
        btn_out_reveal = QPushButton("📁")
        btn_out_reveal.setToolTip("Reveal output directory in file manager")
        btn_out_reveal.setFixedWidth(32)
        btn_out_reveal.clicked.connect(self.reveal_out_dir)
        out_path_lay.addWidget(self.txt_out_dir, 1)
        out_path_lay.addWidget(btn_out_browse)
        out_path_lay.addWidget(btn_out_reveal)
        gen_layout.addLayout(out_path_lay)
        
        # Reconstruction Settings
        self.btn_recon_settings = QPushButton("Reconstruction Settings...")
        self.btn_recon_settings.clicked.connect(self.open_reconstruction_settings)
        gen_layout.addWidget(self.btn_recon_settings)

        chk_recon_report_lay = QHBoxLayout()
        self.chk_run_recon = QCheckBox("Run Reconstruction")
        self.chk_run_recon.setChecked(True)
        self.chk_run_recon.stateChanged.connect(self.update_json_preview)
        chk_recon_report_lay.addWidget(self.chk_run_recon)

        self.chk_create_report = QCheckBox("Create Report")
        self.chk_create_report.setChecked(True)
        self.chk_create_report.stateChanged.connect(self.update_json_preview)
        chk_recon_report_lay.addWidget(self.chk_create_report)
        gen_layout.addLayout(chk_recon_report_lay)
        
        # Metric Configuration
        grp_metric = QGroupBox("ECC Metric Settings")
        metric_lay = QVBoxLayout(grp_metric)
        self.chk_line_integral = QCheckBox("Convert to Line Integral")
        self.chk_line_integral.setChecked(False)
        self.chk_line_integral.stateChanged.connect(self.on_metric_changed)
        
        spacing_lay = QHBoxLayout()
        spacing_lay.addWidget(QLabel("Number of Planes:"))
        self.spin_spacing = QSpinBox()
        self.spin_spacing.setRange(10, 10000)
        self.spin_spacing.setValue(900)
        self.spin_spacing.setSingleStep(100)
        self.spin_spacing.valueChanged.connect(self.on_metric_changed)
        spacing_lay.addWidget(self.spin_spacing)
        
        size_fac_lay = QHBoxLayout()
        size_fac_lay.addWidget(QLabel("Radon dtr Size Factor:"))
        self.spin_size_fac = QDoubleSpinBox()
        self.spin_size_fac.setRange(0.01, 5.0)
        self.spin_size_fac.setValue(1.0)
        self.spin_size_fac.setSingleStep(0.1)
        self.spin_size_fac.valueChanged.connect(self.on_metric_changed)
        size_fac_lay.addWidget(self.spin_size_fac)
        
        radius_lay = QHBoxLayout()
        radius_lay.addWidget(QLabel("Object Radius (mm):"))
        self.spin_object_radius = QDoubleSpinBox()
        self.spin_object_radius.setRange(0.01, 5000.0)
        self.spin_object_radius.setValue(100.0)
        self.spin_object_radius.setSingleStep(1.0)
        self.spin_object_radius.valueChanged.connect(self.on_metric_changed)
        radius_lay.addWidget(self.spin_object_radius)
        
        metric_lay.addWidget(self.chk_line_integral)
        metric_lay.addLayout(spacing_lay)
        metric_lay.addLayout(size_fac_lay)
        metric_lay.addLayout(radius_lay)
        
        self.lbl_dtr_dims = QLabel("DTR Size: alpha = N/A, t = N/A")
        metric_lay.addWidget(self.lbl_dtr_dims)
        
        # JSON Preview Group
        grp_preview = QGroupBox("Main Config JSON Preview")
        preview_lay = QVBoxLayout(grp_preview)
        self.txt_json_preview = QTextEdit()
        self.txt_json_preview.setReadOnly(True)
        self.txt_json_preview.setFont(QFont("Consolas", 9))
        preview_lay.addWidget(self.txt_json_preview)
        
        left_layout.addWidget(grp_general)
        left_layout.addWidget(grp_metric)
        left_layout.addWidget(grp_preview, 1)
        
        # Run Buttons
        self.btn_compare_recon = QPushButton("Compare Reconstructions...")
        self.btn_compare_recon.clicked.connect(self.compare_reconstructions_action)
        self.btn_compare_recon.setStyleSheet("font-weight: bold; background-color: #0284c7; color: white;")
        left_layout.addWidget(self.btn_compare_recon)

        run_lay = QHBoxLayout()
        self.btn_run_opt = QPushButton("Run Optimization...")
        self.btn_run_opt.clicked.connect(self.run_optimization_action)
        self.btn_run_opt.setStyleSheet("font-weight: bold; background-color: #2e7d32; color: white;")
        run_lay.addWidget(self.btn_run_opt)
        self.btn_run_recon = QPushButton("Run Reconstruction...")
        self.btn_run_recon.clicked.connect(self.run_reconstruction_direct_action)
        run_lay.addWidget(self.btn_run_recon)
        left_layout.addLayout(run_lay)
        
        # Right Panel: Stages Manager
        right_panel = QGroupBox("Geometry Calibration Stages")
        right_layout = QVBoxLayout(right_panel)
        
        # Stages Directory Selection
        dir_lay = QHBoxLayout()
        dir_lay.addWidget(QLabel("Stages Directory:"))
        self.txt_stages_dir = QLineEdit()
        self.txt_stages_dir.setReadOnly(True)
        self.txt_stages_dir.setText("Stages Directory Not Set")
        btn_stages_dir_browse = QPushButton("Browse...")
        btn_stages_dir_browse.clicked.connect(self.browse_stages_dir)
        dir_lay.addWidget(self.txt_stages_dir)
        dir_lay.addWidget(btn_stages_dir_browse)
        right_layout.addLayout(dir_lay)
        
        # Stages list widget and buttons side-by-side
        right_layout.addWidget(QLabel("Stages List:"))
        stages_h_lay = QHBoxLayout()
        
        self.list_stages = QListWidget()
        self.list_stages.currentRowChanged.connect(self.select_stage)
        self.list_stages.itemChanged.connect(self.on_stage_item_changed)
        stages_h_lay.addWidget(self.list_stages, 1)
        
        # Stages control buttons (vertical layout on the right)
        btn_stages_v_lay = QVBoxLayout()
        btn_stages_v_lay.setContentsMargins(0, 0, 0, 0)
        btn_add_stage = QPushButton("Add")
        btn_add_stage.clicked.connect(self.add_stage_action)
        btn_rem_stage = QPushButton("Remove")
        btn_rem_stage.clicked.connect(self.remove_stage_action)
        btn_rename_stage = QPushButton("Rename")
        btn_rename_stage.clicked.connect(self.rename_stage_action)
        btn_stage_up = QPushButton("Move Up")
        btn_stage_up.clicked.connect(lambda: self.move_stage(-1))
        btn_stage_down = QPushButton("Move Down")
        btn_stage_down.clicked.connect(lambda: self.move_stage(1))
        
        btn_stages_v_lay.addWidget(btn_add_stage)
        btn_stages_v_lay.addWidget(btn_rem_stage)
        btn_stages_v_lay.addWidget(btn_rename_stage)
        btn_stages_v_lay.addWidget(btn_stage_up)
        btn_stages_v_lay.addWidget(btn_stage_down)
        btn_import_stage = QPushButton("Import...")
        btn_import_stage.clicked.connect(self.import_stage_action)
        btn_export_stage = QPushButton("Export...")
        btn_export_stage.clicked.connect(self.export_stage_action)
        btn_analyze_stage = QPushButton("Analyze...")
        btn_analyze_stage.setStyleSheet("font-weight: bold; background-color: #16a085; color: white;")
        btn_analyze_stage.clicked.connect(self.analyze_stage_action)

        btn_optimize_stage = QPushButton("Optimize...")
        btn_optimize_stage.setStyleSheet("font-weight: bold; background-color: #0284c7; color: white;")
        btn_optimize_stage.clicked.connect(self.optimize_stage_action)
        
        btn_stages_v_lay.addSpacing(6)
        btn_stages_v_lay.addWidget(btn_import_stage)
        btn_stages_v_lay.addWidget(btn_export_stage)
        btn_stages_v_lay.addWidget(btn_analyze_stage)
        btn_stages_v_lay.addWidget(btn_optimize_stage)
        btn_stages_v_lay.addStretch()
        
        stages_h_lay.addLayout(btn_stages_v_lay)
        right_layout.addLayout(stages_h_lay, 1)
        
        # Stage Details panel
        self.stage_details_grp = QGroupBox("Selected Stage Details")
        self.stage_details_lay = QVBoxLayout(self.stage_details_grp)
        
        # Optimizer options in a single horizontal line
        opt_settings_lay = QHBoxLayout()
        
        opt_settings_lay.addWidget(QLabel("Optimizer:"))
        self.combo_optimizer = QComboBox()
        self.combo_optimizer.addItems(["OptimizerPowell", "OptimizerLBFGS"])
        self.combo_optimizer.currentTextChanged.connect(self.update_stage_optimizer)
        opt_settings_lay.addWidget(self.combo_optimizer)
        
        opt_settings_lay.addWidget(QLabel("Max Iterations:"))
        self.spin_maxiter = QSpinBox()
        self.spin_maxiter.setRange(1, 1000)
        self.spin_maxiter.setValue(200)
        self.spin_maxiter.valueChanged.connect(self.update_stage_maxiter)
        opt_settings_lay.addWidget(self.spin_maxiter)
        
        opt_settings_lay.addWidget(QLabel("Tolerance (ftol):"))
        self.txt_ftol = QDoubleSpinBox()
        self.txt_ftol.setDecimals(15)
        self.txt_ftol.setRange(1e-18, 1.0)
        self.txt_ftol.setValue(1e-6)
        self.txt_ftol.setSingleStep(1e-5)
        self.txt_ftol.valueChanged.connect(self.update_stage_ftol)
        opt_settings_lay.addWidget(self.txt_ftol)
        
        opt_settings_lay.addWidget(QLabel("Step Size (eps):"))
        self.txt_eps = QDoubleSpinBox()
        self.txt_eps.setDecimals(6)
        self.txt_eps.setRange(1e-9, 1.0)
        self.txt_eps.setValue(1e-3)
        self.txt_eps.setSingleStep(1e-4)
        self.txt_eps.valueChanged.connect(self.update_stage_eps)
        opt_settings_lay.addWidget(self.txt_eps)
        
        self.stage_details_lay.addLayout(opt_settings_lay)
        
        # Parameterizations List
        self.stage_details_lay.addWidget(QLabel("Parameterizations in Chain:"))
        self.list_params = QListWidget()
        self.list_params.setMaximumHeight(80)
        self.list_params.currentRowChanged.connect(self.select_parameterization)
        self.stage_details_lay.addWidget(self.list_params)
        
        btn_param_lay = QHBoxLayout()
        btn_add_param = QPushButton("Add Param...")
        btn_add_param.clicked.connect(self.add_param_action)
        btn_rem_param = QPushButton("Remove Param")
        btn_rem_param.clicked.connect(self.remove_param_action)
        btn_autoconf = QPushButton("Auto Configure")
        btn_autoconf.clicked.connect(self.auto_configure_action)
        btn_param_lay.addWidget(btn_add_param)
        btn_param_lay.addWidget(btn_rem_param)
        btn_param_lay.addWidget(btn_autoconf)
        self.stage_details_lay.addLayout(btn_param_lay)
        
        # Table of details for selected parameterization
        self.stage_details_lay.addWidget(QLabel("Edit Individual Parameters:"))
        self.table_param_details = QTableWidget()
        self.table_param_details.setColumnCount(7)
        self.table_param_details.setHorizontalHeaderLabels(["Name", "Optimize", "Min Range", "Max Range", "Auto Range", "Sweep", "1D Search"])
        self.table_param_details.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_param_details.setMinimumHeight(220)
        self.stage_details_lay.addWidget(self.table_param_details)
        
        right_layout.addWidget(self.stage_details_grp, 0)
        
        tab_config_layout.addWidget(left_panel, 1)
        tab_config_layout.addWidget(right_panel, 3)
        self.tabs.addTab(tab_config, "Optimization")
        
        # --- TAB 2: DIAGNOSTICS ---
        tab_diag = QWidget()
        tab_diag_layout = QVBoxLayout(tab_diag)
        
        diag_top_lay = QHBoxLayout()
        self.btn_compute_diag = QPushButton("Compute Diagnostics...")
        self.btn_compute_diag.clicked.connect(self.compute_diagnostics_action)
        self.btn_compute_diag.setEnabled(False)
        
        self.lbl_object_radius = QLabel("Estimated Object Size (z_max): N/A")
        self.lbl_planes_needed = QLabel("Expected Planes: N/A")
        
        diag_top_lay.addWidget(self.btn_compute_diag)
        diag_top_lay.addWidget(self.lbl_object_radius)
        diag_top_lay.addWidget(self.lbl_planes_needed)
        diag_top_lay.addStretch()
        tab_diag_layout.addLayout(diag_top_lay)
        
        # Hide the labels initially
        self.lbl_object_radius.hide()
        self.lbl_planes_needed.hide()
        
        # Horizontal layout for Cost Matrix and Weights Matrix
        diag_plots_lay = QHBoxLayout()
        tab_diag_layout.addLayout(diag_plots_lay, 1)  # stretch=1: takes all remaining vertical space
        
        # 1. Cost & Weights Matrix GroupBox
        self.cost_panel = QGroupBox("Cost & Weights Matrix")
        cost_lay = QVBoxLayout(self.cost_panel)
        self.fig_cost = Figure()
        self.canvas_cost = FigureCanvas(self.fig_cost)
        self.ax_cost = self.fig_cost.add_subplot(111)
        self.ax_cost.set_title("Run Diagnostics to see cost matrix")
        cost_lay.addWidget(self.canvas_cost)
        diag_plots_lay.addWidget(self.cost_panel)
        self.canvas_cost.mpl_connect('button_press_event', self.on_cost_matrix_click)

        
        # 2. Zero-Plane Distances GroupBox
        self.weight_panel = QGroupBox("Zero-Plane Distances (px)")
        weight_lay = QVBoxLayout(self.weight_panel)
        self.fig_weight = Figure()
        self.canvas_weight = FigureCanvas(self.fig_weight)
        self.ax_weight = self.fig_weight.add_subplot(111)
        weight_lay.addWidget(self.canvas_weight)
        diag_plots_lay.addWidget(self.weight_panel)
        
        # 2D Parameter Sweep GroupBox (Moved to Diagnostics tab)
        self.sweep_2d_grp = QGroupBox("2D Parameter Sweep")
        sweep_v_lay = QVBoxLayout(self.sweep_2d_grp)
        
        # Row 1: Parameter 1 config
        row1_lay = QHBoxLayout()
        row1_lay.addWidget(QLabel("Class 1:"))
        self.combo_2d_class1 = QComboBox()
        self.combo_2d_class1.currentIndexChanged.connect(self.on_class1_changed)
        row1_lay.addWidget(self.combo_2d_class1, 2)
        
        row1_lay.addWidget(QLabel("Parameter 1:"))
        self.combo_2d_param1 = QComboBox()
        self.combo_2d_param1.currentIndexChanged.connect(self.on_param1_changed)
        row1_lay.addWidget(self.combo_2d_param1, 2)
        
        row1_lay.addWidget(QLabel("Min:"))
        self.spin_2d_min1 = QDoubleSpinBox()
        self.spin_2d_min1.setRange(-10000.0, 10000.0)
        self.spin_2d_min1.setDecimals(4)
        self.spin_2d_min1.setSingleStep(0.5)
        self.spin_2d_min1.valueChanged.connect(self.on_spin_range_changed)
        row1_lay.addWidget(self.spin_2d_min1, 1)
        
        row1_lay.addWidget(QLabel("Max:"))
        self.spin_2d_max1 = QDoubleSpinBox()
        self.spin_2d_max1.setRange(-10000.0, 10000.0)
        self.spin_2d_max1.setDecimals(4)
        self.spin_2d_max1.setSingleStep(0.5)
        self.spin_2d_max1.valueChanged.connect(self.on_spin_range_changed)
        row1_lay.addWidget(self.spin_2d_max1, 1)
        
        sweep_v_lay.addLayout(row1_lay)
        
        # Row 2: Parameter 2 config
        row2_lay = QHBoxLayout()
        row2_lay.addWidget(QLabel("Class 2:"))
        self.combo_2d_class2 = QComboBox()
        self.combo_2d_class2.currentIndexChanged.connect(self.on_class2_changed)
        row2_lay.addWidget(self.combo_2d_class2, 2)
        
        row2_lay.addWidget(QLabel("Parameter 2:"))
        self.combo_2d_param2 = QComboBox()
        self.combo_2d_param2.currentIndexChanged.connect(self.on_param2_changed)
        row2_lay.addWidget(self.combo_2d_param2, 2)
        
        row2_lay.addWidget(QLabel("Min:"))
        self.spin_2d_min2 = QDoubleSpinBox()
        self.spin_2d_min2.setRange(-10000.0, 10000.0)
        self.spin_2d_min2.setDecimals(4)
        self.spin_2d_min2.setSingleStep(0.5)
        self.spin_2d_min2.valueChanged.connect(self.on_spin_range_changed)
        row2_lay.addWidget(self.spin_2d_min2, 1)
        
        row2_lay.addWidget(QLabel("Max:"))
        self.spin_2d_max2 = QDoubleSpinBox()
        self.spin_2d_max2.setRange(-10000.0, 10000.0)
        self.spin_2d_max2.setDecimals(4)
        self.spin_2d_max2.setSingleStep(0.5)
        self.spin_2d_max2.valueChanged.connect(self.on_spin_range_changed)
        row2_lay.addWidget(self.spin_2d_max2, 1)
        
        sweep_v_lay.addLayout(row2_lay)
        
        # Row 3: Samples & Action button
        row3_lay = QHBoxLayout()
        row3_lay.addWidget(QLabel("Samples:"))
        self.spin_2d_samples = QSpinBox()
        self.spin_2d_samples.setRange(3, 100)
        self.spin_2d_samples.setValue(11)
        row3_lay.addWidget(self.spin_2d_samples)
        
        self.btn_plot_2d = QPushButton("Plot 2D Sweep")
        self.btn_plot_2d.clicked.connect(self.plot_2d_sweep_action)
        row3_lay.addWidget(self.btn_plot_2d)
        row3_lay.addStretch()
        
        sweep_v_lay.addLayout(row3_lay)
        
        self.sweep_2d_grp.setFixedHeight(140)
        tab_diag_layout.addWidget(self.sweep_2d_grp)
        
        self.tabs.addTab(tab_diag, "Diagnostics")
        
        # --- TAB 3: EPIPOLAR GEOMETRY ---
        tab_view = QWidget()
        tab_view_layout = QVBoxLayout(tab_view)
        
        top_row_layout = QHBoxLayout()
        tab_view_layout.addLayout(top_row_layout)
        
        # View 1 (Left Viewport) Container
        view1_container = QGroupBox("View 1 (Left)")
        view1_lay = QVBoxLayout(view1_container)
        
        self.viewport1 = InteractiveSvgWidget()
        self.viewport1_wrapper = AspectWrapper(self.viewport1, 1.0)
        self.viewport1_wrapper.setMinimumSize(300, 300)
        
        slider1_lay = QHBoxLayout()
        slider1_lay.addWidget(QLabel("Index:"))
        self.view1_slider = QSlider(Qt.Orientation.Horizontal)
        self.view1_slider.setRange(0, 0)
        self.view1_spin = QSpinBox()
        self.view1_spin.setRange(0, 0)
        
        self.view1_slider.valueChanged.connect(self.view1_spin.setValue)
        self.view1_spin.valueChanged.connect(self.view1_slider.setValue)
        self.view1_slider.valueChanged.connect(self.on_view1_changed)
        self.view1_slider.valueChanged.connect(self.render_viewports)
        
        slider1_lay.addWidget(self.view1_slider)
        slider1_lay.addWidget(self.view1_spin)
        
        view1_lay.addWidget(self.viewport1_wrapper, 1)
        view1_lay.addLayout(slider1_lay)
        
        # View 2 (Right Viewport) Container
        view2_container = QGroupBox("View 2 (Right)")
        view2_lay = QVBoxLayout(view2_container)
        
        self.viewport2 = InteractiveSvgWidget()
        self.viewport2_wrapper = AspectWrapper(self.viewport2, 1.0)
        self.viewport2_wrapper.setMinimumSize(300, 300)
        
        slider2_lay = QHBoxLayout()
        slider2_lay.addWidget(QLabel("Index:"))
        self.view2_slider = QSlider(Qt.Orientation.Horizontal)
        self.view2_slider.setRange(0, 0)
        self.view2_spin = QSpinBox()
        self.view2_spin.setRange(0, 0)
        
        self.view2_slider.valueChanged.connect(self.view2_spin.setValue)
        self.view2_spin.valueChanged.connect(self.view2_slider.setValue)
        self.view2_slider.valueChanged.connect(self.on_view2_changed)
        self.view2_slider.valueChanged.connect(self.render_viewports)
        
        slider2_lay.addWidget(self.view2_slider)
        slider2_lay.addWidget(self.view2_spin)
        
        view2_lay.addWidget(self.viewport2_wrapper, 1)
        view2_lay.addLayout(slider2_lay)
        
        # Radon Intermediate (dtrs) Container (aligned vertically)
        dtrs_container = QGroupBox("Radon Intermediate (dtrs)")
        dtrs_lay = QVBoxLayout(dtrs_container)
        
        self.dtr_viewport1 = InteractiveSvgWidget()
        self.dtr_viewport1.setMinimumSize(150, 150)
        self.dtr_viewport2 = InteractiveSvgWidget()
        self.dtr_viewport2.setMinimumSize(150, 150)
        
        dtrs_lay.addWidget(QLabel("View 1 (Top)"))
        dtrs_lay.addWidget(self.dtr_viewport1, 1)
        dtrs_lay.addWidget(QLabel("View 2 (Bottom)"))
        dtrs_lay.addWidget(self.dtr_viewport2, 1)
        
        top_row_layout.addWidget(view1_container, 2)
        top_row_layout.addWidget(view2_container, 2)
        top_row_layout.addWidget(dtrs_container, 1)
        
        # Bottom Row: Matplotlib Redundancy Profiles Plot
        plot_container = QGroupBox("Epipolar Consistency Redundancy Profiles")
        plot_lay = QVBoxLayout(plot_container)
        
        self.fig_redundancy = Figure()
        self.canvas_redundancy = FigureCanvas(self.fig_redundancy)
        self.canvas_redundancy.setMinimumHeight(300)
        self.ax_redundancy = self.fig_redundancy.add_subplot(111)
        self.ax_redundancy.set_title("Load data to plot redundancy profiles")
        
        plot_lay.addWidget(self.canvas_redundancy)
        tab_view_layout.addWidget(plot_container, 1)
        
        # Click → full render + redundancy plot update
        self.viewport1.click_signal.connect(lambda u, v: self.on_viewport_clicked(1, u, v))
        self.viewport2.click_signal.connect(lambda u, v: self.on_viewport_clicked(2, u, v))
        # Hover → cheap single-line overlay only
        self.viewport1.hover_signal.connect(lambda u, v: self.on_viewport_hover(1, u, v))
        self.viewport2.hover_signal.connect(lambda u, v: self.on_viewport_hover(2, u, v))
        # Leave → restore base SVG (clear dashed hover line)
        self.viewport1.leave_signal.connect(self.on_viewport_leave)
        self.viewport2.leave_signal.connect(self.on_viewport_leave)
        # DTR hover → draw the Radon line in the corresponding image viewport
        self.dtr_viewport1.hover_signal.connect(lambda u, v: self.on_dtr_hover(1, u, v))
        self.dtr_viewport2.hover_signal.connect(lambda u, v: self.on_dtr_hover(2, u, v))
        self.dtr_viewport1.leave_signal.connect(self.on_viewport_leave)
        self.dtr_viewport2.leave_signal.connect(self.on_viewport_leave)
        # Redundancy plot hover → draw epipolar lines at the hovered kappa
        self.canvas_redundancy.mpl_connect('motion_notify_event', self.on_plot_hover)
        self.canvas_redundancy.mpl_connect('axes_leave_event',    self.on_plot_leave)

        self.tabs.addTab(tab_view, "Epipolar Geometry")


        # Status Bar
        self.setStatusBar(QStatusBar())

    def closeEvent(self, event):
        if getattr(self, 'dirty', False):
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Would you like to save them before closing?",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save
            )
            if reply == QMessageBox.StandardButton.Save:
                if not self.save_project_action():
                    event.ignore()
                    return
            elif reply == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return

        if self.console_win:
            if not self.console_win.close():
                event.ignore()
                return
        if self.recon_console_win:
            if not self.recon_console_win.close():
                event.ignore()
                return
        if getattr(self, 'parent_window', None) is not None:
            self.parent_window.show()
        super().closeEvent(event)

    def create_actions_and_menus(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        
        open_action = QAction("Open Reconstruction Config...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.browse_recon_json)
        
        import_geom_action = QAction("Import Geometry Config...", self)
        import_geom_action.setShortcut("Ctrl+I")
        import_geom_action.triggered.connect(self.browse_geom_json)
        
        save_action = QAction("Save Project", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_project_action)
        
        file_menu.addAction(open_action)
        file_menu.addAction(import_geom_action)
        file_menu.addAction(save_action)
        file_menu.addSeparator()
        
        load_ompl_action = QAction("Load OMPL Trajectory...", self)
        load_ompl_action.setShortcut("Ctrl+Shift+O")
        load_ompl_action.triggered.connect(self.load_ompl_action)
        
        save_ompl_action = QAction("Save OMPL Trajectory...", self)
        save_ompl_action.setShortcut("Ctrl+Shift+S")
        save_ompl_action.triggered.connect(self.save_ompl_action)
        
        file_menu.addAction(load_ompl_action)
        file_menu.addAction(save_ompl_action)
        file_menu.addSeparator()

        reapply_action = QAction("Re-Apply Existing Parameterization...", self)
        reapply_action.triggered.connect(self.reapply_parameterization_action)
        file_menu.addAction(reapply_action)
        
        reset_state_action = QAction("Reset Current State", self)
        reset_state_action.triggered.connect(self.reset_current_state_action)
        file_menu.addAction(reset_state_action)
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        
        file_menu.addAction(exit_action)

    def load_default_stages(self):
        try:
            stages_dir = str(ecc.get_data_path("tools", "config", "calibration_correction"))
        except Exception as e:
            stages_dir = os.path.join(os.getcwd(), "calibration_correction")
        
        self.txt_stages_dir.setText(stages_dir)
        self.scan_stages_directory(stages_dir)

    def get_active_stage(self):
        row = self.list_stages.currentRow()
        if row < 0 or row >= self.list_stages.count():
            return None, None
        item = self.list_stages.item(row)
        filename = item.text()
        stages_dir = self.txt_stages_dir.text()
        abs_path = os.path.normpath(os.path.join(stages_dir, filename))
        stage = self.stages_cache.get(abs_path)
        return stage, abs_path

    def select_stage(self, row):
        if row < 0 or row >= self.list_stages.count():
            return
        item = self.list_stages.item(row)
        filename = item.text()
        stages_dir = self.txt_stages_dir.text()
        abs_path = os.path.normpath(os.path.join(stages_dir, filename))
        stage = self.stages_cache.get(abs_path)
        if not stage:
            return
        
        self.combo_optimizer.blockSignals(True)
        self.combo_optimizer.setCurrentText(stage.optimizer)
        self.combo_optimizer.blockSignals(False)
        
        self.spin_maxiter.blockSignals(True)
        self.spin_maxiter.setValue(stage.maxiter)
        self.spin_maxiter.blockSignals(False)
        
        self.txt_ftol.blockSignals(True)
        self.txt_ftol.setValue(stage.ftol)
        self.txt_ftol.blockSignals(False)
        
        self.txt_eps.blockSignals(True)
        self.txt_eps.setValue(stage.eps)
        self.txt_eps.blockSignals(False)
        
        # Update Parameterizations List
        self.list_params.blockSignals(True)
        self.list_params.clear()
        for p in stage.parameterizations:
            if isinstance(p, TimeVariant) and p.ref_inst is not None:
                self.list_params.addItem(f"{p.__class__.__name__} ({p.ref_inst.__class__.__name__})")
            else:
                self.list_params.addItem(p.__class__.__name__)
        self.list_params.blockSignals(False)
        
        if stage.parameterizations:
            self.list_params.setCurrentRow(0)
        else:
            self.table_param_details.setRowCount(0)
            
        # Update 2D parameter sweep options
        self.update_2d_sweep_params()

    def update_stage_optimizer(self, val):
        stage, abs_path = self.get_active_stage()
        if stage:
            stage.optimizer = val
            self.update_json_preview()

    def update_stage_maxiter(self, val):
        stage, _ = self.get_active_stage()
        if stage:
            stage.maxiter = val
            self.update_json_preview()

    def update_stage_ftol(self, val):
        stage, _ = self.get_active_stage()
        if stage:
            stage.ftol = val
            self.update_json_preview()

    def update_stage_eps(self, val):
        stage, _ = self.get_active_stage()
        if stage:
            stage.eps = val
            self.update_json_preview()

    def select_parameterization(self, row):
        stage, _ = self.get_active_stage()
        if not stage or row < 0 or row >= len(stage.parameterizations):
            self.table_param_details.setRowCount(0)
            return
            
        param_obj = stage.parameterizations[row]
        self.table_param_details.blockSignals(True)
        
        # Group parameters ending in _cp{i}
        cp_pattern = re.compile(r'^(.*)_cp(\d+)$')
        
        rows_to_display = []
        groups = {} # base_name -> list of (param_name, config)
        
        for name, config in param_obj.parameters.items():
            match = cp_pattern.match(name)
            if match:
                base_name = match.group(1)
                if base_name not in groups:
                    groups[base_name] = []
                groups[base_name].append((name, config))
            else:
                rows_to_display.append({
                    "is_group": False,
                    "display_name": name,
                    "param_name": name,
                    "config": config
                })
        
        # Add the grouped parameters
        for base_name, param_list in groups.items():
            param_list.sort(key=lambda x: int(cp_pattern.match(x[0]).group(2)))
            N = len(param_list)
            display_name = f"{base_name}_cp[{N}]"
            rep_config = param_list[0][1]
            rows_to_display.append({
                "is_group": True,
                "display_name": display_name,
                "base_name": base_name,
                "param_list": param_list,
                "config": rep_config
            })
            
        self.table_param_details.setRowCount(len(rows_to_display))
        
        for idx, row_data in enumerate(rows_to_display):
            # Name
            item_name = QTableWidgetItem(row_data["display_name"])
            item_name.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.table_param_details.setItem(idx, 0, item_name)
            
            # Optimize (Checkbox)
            chk = QCheckBox()
            chk.setChecked(row_data["config"].get("opt", False))
            chk.checkStateChanged.connect(lambda state, n=row_data["display_name"], p=param_obj: self.on_param_opt_changed(n, p, state))
            self.table_param_details.setCellWidget(idx, 1, chk)
            
            # Range Min
            range_min = QDoubleSpinBox()
            range_min.setRange(-10000.0, 10000.0)
            range_min.setDecimals(4)
            range_min.setValue(row_data["config"].get("range", (-1.0, 1.0))[0])
            range_min.setSingleStep(0.5)
            range_min.valueChanged.connect(lambda val, n=row_data["display_name"], p=param_obj: self.on_param_range_changed(n, p, val, True))
            self.table_param_details.setCellWidget(idx, 2, range_min)
            
            # Range Max
            range_max = QDoubleSpinBox()
            range_max.setRange(-10000.0, 10000.0)
            range_max.setDecimals(4)
            range_max.setValue(row_data["config"].get("range", (-1.0, 1.0))[1])
            range_max.setSingleStep(0.5)
            range_max.valueChanged.connect(lambda val, n=row_data["display_name"], p=param_obj: self.on_param_range_changed(n, p, val, False))
            self.table_param_details.setCellWidget(idx, 3, range_max)
            
            # Auto Range (ComboBox)
            combo = QComboBox()
            combo.addItems(["-", "10% volume", "10% detector", "1% SDD", "+/- 1", "+/- 5", "+/- 10", "+/- 50", "+/- 100"])
            combo.setCurrentText("-")
            combo.activated.connect(lambda index, cb=combo, rd=row_data, p=param_obj: self.apply_auto_range(rd, p, cb.itemText(index)))
            self.table_param_details.setCellWidget(idx, 4, combo)
            
            # Sweep Action Button
            btn_sweep = QPushButton("Sweep")
            if row_data["is_group"]:
                btn_sweep.setEnabled(False)
            else:
                btn_sweep.clicked.connect(lambda checked, n=row_data["param_name"], p=param_obj: self.plot_sweep_action(n, p))
            self.table_param_details.setCellWidget(idx, 5, btn_sweep)
            
            # 1D Search Button
            btn_search = QPushButton("1D Search")
            if row_data["is_group"]:
                btn_search.setEnabled(False)
            else:
                btn_search.clicked.connect(lambda checked, n=row_data["param_name"], p=param_obj: self.one_d_search_action(n, p))
            self.table_param_details.setCellWidget(idx, 6, btn_search)
            
        self.table_param_details.blockSignals(False)

    def one_d_search_action(self, param_name, param_obj):
        if not self.scan:
            QMessageBox.warning(self, "No Trajectory", "Please load a reconstruction config first.")
            return
            
        try:
            # Perform a 1D sweep similar to plot_sweep_action
            p_info = param_obj.parameters[param_name]
            original_val = p_info["value"]
            range_min, range_max = p_info["range"]
            
            samples = np.linspace(range_min, range_max, 51)
            costs = []
            
            # Temporary trajectory chain to isolate target parameter
            temp_chain = ParameterizationChain([param_obj])
            
            # Evaluate costs
            for v in samples:
                p_info["value"] = v
                Ps_temp = temp_chain.apply_to_trajectory(self.P_list_undersampled)
                costs.append(self.scan.compute_ecc_for_projection_matrices([Ps_temp])[0])
                
            p_info["value"] = original_val # Restore original value
            
            # Find the minimum cost value
            min_idx = np.argmin(costs)
            v_opt = samples[min_idx]
            
            # Ask the user for confirmation
            reply = QMessageBox.question(
                self, "Apply 1D Search Results",
                f"1D Search found optimal value {v_opt:.4f} for parameter '{param_name}'.\n"
                f"Do you want to permanently apply this adjustment to all projection matrices in memory and reset the value to 0.0?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                # Create a temporary chain with only this parameterization set to v_opt
                # (and other parameters set to 0.0)
                param_obj_opt = deepcopy(param_obj)
                for k in param_obj_opt.parameters:
                    if k == param_name:
                        param_obj_opt.parameters[k]["value"] = v_opt
                    else:
                        param_obj_opt.parameters[k]["value"] = 0.0
                
                single_chain = ParameterizationChain([param_obj_opt])
                
                # Apply transformation permanently to all matrices in memory
                self.P_list = single_chain.apply_to_trajectory(self.P_list)
                self.dirty = True
                
                # Apply to scan
                if self.scan:
                    self.scan.set_projection_matrices(self.P_list_undersampled)
                
                # Reset this parameter value in the UI to 0.0
                p_info["value"] = 0.0
                
                # Re-select the stage to update table items
                self.select_stage(self.list_stages.currentRow())
                self.render_viewports()
                self.update_json_preview()
                self.statusBar().showMessage(f"Applied 1D search adjustment of {v_opt:.4f} to all matrices.")
                
        except Exception as e:
            QMessageBox.critical(self, "1D Search Error", f"Failed to perform 1D search:\n{e}")

    def on_param_opt_changed(self, name, param_obj, state):
        opt_val = (state == Qt.CheckState.Checked)
        prefix = name.split("[")[0] if "[" in name else name
        for k in param_obj.parameters:
            if k.startswith(prefix):
                param_obj.parameters[k]["opt"] = opt_val
        self.render_viewports()
        self.update_json_preview()
        
        stage, abs_path = self.get_active_stage()
        if stage and abs_path:
            self.save_single_stage(abs_path, stage)

    def on_param_val_changed(self, name, param_obj, val):
        prefix = name.split("[")[0] if "[" in name else name
        for k in param_obj.parameters:
            if k.startswith(prefix):
                param_obj.parameters[k]["value"] = val

        self.render_viewports()
        self.update_json_preview()
        
        stage, abs_path = self.get_active_stage()
        if stage and abs_path:
            self.save_single_stage(abs_path, stage)

    def on_param_range_changed(self, name, param_obj, val, is_min):
        prefix = name.split("[")[0] if "[" in name else name
        for k in param_obj.parameters:
            if k.startswith(prefix):
                r = list(param_obj.parameters[k].get("range", (-1.0, 1.0)))
                if is_min:
                    r[0] = val
                else:
                    r[1] = val
                param_obj.parameters[k]["range"] = tuple(r)
        self.update_json_preview()
        
        stage, abs_path = self.get_active_stage()
        if stage and abs_path:
            self.save_single_stage(abs_path, stage)

    def apply_auto_range(self, row_data, param_obj, option):
        if option == "-":
            return
        R = 1.0
        if option == "10% volume":
            vol_scale = np.linalg.norm(self.model_matrix[:3, 0]) * np.mean(self.voxel_dimensions)
            R = 0.1 * vol_scale
        elif option == "10% detector":
            img_scale = 400.0
            if self.P_list:
                first_p = self.P_list[0]
                det_w = abs(first_p.image_size[0] * first_p.pixel_spacing)
                det_h = abs(first_p.image_size[1] * first_p.pixel_spacing)
                img_scale = np.mean([det_w, det_h])
            elif self.scan:
                img_scale = np.mean(self.scan.detector_size) * abs(self.scan.pixel_spacing)
            R = 0.1 * img_scale
        elif option == "1% SDD":
            sdd = 1000.0
            if self.P_list:
                sdd = np.mean([abs(SourceDetectorGeometry(p).source_detector_distance) for p in self.P_list])
            R = 0.01 * sdd
        elif option == "+/- 1":
            R = 1.0
        elif option == "+/- 5":
            R = 5.0
        elif option == "+/- 10":
            R = 10.0
        elif option == "+/- 50":
            R = 50.0
        elif option == "+/- 100":
            R = 100.0
            
        # Update parameters using prefix matching
        display_name = row_data["display_name"]
        prefix = display_name.split("[")[0] if "[" in display_name else display_name
        for k in param_obj.parameters:
            if k.startswith(prefix):
                val = param_obj.parameters[k].get("value", 0.0)
                param_obj.parameters[k]["range"] = (val - R, val + R)
            
        # Refresh the table display to reflect the new ranges
        current_row = self.list_params.currentRow()
        if current_row >= 0:
            self.select_parameterization(current_row)
            
        self.update_json_preview()
        
        stage, abs_path = self.get_active_stage()
        if stage and abs_path:
            self.save_single_stage(abs_path, stage)

    def add_stage_action(self):
        stages_dir = self.txt_stages_dir.text()
        if not stages_dir or stages_dir == "Stages Directory Not Set":
            QMessageBox.warning(self, "No Stages Directory", "Please open a config or select a stages directory first.")
            return
            
        filename, ok = QInputDialog.getText(self, "Add Stage", "Enter stage JSON filename:", text=f"stage{self.list_stages.count() + 1}.json")
        if ok and filename:
            if not filename.lower().endswith(".json"):
                filename += ".json"
            abs_path = os.path.normpath(os.path.join(stages_dir, filename))
            if abs_path in self.stages_cache:
                QMessageBox.warning(self, "Duplicate", "A stage file with that name already exists.")
                return
            
            # Create default Stage object
            s_new = Stage(name=filename[:-5], optimizer="OptimizerPowell")
            s_new.parameterizations.append(DetectorOrientation())
            
            self.stages_cache[abs_path] = s_new
            
            # Save stage file immediately
            self.save_single_stage(abs_path, s_new)
            
            # Add to list
            self.list_stages.blockSignals(True)
            item = QListWidgetItem(filename)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            item.setCheckState(Qt.CheckState.Checked)
            self.list_stages.addItem(item)
            self.list_stages.blockSignals(False)
            
            self.list_stages.setCurrentItem(item)
            self.update_json_preview()

    def remove_stage_action(self):
        row = self.list_stages.currentRow()
        if row < 0 or row >= self.list_stages.count():
            return
            
        item = self.list_stages.item(row)
        filename = item.text()
        stages_dir = self.txt_stages_dir.text()
        abs_path = os.path.normpath(os.path.join(stages_dir, filename))
        
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Are you sure you want to delete the stage configuration file '{filename}' from disk?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                if os.path.exists(abs_path):
                    os.remove(abs_path)
                if abs_path in self.stages_cache:
                    del self.stages_cache[abs_path]
                self.list_stages.takeItem(row)
                self.update_json_preview()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete file:\n{e}")

    def move_stage(self, offset):
        row = self.list_stages.currentRow()
        if row < 0 or row >= self.list_stages.count():
            return
        target = row + offset
        if 0 <= target < self.list_stages.count():
            self.list_stages.blockSignals(True)
            item = self.list_stages.takeItem(row)
            self.list_stages.insertItem(target, item)
            self.list_stages.setCurrentRow(target)
            self.list_stages.blockSignals(False)
            self.update_json_preview()

    def duplicate_stage_action(self):
        row = self.list_stages.currentRow()
        if row < 0 or row >= self.list_stages.count():
            QMessageBox.warning(self, "No Selection", "Please select a stage to duplicate.")
            return
            
        stages_dir = self.txt_stages_dir.text()
        if not stages_dir or stages_dir == "Stages Directory Not Set":
            QMessageBox.warning(self, "No Stages Directory", "Please open a config or select a stages directory first.")
            return
            
        old_item = self.list_stages.item(row)
        old_filename = old_item.text()
        old_abs_path = os.path.normpath(os.path.join(stages_dir, old_filename))
        
        if old_filename.lower().endswith(".json"):
            base = old_filename[:-5]
        else:
            base = old_filename
        new_filename = base + "_duplicate.json"
        
        new_abs_path = os.path.normpath(os.path.join(stages_dir, new_filename))
        counter = 1
        while new_abs_path in self.stages_cache or os.path.exists(new_abs_path):
            new_filename = f"{base}_duplicate_{counter}.json"
            new_abs_path = os.path.normpath(os.path.join(stages_dir, new_filename))
            counter += 1
            
        orig_stage = self.stages_cache.get(old_abs_path)
        if not orig_stage:
            QMessageBox.warning(self, "Error", f"Could not find stage data for {old_filename} in cache.")
            return
            
        s_dup = deepcopy(orig_stage)
        if s_dup.name:
            s_dup.name = s_dup.name + "_duplicate"
        else:
            s_dup.name = new_filename[:-5]
            
        self.stages_cache[new_abs_path] = s_dup
        self.save_single_stage(new_abs_path, s_dup)
        
        self.list_stages.blockSignals(True)
        item = QListWidgetItem(new_filename)
        item.setFlags(old_item.flags())
        item.setCheckState(old_item.checkState())
        self.list_stages.insertItem(row + 1, item)
        self.list_stages.blockSignals(False)
        
        self.list_stages.setCurrentRow(row + 1)
        self.update_json_preview()

    def rename_stage_action(self):
        row = self.list_stages.currentRow()
        if row < 0 or row >= self.list_stages.count():
            QMessageBox.warning(self, "No Selection", "Please select a stage to rename.")
            return
            
        stages_dir = self.txt_stages_dir.text()
        if not stages_dir or stages_dir == "Stages Directory Not Set":
            QMessageBox.warning(self, "No Stages Directory", "Please open a config or select a stages directory first.")
            return
            
        item = self.list_stages.item(row)
        old_filename = item.text()
        old_abs_path = os.path.normpath(os.path.join(stages_dir, old_filename))
        
        new_filename, ok = QInputDialog.getText(self, "Rename Stage", "Enter new stage JSON filename:", text=old_filename)
        if ok and new_filename:
            if not new_filename.lower().endswith(".json"):
                new_filename += ".json"
            if new_filename == old_filename:
                return
                
            new_abs_path = os.path.normpath(os.path.join(stages_dir, new_filename))
            if new_abs_path in self.stages_cache or os.path.exists(new_abs_path):
                QMessageBox.warning(self, "Duplicate", "A stage file with that name already exists.")
                return
                
            orig_stage = self.stages_cache.get(old_abs_path)
            if not orig_stage:
                QMessageBox.warning(self, "Error", f"Could not find stage data for {old_filename} in cache.")
                return
                
            try:
                if os.path.exists(old_abs_path):
                    os.rename(old_abs_path, new_abs_path)
                    
                self.stages_cache[new_abs_path] = orig_stage
                if old_abs_path in self.stages_cache:
                    del self.stages_cache[old_abs_path]
                    
                orig_stage.name = new_filename[:-5]
                self.save_single_stage(new_abs_path, orig_stage)
                
                self.list_stages.blockSignals(True)
                item.setText(new_filename)
                self.list_stages.blockSignals(False)
                
                self.update_json_preview()
            except Exception as e:
                QMessageBox.critical(self, "Rename Error", f"Failed to rename stage:\n{e}")

    def add_param_action(self):
        stage, _ = self.get_active_stage()
        if not stage:
            return
        
        # Show a context menu of available parameterizations
        menu = QMenu(self)
        for label, cls in AVAILABLE_PARAMS.items():
            act = QAction(label, self)
            act.triggered.connect(lambda checked, c=cls: self.insert_parameterization(stage, c))
            menu.addAction(act)
        menu.exec(self.sender().mapToGlobal(self.sender().rect().bottomLeft()))

    def insert_parameterization(self, stage, cls):
        if issubclass(cls, TimeVariant):
            dlg = QDialog(self)
            dlg.setWindowTitle("Configure Time-Variant Parameter")
            dlg.setModal(True)
            dlg.resize(400, 160)
            
            layout = QVBoxLayout(dlg)
            layout.setSpacing(10)
            layout.setContentsMargins(20, 15, 20, 15)
            
            layout.addWidget(QLabel("Referenced Parameterization Class:"))
            combo_ref = QComboBox()
            for label, c in AVAILABLE_PARAMS.items():
                if not issubclass(c, TimeVariant):
                    combo_ref.addItem(label, c)
            layout.addWidget(combo_ref)
            
            layout.addWidget(QLabel("Number of Control Points:"))
            spin_ncp = QSpinBox()
            spin_ncp.setRange(2, 999)
            spin_ncp.setValue(4)
            layout.addWidget(spin_ncp)
            
            btn_row = QHBoxLayout()
            btn_ok = QPushButton("OK")
            btn_ok.setDefault(True)
            btn_cancel = QPushButton("Cancel")
            btn_row.addStretch()
            btn_row.addWidget(btn_ok)
            btn_row.addWidget(btn_cancel)
            layout.addLayout(btn_row)
            
            btn_ok.clicked.connect(dlg.accept)
            btn_cancel.clicked.connect(dlg.reject)
            
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return
                
            ref_cls = combo_ref.currentData()
            n_cp = spin_ncp.value()
            
            # Check for duplicates of same class AND same referenced_class
            for existing in stage.parameterizations:
                if isinstance(existing, cls) and getattr(existing, 'referenced_class', None) == ref_cls:
                    ref_name = ref_cls.__name__ if ref_cls else "None"
                    QMessageBox.warning(self, "Duplicate", f"{cls.__name__} referencing {ref_name} is already added to this stage.")
                    return
            
            param_inst = cls(referenced_class=ref_cls, num_control_points=n_cp)
        else:
            # Prevent adding duplicates of the same class in a stage for non-TimeVariant
            for existing in stage.parameterizations:
                if isinstance(existing, cls):
                    QMessageBox.warning(self, "Duplicate", f"{cls.__name__} is already added to this stage.")
                    return
            param_inst = cls()
            
        stage.parameterizations.append(param_inst)
        self.select_stage(self.list_stages.currentRow())
        self.list_params.setCurrentRow(len(stage.parameterizations) - 1)
        self.update_json_preview()

    def remove_param_action(self):
        stage, _ = self.get_active_stage()
        if not stage:
            return
        param_row = self.list_params.currentRow()
        if 0 <= param_row < len(stage.parameterizations):
            stage.parameterizations.pop(param_row)
            self.select_stage(self.list_stages.currentRow())
            self.list_params.setCurrentRow(max(0, param_row - 1))
            self.update_json_preview()

    def reapply_parameterization_action(self):
        if not self.P_list:
            QMessageBox.warning(self, "No Trajectory Loaded", "Please load a reconstruction config with a trajectory first.")
            return

        file_path, _ = QFileDialog.getOpenFileName(self, "Load Parameterization JSON", "", "JSON Files (*.json)")
        if not file_path:
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Reconstruct parameterization object
            param_obj = from_dict(data)
            
            # Check for prior knowledge and prevent any automatic estimation
            has_pk = True
            if isinstance(param_obj, ParameterizationChain):
                for p in param_obj.parameterizations:
                    if isinstance(p, DetectorShift):
                        if getattr(p, "prior_knowledge", None) is None:
                            p.prior_knowledge = {}
                    else:
                        if getattr(p, "prior_knowledge", None) is None:
                            has_pk = False
                            break
            else:
                if isinstance(param_obj, DetectorShift):
                    if getattr(param_obj, "prior_knowledge", None) is None:
                        param_obj.prior_knowledge = {}
                else:
                    if getattr(param_obj, "prior_knowledge", None) is None:
                        has_pk = False
            
            if not has_pk:
                QMessageBox.critical(
                    self, "Missing Prior Knowledge", 
                    "The loaded parameterization does not contain the required prior trajectory knowledge (isocenter, rotation axis, etc.).\n"
                    "Re-applying it is not possible because the original alignment framework is missing."
                )
                return

            # Find non-zero parameters
            non_zero_params = []
            for name, p_info in param_obj.items():
                val = p_info.get("value", 0.0)
                if isinstance(val, (list, np.ndarray, tuple)):
                    if any(v != 0.0 for v in val):
                        non_zero_params.append((name, val))
                elif val != 0.0:
                    non_zero_params.append((name, val))
                    
            if not non_zero_params:
                QMessageBox.information(self, "No Non-Zero Parameters", "All parameters in the loaded parameterization are 0.0.")
                return

            # Show dialog to review parameters
            dialog = ReapplyParameterizationDialog(self, non_zero_params)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                # Apply parameterization permanently to all matrices in memory
                self.P_list = param_obj.apply_to_trajectory(self.P_list)
                self.dirty = True
                
                if self.scan:
                    self.scan.set_projection_matrices(self.P_list_undersampled)
                    
                self.statusBar().showMessage(f"Successfully applied parameterization from: {file_path}", 5000)
                self.render_viewports()
                self.update_json_preview()
                
        except Exception as e:
            QMessageBox.critical(self, "Error Loading Parameterization", f"Failed to load or apply parameterization:\n{e}")

    def reset_current_state_action(self):
        traj_path = os.path.normpath(os.path.join(os.path.dirname(self.txt_out_dir.text().strip()), "trajectory.ompl"))
        self.P_list = load_ompl(traj_path)
        self.P_list_original = list(self.P_list)
        self.load_and_undersample()
        for s in self.stages_cache.values():
            for p in s.parameterizations:
                for k in p.parameters: p.parameters[k]["value"] = 0.0
        self.select_stage(self.list_stages.currentRow())

    def auto_configure_action(self):
        stage, stage_path = self.get_active_stage()
        if not stage:
            QMessageBox.warning(self, "No Stage Selected", "Please select an optimization stage first.")
            return

        if not self.scan:
            QMessageBox.warning(self, "No Trajectory", "Please load a reconstruction config first.")
            return

        # Count total parameters across all parameterizations in the stage
        all_params = []
        for param_obj in stage.parameterizations:
            for param_name in param_obj.parameters:
                all_params.append((param_obj, param_name))

        if not all_params:
            QMessageBox.information(self, "No Parameters", "The currently selected stage has no parameters to configure.")
            return

        # Setup Progress Dialog
        progress = QProgressDialog("Running parameter sweeps...", "Cancel", 0, len(all_params), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()

        results = []

        for idx, (param_obj, param_name) in enumerate(all_params):
            if progress.wasCanceled():
                break

            progress.setLabelText(f"Sweeping {param_obj.__class__.__name__}.{param_name}...")
            progress.setValue(idx)
            QApplication.processEvents()

            p_info = param_obj.parameters[param_name]
            original_val = p_info["value"]
            range_min, range_max = p_info["range"]

            # Double the range centered around the midpoint
            center = (range_min + range_max) / 2.0
            width = range_max - range_min
            if width <= 0.0:
                width = 1.0  # fallback
            doubled_min = center - width
            doubled_max = center + width

            samples = np.linspace(doubled_min, doubled_max, 51)
            costs = []

            # Temp chain to isolate
            temp_chain = ParameterizationChain([param_obj])

            # Evaluate ECC cost for each sample
            try:
                for v in samples:
                    p_info["value"] = v
                    Ps_temp = temp_chain.apply_to_trajectory(self.P_list_undersampled)
                    cost = self.scan.compute_ecc_for_projection_matrices([Ps_temp])[0]
                    costs.append(cost)
            except Exception as e:
                # If evaluation fails, fall back/skip or log it
                costs = [0.0] * len(samples)

            # Restore original value
            p_info["value"] = original_val

            # Analyze parameter stability
            costs = np.array(costs)
            cost_range = np.max(costs) - np.min(costs)
            diffs = np.diff(costs)
            threshold = 1e-4 * cost_range if cost_range > 1e-12 else 0.0
            clean_diffs = diffs[np.abs(diffs) > threshold]
            
            sign_changes = 0
            if len(clean_diffs) > 1:
                sign_changes = np.sum(np.diff(np.sign(clean_diffs)) != 0)

            is_noisy = (sign_changes >= 3)
            k_min = np.argmin(costs)
            fraction = k_min / (len(samples) - 1)
            is_near_end = (fraction < 0.05 or fraction > 0.95)
            is_flat = (cost_range < 1e-5 * max(1.0, np.mean(costs)))
            has_min = (costs[k_min] < costs[0] and costs[k_min] < costs[-1] and not is_flat and not is_near_end)

            warnings = []
            if is_noisy:
                warnings.append("Warning: Cost curve is noisy/zig-zagging.")
            if is_flat:
                warnings.append("Warning: Parameter has no significant effect on the cost (flat curve).")
            elif is_near_end:
                warnings.append("Warning: Minimum is near the boundary of the extended sweep range.")
            elif not has_min:
                warnings.append("Warning: No clear minimum found (monotonic or invalid curve shape).")

            enabled = (len(warnings) == 0)
            
            # Update opt state in the actual parameterization!
            p_info["opt"] = enabled

            # Relative distance from 0 to minimum relative to range
            min_val = samples[k_min]
            rel_dist = abs(min_val - 0.0) / width

            # Generate inline SVG
            fig = Figure(figsize=(5, 3.5))
            ax = fig.add_subplot(111)
            ax.plot(samples, costs, color='#1f77b4', linewidth=2)
            ax.axvline(range_min, color='gray', linestyle='--', label='Range Min')
            ax.axvline(range_max, color='gray', linestyle='--', label='Range Max')
            ax.axvline(original_val, color='red', linestyle='-', label=f'Current ({original_val:.4f})')
            ax.set_xlabel(param_name)
            ax.set_ylabel('Cost')
            ax.set_title(f"Sweep: {param_name}")
            ax.legend()
            ax.grid(True, linestyle=':', alpha=0.6)

            buf = io.StringIO()
            fig.savefig(buf, format='svg', bbox_inches='tight')
            svg_string = buf.getvalue()
            svg_start = svg_string.find("<svg")
            svg_code = svg_string[svg_start:] if svg_start != -1 else svg_string

            results.append({
                "param_obj": param_obj,
                "param_name": param_name,
                "display_name": f"{param_obj.__class__.__name__}.{param_name}",
                "enabled": enabled,
                "warnings": warnings,
                "rel_dist": rel_dist,
                "svg_code": svg_code,
                "min_val": min_val
            })

        progress.setValue(len(all_params))

        # 1. Refresh GUI Table & preview
        self.select_parameterization(self.list_params.currentRow())
        self.render_viewports()
        self.update_json_preview()

        # 2. Get output directory
        out_dir = self.txt_out_dir.text().strip()
        if not out_dir or out_dir == "Select Output Folder":
            out_dir = os.getcwd()
        else:
            try:
                os.makedirs(out_dir, exist_ok=True)
            except Exception:
                out_dir = os.getcwd()

        # 3. Determine actual stage name (filename without extension)
        stage_filename = os.path.basename(stage_path)
        stage_name_clean = os.path.splitext(stage_filename)[0]
        html_filename = f"autoconf_{stage_name_clean}.html"
        html_path = os.path.normpath(os.path.join(out_dir, html_filename))

        # 4. Assemble HTML Report
        html_lines = []
        html_lines.append("<html>")
        html_lines.append(f"<head><title>Auto Configuration Report - {stage_name_clean}</title></head>")
        html_lines.append("<body>")
        html_lines.append(f"<h1>Auto Configuration Report - {stage_name_clean}</h1>")

        # First show plots of enabled parameters
        html_lines.append("<h2>Enabled Parameters</h2>")
        enabled_results = [r for r in results if r["enabled"]]
        if enabled_results:
            for r in enabled_results:
                html_lines.append(f"<h3>{r['display_name']}</h3>")
                html_lines.append("<div>" + r["svg_code"] + "</div>")
        else:
            html_lines.append("<p>No parameters were enabled.</p>")

        # Then the table of sorted enabled parameters
        html_lines.append("<h2>Sorted Enabled Parameters</h2>")
        sorted_enabled = sorted(enabled_results, key=lambda x: x["rel_dist"])
        if sorted_enabled:
            html_lines.append("<table border='1' cellpadding='5'>")
            html_lines.append("<tr><th>Parameter</th><th>Optimal Value (Minimum)</th><th>Relative Distance from 0</th></tr>")
            for r in sorted_enabled:
                html_lines.append(f"<tr><td>{r['display_name']}</td><td>{r['min_val']:.6f}</td><td>{r['rel_dist']:.6f}</td></tr>")
            html_lines.append("</table>")
        else:
            html_lines.append("<p>No enabled parameters to display in table.</p>")

        # Then show plots of disabled parameters with warnings
        html_lines.append("<h2>Disabled Parameters</h2>")
        disabled_results = [r for r in results if not r["enabled"]]
        if disabled_results:
            for r in disabled_results:
                html_lines.append(f"<h3>{r['display_name']}</h3>")
                for warning in r["warnings"]:
                    html_lines.append(f"<p><font color='red'><b>{warning}</b></font></p>")
                html_lines.append("<div>" + r["svg_code"] + "</div>")
        else:
            html_lines.append("<p>No parameters were disabled.</p>")

        html_lines.append("</body>")
        html_lines.append("</html>")

        # Write to HTML file
        try:
            with open(html_path, 'w', encoding='utf-8') as f_html:
                f_html.write("\n".join(html_lines))
        except Exception as err:
            QMessageBox.critical(self, "Report Save Error", f"Failed to save HTML report:\n{err}")

        # Show results dialog
        dialog = AutoConfigureResultDialog(self, sorted_enabled, html_path)
        dialog.exec()

    def on_stage_item_changed(self, item):
        self.update_json_preview()

    def update_json_preview(self):
        try:
            out_dir = self.txt_out_dir.text().strip()
            if out_dir and out_dir != "Select Output Folder":
                config_dir = os.path.abspath(out_dir)
            else:
                config_dir = os.getcwd()
            
            # Input data
            U = self.spin_undersample.value()
            undersampled_recon_filename = f"fullscan_{len(self.P_list_undersampled) or 'N'}views_600x400.json"
            
            # Reconstruction Config
            recon_path = self.txt_recon_path.text().strip()
            if recon_path and recon_path != "No file loaded" and os.path.exists(recon_path):
                orig_recon_rel = safe_relpath(recon_path, config_dir)
            else:
                orig_recon_rel = "No input config loaded"
                
            # Output Directory
            out_dir = self.txt_out_dir.text().strip()
            if out_dir and out_dir != "Select Output Folder":
                output_dir_rel = safe_relpath(out_dir, config_dir)
            else:
                output_dir_rel = "output/synthetic_pumpkin"
                
            # Checked stages
            checked_rel_paths = []
            stages_dir = self.txt_stages_dir.text()
            if stages_dir and stages_dir != "Stages Directory Not Set":
                for i in range(self.list_stages.count()):
                    item = self.list_stages.item(i)
                    if item.checkState() == Qt.CheckState.Checked:
                        filename = item.text()
                        abs_path = os.path.normpath(os.path.join(stages_dir, filename))
                        rel_path = safe_relpath(abs_path, config_dir)
                        checked_rel_paths.append(rel_path)
            else:
                for i in range(self.list_stages.count()):
                    item = self.list_stages.item(i)
                    if item.checkState() == Qt.CheckState.Checked:
                        checked_rel_paths.append(f"calibration_correction/{item.text()}")

            dtr_factor = self.spin_size_fac.value()
            default_sigma = 0.0 if abs(dtr_factor - 1.0) < 1e-5 else 1.2
            preview_dict = {
                "input_data": undersampled_recon_filename,
                "output_dir": output_dir_rel,
                "create_report": self.chk_create_report.isChecked(),
                "metric_config": {
                    "convert_to_line_integral": self.chk_line_integral.isChecked(),
                    "dtr_size_factor": dtr_factor,
                    "gaussian_sigma": getattr(self, '_gaussian_sigma', default_sigma),
                    "num_planes": self.spin_spacing.value(),
                    "object_radius_mm": self.spin_object_radius.value()
                },
                "geometry_optimization": {
                    "stages": checked_rel_paths
                }
            }
            if self.chk_run_recon.isChecked():
                preview_dict["reconstruction_config"] = orig_recon_rel
            self.txt_json_preview.setPlainText(json.dumps(preview_dict, indent=2))
        except Exception as e:
            self.txt_json_preview.setPlainText(f"Error generating preview: {str(e)}")

    def update_dtr_size_label(self):
        if self.scan and hasattr(self.scan, "size_alpha") and hasattr(self.scan, "size_t"):
            self.lbl_dtr_dims.setText(
                f"DTR Size: alpha = {self.scan.size_alpha}, t = {self.scan.size_t}"
            )
        else:
            self.lbl_dtr_dims.setText("DTR Size: alpha = N/A, t = N/A")

    def browse_stages_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Stages Directory", self.txt_stages_dir.text() or "")
        if dir_path:
            self.set_stages_dir(dir_path)

    def set_stages_dir(self, dir_path):
        self.txt_stages_dir.setText(dir_path)
        self.scan_stages_directory(dir_path)
        self.update_json_preview()

    def scan_stages_directory(self, folder_path):
        self.stages_cache.clear()
        self.list_stages.blockSignals(True)
        self.list_stages.clear()
        
        if not folder_path or not os.path.exists(folder_path):
            self.list_stages.blockSignals(False)
            return
            
        # Get all JSON files
        json_files = []
        try:
            for f in os.listdir(folder_path):
                if f.lower().endswith(".json"):
                    json_files.append(f)
            json_files.sort()
        except Exception as e:
            QMessageBox.warning(self, "Read Directory Error", f"Failed to list directory:\n{e}")
            self.list_stages.blockSignals(False)
            return

        # Determine checked list
        checked_stages = getattr(self, "initial_stages_to_check", [])
            
        for f in json_files:
            abs_path = os.path.normpath(os.path.join(folder_path, f))
            stage_obj = self.load_stage_file(abs_path)
            if stage_obj:
                self.stages_cache[abs_path] = stage_obj
                
                item = QListWidgetItem(f)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                
                is_checked = False
                if checked_stages:
                    for cs in checked_stages:
                        config_dir = os.path.dirname(self.current_config_path) if self.current_config_path else os.getcwd()
                        cs_abs = os.path.normpath(os.path.abspath(os.path.join(config_dir, cs)))
                        if cs_abs == os.path.abspath(abs_path) or os.path.basename(cs) == f:
                            is_checked = True
                            break
                else:
                    # If checked_stages is not specified (e.g. startup or clean load)
                    # and stage1.json/stage2.json are present in the directory, check only those.
                    # Otherwise, check all files by default.
                    has_standard_stages = any(sf in ["stage1.json", "stage2.json"] for sf in json_files)
                    if has_standard_stages:
                        is_checked = (f in ["stage1.json", "stage2.json"])
                    else:
                        is_checked = True
                    
                item.setCheckState(Qt.CheckState.Checked if is_checked else Qt.CheckState.Unchecked)
                self.list_stages.addItem(item)
                
        self.list_stages.blockSignals(False)
        
        if self.list_stages.count() > 0:
            self.list_stages.setCurrentRow(0)

    def load_stage_file(self, abs_path):
        try:
            with open(abs_path, 'r') as sf:
                stage_json = json.load(sf)
            
            name = stage_json.get("name", os.path.basename(abs_path))
            optimizer = stage_json.get("classname", "OptimizerPowell")
            maxiter = stage_json.get("kwargs", {}).get("options", {}).get("maxiter", 200)
            ftol = stage_json.get("kwargs", {}).get("options", {}).get("ftol", 1e-12)
            
            s = Stage(name=name, optimizer=optimizer, maxiter=maxiter, ftol=ftol)
            
            param_dict = stage_json.get("parameterization", {})
            if param_dict:
                chain = from_dict(param_dict)
                if isinstance(chain, ParameterizationChain):
                    s.parameterizations = chain.parameterizations
                else:
                    s.parameterizations = [chain]
            return s
        except Exception as e:
            print(f"Failed to load stage file {abs_path}: {e}")
            return None

    def save_single_stage(self, path, stage):
        try:
            chain = ParameterizationChain(stage.parameterizations)
            stage_json = {
                "name": stage.name,
                "module": "xray_epipolar_consistency.optimizer",
                "classname": stage.optimizer,
                "parameterization": chain.to_dict(),
                "kwargs": {
                    "options": {
                        "maxiter": stage.maxiter,
                        "ftol": stage.ftol
                    }
                }
            }
            with open(path, 'w') as sf:
                json.dump(stage_json, sf, indent=2)
        except Exception as e:
            print(f"Failed to save stage file {path}: {e}")

    # ------------------------------------------------------------------
    # OMPL load / save (File menu)
    # ------------------------------------------------------------------

    def load_ompl_action(self):
        """Replace the in-memory trajectory (P_list) with one loaded from an
        OMPL file chosen by the user, without changing any other config."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load OMPL Trajectory", "", "OMPL Files (*.ompl);;All Files (*)"
        )
        if not file_path:
            return
        try:
            new_P = load_ompl(file_path)
            if not new_P:
                QMessageBox.warning(self, "Load OMPL", "No projection matrices found in the selected file.")
                return
            self.P_list = new_P
            self.P_list_original = list(new_P)
            self.dirty = True
            if self.scan:
                self.scan.set_projection_matrices(self.P_list_undersampled)
            self.render_viewports()
            self.statusBar().showMessage(f"Loaded OMPL trajectory ({len(new_P)} views) from: {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Load OMPL Error", f"Failed to load OMPL file:\n{traceback.format_exc()}")

    def save_ompl_action(self):
        """Write the current in-memory P_list to an OMPL file chosen by the user."""
        if not self.P_list:
            QMessageBox.warning(self, "Save OMPL", "No trajectory loaded. Please load a config first.")
            return
        default_name = "trajectory.ompl"
        out_dir = self.txt_out_dir.text().strip()
        if out_dir and os.path.isdir(out_dir):
            default_path = os.path.join(out_dir, default_name)
        else:
            default_path = default_name
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save OMPL Trajectory", default_path, "OMPL Files (*.ompl);;All Files (*)"
        )
        if not file_path:
            return
        try:
            save_ompl(
                self.P_list,
                file_path,
                spacing=self.P_list[0].pixel_spacing,
                detector_size_px=self.P_list[0].image_size
            )
            self.statusBar().showMessage(f"Saved OMPL trajectory ({len(self.P_list)} views) to: {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save OMPL Error", f"Failed to save OMPL file:\n{traceback.format_exc()}")

    def save_project_action(self):
        """Save both geometry_correction.json and trajectory.ompl to output directory."""
        if not self.P_list:
            QMessageBox.warning(self, "Save Project", "No trajectory loaded. Please load a config first.")
            return False
        out_dir = self.txt_out_dir.text().strip()
        if not out_dir or out_dir == "Select Output Folder":
            QMessageBox.warning(self, "No Output Folder", "Please select a valid output folder first.")
            return False
            
        os.makedirs(out_dir, exist_ok=True)
        geom_config_path = os.path.join(out_dir, "geometry_correction.json")
        config_path = self.save_geometry_config(geom_config_path)
        if not config_path:
            return False
            
        ompl_path = os.path.join(out_dir, "trajectory.ompl")
        try:
            save_ompl(
                self.P_list,
                ompl_path,
                spacing=self.P_list[0].pixel_spacing,
                detector_size_px=self.P_list[0].image_size
            )
            self.dirty = False
            self.statusBar().showMessage(f"Project saved successfully to {out_dir}", 5000)
            return True
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save trajectory.ompl:\n{traceback.format_exc()}")
            return False

    # ------------------------------------------------------------------
    # Stage import / export
    # ------------------------------------------------------------------

    def import_stage_action(self):
        """Copy a stage JSON file chosen by the user into the current stages
        directory, then refresh the stage list."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Stage", "", "JSON Files (*.json);;All Files (*)"
        )
        if not file_path:
            return
        stages_dir = self.txt_stages_dir.text().strip()
        if not stages_dir or not os.path.isdir(stages_dir):
            QMessageBox.warning(self, "Import Stage",
                                "Please set a valid Stages Directory before importing.")
            return
        # Validate that the file actually looks like a stage JSON
        try:
            with open(file_path, 'r') as fh:
                data = json.load(fh)
            if "parameterization" not in data:
                QMessageBox.warning(self, "Import Stage",
                                    "The selected file does not appear to be a stage JSON "
                                    "(missing 'parameterization' key).")
                return
        except Exception as e:
            QMessageBox.critical(self, "Import Stage", f"Failed to read file:\n{e}")
            return

        dest = os.path.join(stages_dir, os.path.basename(file_path))
        if os.path.exists(dest):
            reply = QMessageBox.question(
                self, "Import Stage",
                f"A file named '{os.path.basename(file_path)}' already exists in the stages directory.\n"
                "Overwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        try:
            shutil.copy2(file_path, dest)
            self.scan_stages_directory(stages_dir)
            self.statusBar().showMessage(f"Imported stage: {dest}")
        except Exception as e:
            QMessageBox.critical(self, "Import Stage Error", f"Failed to copy file:\n{e}")

    def export_stage_action(self):
        """Save the currently selected stage to a JSON file chosen by the user."""
        stage, abs_path = self.get_active_stage()
        if not stage:
            QMessageBox.warning(self, "Export Stage", "Please select a stage to export.")
            return
        default_name = os.path.basename(abs_path) if abs_path else "stage.json"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Stage", default_name, "JSON Files (*.json);;All Files (*)"
        )
        if not file_path:
            return
        self.save_single_stage(file_path, stage)
        self.statusBar().showMessage(f"Exported stage '{stage.name}' to: {file_path}")

    def on_undersample_changed(self):
        self.load_and_undersample()
        self.update_json_preview()

    def on_view1_changed(self, val):
        self._base_svg.clear()
        if self.view1_slider.maximum() <= 0:
            return
        if val == self.view2_slider.value():
            new_val = val + 1
            if new_val > self.view2_slider.maximum():
                new_val = val - 1
            self.view2_slider.setValue(max(0, new_val))
            
    def on_view2_changed(self, val):
        self._base_svg.clear()
        if self.view2_slider.maximum() <= 0:
            return
        if val == self.view1_slider.value():
            new_val = val + 1
            if new_val > self.view1_slider.maximum():
                new_val = val - 1
            self.view1_slider.setValue(max(0, new_val))
        
    def on_metric_changed(self):
        if self.scan:
            # If the user changed the spacing manually, remember that value
            if self.sender() == self.spin_spacing:
                self._loaded_num_planes = self.spin_spacing.value()
            # If the user changed the object radius manually, remember that value
            elif self.sender() == self.spin_object_radius:
                self._loaded_object_radius = self.spin_object_radius.value()
            # If the size factor changed, and they are using the default spacing, automatically update it
            elif self.sender() == self.spin_size_fac and getattr(self, "_loaded_num_planes", None) is None:
                if self.images_undersampled:
                    first_img = self.images_undersampled[0]
                    dtr_size_factor = self.spin_size_fac.value()
                    size_t = int(np.hypot(*first_img.shape[:2]) * dtr_size_factor)
                    size_alpha = int(np.ceil((np.pi / 2.0) * size_t) // 2)
                    planes_val = int(round((size_alpha + size_t) * 0.5))
                    self.spin_spacing.blockSignals(True)
                    self.spin_spacing.setValue(planes_val)
                    self.spin_spacing.blockSignals(False)

            dtr_factor = self.spin_size_fac.value()
            default_sigma = 0.0 if abs(dtr_factor - 1.0) < 1e-5 else 1.2
            try:
                with QtProgressBarContext(self, "Updating Metric Settings"):
                    self.scan.init_epipolar_consistency(
                        convert_to_line_integral=self.chk_line_integral.isChecked(),
                        gaussian_sigma=getattr(self, '_gaussian_sigma', default_sigma),
                        dtr_size_factor=dtr_factor,
                        num_planes=self.spin_spacing.value(),
                        object_radius_mm=self.spin_object_radius.value()
                    )
            except KeyboardInterrupt:
                self.statusBar().showMessage("Metric update canceled by user.")
                return
        self.update_dtr_size_label()
        self.update_json_preview()

    def browse_recon_json(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Config JSON", "", "JSON Files (*.json)"
        )
        if not file_path:
            return
        
        self.load_config_file(file_path)

    def browse_geom_json(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Geometry Config", "", "JSON Files (*.json)"
        )
        if not file_path:
            return
        
        self.load_config_file(file_path)

    def on_recon_path_edited(self):
        file_path = self.txt_recon_path.text().strip()
        if file_path and file_path != "No file loaded" and os.path.exists(file_path):
            self.load_config_file(file_path)
        else:
            self.update_json_preview()

    def compute_default_output_dir(self, recon_path):
        if not recon_path or not os.path.exists(recon_path):
            return ""
        recon_path_abs = os.path.abspath(recon_path)
        parent_dir = os.path.dirname(recon_path_abs)
        parent_name = os.path.basename(parent_dir)
        timestamp = datetime.datetime.now().strftime("ecc_%Y%m%d_%H%M")
        
        if re.match(r'^ecc_\d{8}_\d{4}$', parent_name):
            grandparent_dir = os.path.dirname(parent_dir)
            return os.path.normpath(os.path.join(grandparent_dir, timestamp))
        else:
            return os.path.normpath(os.path.join(parent_dir, timestamp))

    def load_config_file(self, file_path):
        try:
            config_dir = os.path.dirname(os.path.abspath(file_path))
            with open(file_path, 'r') as f:
                config = json.load(f)

            # Resolve the reconstruction config path (original dataset config)
            recon_json_path = None
            recon_config = config
            if "input_data" in config:
                # Check for reconstruction_config or source_reconstruction_config key in geometry_correction.json
                recon_config_rel = config.get("reconstruction_config") or config.get("source_reconstruction_config")
                if recon_config_rel:
                    test_path = os.path.normpath(os.path.join(config_dir, recon_config_rel))
                    # Only accept if it is not in the same output directory
                    if os.path.exists(test_path) and os.path.dirname(os.path.abspath(test_path)) != config_dir:
                        recon_json_path = test_path
                
                # If not found or was in the same output directory, try to use ../reconstruction.json
                if not recon_json_path:
                    parent_recon_path = os.path.normpath(os.path.join(config_dir, "..", "reconstruction.json"))
                    if os.path.exists(parent_recon_path):
                        recon_json_path = parent_recon_path
                
                # If neither exists, prompt the user to select the complete reconstruction JSON file containing all images
                if not recon_json_path:
                    msg_box = QMessageBox(self)
                    msg_box.setWindowTitle("Select Reconstruction JSON")
                    msg_box.setIcon(QMessageBox.Icon.Question)
                    msg_box.setText("Could not locate the original reconstruction config automatically.\n\nWould you like to manually select the complete reconstruction JSON config containing all images?")
                    msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                    msg_box.setDefaultButton(QMessageBox.StandardButton.Yes)
                    if msg_box.exec() == QMessageBox.StandardButton.Yes:
                        chosen_path, _ = QFileDialog.getOpenFileName(
                            self, "Select Reconstruction Config JSON", "", "JSON Files (*.json)"
                        )
                        if chosen_path and os.path.exists(chosen_path):
                            recon_json_path = chosen_path
                
                # Set original_recon_config_path to the found full config path (only if outside output directory)
                if recon_json_path and os.path.dirname(os.path.abspath(recon_json_path)) != config_dir:
                    self.original_recon_config_path = recon_json_path
                else:
                    self.original_recon_config_path = None
                
                # Fallback to input_data if neither exists (and user did not select one)
                if not recon_json_path:
                    input_data_rel = config.get("input_data")
                    if input_data_rel:
                        recon_json_path = os.path.normpath(os.path.join(config_dir, input_data_rel))
                        if not os.path.exists(recon_json_path):
                            recon_json_path = file_path
                    else:
                        recon_json_path = file_path

                if os.path.exists(recon_json_path) and recon_json_path != file_path:
                    with open(recon_json_path, 'r') as rf:
                        recon_config = json.load(rf)
            
            recon_config_for_peek = recon_config

            # Skip the dataset dialog if the caller set this flag
            # (e.g. returning from the Reconstruction GUI — settings are unchanged)
            if not getattr(self, '_skip_load_dialog', False):
                metric_cfg = config.get("metric_config", {})
                init_dtr_factor = metric_cfg.get("dtr_size_factor", self.spin_size_fac.value())
                init_gaussian_sigma = metric_cfg.get("gaussian_sigma", getattr(self, '_gaussian_sigma', None))

                dlg = ConfigLoadDialog(
                    self,
                    recon_config=recon_config_for_peek,
                    recon_config_path=recon_json_path or file_path,
                    current_undersample=self.spin_undersample.value(),
                    current_dtr_factor=init_dtr_factor,
                    current_gaussian_sigma=init_gaussian_sigma,
                )
                if not dlg.exec():
                    return  # user cancelled

                chosen_undersample   = dlg.chosen_undersample()
                chosen_dtr_factor    = dlg.chosen_dtr_factor()
                self._gaussian_sigma = dlg.chosen_gaussian_sigma()

                self.spin_undersample.blockSignals(True)
                self.spin_undersample.setValue(chosen_undersample)
            else:
                # Dialog skipped — keep current spin values unchanged
                chosen_undersample = self.spin_undersample.value()
                chosen_dtr_factor  = self.spin_size_fac.value()
            self._skip_load_dialog = False  # always reset after load
            self.spin_undersample.blockSignals(False)

            self.spin_size_fac.blockSignals(True)
            self.spin_size_fac.setValue(chosen_dtr_factor)
            self.spin_size_fac.blockSignals(False)
            # ---- end of dialog-sourced settings ----


            metric_cfg = config.get("metric_config", {})
            self.chk_line_integral.blockSignals(True)
            # Use root level convert_to_line_integral or metric_cfg one
            convert_val = metric_cfg.get("convert_to_line_integral")
            if convert_val is None:
                convert_val = config.get("convert_to_line_integral", False)
            self.chk_line_integral.setChecked(bool(convert_val))
            self.chk_line_integral.blockSignals(False)
            
            if "dtr_size_factor" in metric_cfg:
                self.spin_size_fac.blockSignals(True)
                self.spin_size_fac.setValue(metric_cfg["dtr_size_factor"])
                self.spin_size_fac.blockSignals(False)

            if "gaussian_sigma" in metric_cfg:
                self._gaussian_sigma = metric_cfg["gaussian_sigma"]
            
            # Store loaded values to apply them after the scan/images are loaded
            self._loaded_num_planes = metric_cfg.get("num_planes")
            self._loaded_object_radius = metric_cfg.get("object_radius_mm")
            if self._loaded_object_radius is not None and self._loaded_object_radius <= 0.0:
                self._loaded_object_radius = None
            
            self.spin_spacing.blockSignals(True)
            self.spin_spacing.setValue(self._loaded_num_planes if self._loaded_num_planes is not None else 900)
            self.spin_spacing.blockSignals(False)
            
            self.spin_object_radius.blockSignals(True)
            self.spin_object_radius.setValue(self._loaded_object_radius if self._loaded_object_radius is not None else 100.0)
            self.spin_object_radius.blockSignals(False)
            
            self.initial_stages_to_check = []
            
            # Check if this is a geometry optimization config (has input_data, and optionally geometry_optimization)
            if "input_data" in config:
                config_dir = os.path.dirname(os.path.abspath(file_path))
                
                # Check for stages to select
                geom_opt = config.get("geometry_optimization", {})
                stages_to_check = geom_opt.get("stages", [])
                self.initial_stages_to_check = stages_to_check
                
                # Use reconstruction config as our base (resolved in the peek section above)
                self.current_recon_json = recon_config
                
                # Show the original reconstruction config path if available, else recon_json_path
                display_recon_path = self.original_recon_config_path or recon_json_path
                self.txt_recon_path.blockSignals(True)
                self.txt_recon_path.setText(display_recon_path if display_recon_path and os.path.exists(display_recon_path) else file_path)
                self.txt_recon_path.blockSignals(False)
                
                self.current_config_path = file_path
                
                # Restore reconstruction checkbox state based on run_reconstruction key (or fallback to reconstruction_config existence)
                run_recon = config.get("run_reconstruction")
                if run_recon is None:
                    run_recon = "reconstruction_config" in config
                self.chk_run_recon.blockSignals(True)
                self.chk_run_recon.setChecked(bool(run_recon))
                self.chk_run_recon.blockSignals(False)

                self.chk_create_report.blockSignals(True)
                self.chk_create_report.setChecked(config.get("create_report", True))
                self.chk_create_report.blockSignals(False)

                # Restore output directory if specified
                out_dir_rel = config.get("output_dir")
                if out_dir_rel:
                    out_dir_abs = os.path.normpath(os.path.join(config_dir, out_dir_rel))
                    self.txt_out_dir.blockSignals(True)
                    self.txt_out_dir.setText(out_dir_abs)
                    self.txt_out_dir.blockSignals(False)
                else:
                    default_out = self.compute_default_output_dir(recon_json_path if (recon_json_path and os.path.exists(recon_json_path)) else file_path)
                    self.txt_out_dir.blockSignals(True)
                    self.txt_out_dir.setText(default_out)
                    self.txt_out_dir.blockSignals(False)
            else:
                # 2. Reconstruction Config (original workflow)
                self.current_recon_json = config
                self.original_recon_config_path = file_path
                self.txt_recon_path.blockSignals(True)
                self.txt_recon_path.setText(file_path)
                self.txt_recon_path.blockSignals(False)
                
                default_out = self.compute_default_output_dir(file_path)
                # Automatically set default geometry config path in the output directory
                geom_default = os.path.normpath(os.path.join(default_out, "geometry_correction.json"))
                self.current_config_path = geom_default
                
                self.txt_out_dir.blockSignals(True)
                self.txt_out_dir.setText(default_out)
                self.txt_out_dir.blockSignals(False)
            
            # Setup stages directory: look if a stages subdirectory exists next to config file,
            # otherwise fall back to package defaults.
            stages_dir = os.path.normpath(os.path.join(config_dir, "stages"))
            if not os.path.exists(stages_dir):
                stages_dir = str(ecc.get_data_path("tools", "config", "calibration_correction"))
            self.txt_stages_dir.setText(stages_dir)
            
            # Reset cache when loading new config
            self.raw_images = None
            self.P_list = []
            self.P_list_original = []
            self.dirty = False
            
            self.load_and_undersample()
            self.scan_stages_directory(stages_dir)
            self.update_json_preview()
            
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load JSON config:\n{e}")

    def browse_out_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.txt_out_dir.text() or "")
        if dir_path:
            self.txt_out_dir.setText(dir_path)

    def reveal_out_dir(self):
        out_dir = self.txt_out_dir.text().strip()
        if not out_dir:
            self.statusBar().showMessage("No output directory set.")
            return
        if not os.path.isdir(out_dir):
            try:
                os.makedirs(out_dir, exist_ok=True)
            except Exception as e:
                QMessageBox.warning(self, "Reveal Directory", f"Could not create directory:\n{e}")
                return
        QDesktopServices.openUrl(QUrl.fromLocalFile(out_dir))

    def load_and_undersample(self):
        if not self.current_recon_json:
            return
            
        try:
            config = self.current_recon_json
            recon_config_path = self.txt_recon_path.text()
            recon_dir = os.path.dirname(os.path.abspath(recon_config_path)) if recon_config_path else os.getcwd()
            
            # Load OMPL trajectory (cached)
            if not self.P_list:
                # Check for existing OMPL files in the output directory in priority order
                opt_ompl_path = None
                out_dir = self.txt_out_dir.text().strip()
                if out_dir and os.path.isdir(out_dir):
                    for opt_name in ["trajectory_optimized.ompl", "trajectory.ompl", "reconstructed_trajectory.ompl"]:
                        test_path = os.path.normpath(os.path.join(out_dir, opt_name))
                        if os.path.exists(test_path):
                            opt_ompl_path = test_path
                            break
                
                if opt_ompl_path:
                    print(f"Restoring complete trajectory from: {opt_ompl_path}")
                    self.P_list = load_ompl(opt_ompl_path)
                else:
                    ompl_file = config.get("ompl_file")
                    if not ompl_file:
                        raise KeyError("Reconstruction config lacks 'ompl_file'.")
                        
                    if os.path.isabs(ompl_file):
                        ompl_path = ompl_file
                    else:
                        ompl_path = os.path.normpath(os.path.join(recon_dir, ompl_file))
                    print(f"Loading complete trajectory from: {ompl_path}")
                    self.P_list = load_ompl(ompl_path)
                self.P_list_original = list(self.P_list)
            
            # Set voxel geometry
            self.voxel_dimensions = config.get("voxel_dimensions", [100, 100, 100])
            self.model_matrix = np.array(config.get("model_matrix", np.eye(4).tolist()))
            
            # Perform undersampling
            U = self.spin_undersample.value()
            
            # Load images (cached)
            if not self.raw_images:
                image_files = config.get("image_files", [])
                data_dir = config.get("data_dir", "./")
                if os.path.isabs(data_dir):
                    data_dir_path = data_dir
                else:
                    data_dir_path = os.path.normpath(os.path.join(recon_dir, data_dir))
                
                raw_images = []
                if len(image_files) == 1 and image_files[0].lower().endswith('.nrrd'):
                    if os.path.isabs(image_files[0]):
                        image_path = image_files[0]
                    else:
                        image_path = os.path.normpath(os.path.join(data_dir_path, image_files[0]))
                    data, header = nrrd.read(image_path)
                    # data shape is (W, H, num_views) -> transpose to (num_views, H, W)
                    projs = np.transpose(data, (2, 1, 0))
                    raw_images = [projs[i] for i in range(projs.shape[0])]
                else:
                    for img_name in image_files:
                        if os.path.isabs(img_name):
                            img_path = img_name
                        else:
                            img_path = os.path.normpath(os.path.join(data_dir_path, img_name))
                        if img_path.lower().endswith('.nrrd'):
                            img, _ = nrrd.read(img_path)
                            img = np.squeeze(img)
                            if img.ndim == 2:
                                img = img.T
                            raw_images.append(img)
                        else:
                            raw_images.append(imread(img_path))
                self.raw_images = raw_images
                        
            # Undersample images
            self.images_undersampled = self.raw_images[::U]
            
            # Crop trajectory lists if sizes mismatch
            min_size = min(len(self.P_list_undersampled), len(self.images_undersampled))
            self.images_undersampled = self.images_undersampled[:min_size]
            
            # Instantiate Scan object
            self.scan = Scan(self.images_undersampled, self.P_list_undersampled)
            
            # Determine and set default or loaded planes in spin box
            if getattr(self, "_loaded_num_planes", None) is not None:
                planes_val = self._loaded_num_planes
            else:
                first_img = self.images_undersampled[0]
                dtr_size_factor = self.spin_size_fac.value()
                size_t = int(np.hypot(*first_img.shape[:2]) * dtr_size_factor)
                size_alpha = int(np.ceil((np.pi / 2.0) * size_t) // 2)
                planes_val = int(round((size_alpha + size_t) * 0.5))
            
            self.spin_spacing.blockSignals(True)
            self.spin_spacing.setValue(planes_val)
            self.spin_spacing.blockSignals(False)
            
            # Determine and set default or loaded object_radius_mm in spin box
            if getattr(self, "_loaded_object_radius", None) is not None and self._loaded_object_radius > 0.0:
                radius_val = self._loaded_object_radius
            else:
                est_geom = self.scan._estimate_iso_center_and_object_radius()
                radius_val = est_geom.get("object_radius_mm", 100.0)
                if radius_val <= 0.0:
                    radius_val = 100.0
            
            self.spin_object_radius.blockSignals(True)
            self.spin_object_radius.setValue(radius_val)
            self.spin_object_radius.blockSignals(False)
            
            # Initialize epipolar consistency on GPU
            dtr_factor = self.spin_size_fac.value()
            default_sigma = 0.0 if abs(dtr_factor - 1.0) < 1e-5 else 1.2
            try:
                with QtProgressBarContext(self, "Initializing Epipolar Consistency"):
                    self.scan.init_epipolar_consistency(
                        convert_to_line_integral=self.chk_line_integral.isChecked(),
                        gaussian_sigma=getattr(self, '_gaussian_sigma', default_sigma),
                        dtr_size_factor=dtr_factor,
                        num_planes=self.spin_spacing.value(),
                        object_radius_mm=self.spin_object_radius.value()
                    )
            except KeyboardInterrupt:
                self.statusBar().showMessage("Scan initialization canceled by user.")
                self.scan = None
                return
            
            # Update info labels in Tab 3
            self.lbl_object_radius.setText(f"Estimated Object Size (z_max): {self.scan.object_radius_mm:.2f} mm")
            self.lbl_planes_needed.setText(f"Expected Planes: {self.scan.size_t + self.scan.size_alpha}")
            self.btn_compute_diag.setEnabled(True)
            self.update_dtr_size_label()
            
            # Setup view 1 limits
            self.view1_slider.setRange(0, min_size - 1)
            self.view1_spin.setRange(0, min_size - 1)
            self.view1_slider.setValue(0)
            
            # Setup view 2 limits
            self.view2_slider.setRange(0, min_size - 1)
            self.view2_spin.setRange(0, min_size - 1)
            self.view2_slider.setValue(min(1, min_size - 1))
            
            # Reset clicked point when new data is loaded
            self.clicked_point_viewport = None
            self.clicked_point_coords = None
            
            if self.images_undersampled:
                first_img = self.images_undersampled[0]
                H, W = first_img.shape
                aspect = W / H
                self.viewport1_wrapper.setAspectRatio(aspect)
                self.viewport2_wrapper.setAspectRatio(aspect)
                self.viewport1.setImageSize(W, H)
                self.viewport2.setImageSize(W, H)
            
            # Trigger render
            self.render_viewports()
            
            self.statusBar().showMessage(f"Loaded config. Computed RadonIntermediate functions for {min_size} views.")
            
        except Exception as e:
            QMessageBox.critical(self, "Loading Error", f"Failed to initialize trajectory/images:\n{e}")

    def on_viewport_hover(self, view_idx, u, v):
        """Fast path: just draw a dashed hover line on the cached base SVG."""
        self.hover_point_viewport = view_idx
        self.hover_point_coords   = (u, v)
        self.render_hover_overlay()

    def on_plot_hover(self, event):
        """Matplotlib motion_notify_event: x-axis IS kappa in degrees — draw
        the corresponding epipolar lines in both image viewports and dots in
        both DTR viewports with zero GPU work."""
        if event.xdata is None or not self._base_svg:
            return
        kappa = np.radians(event.xdata)
        self._draw_kappa_overlay(kappa)
        # Update the vline artist in-place
        if self._redundancy_artists is not None:
            try:
                self._redundancy_artists[2].set_xdata([event.xdata])
                self._redundancy_artists[2].set_alpha(0.85)
                self.canvas_redundancy.draw_idle()
            except Exception:
                pass

    def on_plot_leave(self, event):
        """Mouse left the redundancy plot axes — restore base SVGs."""
        self.on_viewport_leave()

    def _draw_kappa_overlay(self, kappa):
        """Shared helper: given a kappa angle (radians), inject one dashed cyan
        epipolar line into each image viewport SVG and one dashed dot into each
        DTR viewport SVG. Uses only cached data — no GPU, no matplotlib redraw."""
        # ---- image viewports ----
        for view_idx, widget in ((self.view1_slider.value(), self.viewport1),
                                 (self.view2_slider.value(), self.viewport2)):
            cache = self._base_svg.get(view_idx)
            if cache is None:
                continue
            base_svg, P_invT, K_pencil, E0, E90, W, H = cache
            try:
                l = hessianNormalForm(
                    (P_invT @ (K_pencil @ [np.cos(kappa), np.sin(kappa)])).flatten()
                ).flatten()
                a, b, c = l
                if abs(b) > abs(a):
                    x0, x1 = 0.0, float(W)
                    y0, y1 = -(a * x0 + c) / b, -(a * x1 + c) / b
                else:
                    y0, y1 = 0.0, float(H)
                    x0, x1 = -(b * y0 + c) / a, -(b * y1 + c) / a
                elem = (
                    f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" '
                    f'stroke="#00eeff" stroke-width="2" stroke-dasharray="8,6" />'
                )
                widget.load(base_svg.replace("</svg>", elem + "\n</svg>", 1).encode('utf-8'))
            except Exception:
                pass

        # ---- DTR viewports ----
        for view_idx, widget in ((self.view1_slider.value(), self.dtr_viewport1),
                                 (self.view2_slider.value(), self.dtr_viewport2)):
            dtr_cache = self._dtr_base_svg.get(view_idx)
            if dtr_cache is None:
                continue
            dtr_svg_str, P_invT_dtr, K_pencil_d, E0_d, E90_d, w, h, size_alpha, size_t, range_t = dtr_cache
            try:
                l = (P_invT_dtr @ (K_pencil_d @ [np.cos(kappa), np.sin(kappa)])).flatten()
                T_c = np.array([[1, 0, 0], [0, 1, 0], [w * 0.5, h * 0.5, 1.0]])
                l_c = hessianNormalForm(T_c @ l).flatten()
                a_val = np.arctan2(l_c[1], l_c[0]) / np.pi
                if a_val < 0:
                    a_val += 2.0
                t_val = -l_c[2] / range_t + 0.5
                if a_val > 1.0:
                    a_val -= 1.0
                    t_val = 1.0 - t_val
                a_px = a_val * (size_alpha - 1)
                t_px = t_val * (size_t - 1)
                dot = (
                    f'<circle cx="{a_px:.2f}" cy="{t_px:.2f}" r="12" '
                    f'fill="none" stroke="#00eeff" stroke-width="2" stroke-dasharray="4,3" />'
                )
                widget.load(dtr_svg_str.replace("</svg>", dot + "\n</svg>", 1).encode('utf-8'))
            except Exception:
                pass

    def on_viewport_leave(self):
        """Restore both image viewports to their base SVGs (remove hover overlay)."""
        for view_idx, widget in ((self.view1_slider.value(), self.viewport1),
                                 (self.view2_slider.value(), self.viewport2)):
            cache = self._base_svg.get(view_idx)
            if cache:
                widget.load(cache[0].encode('utf-8'))
        # Also restore DTR viewports
        for view_idx, widget in ((self.view1_slider.value(), self.dtr_viewport1),
                                 (self.view2_slider.value(), self.dtr_viewport2)):
            cache = self._dtr_base_svg.get(view_idx)
            if cache:
                widget.load(cache[0].encode('utf-8'))
        # Hide the hover vline in the plot
        if self._redundancy_artists is not None:
            try:
                self._redundancy_artists[2].set_alpha(0.0)
                self.canvas_redundancy.draw_idle()
            except Exception:
                pass

    def on_dtr_hover(self, dtr_view_idx, u, v):
        """Mouse moved over a DTR viewport.
        Compute the (alpha, t) → image line and draw it as a dashed cyan line
        in the *corresponding* image viewport only (Radon space belongs to one view)."""
        dtr_cache = self._dtr_base_svg.get(
            self.view1_slider.value() if dtr_view_idx == 1
            else self.view2_slider.value()
        )
        if dtr_cache is None:
            return
        dtr_svg_str, P_invT_dtr, K_pencil_d, E0_d, E90_d, w, h, size_alpha, size_t, range_t = dtr_cache

        # Convert pixel position to (a_val, t_val) in [0,1]
        a_val = u / max(1, size_alpha - 1)
        t_val = v / max(1, size_t - 1)

        # Inverse of map_line_to_Radon_space
        # Undo the >1 wrap (we don't know which branch was taken, but a_val ∈ [0,1])
        theta = a_val * np.pi                          # angle of line normal
        t_signed = -(0.5 - t_val) * range_t           # signed distance from image centre

        # Reconstruct centered homogeneous line [cos θ, sin θ, -t]
        l_c = np.array([np.cos(theta), np.sin(theta), -t_signed])

        # Undo centering: T_center_star^{-1} = [[1,0,0],[0,1,0],[-cx,-cy,1]]
        cx, cy = w * 0.5, h * 0.5
        T_inv = np.array([[1, 0, 0], [0, 1, 0], [-cx, -cy, 1.0]])
        l_img = T_inv @ l_c  # line in image pixel coordinates

        # Draw it in the matching image viewport only
        img_widget = self.viewport1 if dtr_view_idx == 1 else self.viewport2
        img_cache = self._base_svg.get(
            self.view1_slider.value() if dtr_view_idx == 1
            else self.view2_slider.value()
        )
        if img_cache is None:
            return
        base_svg, _, _, _, _, W, H = img_cache
        try:
            a, b, c = l_img
            if abs(b) > abs(a):
                x0, x1 = 0.0, float(W)
                y0, y1 = -(a * x0 + c) / b, -(a * x1 + c) / b
            else:
                y0, y1 = 0.0, float(H)
                x0, x1 = -(b * y0 + c) / a, -(b * y1 + c) / a
            elem = (
                f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" '
                f'stroke="#00eeff" stroke-width="2" stroke-dasharray="8,6" />'
            )
            img_widget.load(base_svg.replace("</svg>", elem + "\n</svg>", 1).encode('utf-8'))
        except Exception:
            pass

        # Also update the plot hover vline using the kappa that corresponds to this Radon line
        # (the DTR stores lines indexed by kappa — the curve in the DTR IS the kappa mapping)
        # We don't have a direct kappa here, but we can derive it from E0/E90 if needed.
        # For now just hide the plot vline when hovering the DTR (different interaction mode).
        if self._redundancy_artists is not None:
            try:
                self._redundancy_artists[2].set_alpha(0.0)
                self.canvas_redundancy.draw_idle()
            except Exception:
                pass

    def on_viewport_clicked(self, view_idx, u, v):
        """Slow path: update the clicked point and do a full render + plot."""
        self.clicked_point_viewport = view_idx
        self.clicked_point_coords   = (u, v)
        self.render_viewports()

    def render_viewports(self):
        if not self.scan or not self.P_list_undersampled:
            return
            
        idx1 = self.view1_slider.value()
        idx2 = self.view2_slider.value()
        if idx1 >= len(self.P_list_undersampled) or idx2 >= len(self.P_list_undersampled):
            return
            
        if idx1 == idx2:
            return
            
        stage, _ = self.get_active_stage()
        if stage:
            chain = ParameterizationChain(stage.parameterizations)
            P1_obj = chain.apply_to_trajectory([self.P_list_undersampled[idx1]])[0]
            P2_obj = chain.apply_to_trajectory([self.P_list_undersampled[idx2]])[0]
        else:
            P1_obj = self.P_list_undersampled[idx1]
            P2_obj = self.P_list_undersampled[idx2]
            
        P1 = P1_obj.P
        C1 = P1_obj.getCenterOfProjection().flatten()
        P2 = P2_obj.P
        C2 = P2_obj.getCenterOfProjection().flatten()
        
        # Baseline joining camera centers
        B = pluecker.join_points(C1, C2)
        Bx_dual = pluecker.matrixDual(B)
        E0 = pluecker.join(B, np.array([0, 0, 0, 1]))
        E90 = Bx_dual @ Bx_dual @ np.array([0, 0, 0, 1])
        E0 = hessianNormalForm(E0).flatten()
        E90 = hessianNormalForm(E90).flatten()
        K_pencil = np.column_stack([E0, E90])
        
        # Get active pair consistency & profiles
        stage, _ = self.get_active_stage()
        if stage:
            chain = ParameterizationChain(stage.parameterizations)
            Ps_active = chain.apply_to_trajectory(self.P_list_undersampled)
        else:
            Ps_active = self.P_list_undersampled
            
        Ps_aligned = [P.P @ self.scan.T_norm for P in Ps_active]
        self.scan.metric.setProjectionMatrices(Ps_aligned)
        
        cost, v0, v1, kappas, _ = self.scan.metric.getRedundantSignalsForViews(idx1, idx2)

        
        # Update viewports (Image 1, Image 2, Dtr 1, Dtr 2)
        self.render_single_viewport(idx1, self.viewport1, P1_obj, P2_obj, K_pencil, E0, E90, kappas)
        self.render_single_viewport(idx2, self.viewport2, P1_obj, P2_obj, K_pencil, E0, E90, kappas)
        self.render_single_dtr_viewport(idx1, self.dtr_viewport1, True, P1_obj, P2_obj, K_pencil, E0, E90, kappas)
        self.render_single_dtr_viewport(idx2, self.dtr_viewport2, False, P1_obj, P2_obj, K_pencil, E0, E90, kappas)
        
        # Update Matplotlib Redundancy Profiles Plot
        kappas_deg = np.degrees(kappas)
        v0_data = v0[:len(kappas)]
        v1_data = v1[:len(kappas)]

        if self._redundancy_artists is None:
            # First render: create persistent artists
            self.ax_redundancy.clear()
            line0, = self.ax_redundancy.plot(kappas_deg, v0_data,
                color='red', alpha=0.7, label=f'View {idx1} (Left)')
            line1, = self.ax_redundancy.plot(kappas_deg, v1_data,
                color='green', alpha=0.7, label=f'View {idx2} (Right)')
            vline_hover = self.ax_redundancy.axvline(
                x=0, color='cyan', linestyle='--', linewidth=1.5,
                alpha=0.0, label='_hover')

            vline_click = self.ax_redundancy.axvline(
                x=0, color='yellow', linestyle='--', linewidth=2,
                alpha=0.0, label='Clicked')
            self.ax_redundancy.set_xlabel('Kappa angle (degrees)')
            self.ax_redundancy.set_ylabel('[a.u.]')
            self.ax_redundancy.yaxis.set_major_formatter(NullFormatter())
            self.ax_redundancy.grid(True, linestyle=':', alpha=0.6)
            self._redundancy_artists = (line0, line1, vline_hover, vline_click)
        else:
            line0, line1, vline_hover, vline_click = self._redundancy_artists
            line0.set_xdata(kappas_deg)
            line0.set_ydata(v0_data)
            line0.set_label(f'View {idx1} (Left)')
            line1.set_xdata(kappas_deg)
            line1.set_ydata(v1_data)
            line1.set_label(f'View {idx2} (Right)')
            self.ax_redundancy.relim()
            self.ax_redundancy.autoscale_view()

        self.ax_redundancy.set_title(
            f'Epipolar Consistency Redundancy Profiles  (ECC cost: {cost:.4e})')

        # Clicked kappa vline
        if self.clicked_point_viewport is not None and self.clicked_point_coords is not None:
            c_view = self.clicked_point_viewport
            cu, cv = self.clicked_point_coords
            x_click = np.array([cu, cv, 1.0])
            P_src = P1_obj if c_view == 1 else P2_obj
            X_pixel = P_src.pseudoinverse() @ x_click
            kappa_click = (np.arctan2(-np.dot(E0, X_pixel),
                                       np.dot(E90, X_pixel))
                           + np.pi / 2) % np.pi - np.pi / 2
            vline_click.set_xdata([np.degrees(kappa_click)])
            vline_click.set_alpha(1.0)
            vline_click.set_label(f'Clicked ({np.degrees(kappa_click):.1f}°)')
        else:
            vline_click.set_alpha(0.0)

        vline_hover.set_alpha(0.0)  # hidden until mouse moves
        self.ax_redundancy.legend(loc='upper right', fontsize='small')
        self.canvas_redundancy.draw_idle()


    def render_single_viewport(self, active_idx, viewport_widget, P1_obj, P2_obj, K_pencil, E0, E90, kappas):
        if not self.scan or not self.P_list_undersampled:
            return
            
        try:
            if active_idx >= len(self.P_list_undersampled):
                return
                
            image = self.images_undersampled[active_idx]
            H, W = image.shape
            
            # Convert NumPy image to PIL background
            img_min, img_max = image.min(), image.max()
            img_norm = ((image - img_min) / (img_max - img_min + 1e-8) * 255.0).astype(np.uint8)
            pil_img = Image.fromarray(img_norm)
            
            # Create Composer with canvas dimensions and add the image manually with explicit width/height
            svg = Composer((W, H))
            svg.add(svg_image, data=pil_img, width=W, height=H)
            
            # Determine colors based on viewport (View 1 = red border, green lines; View 2 = green border, red lines)
            is_view1 = (viewport_widget == self.viewport1)
            border_color = "red" if is_view1 else "green"
            family_color = "#2dff5560" if is_view1 else "#ff2d5560"
            
            # Draw border rectangle using rect from svg_snip.Elements
            svg.add(rect, x=0, y=0, width=W, height=H, stroke=border_color, stroke_width=5, fill="none")
            
            # Accumulate all parameterizations up to the active stage/chain
            # (or use the currently selected stage parameterizations)
            stage, _ = self.get_active_stage()
            if stage:
                chain = ParameterizationChain(stage.parameterizations)
                
                # Apply parameterizations to active matrix
                P_active_orig = self.P_list_undersampled[active_idx]
                P_active = chain.apply_to_trajectory([P_active_orig])[0]
            else:
                P_active = self.P_list_undersampled[active_idx]
                
            # Project volume cube onto image coordinates
            Nx, Ny, Nz = self.voxel_dimensions
            P_local = P_active.P @ self.model_matrix
            
            # Draw wireframe cube (blue color as requested by user)
            svg.add(
                e3d.wire_cube,
                P=P_local,
                min=[0, 0, 0],
                max=[Nx, Ny, Nz],
                stroke="#0055ff",
                stroke_width=1.5
            )
            
            # Sample kappas to avoid rendering too many lines
            step = max(1, len(kappas) // 25)
            kappas_py = kappas[::step]
            
            # Project family of lines onto this viewport
            P_invT = P_active.pseudoinverse().T
            
            for k in kappas_py:
                E_kappa = K_pencil @ [np.cos(k), np.sin(k)]
                l = (P_invT @ E_kappa).flatten()
                l = hessianNormalForm(l).flatten()
                svg.add(svg_homogeneous_line, l=l, stroke=family_color, stroke_width=1.0)
            
            # Draw epipole if it is inside the detector
            P_other = P2_obj if is_view1 else P1_obj
            C_other = P_other.getCenterOfProjection()
            epipole_hom = (P_active.P @ C_other).flatten()
            if abs(epipole_hom[2]) > 1e-8:
                ex = epipole_hom[0] / epipole_hom[2]
                ey = epipole_hom[1] / epipole_hom[2]
                if 0 <= ex < W and 0 <= ey < H:
                    def draw_epipole(**kwargs):
                        return f'<circle cx="{ex:.2f}" cy="{ey:.2f}" r="6" fill="#00ffff" stroke="white" stroke-width="1.5" />'
                    svg.add(draw_epipole)
            
            # Draw clicked epipolar line if click is active
            if self.clicked_point_viewport is not None and self.clicked_point_coords is not None:
                c_view = self.clicked_point_viewport
                cu, cv = self.clicked_point_coords
                
                x_click = np.array([cu, cv, 1.0])
                if c_view == 1:
                    P1_pinv = P1_obj.pseudoinverse()
                    X_pixel = P1_pinv @ x_click
                else:
                    P2_pinv = P2_obj.pseudoinverse()
                    X_pixel = P2_pinv @ x_click
                    
                # Find kappa corresponding to X_pixel
                val0 = np.dot(E0, X_pixel)
                val90 = np.dot(E90, X_pixel)
                kappa_click = np.arctan2(-val0, val90)
                # Map kappa_click to principal interval [-pi/2, pi/2]
                kappa_click = (kappa_click + np.pi/2) % np.pi - np.pi/2
                
                # Compute specific epipolar line for this view
                E_kappa_click = K_pencil @ [np.cos(kappa_click), np.sin(kappa_click)]
                l_click = (P_invT @ E_kappa_click).flatten()
                l_click = hessianNormalForm(l_click).flatten()
                
                # Highlight clicked line in yellow
                svg.add(svg_homogeneous_line, l=l_click, stroke="yellow", stroke_width=3.0)
                
                if (is_view1 and c_view == 1) or (not is_view1 and c_view == 2):
                    def draw_clicked_point(**kwargs):
                        return f'<circle cx="{cu:.2f}" cy="{cv:.2f}" r="5" fill="yellow" stroke="white" stroke-width="1.5" />'
                    svg.add(draw_clicked_point)
            
            # Render SVG and display
            raw_svg = svg.render()
            # Cache for hover overlay (7-tuple consumed by render_hover_overlay)
            self._base_svg[active_idx] = (raw_svg, P_invT, K_pencil, E0, E90, W, H)
            viewport_widget.load(raw_svg.encode('utf-8'))
            
        except Exception as e:
            self.statusBar().showMessage(f"Viewport rendering error: {str(e)}")

    def render_hover_overlay(self):
        """Back-project the hover pixel → kappa, then delegate to _draw_kappa_overlay."""
        if not self._base_svg:
            return
        hv = self.hover_point_viewport
        if hv is None or self.hover_point_coords is None:
            return
        hu, hv_coord = self.hover_point_coords

        first_cache = next(iter(self._base_svg.values()))
        _, _, K_pencil, E0, E90, _, _ = first_cache

        try:
            P_src = self.P_list_undersampled[
                self.view1_slider.value() if hv == 1 else self.view2_slider.value()
            ]
            X_hover = P_src.pseudoinverse() @ np.array([hu, hv_coord, 1.0])
            kappa = (np.arctan2(-np.dot(E0, X_hover), np.dot(E90, X_hover))
                     + np.pi / 2) % np.pi - np.pi / 2
        except Exception:
            return

        self._draw_kappa_overlay(kappa)

        if self._redundancy_artists is not None:
            try:
                self._redundancy_artists[2].set_xdata([np.degrees(kappa)])
                self._redundancy_artists[2].set_alpha(0.85)
                self.canvas_redundancy.draw_idle()
            except Exception:
                pass


    def render_single_dtr_viewport(self, active_idx, viewport_widget, is_view1, P1_obj, P2_obj, K_pencil, E0, E90, kappas):
        if not self.scan or not self.P_list_undersampled:
            return
            
        try:
            dtr_obj = self.scan.dtrs[active_idx]
            raw_dtr_data = dtr_obj.get_data()
            
            dtr_min, dtr_max = raw_dtr_data.min(), raw_dtr_data.max()
            dtr_norm = np.clip((raw_dtr_data - dtr_min) / (dtr_max - dtr_min + 1e-8) * 255.0, 0, 255).astype(np.uint8)
            pil_dtr = Image.fromarray(dtr_norm)
            
            size_alpha = self.scan.size_alpha
            size_t = self.scan.size_t
            dtr_svg = Composer((size_alpha, size_t))
            
            # Add background image
            dtr_svg.add(svg_image, data=pil_dtr, width=size_alpha, height=size_t)
            
            # Get project image size for Radon mapping
            w = self.images_undersampled[active_idx].shape[1]
            h = self.images_undersampled[active_idx].shape[0]
            range_t = np.hypot(w, h)
            
            def map_line_to_Radon_space(l):
                T_center_star = np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [w * 0.5, h * 0.5, 1.0]
                ])
                l_centered = hessianNormalForm(T_center_star @ l).flatten()

                a_val = np.atan2(l_centered[1], l_centered[0]) / np.pi
                if a_val < 0:
                    a_val += 2.0
                t_val = -l_centered[2] / range_t + 0.5
                sign = 1.0
                if a_val > 1.0:
                    a_val -= 1.0
                    t_val = 1.0 - t_val
                    sign = -1.0
                t_idx = t_val * (size_t - 1)
                a_idx = a_val * (size_alpha - 1)
                return a_idx, t_idx, sign
            
            # Active projection matrix P_active
            stage, _ = self.get_active_stage()
            if stage:
                chain = ParameterizationChain(stage.parameterizations)
                P_active = chain.apply_to_trajectory([self.P_list_undersampled[active_idx]])[0]
            else:
                P_active = self.P_list_undersampled[active_idx]
                
            P_invT = P_active.pseudoinverse().T
            
            pts_list = []
            for k in kappas:
                E_kappa = K_pencil @ [np.cos(k), np.sin(k)]
                l = (P_invT @ E_kappa).flatten()
                a_idx, t_idx, _ = map_line_to_Radon_space(l)
                pts_list.append((a_idx, t_idx))
                
            # Draw the curve as a polyline
            pts_str = " ".join([f"{a:.2f},{t:.2f}" for a, t in pts_list])
            curve_color = "green" if is_view1 else "red"
            dtr_svg.add(polyline, points=pts_str, stroke=curve_color, stroke_width=2.0, fill="none")
            
            # Draw clicked point on the curve if click is active
            if self.clicked_point_viewport is not None and self.clicked_point_coords is not None:
                c_view = self.clicked_point_viewport
                cu, cv = self.clicked_point_coords
                
                # Backproject clicked point to 3D point
                x_click = np.array([cu, cv, 1.0])
                if c_view == 1:
                    P1_pinv = P1_obj.pseudoinverse()
                    X_pixel = P1_pinv @ x_click
                else:
                    P2_pinv = P2_obj.pseudoinverse()
                    X_pixel = P2_pinv @ x_click
                    
                val0 = np.dot(E0, X_pixel)
                val90 = np.dot(E90, X_pixel)
                kappa_click = np.arctan2(-val0, val90)
                # Map kappa_click to principal interval [-pi/2, pi/2]
                kappa_click = (kappa_click + np.pi/2) % np.pi - np.pi/2
                
                E_kappa_click = K_pencil @ [np.cos(kappa_click), np.sin(kappa_click)]
                l_click = (P_invT @ E_kappa_click).flatten()
                a_idx_click, t_idx_click, _ = map_line_to_Radon_space(l_click)
                
                # Draw a circle on the curve
                def draw_clicked_point_dtr(**kwargs):
                    return f'<circle cx="{a_idx_click:.2f}" cy="{t_idx_click:.2f}" r="10" fill="yellow" stroke="white" stroke-width="1.5" />'
                dtr_svg.add(draw_clicked_point_dtr)
            
            # Render SVG and display; also cache for hover dot overlay
            raw_svg = dtr_svg.render()
            stage2, _ = self.get_active_stage()
            if stage2:
                chain2 = ParameterizationChain(stage2.parameterizations)
                P_invT_dtr = chain2.apply_to_trajectory([self.P_list_undersampled[active_idx]])[0].pseudoinverse().T
            else:
                P_invT_dtr = self.P_list_undersampled[active_idx].pseudoinverse().T
            self._dtr_base_svg[active_idx] = (
                raw_svg, P_invT_dtr, K_pencil, E0, E90,
                w, h, size_alpha, size_t, range_t
            )
            # Tell the widget its logical image size so hover coords are correct
            if hasattr(viewport_widget, 'setImageSize'):
                viewport_widget.setImageSize(size_alpha, size_t)
            viewport_widget.load(raw_svg.encode('utf-8'))


        except Exception as e:
            self.statusBar().showMessage(f"Dtr Viewport rendering error: {str(e)}")

    def plot_sweep_action(self, param_name, param_obj):
        if not self.scan:
            QMessageBox.warning(self, "No Trajectory", "Please load a reconstruction config first.")
            return
            
        try:
            # Generate 1D Parameter sweep of epipolar consistency cost
            p_info = param_obj.parameters[param_name]
            original_val = p_info["value"]
            range_min, range_max = p_info["range"]
            
            samples = np.linspace(range_min, range_max, 51)
            costs = []
            
            # Temporary trajectory chain to isolate target parameter
            temp_chain = ParameterizationChain([param_obj])
            
            # Evaluate costs
            for v in samples:
                p_info["value"] = v
                Ps_temp = temp_chain.apply_to_trajectory(self.P_list_undersampled)
                costs.append(self.scan.compute_ecc_for_projection_matrices([Ps_temp])[0])
                
            p_info["value"] = original_val # Restore original value
            
            # Clean up closed windows from the list
            active = []
            for w in self.sweep_windows:
                try:
                    if w.isVisible():
                        active.append(w)
                except RuntimeError:
                    pass
            self.sweep_windows = active
            
            # Initialize new separate sweep window
            sweep_win = SweepWindow(self, f"1D Consistency Sweep Plot: {param_name}")
            self.sweep_windows.append(sweep_win)
            sweep_win.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
                
            # Draw plot on the matplotlib canvas
            sweep_win.ax.clear()
            sweep_win.ax.plot(samples, costs, color='#1f77b4', linewidth=2)
            sweep_win.ax.axvline(original_val, color='red', linestyle='--', label=f'Current ({original_val:.4f})')
            sweep_win.ax.set_xlabel(param_name)
            sweep_win.ax.set_ylabel('Consistency cost [a.u.]')
            sweep_win.ax.set_title(f"ECC Sweep vs {param_name}")
            sweep_win.ax.legend()
            sweep_win.ax.grid(True, linestyle=':', alpha=0.6)
            sweep_win.canvas.draw()
            
            sweep_win.show()
            sweep_win.raise_()
            
        except Exception as e:
            QMessageBox.critical(self, "Sweep Error", f"Failed to compute sweep:\n{e}")

    def update_2d_sweep_params(self):
        stage, _ = self.get_active_stage()
        self.combo_2d_class1.blockSignals(True)
        self.combo_2d_class2.blockSignals(True)
        self.combo_2d_class1.clear()
        self.combo_2d_class2.clear()
        

            
        # First, scan stage parameterizations and update our instances pool
        if stage:
            for p_obj in stage.parameterizations:
                self.sweep_param_instances[p_obj.__class__] = p_obj
                
        # Now, populate dropdowns with all non-TimeVariant classes from AVAILABLE_PARAMS
        for label, cls in AVAILABLE_PARAMS.items():
            if issubclass(cls, TimeVariant):
                continue
                
            # Get or create instance
            p_obj = self.sweep_param_instances.get(cls)
            if p_obj is None:
                p_obj = cls()
                self.sweep_param_instances[cls] = p_obj
                
            self.combo_2d_class1.addItem(label, p_obj)
            self.combo_2d_class2.addItem(label, p_obj)
            
        self.combo_2d_class1.blockSignals(False)
        self.combo_2d_class2.blockSignals(False)
        
        # Trigger updates to parameters dropdowns
        self.on_class1_changed()
        self.on_class2_changed()

    def on_class1_changed(self):
        p_obj = self.combo_2d_class1.currentData()
        self.combo_2d_param1.blockSignals(True)
        self.combo_2d_param1.clear()
        if p_obj:
            for p_name in p_obj.parameters.keys():
                self.combo_2d_param1.addItem(p_name, (p_obj, p_name))
        self.combo_2d_param1.blockSignals(False)
        self.on_param1_changed()

    def on_class2_changed(self):
        p_obj = self.combo_2d_class2.currentData()
        self.combo_2d_param2.blockSignals(True)
        self.combo_2d_param2.clear()
        if p_obj:
            for p_name in p_obj.parameters.keys():
                self.combo_2d_param2.addItem(p_name, (p_obj, p_name))
        self.combo_2d_param2.blockSignals(False)
        self.on_param2_changed()

    def on_param1_changed(self):
        data = self.combo_2d_param1.currentData()
        self.spin_2d_min1.blockSignals(True)
        self.spin_2d_max1.blockSignals(True)
        if data:
            p_obj, p_name = data
            config = p_obj.parameters.get(p_name, {})
            r = config.get("range", (-1.0, 1.0))
            self.spin_2d_min1.setValue(r[0])
            self.spin_2d_max1.setValue(r[1])
            self.spin_2d_min1.setEnabled(True)
            self.spin_2d_max1.setEnabled(True)
        else:
            self.spin_2d_min1.setValue(0.0)
            self.spin_2d_max1.setValue(0.0)
            self.spin_2d_min1.setEnabled(False)
            self.spin_2d_max1.setEnabled(False)
        self.spin_2d_min1.blockSignals(False)
        self.spin_2d_max1.blockSignals(False)

    def on_param2_changed(self):
        data = self.combo_2d_param2.currentData()
        self.spin_2d_min2.blockSignals(True)
        self.spin_2d_max2.blockSignals(True)
        if data:
            p_obj, p_name = data
            config = p_obj.parameters.get(p_name, {})
            r = config.get("range", (-1.0, 1.0))
            self.spin_2d_min2.setValue(r[0])
            self.spin_2d_max2.setValue(r[1])
            self.spin_2d_min2.setEnabled(True)
            self.spin_2d_max2.setEnabled(True)
        else:
            self.spin_2d_min2.setValue(0.0)
            self.spin_2d_max2.setValue(0.0)
            self.spin_2d_min2.setEnabled(False)
            self.spin_2d_max2.setEnabled(False)
        self.spin_2d_min2.blockSignals(False)
        self.spin_2d_max2.blockSignals(False)

    def on_spin_range_changed(self):
        # Update underlying config when spin box values are edited
        data1 = self.combo_2d_param1.currentData()
        if data1:
            p_obj, p_name = data1
            p_obj.parameters[p_name]["range"] = (self.spin_2d_min1.value(), self.spin_2d_max1.value())
            
        data2 = self.combo_2d_param2.currentData()
        if data2:
            p_obj, p_name = data2
            p_obj.parameters[p_name]["range"] = (self.spin_2d_min2.value(), self.spin_2d_max2.value())
            
        # Update GUI displays (preview / parameter details table)
        self.update_json_preview()
        # If the parameter details table has the modified parameter active, redraw it
        row = self.list_params.currentRow()
        if row >= 0:
            self.select_parameterization(row)

    def plot_2d_sweep_action(self):
        if not self.scan:
            QMessageBox.warning(self, "No Trajectory", "Please load a reconstruction config first.")
            return
            
        stage, _ = self.get_active_stage()
        if not stage:
            QMessageBox.warning(self, "No Active Stage", "No active optimization stage found.")
            return
            
        data1 = self.combo_2d_param1.currentData()
        data2 = self.combo_2d_param2.currentData()
        if not data1 or not data2:
            QMessageBox.warning(self, "No Parameters Selected", "Please select two parameters for 2D sweep.")
            return
            
        p_obj1, name1 = data1
        p_obj2, name2 = data2
        
        try:
            num_samples = self.spin_2d_samples.value()
            
            range_min1, range_max1 = self.spin_2d_min1.value(), self.spin_2d_max1.value()
            range_min2, range_max2 = self.spin_2d_min2.value(), self.spin_2d_max2.value()
            
            samples1 = np.linspace(range_min1, range_max1, num_samples)
            samples2 = np.linspace(range_min2, range_max2, num_samples)
            
            X, Y = np.meshgrid(samples1, samples2)
            
            # Build parameterization chain including stage parameterizations and the swept classes
            param_list = list(stage.parameterizations)
            if p_obj1 not in param_list:
                param_list.append(p_obj1)
            if p_obj2 not in param_list and p_obj2 != p_obj1:
                param_list.append(p_obj2)
                
            chain = ParameterizationChain(param_list)
            
            # Show progress dialog
            progress = QProgressDialog("Computing 2D sweep...", "Cancel", 0, len(X.ravel()), self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.show()
            
            Ps_list = []
            cancelled = False
            
            # Save original values
            orig1 = p_obj1.parameters[name1]["value"]
            orig2 = p_obj2.parameters[name2]["value"]
            
            for idx, (v1, v2) in enumerate(zip(X.ravel(), Y.ravel())):
                if progress.wasCanceled():
                    cancelled = True
                    break
                p_obj1.parameters[name1]["value"] = v1
                p_obj2.parameters[name2]["value"] = v2
                # Apply the chain
                Ps_list.append(chain.apply_to_trajectory(self.P_list_undersampled))
                progress.setValue(idx)
                
            # Restore values
            p_obj1.parameters[name1]["value"] = orig1
            p_obj2.parameters[name2]["value"] = orig2
            
            if cancelled:
                return
                
            # Compute costs
            progress.setLabelText("Evaluating ECC costs...")
            costs = self.scan.compute_ecc_for_projection_matrices(Ps_list)
            progress.setValue(len(X.ravel()))
            progress.close()
            
            Z = np.array(costs).reshape(X.shape)
            
            # Clean up closed windows from the list
            active = []
            for w in self.sweep_windows:
                try:
                    if w.isVisible():
                        active.append(w)
                except RuntimeError:
                    pass
            self.sweep_windows = active
            
            # Create a Sweep2DWindow
            sweep_win = Sweep2DWindow(self, name1, name2)
            self.sweep_windows.append(sweep_win)
            sweep_win.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            
            # Draw on the 3D canvas
            sweep_win.plot_surface(X, Y, Z, name1, name2, orig1, orig2)
            
            sweep_win.show()
            sweep_win.raise_()
            
        except Exception as e:
            QMessageBox.critical(self, "Sweep 2D Error", f"Failed to compute 2D sweep:\n{e}")

    def compute_diagnostics_action(self):
        if not self.scan:
            return
            
        num_views = len(self.P_list_undersampled)
        if num_views > 50:
            reply = QMessageBox.question(
                self,
                "Confirm CPU Diagnostics",
                f"You are about to compute diagnostics for {num_views} views.\n\n"
                "Evaluating pairwise diagnostics on the CPU can take a significant amount of time "
                "compared to the CUDA implementation.\n\n"
                "Are you sure you want to proceed?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
                
        try:
            # Set matrices of Scan based on current tab parameters
            stage, _ = self.get_active_stage()
            if stage:
                chain = ParameterizationChain(stage.parameterizations)
                Ps_opt = chain.apply_to_trajectory(self.P_list_undersampled)
                self.scan.set_projection_matrices(Ps_opt)
            else:
                self.scan.set_projection_matrices(self.P_list_undersampled)
                
            # Run diagnostics
            try:
                with QtProgressBarContext(self, "Running Diagnostics"):
                    diag = self.scan.compute_diagnostics()
            except KeyboardInterrupt:
                self.statusBar().showMessage("Diagnostics canceled by user.")
                return
            self.cost_matrix = diag["cost_matrix"]
            self.zero_plane_distances = diag["zero_plane_distances"]

            # Combine cost matrix (lower triangle) and weight matrix (upper triangle)
            weight_matrix = diag["weight_matrix"]
            n_views = self.cost_matrix.shape[0]
            combined_matrix = np.zeros_like(self.cost_matrix)

            lower_indices = np.tril_indices(n_views, k=-1)
            cost_lower = np.nan_to_num(self.cost_matrix[lower_indices], nan=0.0)
            max_cost_val = float(np.nanmax(cost_lower)) if cost_lower.size > 0 else 1.0
            if max_cost_val <= 0:
                max_cost_val = 1.0
            combined_matrix[lower_indices] = cost_lower / max_cost_val

            upper_indices = np.triu_indices(n_views, k=1)
            weight_upper = np.nan_to_num(weight_matrix[upper_indices], nan=0.0)
            max_weight_val = float(np.nanmax(weight_upper)) if weight_upper.size > 0 else 1.0
            if max_weight_val <= 0:
                max_weight_val = 1.0
            combined_matrix[upper_indices] = weight_upper / max_weight_val

            np.fill_diagonal(combined_matrix, -1)
            
            print(f"[Diagnostics] Cost matrix size: {n_views}x{n_views}")
            print(f"[Diagnostics] Max cost: {max_cost_val:.6e} | Max weight: {max_weight_val:.6e}")
            sys.stdout.flush()

            # Show the labels now that data is computed
            self.lbl_object_radius.show()
            self.lbl_planes_needed.show()
            
            # Update info labels
            self.lbl_object_radius.setText(f"Estimated Object Size (z_max): {self.scan.object_radius_mm:.2f} mm")
            self.lbl_planes_needed.setText(f"Expected Planes: {diag['max_sample_count']} | {diag['info']}")
            
            # 1. Plot Cost & Weights Matrix
            self.ax_cost.clear()
            self.ax_cost.imshow(combined_matrix, cmap='gray', interpolation='nearest', vmin=0.0, vmax=1.0)
            self.ax_cost.axis('off')
            self.canvas_cost.draw()

            # 2. Plot Zero-Plane Distance Matrix
            self.fig_weight.clear()
            self.ax_weight = self.fig_weight.add_subplot(111)
            im = self.ax_weight.imshow(self.zero_plane_distances, cmap='gray', interpolation='nearest')
            self.ax_weight.axis('off')
            self.fig_weight.colorbar(im, ax=self.ax_weight, label="Zero-Plane Distance (px)")
            self.canvas_weight.draw()

            
            self.statusBar().showMessage(f"Diagnostics complete. Max sample count per plane: {diag['max_sample_count']}")
            
        except Exception as e:
            QMessageBox.critical(self, "Diagnostics Error", f"Failed to run diagnostics:\n{e}")

    def on_cost_matrix_click(self, event):
        """Click on the cost matrix → jump to the corresponding view pair in
        the Epipolar Geometry tab.  Matrix is (N×N); cell (row=i, col=j) with
        i < j is a valid upper-triangle entry."""
        if event.xdata is None or event.ydata is None:
            return
        if self.cost_matrix is None:
            return

        j = int(round(event.xdata))   # column → view index 2
        i = int(round(event.ydata))   # row    → view index 1
        n = self.cost_matrix.shape[0]

        if not (0 <= i < n and 0 <= j < n):
            return

        if i == j:
            self.statusBar().showMessage("Diagonal cell selected — no view pair.")
            return

        # Ensure i < j (upper triangle convention); swap if needed
        if i > j:
            i, j = j, i

        # Guard against indices beyond the loaded trajectory
        max_idx = len(self.P_list_undersampled) - 1
        if i > max_idx or j > max_idx:
            self.statusBar().showMessage(
                f"Cell ({i},{j}) out of range for {max_idx+1} loaded views.")
            return

        # Set the view spinboxes (blockSignals to avoid double-render)
        self.view1_spin.blockSignals(True)
        self.view2_spin.blockSignals(True)
        self.view1_slider.blockSignals(True)
        self.view2_slider.blockSignals(True)

        self.view1_spin.setValue(i)
        self.view1_slider.setValue(i)
        self.view2_spin.setValue(j)
        self.view2_slider.setValue(j)

        self.view1_spin.blockSignals(False)
        self.view2_spin.blockSignals(False)
        self.view1_slider.blockSignals(False)
        self.view2_slider.blockSignals(False)

        # Switch to the Epipolar Geometry tab (index 2) and render
        self.tabs.setCurrentIndex(2)
        self._base_svg.clear()
        self._dtr_base_svg.clear()
        self._redundancy_artists = None
        self.render_viewports()
        self.statusBar().showMessage(f"Showing epipolar geometry for view pair ({i}, {j})")

    def save_config_action(self):
        if self.current_config_path:
            return self.save_geometry_config(self.current_config_path)
        else:
            return self.save_geometry_config()

    def save_config_as_action(self):
        old_path = self.current_config_path
        self.current_config_path = None
        res = self.save_geometry_config()
        if not res:
            self.current_config_path = old_path
        return res

    def save_geometry_config(self, file_path=None):
        if not file_path:
            # Get path to save
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Geometry Correction Config", self.current_config_path or "", "JSON Files (*.json)"
            )
            if not file_path:
                return False
            
        try:
            config_dir = os.path.dirname(os.path.abspath(file_path))
            
            # Update stages directory to be next to the new config path if needed
            stages_dir = os.path.join(config_dir, "stages")
            if os.path.exists(stages_dir):
                try:
                    shutil.rmtree(stages_dir)
                except Exception:
                    pass
            os.makedirs(stages_dir, exist_ok=True)
            self.txt_stages_dir.setText(stages_dir)
            
            # Migrate and save stage JSON files for checked ones only, keeping their original filenames
            checked_rel_paths = []
            new_stages_cache = {}
            for i in range(self.list_stages.count()):
                item = self.list_stages.item(i)
                if item.checkState() == Qt.CheckState.Checked:
                    filename = item.text()
                    # Find matching old path in stages_cache
                    old_path = None
                    for op in self.stages_cache.keys():
                        if os.path.basename(op) == filename:
                            old_path = op
                            break
                    
                    if old_path:
                        stage = self.stages_cache[old_path]
                        new_path = os.path.normpath(os.path.join(stages_dir, filename))
                        new_stages_cache[new_path] = stage
                        self.save_single_stage(new_path, stage)
                        rel_path = safe_relpath(new_path, config_dir)
                        checked_rel_paths.append(rel_path)
            self.stages_cache = new_stages_cache
            
            # Create undersampled data JSON file
            recon_config_abs = self.original_recon_config_path or self.txt_recon_path.text()
            if recon_config_abs and recon_config_abs != "No file loaded" and os.path.exists(recon_config_abs):
                orig_recon_rel = safe_relpath(recon_config_abs, config_dir)
            else:
                orig_recon_rel = ""
                
            if self.current_recon_json:
                U = self.spin_undersample.value()
                undersampled_recon_filename = f"fullscan_{len(self.P_list_undersampled)}views_600x400.json"
                undersampled_recon_path = os.path.join(config_dir, undersampled_recon_filename)
                
                # Construct the undersampled trajectory details
                undersampled_ompl_filename = f"fullscan_{len(self.P_list_undersampled)}views_600x400.ompl"
                undersampled_ompl_path = os.path.join(config_dir, undersampled_ompl_filename)
                save_ompl(
                    self.P_list_undersampled,
                    undersampled_ompl_path,
                    spacing=self.P_list_undersampled[0].pixel_spacing,
                    detector_size_px=self.P_list_undersampled[0].image_size
                )
                
                # Sliced NRRD if applicable
                orig_image_files = self.current_recon_json.get("image_files", [])
                data_dir = self.current_recon_json.get("data_dir", "./")
                
                recon_dir = os.path.dirname(recon_config_abs)
                if os.path.isabs(data_dir):
                    data_dir_path_abs = data_dir
                else:
                    data_dir_path_abs = os.path.normpath(os.path.join(recon_dir, data_dir))
                
                if len(orig_image_files) == 1 and orig_image_files[0].lower().endswith('.nrrd'):
                    orig_nrrd_path = os.path.join(data_dir_path_abs, orig_image_files[0])
                    data, header = nrrd.read(orig_nrrd_path)
                    data_sliced = data[:, :, ::U]
                    
                    undersampled_nrrd_filename = f"fullscan_{len(self.P_list_undersampled)}views_600x400.nrrd"
                    undersampled_nrrd_path = os.path.join(config_dir, undersampled_nrrd_filename)
                    nrrd.write(undersampled_nrrd_path, data_sliced, header)
                    
                    image_files_field = [undersampled_nrrd_filename]
                else:
                    image_files_field = [safe_relpath(os.path.join(data_dir_path_abs, f), config_dir) for f in orig_image_files[::U]]
                    
                undersampled_recon_dict = {
                    "data_dir": "./",
                    "ompl_file": undersampled_ompl_filename,
                    "image_files": image_files_field,
                    "voxel_dimensions": self.voxel_dimensions,
                    "model_matrix": self.model_matrix.tolist(),
                    "output_file": "reconstruction.nrrd",
                    "convert_to_line_integral": self.chk_line_integral.isChecked()
                }
                with open(undersampled_recon_path, 'w') as uf:
                    json.dump(undersampled_recon_dict, uf, indent=2)
            else:
                undersampled_recon_filename = ""
                
            # Finally create the Main configuration file
            dtr_factor = self.spin_size_fac.value()
            default_sigma = 0.0 if abs(dtr_factor - 1.0) < 1e-5 else 1.2
            config_json = {
                "input_data": undersampled_recon_filename,
                "output_dir": safe_relpath(self.txt_out_dir.text(), config_dir),
                "create_report": self.chk_create_report.isChecked(),
                "run_reconstruction": self.chk_run_recon.isChecked(),
                "metric_config": {
                    "convert_to_line_integral": self.chk_line_integral.isChecked(),
                    "dtr_size_factor": dtr_factor,
                    "gaussian_sigma": getattr(self, '_gaussian_sigma', default_sigma),
                    "num_planes": self.spin_spacing.value(),
                    "object_radius_mm": self.spin_object_radius.value()
                },
                "geometry_optimization": {
                    "stages": checked_rel_paths
                }
            }
            if orig_recon_rel:
                config_json["reconstruction_config"] = orig_recon_rel
            
            with open(file_path, 'w') as mf:
                json.dump(config_json, mf, indent=2)
                
            self.current_config_path = file_path
            self.statusBar().showMessage(f"Saved configuration to: {file_path}")
            self.update_json_preview()
            return file_path
            
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save geometry correction configuration:\n{e}")
            return False

    def run_optimization_action(self):
        if self.chk_run_recon.isChecked():
            recon_path = self.txt_recon_path.text().strip()
            if not recon_path or recon_path == "No file loaded" or not os.path.exists(recon_path):
                QMessageBox.warning(self, "Missing Reconstruction Config", "Please load a valid reconstruction config first or uncheck 'Run Reconstruction'.")
                return

        # Save config first
        out_dir = self.txt_out_dir.text().strip()
        geom_config_path = os.path.join(out_dir, "geometry_correction.json")
        config_path = self.save_geometry_config(geom_config_path)
        if not config_path:
            return
            
        # Print the final geometry correction JSON data to the terminal console
        try:
            with open(config_path, 'r') as f:
                geom_json_data = json.load(f)
            print("\n==============================================================")
            print("FINAL GEOMETRY CORRECTION JSON PASSED TO geometry_correction.py:")
            print("==============================================================")
            print(json.dumps(geom_json_data, indent=2))
            print("==============================================================\n")
            sys.stdout.flush()
        except Exception as e:
            print(f"Failed to print geometry correction JSON: {e}")
            
        # Spawn process console window
            
        cmd = f'"{sys.executable}" -u -m xray_epipolar_consistency.tools.geometry_correction "{config_path}"'
        self.console_win = ProcessConsoleWindow(
            command=cmd,
            working_dir=os.path.dirname(config_path),
            title="Geometry Calibration Console",
            parent=self,
            show_progress=True
        )
        
        self.console_win.finished_signal.connect(
            lambda exit_code, log: self.on_correction_finished_new(exit_code == 0, log)
        )
        self.console_win.finished.connect(lambda result: self.on_correction_dialog_closed(result))
        self.console_win.show()
        self.hide()

    def optimize_stage_action(self):
        stage, abs_path = self.get_active_stage()
        if not stage:
            QMessageBox.warning(self, "No Stage Selected", "Please select a stage to optimize.")
            return
            
        if not self.scan or not self.P_list_undersampled:
            QMessageBox.warning(self, "No Data", "Please load a scan first.")
            return

        # Get list of enabled (optimized) parameters for this stage
        enabled_params = []
        for p in stage.parameterizations:
            p_class = p.__class__.__name__
            for name, config in p.parameters.items():
                if config.get("opt", False):
                    enabled_params.append(f"{p_class}.{name}")
                    
        params_str = ", ".join(enabled_params) if enabled_params else "(none)"
        
        reply = QMessageBox.question(
            self,
            "Confirm Stage Optimization",
            f"Are you sure you want to run the selected stage {stage.name}: ({params_str})?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return

        dtr_factor = self.spin_size_fac.value()
        default_sigma = 0.0 if abs(dtr_factor - 1.0) < 1e-5 else 1.2
        metric_config = {
            "convert_to_line_integral": self.chk_line_integral.isChecked(),
            "dtr_size_factor": dtr_factor,
            "gaussian_sigma": getattr(self, '_gaussian_sigma', default_sigma),
            "num_planes": self.spin_spacing.value(),
            "object_radius_mm": self.spin_object_radius.value()
        }

        # Save current stage state to its json file on disk first
        self.save_single_stage(abs_path, stage)

        dialog = OptimizationProgressDialog(self, self.scan, stage, metric_config)
        dialog.start()
        if dialog.exec() == QDialog.DialogCode.Accepted:
            result = dialog.optimization_result
            if result:
                # Apply changes to current state
                try:
                    opt_chain = from_dict(result["optimized_parameterization"])
                    
                    # Apply optimized parameters to original full trajectory in memory (P_list)
                    self.P_list = opt_chain.apply_to_trajectory(self.P_list)
                    self.dirty = True
                    
                    if self.scan:
                        self.scan.set_projection_matrices(self.P_list_undersampled)
                        
                    # Reset the selected stage's parameter values to 0.0 in the GUI,
                    # since they are now applied directly to the matrices.
                    for param_obj in stage.parameterizations:
                        for name in param_obj.parameters:
                            param_obj.parameters[name]["value"] = 0.0
                            
                    # Refresh GUI
                    self.select_stage(self.list_stages.currentRow())
                    self.render_viewports()
                    self.update_json_preview()
                    
                    # Also save the updated parameterization.json and trajectory_optimized.ompl in out_dir if it exists
                    out_dir = self.txt_out_dir.text().strip()
                    if out_dir and os.path.isdir(out_dir):
                        opt_ompl = os.path.join(out_dir, "trajectory_optimized.ompl")
                        save_ompl(
                            self.P_list,
                            opt_ompl,
                            spacing=self.P_list[0].pixel_spacing,
                            detector_size_px=self.P_list[0].image_size
                        )
                    
                    self.statusBar().showMessage("Applied optimization results to the current state.", 5000)
                except Exception as e:
                    QMessageBox.critical(self, "Application Error", f"Failed to apply optimization results:\n{e}")

    def analyze_stage_action(self):
        stage, abs_path = self.get_active_stage()
        if not stage:
            QMessageBox.warning(self, "No Stage Selected", "Please select a stage to analyze.")
            return
            
        if not self.current_config_path:
            QMessageBox.warning(self, "No Dataset Loaded", "Please load a dataset configuration first.")
            return

        recon_config_abs = self.original_recon_config_path or self.txt_recon_path.text()
        if not recon_config_abs or not os.path.exists(recon_config_abs):
            QMessageBox.warning(self, "No Reconstruction Config", "Could not locate the original reconstruction.json file.")
            return

        # Query user for analysis parameters
        dialog_param = AnalysisParametersDialog(self)
        if dialog_param.exec() != QDialog.DialogCode.Accepted:
            return
        params = dialog_param.get_values()

        # Save stage config to disk first to capture latest UI changes
        self.save_single_stage(abs_path, stage)

        # Output directory is where other files for the ECC correction are stored
        out_dir = self.txt_out_dir.text().strip()
        if not out_dir or out_dir == "Select Output Folder":
            out_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(self.current_config_path)), "analysis"))
        os.makedirs(out_dir, exist_ok=True)

        # Show progress dialog
        progress = QProgressDialog("Running geometric identifiability & advisor analysis...", "Cancel", 0, 100, self)
        progress.setWindowTitle("Geometric Analysis")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(5)
        QApplication.processEvents()

        try:
            import shutil
            # Resolve executable commands
            # We can run them via the python interpreter modules to be more robust
            exe_ident = shutil.which("sloppy-direction-analysis")
            if exe_ident:
                cmd_ident_base = [exe_ident]
            else:
                p = os.path.join(os.path.dirname(sys.executable), "sloppy-direction-analysis")
                if os.path.exists(p):
                    cmd_ident_base = [p]
                else:
                    cmd_ident_base = [sys.executable, "-m", "xray_epipolar_consistency.tools.sloppy_direction_analysis"]

            exe_advisor = shutil.which("optimization-advisor")
            if exe_advisor:
                cmd_advisor_base = [exe_advisor]
            else:
                p = os.path.join(os.path.dirname(sys.executable), "optimization-advisor")
                if os.path.exists(p):
                    cmd_advisor_base = [p]
                else:
                    cmd_advisor_base = [sys.executable, "-m", "xray_epipolar_consistency.tools.optimization_advisor"]

            # Run sloppy direction analysis
            progress.setValue(20)
            progress.setLabelText("Running Sloppy Direction Analysis...")
            QApplication.processEvents()
            
            cmd_ident = cmd_ident_base + [
                recon_config_abs,
                abs_path,
                "--output_dir", out_dir,
                "--num_samples", str(params["num_samples"]),
                "--seed", str(params["seed"]),
                "--delta_p", str(params["delta_p"]),
                "--max_epipolar_views", str(params["max_epipolar_views"]),
                "--gap_threshold", str(params["gap_threshold"])
            ]
            res_ident = subprocess.run(cmd_ident, capture_output=True, text=True)
            if res_ident.returncode != 0:
                raise RuntimeError(f"Sloppy Direction Analysis failed:\n{res_ident.stderr}")

            # Run optimization advisor
            progress.setValue(60)
            progress.setLabelText("Running Optimization Advisor...")
            QApplication.processEvents()

            cmd_advisor = cmd_advisor_base + [
                recon_config_abs,
                abs_path,
                "--output_dir", out_dir,
                "--d_max", str(params["d_max"]),
                "--num_samples", str(params["num_samples"]),
                "--seed", str(params["seed"]),
                "--delta_p", str(params["delta_p"]),
                "--max_epipolar_views", str(params["max_epipolar_views"])
            ]
            res_advisor = subprocess.run(cmd_advisor, capture_output=True, text=True)
            if res_advisor.returncode != 0:
                raise RuntimeError(f"Optimization Advisor failed:\n{res_advisor.stderr}")

            progress.setValue(100)
            progress.close()

            # Show results dialog
            html_path1 = os.path.join(out_dir, "report_identifiability.html")
            html_path2 = os.path.join(out_dir, "report_optimization_advisor.html")
            suggested_json = os.path.join(out_dir, "suggested_optimizer_config.json")
            
            dlg = AnalysisResultsDialog(self, out_dir, html_path1, html_path2, suggested_json, abs_path)
            dlg.exec()
            
        except Exception as e:
            progress.close()
            QMessageBox.critical(self, "Analysis Failed", f"An error occurred during analysis:\n{e}")

    def run_reconstruction_action(self):
        if not self.current_recon_json:
            QMessageBox.warning(self, "No Config", "Please load a reconstruction config first.")
            self.show()
            return
            
        try:
            out_dir = self.txt_out_dir.text().strip()
            os.makedirs(out_dir, exist_ok=True)
            
            # Save the full trajectory in memory to an OMPL file
            ompl_filename = "reconstructed_trajectory.ompl"
            ompl_path = os.path.join(out_dir, ompl_filename)
            save_ompl(
                self.P_list,
                ompl_path,
                spacing=self.P_list[0].pixel_spacing,
                detector_size_px=self.P_list[0].image_size
            )
            
            # Construct the new reconstruction.json
            recon_dict = dict(self.current_recon_json)
            
            orig_recon_dir = os.path.dirname(os.path.abspath(self.txt_recon_path.text()))
            orig_data_dir = self.current_recon_json.get("data_dir", "./")
            if not os.path.isabs(orig_data_dir):
                abs_data_dir = os.path.normpath(os.path.join(orig_recon_dir, orig_data_dir))
            else:
                abs_data_dir = orig_data_dir
                
            recon_dict["data_dir"] = abs_data_dir
            recon_dict["ompl_file"] = os.path.abspath(ompl_path)
            
            new_recon_json_path = os.path.join(out_dir, "reconstruction.json")
            with open(new_recon_json_path, 'w') as rf:
                json.dump(recon_dict, rf, indent=2)
                
            # Print the final reconstruction JSON data to the terminal console
            try:
                print("\n==============================================================")
                print("FINAL RECONSTRUCTION JSON PASSED TO RECONSTRUCTION GUI:")
                print("==============================================================")
                print(json.dumps(recon_dict, indent=2))
                print("==============================================================\n")
                sys.stdout.flush()
            except Exception as e:
                print(f"Failed to print reconstruction JSON: {e}")
                
            # Open the Reconstruction GUI window and hide the main window
            self.hide()
            self.recon_win = InteractiveReconstructionGUI(new_recon_json_path, self, edit_config_only=False)
            self.recon_win.show()
            self.statusBar().showMessage(f"Opened Reconstruction GUI with: {new_recon_json_path}")

        except Exception as e:
            QMessageBox.critical(self, "Reconstruction Launch Error", f"Failed to run reconstruction:\n{e}")
            self.show()

    def open_reconstruction_settings(self):
        """Open the *input* reconstruction JSON in the Reconstruction GUI
        (edit-only mode) so the user can set volume geometry, filters, etc."""
        file_path = self.txt_recon_path.text()
        if not file_path or file_path == "No file loaded" or not os.path.exists(file_path):
            QMessageBox.warning(self, "No Config File", "Please load a valid reconstruction config JSON first.")
            return

        self.hide()
        self.recon_win = InteractiveReconstructionGUI(file_path, self, edit_config_only=True)
        self.recon_win.show()
        self.statusBar().showMessage(f"Opened Reconstruction Settings for: {file_path}")

    # ------------------------------------------------------------------
    # Direct Reconstruction (Run Reconstruction... button)
    # ------------------------------------------------------------------

    def _ask_reconstruction_suffix(self):
        """Show a small dialog that lets the user pick (or type) a suffix
        for the ompl / output-nrrd files.  Returns the suffix string, or
        None if the user cancelled."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Run Reconstruction")
        dlg.setModal(True)
        dlg.resize(400, 130)

        layout = QVBoxLayout(dlg)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 15, 20, 15)

        layout.addWidget(QLabel("Filename suffix for trajectory and output NRRD:"))

        txt = QLineEdit("_intermediate")
        layout.addWidget(txt)

        btn_row = QHBoxLayout()
        btn_initial   = QPushButton("_initial")
        btn_optimized = QPushButton("_optimized")
        btn_ok        = QPushButton("OK")
        btn_ok.setDefault(True)
        btn_cancel    = QPushButton("Cancel")

        btn_row.addWidget(btn_initial)
        btn_row.addWidget(btn_optimized)
        btn_row.addStretch()
        btn_row.addWidget(btn_ok)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)

        def pick(suffix):
            txt.setText(suffix)
            dlg.accept()

        btn_initial.clicked.connect(lambda: pick("_initial"))
        btn_optimized.clicked.connect(lambda: pick("_optimized"))
        btn_ok.clicked.connect(dlg.accept)
        btn_cancel.clicked.connect(dlg.reject)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        return txt.text().strip() or "_intermediate"

    def run_reconstruction_direct_action(self):
        """Prompt for a suffix, write a reconstruction JSON with the
        current in-memory trajectory, then run FDK in a background thread
        showing a console window — exactly like the reconstruction GUI."""
        if not self.current_recon_json:
            QMessageBox.warning(self, "No Config", "Please load a reconstruction config first.")
            return
        if not self.P_list:
            QMessageBox.warning(self, "No Trajectory", "No projection matrices loaded yet.")
            return

        suffix = self._ask_reconstruction_suffix()
        if suffix is None:  # user cancelled
            return

        # Default saving behavior: save geometry_correction.json and trajectory.ompl
        if not self.save_project_action():
            return

        try:
            out_dir = self.txt_out_dir.text().strip()
            os.makedirs(out_dir, exist_ok=True)

            # Save the current in-memory trajectory to an OMPL file
            ompl_filename = f"trajectory{suffix}.ompl"
            ompl_path = os.path.join(out_dir, ompl_filename)
            save_ompl(
                self.P_list,
                ompl_path,
                spacing=self.P_list[0].pixel_spacing,
                detector_size_px=self.P_list[0].image_size
            )

            # Build the reconstruction JSON from the input config
            recon_dict = dict(self.current_recon_json)

            # Resolve data_dir to an absolute path so the new JSON is portable
            orig_recon_dir = os.path.dirname(os.path.abspath(self.txt_recon_path.text()))
            orig_data_dir = self.current_recon_json.get("data_dir", "./")
            if not os.path.isabs(orig_data_dir):
                abs_data_dir = os.path.normpath(os.path.join(orig_recon_dir, orig_data_dir))
            else:
                abs_data_dir = orig_data_dir

            recon_dict["data_dir"] = abs_data_dir
            recon_dict["ompl_file"] = os.path.abspath(ompl_path)

            # Derive output filename from the suffix
            base_output = self.current_recon_json.get("output_file", "reconstruction.nrrd")
            recon_config_abs = self.txt_recon_path.text().strip()
            if recon_config_abs and recon_config_abs != "No file loaded" and os.path.exists(recon_config_abs):
                orig_output_path_abs = os.path.abspath(os.path.join(os.path.dirname(recon_config_abs), base_output))
            else:
                orig_output_path_abs = os.path.abspath(base_output)
            
            orig_dir = os.path.dirname(orig_output_path_abs)
            orig_name = os.path.basename(orig_output_path_abs)
            stem, ext = os.path.splitext(orig_name)
            out_dir_name = os.path.basename(out_dir)
            
            out_vol_abs = os.path.join(orig_dir, f"{stem}_{out_dir_name}{suffix}{ext}")
            recon_dict["output_file"] = safe_relpath(out_vol_abs, out_dir)

            new_recon_json_path = os.path.join(out_dir, f"reconstruction{suffix}.json")
            with open(new_recon_json_path, 'w') as rf:
                json.dump(recon_dict, rf, indent=2)

            # Resolve reconstruct.py path
            reconstruct_file = self.resolve_reconstruct_file()
            if not reconstruct_file:
                QMessageBox.critical(self, "Error", "Could not locate reconstruct.py")
                self.show()
                return
                
            if reconstruct_file.endswith('.pyc'):
                reconstruct_file = reconstruct_file[:-1]
                
            cmd = f'"{sys.executable}" -u "{reconstruct_file}" "{new_recon_json_path}"'
            self.recon_console_win = ProcessConsoleWindow(
                command=cmd,
                working_dir=os.path.dirname(new_recon_json_path),
                title="Reconstruction Progress",
                parent=self,
                show_progress=True
            )
            self.recon_console_win.finished_signal.connect(
                lambda exit_code, log: self.on_reconstruction_direct_finished_new(exit_code == 0, new_recon_json_path)
            )
            self.recon_console_win.finished.connect(lambda result: self.show())
            self.recon_console_win.show()
            self.hide()
            self.statusBar().showMessage(f"Reconstruction started: {new_recon_json_path}")

        except Exception as e:
            QMessageBox.critical(self, "Reconstruction Error", f"Failed to start reconstruction:\n{traceback.format_exc()}")
            self.show()

    def resolve_reconstruct_file(self):
        try:
            return os.path.abspath(reconstruct.__file__)
        except Exception:
            pass
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.normpath(os.path.join(cur_dir, "reconstruct.py")),
            os.path.normpath(os.path.join(cur_dir, "..", "reconstruct.py")),
            os.path.normpath(os.path.join(cur_dir, "..", "reconstruct", "reconstruct.py")),
            "/run/media/aaichert/Intenso/reconstruct/reconstruct.py"
        ]
        for p in possible_paths:
            if os.path.exists(p):
                return p
        return None

    def resolve_nrrd_view_3d_file(self):
        try:
            return os.path.abspath(nrrd_view_3d.__file__)
        except Exception:
            pass
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.normpath(os.path.join(cur_dir, "gui", "nrrd_view_3d.py")),
            os.path.normpath(os.path.join(cur_dir, "..", "reconstruct", "gui", "nrrd_view_3d.py")),
            "/run/media/aaichert/Intenso/reconstruct/gui/nrrd_view_3d.py"
        ]
        for p in possible_paths:
            if os.path.exists(p):
                return p
        return None

    def compare_reconstructions_action(self):
        recon_path = self.txt_recon_path.text().strip()
        if not recon_path or recon_path == "No file loaded" or not os.path.exists(recon_path):
            QMessageBox.warning(self, "No Config Loaded", "Please load a valid reconstruction config first.")
            return

        recon_dir = os.path.dirname(os.path.abspath(recon_path))
        try:
            nrrd_files = [f for f in os.listdir(recon_dir) if f.lower().endswith('.nrrd')]
        except Exception as e:
            QMessageBox.critical(self, "Error Reading Directory", f"Failed to read reconstruction directory:\n{e}")
            return

        if not nrrd_files:
            QMessageBox.information(self, "No Files Found", f"No .nrrd files found in:\n{recon_dir}")
            return

        # Show Selection Dialog
        dialog = CompareReconstructionsDialog(self, sorted(nrrd_files))
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_names = dialog.get_selected_files()
            if not selected_names:
                return
            
            abs_paths = [os.path.normpath(os.path.join(recon_dir, name)) for name in selected_names]
            
            # Resolve nrrd_view_3d.py or fallback
            nrrd_view_3d_file = self.resolve_nrrd_view_3d_file()
            if nrrd_view_3d_file:
                cmd = [sys.executable, nrrd_view_3d_file] + abs_paths
            else:
                cmd = ["NrrdView3D"] + abs_paths
                
            try:
                subprocess.Popen(cmd)
                self.statusBar().showMessage(f"Launched NrrdView3D for comparison of {len(abs_paths)} files.", 5000)
            except Exception as e:
                QMessageBox.critical(self, "Error Launching Viewer", f"Could not launch NrrdView3D:\n{e}")

    def on_reconstruction_direct_finished_new(self, success, config_path):
        if success:
            try:
                with open(config_path, 'r') as f:
                    cfg = json.load(f)
                output_file = cfg.get("output_file", "")
                if not os.path.isabs(output_file):
                    output_file = os.path.join(os.path.dirname(config_path), output_file)
                output_file = os.path.normpath(output_file)
                if os.path.exists(output_file):
                    self.nrrd_win = NrrdView3DWindow()
                    self.nrrd_win.show()
                    self.nrrd_win.open_file(output_file)
                else:
                    QMessageBox.warning(
                        self, "Output Not Found",
                        f"Reconstruction finished but output file not found:\n{output_file}"
                    )
            except Exception as ex:
                QMessageBox.critical(self, "Post-Reconstruction Error",
                                     f"Failed to open NrrdView3D:\n{traceback.format_exc()}")
        else:
            QMessageBox.critical(self, "Reconstruction Error", "Reconstruction process failed. Please check the logs.")

    def on_correction_finished_new(self, success, log):
        if success:
            # Launch report.html in default browser
            try:
                out_dir = self.txt_out_dir.text().strip()
                report_path = os.path.join(out_dir, "report.html")
                if os.path.exists(report_path):
                    webbrowser.open(os.path.abspath(report_path))
            except Exception as e:
                print(f"Failed to open report in browser: {e}")

    def on_correction_dialog_closed(self, result):
        self.reload_after_optimization()
        
        success = (self.console_win.process.exitCode() == 0 and 
                   self.console_win.process.exitStatus() == QProcess.ExitStatus.NormalExit)
        
        if success:
            if self.chk_run_recon.isChecked():
                out_dir = self.txt_out_dir.text().strip()
                recon_config_abs = self.txt_recon_path.text().strip()
                
                recon_misaligned = os.path.normpath(os.path.join(out_dir, "recon_misaligned.nrrd"))
                recon_optimized = os.path.normpath(os.path.join(out_dir, "recon_optimized.nrrd"))
                
                if recon_config_abs and recon_config_abs != "No file loaded" and os.path.exists(recon_config_abs):
                    try:
                        with open(recon_config_abs, 'r') as rf:
                            recon_config = json.load(rf)
                        orig_output_file = recon_config.get("output_file", "reconstruction.nrrd")
                        orig_output_path_abs = os.path.abspath(os.path.join(os.path.dirname(recon_config_abs), orig_output_file))
                        orig_dir = os.path.dirname(orig_output_path_abs)
                        orig_name = os.path.basename(orig_output_path_abs)
                        orig_base, orig_ext = os.path.splitext(orig_name)
                        out_dir_name = os.path.basename(out_dir)
                        
                        recon_misaligned = orig_output_path_abs
                        recon_optimized = os.path.normpath(os.path.join(orig_dir, f"{orig_base}_{out_dir_name}{orig_ext}"))
                    except Exception as e:
                        print(f"Error resolving reconstructed paths in GUI: {e}")
                
                if os.path.exists(recon_misaligned) and os.path.exists(recon_optimized):
                    class InteractiveNrrdView3DWindow(NrrdView3DWindow):
                        def __init__(self, parent_window):
                            self.parent_window = parent_window
                            super().__init__()
                        def closeEvent(self, event):
                            super().closeEvent(event)
                            if self.parent_window:
                                self.parent_window.show()
                    
                    self.hide()
                    self.nrrd_win_opt = InteractiveNrrdView3DWindow(self)
                    self.nrrd_win_opt.show()
                    self.nrrd_win_opt.open_file([recon_misaligned, recon_optimized])
                else:
                    QMessageBox.warning(
                        self, "Reconstruction Not Found",
                        f"Expected reconstruction volumes not found:\n- {recon_misaligned}\n- {recon_optimized}\n"
                        "Please check if ASTRA / reconstruct is installed correctly."
                    )
                    self.show()
            else:
                self.run_reconstruction_action()
        else:
            self.show()

    def reload_after_optimization(self):
        # Once calibration is completed, reload optimized matrices in the viewport if available
        out_dir = self.txt_out_dir.text()
        opt_param_path = os.path.join(out_dir, "parameterization.json")
        
        # Ensure we have the original full trajectory loaded
        if not self.P_list_original:
            self.P_list_original = list(self.P_list)
            
        if os.path.exists(opt_param_path):
            try:
                with open(opt_param_path, 'r') as f:
                    opt_param_dict = json.load(f)
                opt_chain = from_dict(opt_param_dict)
                
                # Apply optimized parameters to original full trajectory
                P_opt_all = opt_chain.apply_to_trajectory(self.P_list_original)
                self.P_list = P_opt_all
                
                # Save optimized trajectory (all views) to trajectory_optimized.ompl
                opt_ompl = os.path.join(out_dir, "trajectory_optimized.ompl")
                save_ompl(
                    self.P_list,
                    opt_ompl,
                    spacing=self.P_list_original[0].pixel_spacing,
                    detector_size_px=self.P_list_original[0].image_size
                )
                
                if self.scan:
                    self.scan.set_projection_matrices(self.P_list_undersampled)
                
                # Reset all parameters to 0 since matrices are now updated to the optimized state
                for stage in self.stages_cache.values():
                    for param_obj in stage.parameterizations:
                        for name in param_obj.parameters:
                            param_obj.parameters[name]["value"] = 0.0
                            
                self.select_stage(self.list_stages.currentRow())
                self.render_viewports()
                self.statusBar().showMessage("Reloaded optimized geometry into viewport successfully.")
            except Exception as e:
                self.statusBar().showMessage(f"Failed to reload optimized geometry: {str(e)}")


class StartupWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Geometry Correction GUI - Startup")
        self.resize(450, 260)
        self.main_win = None
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(15)
        
        lbl_header = QLabel("X-Ray Epipolar Consistency Geometry Calibration")
        lbl_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_header)
        
        lbl_subheader = QLabel("Select a Reconstruction Config file to begin:")
        lbl_subheader.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_subheader)
        
        self.btn_recon = QPushButton("Open Reconstruction Config (.json)")
        self.btn_recon.clicked.connect(self.open_recon_config)
        layout.addWidget(self.btn_recon)
        
        self.btn_example = QPushButton("Load Example")
        self.btn_example.clicked.connect(self.load_example_action)
        layout.addWidget(self.btn_example)

        self.btn_walnut = QPushButton("Load Walnut Dataset (Example)")
        self.btn_walnut.clicked.connect(self.load_walnut_action)
        layout.addWidget(self.btn_walnut)
        
        layout.addStretch()
        
    def open_recon_config(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Reconstruction Config JSON", "", "JSON Files (*.json)"
        )
        if file_path:
            self.launch_main_gui(file_path)
            
    def load_example_action(self):
        # Resolve path to the sibling reconstruct repository's example data
        gui_dir = os.path.dirname(os.path.abspath(__file__))
        reconstruct_example_data_dir = os.path.normpath(
            os.path.join(os.path.dirname(gui_dir), "reconstruct", "example_data")
        )
        
        config_path = os.path.join(reconstruct_example_data_dir, "fullscan_180views_600x400.json")
        nrrd_path = os.path.join(reconstruct_example_data_dir, "fullscan_180views_600x400.nrrd")

        if not os.path.exists(config_path):
            QMessageBox.warning(self, "Example Data Not Available", f"File not found: {config_path}")
            return

        if not os.path.exists(nrrd_path):
            QMessageBox.warning(self, "Example Data Missing", f"File not found: {nrrd_path}")
            return

        self.launch_main_gui(config_path)

    def load_walnut_action(self):
        """Load the walnut example dataset shipped with xray-epipolar-consistency."""
        try:
            walnut_data_dir = str(ecc.get_data_path("example_data", "20201111_walnut_raw_data"))
        except Exception as e:
            QMessageBox.warning(
                self,
                "Walnut Dataset Not Found",
                f"Could not locate the walnut example data directory:\n{e}"
            )
            return

        config_path = os.path.join(walnut_data_dir, "walnut_360_4x4.json")
        raw_images_dir = os.path.join(walnut_data_dir, "20201111_walnut_raw_data")
        readme_path = os.path.join(walnut_data_dir, "README.md")

        if not os.path.exists(config_path):
            QMessageBox.warning(
                self,
                "Walnut Config Not Found",
                f"Expected configuration file not found:\n{config_path}"
            )
            return

        # Check that raw TIFF images exist in the images sub-directory
        images_present = (
            os.path.isdir(raw_images_dir)
            and any(
                fname.lower().endswith(".tif") or fname.lower().endswith(".tiff")
                for fname in os.listdir(raw_images_dir)
            )
        )

        if not images_present:
            msg = QMessageBox(self)
            msg.setWindowTitle("Walnut Images Not Found")
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText(
                "The walnut raw projection images (.tif) could not be found.\n\n"
                "Please download the dataset from Zenodo as described in the README file "
                "and place the .tif files in the correct directory.\n\n"
                f"README: {readme_path}\n\n"
                f"Expected images in:\n{raw_images_dir}"
            )
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            return

        self.launch_main_gui(config_path)
            
    def launch_main_gui(self, file_path):
        self.hide()  # disappear before the dataset load dialog appears
        self.main_win = GeometryCorrectionGUI()
        self.main_win.load_config_file(file_path)
        # If load was cancelled (P_list still empty and no scan) show startup again
        if not self.main_win.P_list:
            self.main_win = None
            self.show()
            return
        self.main_win.show()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    if len(sys.argv) > 1:
        file_path = os.path.abspath(sys.argv[1])
        if os.path.exists(file_path):
            window = GeometryCorrectionGUI()
            window.load_config_file(file_path)
            window.show()
            sys.exit(app.exec())
        else:
            print(f"Provided config file does not exist: {file_path}")
            
    startup = StartupWindow()
    startup.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
