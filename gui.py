from pathlib import Path

import numpy as np
import OpenGL.GL as gl
from imgui_bundle import hello_imgui, imgui as gui, imgui_ctx, implot, ImVec2
from imgui_bundle.python_backends.opengl_backend import ProgrammablePipelineRenderer
from PyQt6 import QtCore, QtOpenGLWidgets, QtWidgets

import gui_utils
import imguio as guio
from core import AdamPhysics, CBDPhysics, Petri, PhysicsConfig, Run, SGDPhysics, TissueConfig
from models.basinae import BasinTissue
from models.synapse import SynapseTissue
from models.retnetae import RetNetTissue
from optimizers.dadam import DivineAdamPhysics
from TexturePool import TexturePool

def cached_workvars(**kwargs):
    """
    Decorator that attaches persistent static data to a function.
    Data remains accessible between calls through fn._data.

    Args:
        fn: Target function
        **kwargs: Initial static variable values
    """
    def decorator(fn):
        class DataContainer:
            pass

        fn._data = DataContainer()
        for key, value in kwargs.items():
            setattr(fn._data, key, value)
        return fn

    return decorator

# Nomenclature:
#   tissue: a self-organizing computational structure and mathematical substrate (formerly “model”), capable of emergent dynamics (e.g., neural networks, matrix systems, etc.)
#   physics: gradient integration and weight adjustment mechanics (formerly “optimizer”)
#   petri: an editing viewport for a tissue, built with imgui for live modification
#   run: an encapsulated training environment combining a petri, tissue, and physics engine; can be stored, reloaded, and altered efficiently


# ----------------------------------------
# PETRI WINDOW
# ----------------------------------------

class PlotUI(guio.IGUI):
    """
    Visualization panel for displaying RunHistory metrics with selectable values.
    Uses a sidebar for metric toggles and an interactive plotting window.
    """

    def __init__(self, run: Run):
        self.run = run
        self.value_names = {
            'loss': 'Training Loss',
            'grad_norm': 'Gradient Norm',
            'param_norm': 'Parameter Norm',
            'learning_rate': 'Learning Rate',
            'batch_time': 'Batch Duration',
            'memory_used': 'Memory Usage',
            'val_loss': 'Validation Loss',
            'val_accuracy': 'Validation Accuracy',
            'activation_sparsity': 'Activation Sparsity',
            'weight_sparsity': 'Weight Sparsity'
        }

        self._display_i = 0
        self._display_plots = {'loss', 'val_loss'}
        self._display_arrays = []
        self.refresh_plots()

    @property
    def stats(self):
        return self.run.stats

    @property
    def selected_values(self):
        return self._display_plots

    def refresh_plots(self):
        """Rebuilds plot data arrays from run statistics"""
        self._display_i = self.run
        self._display_arrays = []
        for attr in self.selected_values:
            values = getattr(self.stats, attr)
            if values is not None and len(values) > 0:
                self._display_arrays.append(values)

    def gui(self) -> 'PlotUI':
        """Render plot GUI with sidebar metric selection"""
        ret_plot = None

        def gui_left_checkboxes():
            nonlocal ret_plot
            for attr_key, name in self.value_names.items():
                shift_held = gui.get_io().key_shift
                clicked = gui.selectable(attr_key, attr_key in self.selected_values)[0]

                if clicked and shift_held:
                    new_plot = PlotUI(self)
                    new_plot.selected_values.add(attr_key)
                    ret_plot = new_plot
                elif clicked:
                    if attr_key in self.selected_values:
                        self.selected_values.remove(attr_key)
                    else:
                        self.selected_values.add(attr_key)
                    self.refresh_plots()

        def gui_right_plot():
            avail_width, avail_height = gui.get_content_region_avail()
            if not self._display_arrays:
                if implot.begin_plot("##empty_plot", ImVec2(avail_width, avail_height)):
                    implot.setup_axes("Steps", "Metric")
                    implot.end_plot()
            else:
                if implot.begin_plot("##metrics_plot", ImVec2(avail_width, avail_height)):
                    implot.setup_axes("Steps", "Value")
                    for name, values in zip(self._display_plots, self._display_arrays):
                        implot.plot_line(name, values)
                    implot.end_plot()

        if self._display_i != self.run.i:
            self.refresh_plots()

        guio.imgui_sidebar_layout(200, gui_left_checkboxes, gui_right_plot)
        return ret_plot


class RunWindow(guio.IGUI):
    """
    Represents a single Petri simulation run or interactive training dashboard.
    """

    def __init__(self, run: Run):
        assert isinstance(run, Run)
        self.run: Run = run
        self.petri: Petri | None = None
        self.plots = [PlotUI(run)]
        self.is_autostepping = False
        self.texture_pool = TexturePool()

    def insert_plot(self, index: int, plot: PlotUI):
        self.plots.insert(index, plot)

    def gui(self):
        """Render Petri window interface"""
        if not gui.begin(f"Petri - {self.run.run_id}", True):
            gui.end()
            return False

        self.petri.consume_schedule()

        if gui.begin_tab_bar("PetriTabs"):
            if gui.begin_tab_item("Training")[0]:
                guio.imgui_sidebar_layout(300, self.run_left_dash, self.gui_right_viz)
                gui.end_tab_item()
            if gui.begin_tab_item("Tensors")[0]:
                if gui.begin_child("inference_view", ImVec2(0, 0)):
                    avail_width = gui.get_content_region_avail().x
                    avail_height = gui.get_content_region_avail().y
                    tensor_height = avail_height / len(self.petri.tensorviews) if len(self.petri.tensorviews) > 0 else 0
                    for id, view in self.petri.tensorviews.items():
                        if gui.begin_child(f"view_{id}", ImVec2(0, tensor_height)):
                            view.gui()
                        gui.end_child()
                    if len(self.petri.tensorviews) == 0:
                        gui.text("No tensor data available.")
                gui.end_child()
                gui.end_tab_item()
            gui.end_tab_bar()

        gui.end()
        return True

    def run_left_dash(self):
        """Render left sidebar with run and checkpoint controls"""
        if not self.petri.tissue:
            self.run.init_tissue = gui_utils.gui_object_header("Tissue Config", self.run.schedule_config.tissue, TissueConfig)
        else:
            self.run.schedule_config.tissue = gui_utils.gui_object_header("Tissue Config", self.run.schedule_config.tissue, TissueConfig)

        if not self.petri.tissue:
            self.run.init_physics = gui_utils.gui_object_header("Physics Config", self.run.init_tissue, PhysicsConfig)
        else:
            self.run.schedule_config.physics = gui_utils.gui_object_header("Physics Config", self.run.schedule_config.physics, PhysicsConfig)


# ----------------------------------------
# PETRI APP
# ----------------------------------------

class PetriApp:
    """Main application for managing multiple Petri run windows."""

    def __init__(self):
        self.petris: dict[str, RunWindow] = {}
        self.run_names = self.get_available_runs()
        self.maximizing_window = None

        from models.autoencodermodern import AutoencoderTissue
        wnd = self.on_btn_new_run()
        wnd.run.init_tissue = AutoencoderTissue()
        wnd.run.init_physics = AdamPhysics()
        wnd.petri = wnd.run.create_petri()
        wnd.petri.init()

    def get_available_runs(self) -> list[str]:
        """Return a list of available run directories."""
        runs_dir = Path("runs")
        if not runs_dir.exists():
            return []
        return [d.name for d in runs_dir.iterdir() if d.is_dir() and (d / "config.json").exists()]

    def gui_init(self):
        hello_imgui.apply_theme(hello_imgui.ImGuiTheme_.photoshop_style)

    def gui(self):
        """Render menu bar and Petri windows."""
        if gui.begin_main_menu_bar():
            if gui.button("new"):
                self.on_btn_new_run()
            gui.text(f"Runs: {len(self.petris)}")
            gui.set_next_item_width(200)
            if gui.begin_combo("##load_run", ""):
                for run_name in self.run_names:
                    is_selected = gui.selectable(run_name, False)[0]
                    if is_selected:
                        self.on_load_petri_name(run_name)
                gui.end_combo()
            gui.end_main_menu_bar()

        to_close = []
        for run_id, window in self.petris.items():
            if not window.gui():
                to_close.append(run_id)
        for run_id in to_close:
            del self.petris[run_id]

    def add_petri(self, run: Run) -> RunWindow:
        wnd = RunWindow(run)
        self.petris[run.run_id] = wnd
        return wnd

    def on_btn_new_run(self):
        physics_config = AdamPhysics()
        run = Run.NewRun(None, physics_config)
        wnd = self.add_petri(run)
        return wnd

    def on_load_petri_name(self, run_name):
        run = Run.LoadRun(run_name)
        self.petris[run.run_id] = RunWindow(run)


# ----------------------------------------
# MAIN ENTRY
# ----------------------------------------

def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QMainWindow()
    synapse_app = PetriApp()
    window.setWindowTitle("Synapse Petri")
    window.resize(1280, 720)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
