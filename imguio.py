# synapse_ui.py — Interactive visualization utilities
# ----------------------------------------
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union

from imgui_bundle import imgui, imgui_ctx, ImVec2


class ISynapseGUI(ABC):
    @abstractmethod
    def render(self):
        raise NotImplementedError


class ISynapsePlottable(ABC):
    @abstractmethod
    def render(self):
        raise NotImplementedError


def render_header(label: str, spacing: bool = True):
    if spacing:
        imgui.dummy(imgui.ImVec2(0, 10))
    imgui.text(label)
    imgui.separator()


def toggle_button(label: str, state: bool) -> Tuple[bool, bool]:
    default_color = imgui.get_style_color_vec4(imgui.Col_.button.value)
    active_color = imgui.get_style_color_vec4(imgui.Col_.button_active.value)
    color = active_color if state else default_color
    with imgui_ctx.push_style_color(imgui.Col_.button.value, color):
        clicked = imgui.button(label)
        if clicked:
            state = not state
    return clicked, state


def synapse_sidebar_layout(width, sidebar_ui, main_ui):
    with imgui_ctx.begin_child("Sidebar", ImVec2(width, 0), imgui.ChildFlags_.border.value):
        sidebar_ui()

    imgui.same_line()
    with imgui_ctx.begin_child("MainPanel", ImVec2(0, 0)):
        main_ui()


def format_float(value: float) -> str:
    return f"{value:.2f}"


def format_float_gb(value: float) -> str:
    return f"{value:.1f} GB"


def format_int(value: int) -> str:
    return str(value)


def format_str(value: str) -> str:
    return str(value)


@dataclass
class ColumnDescriptor:
    key: str
    label: str
    format_spec: Union[Callable[[Any], str], str, None] = None
    min: float = 0
    max: float = 0
    inverted: bool = False


def get_nested_attr(obj: Any, attr_path: str) -> Any:
    """Retrieve nested attributes safely using dot notation."""
    attrs = attr_path.split('.')
    value = obj
    for attr in attrs:
        try:
            value = getattr(value, attr)
        except AttributeError:
            return None
    return value


def draw_sortable_table(
    items: List[Any],
    columns: List[ColumnDescriptor],
    table_name: str,
    selectable: bool = False,
    flags=None
) -> Optional[Any]:
    selected = None

    if flags is None:
        flags = (
            imgui.TableFlags_.scroll_y.value
            | imgui.TableFlags_.row_bg.value
            | imgui.TableFlags_.borders_outer.value
            | imgui.TableFlags_.borders_v.value
            | imgui.TableFlags_.resizable.value
            | imgui.TableFlags_.reorderable.value
            | imgui.TableFlags_.hideable.value
            | imgui.TableFlags_.sortable.value
        )

    if imgui.begin_table(table_name, len(columns), flags):
        for col in columns:
            imgui.table_setup_column(col.label)
        imgui.table_headers_row()

        sort_specs = imgui.table_get_sort_specs()
        if sort_specs and sort_specs.specs_dirty:
            if sort_specs.specs_count > 0:
                specs = sort_specs.get_specs(0)
                sort_key = columns[specs.column_index].key
                items.sort(
                    key=lambda x: get_nested_attr(x, sort_key) or 0,
                    reverse=specs.sort_direction == imgui.SortDirection.descending.value,
                )
            sort_specs.specs_dirty = False

        for j, obj in enumerate(items):
            imgui.table_next_row()
            for i, col in enumerate(columns):
                value = get_nested_attr(obj, col.key)
                if value is None:
                    value = 0

                is_percent = col.min != col.max
                percent = value / col.max if col.max > 0 else 0
                if col.inverted:
                    percent = 1 - percent

                imgui.table_set_column_index(i)
                if isinstance(col.format_spec, Callable):
                    formatted_value = col.format_spec(value)
                elif isinstance(col.format_spec, str):
                    formatted_value = col.format_spec.format(value)
                else:
                    formatted_value = str(value)

                if selectable and i == 0:
                    if imgui.selectable(
                        f"#{j}: {formatted_value}",
                        False,
                        imgui.SelectableFlags_.span_all_columns.value,
                    )[0]:
                        selected = obj
                else:
                    if is_percent:
                        color = imgui.ImVec4(1 - percent, 1, 1 - percent, 1)
                        imgui.text_colored(color, formatted_value)
                    else:
                        imgui.text(formatted_value)

        imgui.end_table()

    return selected
