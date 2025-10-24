# synapse_gui_framework.py — Dynamic ImGui reflection system for Synapse visual editing
# ------------------------------------------------------------------------------------
from dataclasses import is_dataclass, fields
from typing import Any, Dict, List, Optional, Set
from imgui_bundle import imgui, imgui_ctx

import gui


def gui_none(name: str, obj: Any, obj_type: Any) -> Any:
    """Render a type selector for None values, allowing instantiation of obj_type or its subclasses."""
    def collect_all_subclasses(cls):
        subclasses = []
        for subclass in cls.__subclasses__():
            subclasses.append(subclass)
            subclasses.extend(collect_all_subclasses(subclass))
        return subclasses

    subclasses = [obj_type] + collect_all_subclasses(obj_type)
    current_class = obj_type.__name__
    imgui.text(f"{name}:")
    imgui.same_line()
    if imgui.begin_combo("##type_selector", current_class):
        for cls in subclasses:
            selected = current_class == cls.__name__
            if imgui.selectable(cls.__name__, selected)[0]:
                obj = cls()
            if selected:
                imgui.set_item_default_focus()
        imgui.end_combo()
    return obj


def gui_any(name: str, obj: Any, obj_type: Any = None, collapsible_header: bool = False) -> Any:
    """Render a generic ImGui representation for various data structures."""
    if is_dataclass(obj):
        return gui_object(name, obj, collapsible_header)
    elif isinstance(obj, dict):
        return gui_dict(name, obj)
    elif isinstance(obj, list):
        return gui_list(name, obj)
    elif isinstance(obj, tuple):
        return gui_tuple(name, obj)
    elif isinstance(obj, set):
        return gui_set(name, obj)
    elif isinstance(obj, (int, float)):
        _, value = imgui.drag_float(f"{name}", obj, 0.001)
        return value
    elif isinstance(obj, str):
        _, value = imgui.input_text(f"{name}", obj, 256)
        return value
    elif isinstance(obj, bool):
        _, value = imgui.checkbox(f"{name}", obj)
        return value
    elif obj_type is not None:
        imgui.pop_id()
        return gui_none(name, obj, obj_type)
    else:
        imgui.text(f"{name}: {str(obj)}")
        imgui.pop_id()
        return obj


def gui_object(label: str, obj: Any, obj_type: Any = None, fields_to_render: Optional[List[str]] = None, enable: bool = True) -> Any:
    """Render a dataclass or structured object for editing in ImGui."""
    if not enable:
        imgui.begin_disabled()

    if obj is None and obj_type is not None:
        return gui_none(label, obj, obj_type)
    elif is_dataclass(obj):
        with imgui_ctx.push_id(label):
            for field in fields(obj):
                if fields_to_render is None or field.name in fields_to_render:
                    val = getattr(obj, field.name)
                    new_val = gui_any(field.name, val)
                    setattr(obj, field.name, new_val)
        result = obj
    elif isinstance(obj, dict):
        result = gui_dict(label, obj)
    else:
        result = gui_any(label, obj, obj_type)

    if not enable:
        imgui.end_disabled()

    return result


def gui_object_header(name: str, obj: Any, obj_type: Any = None, enable: bool = True) -> Any:
    """Render an expandable dataclass section with header."""
    with imgui_ctx.push_id(name):
        label = f"{name} ({obj_type.__name__ if obj is None else obj.__class__.__name__})"
        if imgui.collapsing_header(label):
            gui_object(name, obj, obj_type, enable=enable)
        if imgui.begin_popup_context_item(f"{name}_header_popup"):
            obj = gui_none(f"Change {name} Type", obj, obj_type)
            imgui.end_popup()
    return obj


def gui_object_node(name: str, obj: Any, obj_type: Any = None, enable: bool = True) -> Any:
    """Render an expandable tree node for a dataclass object."""
    with imgui_ctx.push_id(name):
        label = f"{name} ({obj_type.__name__ if obj is None else obj.__class__.__name__})"
        if imgui.tree_node(label):
            obj = gui_object(name, obj, obj_type, enable=enable)
            imgui.tree_pop()
        if imgui.begin_popup_context_item(f"{name}_node_popup"):
            obj = gui_none(f"Change {name} Type", obj, obj_type)
            imgui.end_popup()
    return obj


# TUPLES
# ------------------------------------------------------------------------------------

def gui_tuple(name: str, t: tuple) -> tuple:
    """Render and edit tuple values in-line."""
    if len(t) <= 4 and all(isinstance(x, (int, float)) for x in t):
        if len(t) == 2:
            changed, values = imgui.input_float2(f"{name}##value", t)
        elif len(t) == 3:
            changed, values = imgui.input_float3(f"{name}##value", t)
        elif len(t) == 4:
            changed, values = imgui.input_float4(f"{name}##value", t)
        else:
            changed, values = False, t
        return tuple(values) if changed else t

    modified = False
    values = list(t)
    for i, item in enumerate(values):
        imgui.push_id(str(i))
        new_val = gui_any(f"Item {i}:", item)
        if new_val != item:
            values[i] = new_val
            modified = True
        imgui.pop_id()
    return tuple(values) if modified else t


def gui_tuple_node(name: str, t: tuple) -> tuple:
    label = f"{name} (Tuple: {len(t)} items)"
    if imgui.tree_node(label):
        result = gui_tuple(name, t)
        imgui.tree_pop()
        return result
    return t


def gui_tuple_header(name: str, t: tuple) -> tuple:
    label = f"{name} (Tuple: {len(t)} items)"
    if imgui.collapsing_header(label):
        return gui_tuple(name, t)
    return t


# DICTIONARIES
# ------------------------------------------------------------------------------------

def gui_dict(name: str, d: Dict[Any, Any]) -> Dict[Any, Any]:
    """Render a dictionary inline with editable key-value pairs."""
    modified = False
    updated_dict = d.copy()
    for key, value in d.items():
        imgui.text(f"{key}:")
        imgui.same_line()
        new_val = gui_any(key, value)
        if new_val != value:
            updated_dict[key] = new_val
            modified = True
    return updated_dict if modified else d


def gui_dict_node(name: str, d: Dict[Any, Any]) -> Dict[Any, Any]:
    label = f"{name} (Dict: {len(d)} entries)"
    if imgui.tree_node(label):
        result = gui_dict(name, d)
        imgui.tree_pop()
        return result
    return d


def gui_dict_header(name: str, d: Dict[Any, Any]) -> Dict[Any, Any]:
    label = f"{name} (Dict: {len(d)} entries)"
    if imgui.collapsing_header(label):
        return gui_dict(name, d)
    return d


# LISTS
# ------------------------------------------------------------------------------------

def gui_list(name: str, l: List[Any]) -> List[Any]:
    """Render and edit lists inline with add/remove/reorder controls."""
    modified = False
    actions = []

    if imgui.button(f"Add to {name}"):
        actions.append(("add", len(l), None))
        modified = True

    for i, item in enumerate(l):
        imgui.push_id(str(i))
        if imgui.button("×"):
            actions.append(("delete", i, None))
            modified = True
        imgui.same_line()
        if i > 0 and imgui.arrow_button("up", imgui.Dir.up):
            actions.append(("swap", i, i - 1))
            modified = True
        imgui.same_line()
        if i < len(l) - 1 and imgui.arrow_button("down", imgui.Dir.down):
            actions.append(("swap", i, i + 1))
            modified = True
        imgui.same_line()

        new_val = gui_any(f"Item {i}", item)
        if new_val != item:
            actions.append(("edit", i, new_val))
            modified = True
        imgui.pop_id()

    if modified:
        for op, idx, val in reversed(actions):
            if op == "swap":
                l[idx], l[val] = l[val], l[idx]
            elif op == "delete":
                l.pop(idx)
            elif op == "add":
                l.append(val if val is not None else "New Item")
            elif op == "edit":
                l[idx] = val
    return l


def gui_list_node(name: str, l: List[Any]) -> List[Any]:
    label = f"{name} (List: {len(l)} items)"
    if imgui.tree_node(label):
        result = gui_list(name, l)
        imgui.tree_pop()
        return result
    return l


def gui_list_header(name: str, l: List[Any]) -> List[Any]:
    label = f"{name} (List: {len(l)} items)"
    if imgui.collapsing_header(label):
        return gui_list(name, l)
    return l


# SETS
# ------------------------------------------------------------------------------------

def gui_set(name: str, s: Set[Any]) -> Set[Any]:
    """Render and edit sets inline."""
    modified = False
    new_set = set()
    for item in s:
        imgui.indent()
        new_val = gui_any(name, item)
        if new_val != item:
            modified = True
        new_set.add(new_val)
        imgui.unindent()
    return new_set if modified else s


def gui_set_node(name: str, s: Set[Any]) -> Set[Any]:
    label = f"{name} (Set: {len(s)} items)"
    if imgui.tree_node(label):
        result = gui_set(name, s)
        imgui.tree_pop()
        return result
    return s


def gui_set_header(name: str, s: Set[Any]) -> Set[Any]:
    label = f"{name} (Set: {len(s)} items)"
    if imgui.collapsing_header(label):
        return gui_set(name, s)
    return s
