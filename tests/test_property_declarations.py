"""Every node declares its properties, and each property's input port sets it.

The checks of nonoha_core._testing (nonoha-core 1.2), written out here because
these tests also run against the nonoha-core releases before it.
"""

import numpy as np
from nonoha_core.data_type import name_to_port_value_type
from nonoha_core.node_registry import NodeRegistry

import audio_nodes


def _nodes():
    registry = NodeRegistry()
    audio_nodes.register_nodes(registry)
    for definition in registry.node_definitions.values():
        node = definition.factory(f"check_{definition.type_name}")
        yield definition.type_name, node
        node.finalize()


def test_all_properties_are_declared():
    """PROPERTIES holds every property: description, data_type, default_value; each has its port."""
    errors = []
    for type_name, node in _nodes():
        declared = type(node).PROPERTIES
        errors += [f"{type_name}.{key}: not in PROPERTIES" for key in node.properties.keys() - declared.keys()]
        for key, description in declared.items():
            where = f"{type_name}.{key}"
            if not description.get("description"):
                errors.append(f"{where}: no description")
            if "data_type" not in description:
                errors.append(f"{where}: no data_type")
            else:
                name_to_port_value_type(description["data_type"])
                if node.PROPERTY_PORT_PREFIX + key not in node.inputs:
                    errors.append(f"{where}: no input port")
            if "default_value" not in description:
                errors.append(f"{where}: no default_value")
            elif node.properties.get(key) != description["default_value"]:
                errors.append(f"{where}: starts as {node.properties.get(key)!r}, not as its default_value")
    assert errors == []


def test_property_ports_apply():
    """A port hands over a numpy scalar: the node takes its own default that way,
    keeps the property a plain int, float, bool or str (properties are saved as
    JSON), and keeps its property ports."""
    errors = []
    for type_name, node in _nodes():
        prefix = node.PROPERTY_PORT_PREFIX
        ports = [name for name in node.inputs if name.startswith(prefix)]
        for name in ports:
            key = name[len(prefix):]
            before = node.properties[key]
            try:
                node.set_properties({key: np.array([before])[-1]})
            except Exception as e:
                errors.append(f"{type_name}.{key}: refuses its default from the port ({type(e).__name__}: {e})")
                continue
            after = node.properties.get(key)
            if not isinstance(after, (bool, int, float, str)):
                errors.append(f"{type_name}.{key}: becomes {type(after).__name__}")
            elif after != before:
                errors.append(f"{type_name}.{key}: {before!r} becomes {after!r}")
            if any(port not in node.inputs for port in ports):
                errors.append(f"{type_name}.{key}: setting it removes property ports")
                break
    assert errors == []
