from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_task_node_loads_valid_task():
    from tactile_task.task_node import TaskNode

    node = object.__new__(TaskNode)
    node.task_package = "tactile_task.tasks"

    module, args = TaskNode._load_task(node, "pick_place", '{"place":{"x":0.1}}')

    assert callable(module.run)
    assert args["place"]["x"] == pytest.approx(0.1)


def test_task_node_rejects_invalid_task_name():
    from tactile_task.task_node import TaskNode

    node = object.__new__(TaskNode)
    node.task_package = "tactile_task.tasks"

    with pytest.raises(RuntimeError, match="invalid task name"):
        TaskNode._load_task(node, "../pick_place", "{}")


def test_task_node_rejects_non_object_json():
    from tactile_task.task_node import TaskNode

    node = object.__new__(TaskNode)
    node.task_package = "tactile_task.tasks"

    with pytest.raises(RuntimeError, match="JSON object"):
        TaskNode._load_task(node, "pick_place", "[]")
