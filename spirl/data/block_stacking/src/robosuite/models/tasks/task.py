from spirl.data.block_stacking.src.robosuite.models.world import MujocoWorldBase


class Task(MujocoWorldBase):
    """
    Base class for creating MJCF model of a task.

    A task typically involves a robot interacting with tasks in an arena
    (workshpace). The purpose of a task class is to generate a MJCF model
    of the task by combining the MJCF models of each component together and
    place them to the right positions. Object placement can be done by
    ad-hoc methods or placement samplers.
    """

    def merge_robot(self, mujoco_robot):
        """Adds robot model to the MJCF model."""
        pass

    def merge_arena(self, mujoco_arena):
        """Adds arena model to the MJCF model."""
        pass

    def merge_objects(self, mujoco_objects):
        """Adds physical tasks to the MJCF model."""
        pass

    def merge_visual(self, mujoco_objects):
        """Adds visual tasks to the MJCF model."""

    def place_objects(self):
        """Places tasks randomly until no collisions or max iterations hit."""
        pass

    def place_visual(self):
        """Places visual tasks randomly until no collisions or max iterations hit."""
        pass
