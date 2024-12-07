import gymnasium as gym
import gymnasium_robotics
import numpy as np
from typing import List, Union, Optional, Dict
from os import path
from gymnasium import spaces
from gymnasium.utils.ezpickle import EzPickle
from gymnasium_robotics.envs.maze.maze_v4 import MazeEnv
from gymnasium_robotics.envs.maze.point import PointEnv
from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames
from gymnasium_robotics.envs.maze.point_maze import PointMazeEnv
from gymnasium.envs.registration import register
import imageio
from gymnasium_robotics.envs.maze.maps import U_MAZE

from gymnasium_robotics.envs.maze.maze_v4 import Maze
import xml.etree.ElementTree as ET
from gymnasium_robotics.envs.maze.point_maze import PointMazeEnv
from gymnasium_robotics.envs.maze.maps import COMBINED, GOAL, RESET, U_MAZE
import tempfile
import time
TRAP = T = "t"

class modifyMaze(Maze):
    def __init__(
        self,
        maze_map: List[List[Union[str, int]]],
        maze_size_scaling: float,
        maze_height: float,
    ):

        self._maze_map = maze_map
        self._maze_size_scaling = maze_size_scaling
        self._maze_height = maze_height

        self._unique_goal_locations = []
        self._unique_reset_locations = []
        self._combined_locations = []
        self._trap_locations = []

        # Get the center cell Cartesian position of the maze. This will be the origin
        self._map_length = len(maze_map)
        self._map_width = len(maze_map[0])
        self._x_map_center = self.map_width / 2 * maze_size_scaling
        self._y_map_center = self.map_length / 2 * maze_size_scaling

    @classmethod
    def make_maze(
        cls,
        agent_xml_path: str,
        maze_map: list,
        maze_size_scaling: float,
        maze_height: float,
    ):
        tree = ET.parse(agent_xml_path)
        worldbody = tree.find(".//worldbody")

        maze = cls(maze_map, maze_size_scaling, maze_height)
        empty_locations = []
        for i in range(maze.map_length):
            for j in range(maze.map_width):
                struct = maze_map[i][j]
                # Store cell locations in simulation global Cartesian coordinates
                x = (j + 0.5) * maze_size_scaling - maze.x_map_center
                y = maze.y_map_center - (i + 0.5) * maze_size_scaling
                if struct == 1:  # Unmovable block.
                    # Offset all coordinates so that maze is centered.
                    ET.SubElement(
                        worldbody,
                        "geom",
                        name=f"block_{i}_{j}",
                        pos=f"{x} {y} {maze_height / 2 * maze_size_scaling}",
                        size=f"{0.5 * maze_size_scaling} {0.5 * maze_size_scaling} {maze_height / 2 * maze_size_scaling}",
                        type="box",
                        material="",
                        contype="1",
                        conaffinity="1",
                        rgba="0.7 0.5 0.3 1.0",
                    )

                elif struct == RESET:
                    maze._unique_reset_locations.append(np.array([x, y]))
                elif struct == GOAL:
                    maze._unique_goal_locations.append(np.array([x, y]))
                elif struct == COMBINED:
                    maze._combined_locations.append(np.array([x, y]))
                elif struct == 0:
                    empty_locations.append(np.array([x, y]))
                elif struct == TRAP:  
                    ET.SubElement(
                        worldbody,
                        "site",
                        name=f"trap_{i}_{j}",
                        pos=f"{x} {y} {maze_height / 2 * maze_size_scaling}",
                        size=f"{0.2 * maze_size_scaling}",
                        rgba="0 0 1 0.7",  
                        type="sphere",
                    )
                    maze._trap_locations.append(np.array([x, y]))  
                elif struct == 0:
                    empty_locations.append(np.array([x, y]))


        # Add target site for visualization
        ET.SubElement(
            worldbody,
            "site",
            name="target",
            pos=f"0 0 {maze_height / 2 * maze_size_scaling}",
            size=f"{0.2 * maze_size_scaling}",
            rgba="1 0 0 0.7",
            type="sphere",
        )

        # Add the combined cell locations (goal/reset) to goal and reset
        if (
            not maze._unique_goal_locations
            and not maze._unique_reset_locations
            and not maze._combined_locations
        ):
            # If there are no given "r", "g" or "c" cells in the maze data structure,
            # any empty cell can be a reset or goal location at initialization.
            maze._combined_locations = empty_locations
        elif not maze._unique_reset_locations and not maze._combined_locations:
            # If there are no given "r" or "c" cells in the maze data structure,
            # any empty cell can be a reset location at initialization.
            maze._unique_reset_locations = empty_locations
        elif not maze._unique_goal_locations and not maze._combined_locations:
            # If there are no given "g" or "c" cells in the maze data structure,
            # any empty cell can be a gaol location at initialization.
            maze._unique_goal_locations = empty_locations

        maze._unique_goal_locations += maze._combined_locations
        maze._unique_reset_locations += maze._combined_locations

        # Save new xml with maze to a temporary file
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_xml_name = f"ant_maze{str(time.time())}.xml"
            temp_xml_path = path.join(path.dirname(tmp_dir), temp_xml_name)
            tree.write(temp_xml_path)

        return maze, temp_xml_path

class modifyPointMazeEnv(PointMazeEnv):
    def __init__(
        self,
        agent_xml_path=None,
        maze_map=None,
        maze_size_scaling=1,
        maze_height=0.4,
        position_noise_range: float = 0.25,
        render_mode: Optional[str] = None,
        reward_type=None,
        continuing_task=None,
        reset_target=None,
        point_xml_file_path = None,
        **kwargs,
    ):

        self.reward_type = reward_type
        self.continuing_task = continuing_task
        self.reset_target = reset_target
        self.maze, self.tmp_xml_file_path = modifyMaze.make_maze(
            agent_xml_path , maze_map, maze_size_scaling, maze_height
        )

        self.position_noise_range = position_noise_range

        maze_length = len(maze_map)
        default_camera_config = {"distance": 12.5 if maze_length > 8 else 8.8}

        self.point_env = PointEnv(
            xml_file=self.tmp_xml_file_path,
            render_mode=render_mode,
            default_camera_config=default_camera_config,
            **kwargs,
        )
        self._model_names = MujocoModelNames(self.point_env.model)
        self.target_site_id = self._model_names.site_name2id["target"]

        self.action_space = self.point_env.action_space
        obs_shape: tuple = self.point_env.observation_space.shape
        self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(
                    -np.inf, np.inf, shape=obs_shape, dtype="float64"
                ),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=(2,), dtype="float64"),
                desired_goal=spaces.Box(-np.inf, np.inf, shape=(2,), dtype="float64"),
            )
        )

        self.render_mode = render_mode

        EzPickle.__init__(
            self,
            maze_map,
            render_mode,
            reward_type,
            continuing_task,
            reset_target,
            **kwargs,
        )

class TrapMazeEnv(modifyPointMazeEnv):
    def __init__(self, *args, **kwargs):
        self.in_trap = False 
        agent_xml_path = path.join(
            path.dirname(path.realpath(__file__)), "/Users/yuxuanli/Maze/point.xml"
        )
        super().__init__(*args, agent_xml_path=agent_xml_path, **kwargs)

    def step(self, action):
        current_position = self.point_env.data.qpos[:2].copy()

        for trap_pos in self.maze._trap_locations:
            if np.linalg.norm(current_position - trap_pos) <= 0.01:  
                self.in_trap = True
                break

        if self.in_trap:
            self.point_env.data.qvel[:] = 0  
            action = np.zeros_like(action)  

        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self.in_trap = False  
        return super().reset(*args, **kwargs)

example_map = [
    [1, 1, 1, 1, 1],
    [1, 0, 0, 'g', 1], 
    [1, 1, 't', 0, 1],
    [1, 'r', 0, 't', 1],
    [1, 1, 1, 1, 1]
]

register(
    id="TrapMazeEnv",  
    entry_point="__main__:TrapMazeEnv", 
    max_episode_steps=100,  
)
env = gym.make('TrapMazeEnv', maze_map=example_map, render_mode="human", max_episode_steps=100, camera_name="topview")

obs, info = env.reset()
done = False
for _ in range(1):
    state, _ = env.reset()
    truncated, done = False, False

    while not (truncated or done) :
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
env.close()