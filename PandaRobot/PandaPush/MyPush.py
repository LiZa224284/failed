# from panda_gym.envs.panda_tasks import PandaPushEnv
from panda_gym.pybullet import PyBullet
from panda_gym.envs.robots.panda import Panda
from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.tasks.push import Push
from typing import Optional
import numpy as np
import imageio

class MyPush(Push):
    def reset(self) -> None:
        self.goal = self._sample_goal()
        # object_position = self._sample_object()
        object_position = np.array([0., 0. ,0.02])
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object", object_position, np.array([0.0, 0.0, 0.0, 1.0]))

class MyPandaPushEnv(RobotTaskEnv):
    def __init__(
        self,
        render_mode: str = "rgb_array",
        reward_type: str = "sparse",
        control_type: str = "ee",
        renderer: str = "Tiny",
        render_width: int = 720,
        render_height: int = 480,
        render_target_position: Optional[np.ndarray] = None,
        render_distance: float = 1.4,
        render_yaw: float = 45,
        render_pitch: float = -30,
        render_roll: float = 0,
    ) -> None:
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        robot = Panda(sim, block_gripper=True, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = MyPush(sim, reward_type=reward_type)
        super().__init__(
            robot,
            task,
            render_width=render_width,
            render_height=render_height,
            render_target_position=render_target_position,
            render_distance=render_distance,
            render_yaw=render_yaw,
            render_pitch=render_pitch,
            render_roll=render_roll,
        )

env = MyPandaPushEnv(render_mode="rgb_array")

observation, info = env.reset()

frames = []

num_episodes = 3
for episode in range(num_episodes):
    observation, info = env.reset()
    # terminated = truncated = False
    done = False
    # while not done:
    for _ in range (20):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        frame = env.render()
        frames.append(frame)

        done = terminated or truncated
    print('record one episode')

env.close()
imageio.mimsave('/home/yuxuanli/failed_IRL_new/PandaRobot/PandaPush/MyPush.mp4',frames, fps=10)
