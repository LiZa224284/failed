import gymnasium as gym
import panda_gym
import imageio

# Initialize environment and set the desired number of episodes
env = gym.make("PandaReach-v3", render_mode="rgb_array")
total_episodes = 10  # Set the total number of episodes
episode_count = 0
frames = []

while episode_count < total_episodes:
    observation, info = env.reset()
    
    # Run the episode until it terminates or is truncated
    while True:
        current_position = observation["observation"][0:3]
        desired_position = observation["desired_goal"][0:3]
        action = 5.0 * (desired_position - current_position)
        observation, reward, terminated, truncated, info = env.step(action)

        # Render and store frames
        frame = env.render()
        frames.append(frame)

        # Check if episode ended
        if terminated or truncated:
            episode_count += 1
            print(f"Episode {episode_count} completed")
            break

# Close the environment and save the video
env.close()
imageio.mimsave("/files1/Yuxuan_Li/failed_demos/Experiments/Robots/ManualControl.mp4", frames, fps=30)
print(f"Video saved with {total_episodes} episodes")