import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import os

# Paths
success_demo_path = "/home/yuxuanli/failed_IRL_new/FL/models/successful_demonstrations_1000.pkl"
failed_demo_path = "/home/yuxuanli/failed_IRL_new/FL/models/failed_demonstrations_1000.pkl"
model_save_path = "/home/yuxuanli/failed_IRL_new/Deep_Model/KR_Deep_Model_SFD_100000.pth"

# Check if CUDA is available and set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the neural kernel regression model
class NeuralKernelRegression(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(NeuralKernelRegression, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # Predicts the weight

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Function to preprocess the transitions and compute weights for success and failure
def prepare_data(successful_demos, failed_demos, exp_k=1.0):
    states = []
    weights = []
    
    # Process successful demonstrations with positive weights
    for traj in successful_demos:
        demo_length = len(traj)
        for i, (obs, act, reward, next_obs, done, success) in enumerate(traj):
            # Use only the obs (state) as the input feature
            flat_trans = obs.flatten()
            weight = (i + 1) / demo_length * np.exp(exp_k * (i + 1) / demo_length)
            states.append(flat_trans)
            weights.append(weight)
    
    # Process failed demonstrations with negative weights
    for traj in failed_demos:
        demo_length = len(traj)
        for i, (obs, act, reward, next_obs, done, success) in enumerate(traj):
            # Use only the obs (state) as the input feature
            flat_trans = obs.flatten()
            weight = -(i + 1) / demo_length * np.exp(exp_k * (i + 1) / demo_length)  # Negative weight
            states.append(flat_trans)
            weights.append(weight)
    
    # Convert to numpy arrays
    states = np.array(states)
    weights = np.array(weights).reshape(-1, 1)
    
    return states, weights

# Load the successful and failed demonstrations
if os.path.exists(success_demo_path) and os.path.exists(failed_demo_path):
    with open(success_demo_path, "rb") as f:
        successful_demos = pickle.load(f)
    with open(failed_demo_path, "rb") as f:
        failed_demos = pickle.load(f)
else:
    raise FileNotFoundError("One or both of the demonstration files not found. Run demonstration collection first.")

# Prepare data for training
states, weights = prepare_data(successful_demos, failed_demos)

# Convert data to PyTorch tensors and move them to the device (GPU if available)
states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

# Initialize and train the neural kernel regression model
input_dim = states_tensor.shape[1]
hidden_dim = 64
model = NeuralKernelRegression(input_dim, hidden_dim).to(device)  # Move model to GPU
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop
num_epochs = 100000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    predictions = model(states_tensor)
    loss = criterion(predictions, weights_tensor)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Save the trained model
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# To load the model later:
model = NeuralKernelRegression(input_dim, hidden_dim).to(device)
model.load_state_dict(torch.load(model_save_path))
model.eval()