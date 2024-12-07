import torch
import torch.nn as nn
import torch.optim as optim
import gpytorch
import numpy as np
import pickle
import os

# Paths
success_demo_path = "/home/yuxuanli/failed_IRL_new/FL/models/successful_demonstrations_1000.pkl"
failed_demo_path = "/home/yuxuanli/failed_IRL_new/FL/models/failed_demonstrations_1000.pkl"
model_save_path = "/home/yuxuanli/failed_IRL_new/Deep_Model/DKL_GPR_Model_SFD_100000.pth"

# Check if CUDA is available and set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the deep neural kernel model (feature extractor)
class DeepFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DeepFeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the DKL Gaussian Process Regression Model
class DKLGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, feature_extractor, likelihood):
        super(DKLGPModel, self).__init__(train_x, train_y, likelihood)
        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        # Pass through the feature extractor first
        x = self.feature_extractor(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Function to preprocess the transitions and compute weights for success and failure
def prepare_data(successful_demos, failed_demos, exp_k=1.0):
    states = []
    weights = []
    
    # Process successful demonstrations with positive weights
    for traj in successful_demos:
        demo_length = len(traj)
        for i, (obs, act, reward, next_obs, done, success) in enumerate(traj):
            flat_trans = obs.flatten()
            weight = (i + 1) / demo_length * np.exp(exp_k * (i + 1) / demo_length)
            states.append(flat_trans)
            weights.append(weight)
    
    # Process failed demonstrations with negative weights
    for traj in failed_demos:
        demo_length = len(traj)
        for i, (obs, act, reward, next_obs, done, success) in enumerate(traj):
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
states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
weights_tensor = torch.tensor(weights, dtype=torch.float32).squeeze().to(device)  # Squeeze to 1D

# Initialize the deep feature extractor and DKL-GP model
input_dim = states_tensor.shape[1]
hidden_dim = 64
feature_extractor = DeepFeatureExtractor(input_dim, hidden_dim).to(device)
likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
model = DKLGPModel(states_tensor, weights_tensor, feature_extractor, likelihood).to(device)

# Set the model to training mode
model.train()
likelihood.train()

# Optimizer for both the model parameters and the likelihood
optimizer = torch.optim.Adam([
    {'params': model.feature_extractor.parameters()},
    {'params': model.covar_module.parameters()},
    {'params': model.mean_module.parameters()},
    {'params': likelihood.parameters()},
], lr=0.01)

# Loss function (marginal log likelihood for GPs)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    likelihood.train()
    optimizer.zero_grad()
    
    # Forward pass
    output = model(states_tensor)
    loss = -mll(output, weights_tensor)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Save the trained model and likelihood
torch.save({
    'model_state_dict': model.state_dict(),
    'likelihood_state_dict': likelihood.state_dict(),
    'feature_extractor_state_dict': feature_extractor.state_dict()
}, model_save_path)
print(f"Model saved to {model_save_path}")

# To load the model later:
model = DKLGPModel(states_tensor, weights_tensor, feature_extractor, likelihood).to(device)
checkpoint = torch.load(model_save_path)
model.load_state_dict(checkpoint['model_state_dict'])
likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
model.eval()
likelihood.eval()