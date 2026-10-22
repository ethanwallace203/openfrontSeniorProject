from openfront_il.models.checkpoint_format import load_checkpoint
from pathlib import Path
import torch
from torch.distributions import Categorical

def load_agent(checkpoint_path):
    policy, metadata = load_checkpoint(Path(checkpoint_path))
    policy.eval()
    return policy

def choose_action(policy, spatial, flat):
    spatialTensor = torch.as_tensor(spatial).float().unsqueeze(0)
    flatTensor = torch.as_tensor(flat).float().unsqueeze(0)
    with torch.no_grad():
        outputLogits = policy(spatialTensor, flatTensor)
    actionTypeDistribution = Categorical(logits=outputLogits["action_type"])
    unitTypeDistribution = Categorical(logits=outputLogits["unit_type"])

    return actionTypeDistribution.sample().item(), unitTypeDistribution.sample().item()