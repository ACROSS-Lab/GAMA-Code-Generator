import numpy as np

def proxy_tuning(base_logits, expert_logits, anti_expert_logits):
    # Compute the logit scores for the proxy-tuned model
    tuned_logits = base_logits + expert_logits - anti_expert_logits
    
    # Apply softmax to obtain the probability distribution
    tuned_probs = softmax(tuned_logits)
    
    return tuned_probs

def softmax(x):
    # Compute softmax values for each set of scores in x
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)

# Example usage
base_model_logits = np.array([0.1, 0.5, 0.3])  # Replace with actual logits from your base model
expert_model_logits = np.array([0.2, 0.4, 0.1])  # Replace with actual logits from your expert model
anti_expert_model_logits = np.array([0.3, 0.2, 0.5])  # Replace with actual logits from your anti-expert model

# Proxy-tuning
tuned_probs = proxy_tuning(base_model_logits, expert_model_logits, anti_expert_model_logits)

print("Proxy-tuned probabilities:", tuned_probs)

