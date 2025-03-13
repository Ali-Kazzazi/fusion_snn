import numpy as np
from tqdm import tqdm

import chainer
from utils import *
from run import DiehlAndCookNetwork



# Simulation Parameters
dt = 1e-3  # Time step (sec)
stimulus_duration = 0.350  # Stimulus input duration (sec)
rest_duration = 0.150  # rest input duration (sec)
stimulus_steps = round(stimulus_duration / dt)
rest_steps = round(rest_duration / dt)

# Network Parameters
num_neurons = 100  # Number of exc/inh neurons
num_labels = 10  # Number of labels

num_testing_samples = 1000  # Number of testing samples

# Load MNIST dataset
_, test_dataset = chainer.datasets.get_mnist()
labels = np.array([test_dataset[i][1] for i in range(num_testing_samples)])  # Extract labels

# Initialize Network
network = DiehlAndCookNetwork(input_size=784, neuron_count=num_neurons, exc_weight=8, inh_weight=6, dt=dt)

selected_epoch = 15

results_directory = "./results/"

network.initialize_states()
network.input_connection.weights = np.load(results_directory+"weight_epoch"+str(selected_epoch)+".npy")
network.exc_neurons.adaptive_threshold_theta = np.load(results_directory+"exc_neurons_theta_epoch"+str(selected_epoch)+".npy")
network.exc_neurons.threshold_increment = 0

neuron_spike_counts = np.zeros((num_testing_samples, num_neurons), dtype=np.uint8)  # Track spikes
rest_input = np.zeros(784)  # rest stimulus
initial_max_firing_rate = 32  # Initial maximum firing rate for Poisson spikes


# Test the network
for sample_idx in tqdm(range(num_testing_samples)):
    max_firing_rate = initial_max_firing_rate
    while True:
        # Generate input spikes dynamically
        input_spikes = encode_dataset_spikes(
            test_dataset, sample_idx, dt, stimulus_steps, max_firing_rate
        )
        spike_record = []
        
        # Apply image stimulus
        for t in range(stimulus_steps):
            exc_spikes = network(input_spikes[t], apply_stdp=False)
            spike_record.append(exc_spikes)
        
        neuron_spike_counts[sample_idx] = np.sum(np.array(spike_record), axis=0)
        
        # Apply rest stimulus
        for _ in range(rest_steps):
            _ = network(rest_input, apply_stdp=False)
        
        total_spikes = np.sum(np.array(spike_record))
        if total_spikes >= 5:
            break  # Move to the next sample if sufficient spikes occur
        else:
            max_firing_rate += 16  # Increase input firing rate if too few spikes

np.save(results_directory+"features_test_epoch"+str(selected_epoch)+".npy", np.array(neuron_spike_counts))

neuron_assignments = np.load(results_directory+"assignments_epoch"+str(selected_epoch)+".npy")

# Predict labels from spike activity
predicted_labels = predict_labels(neuron_spike_counts, neuron_assignments, num_labels)

 # Compute testing accuracy
accuracy = np.mean((labels == predicted_labels).astype(np.float16))
print("Epoch:", selected_epoch, "Accuracy:", accuracy)