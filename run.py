import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import chainer
import os 
import copy

from utils import *

np.random.seed(seed=0)


class DiehlAndCookNetwork:
    def __init__(self, input_size=784, neuron_count=100, exc_weight=2.25, inh_weight=0.875,
                 dt=1e-3, weight_min=0.0, weight_max=5e-2, learning_rates=(1e-2, 1e-4),
                 update_interval=100):
        """
        Diehl and Cook's 2015 Spiking Neural Network Model
        https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full
        
        Args:
            input_size: Number of input neurons (e.g., pixel count for image input).
            neuron_count: Number of exc and inh neurons.
            exc_weight: Strength of synapses from exc to inh neurons.
            inh_weight: Strength of synapses from inh to exc neurons.
            dt: Simulation time step.
            learning_rates: Tuple of learning rates for pre- and post-synaptic updates.
            weight_min: Minimum synaptic weight.
            weight_max: Maximum synaptic weight.
            update_interval: Time steps before updating weights.
        """
        
        self.dt = dt
        self.learning_rate_pre, self.learning_rate_post = learning_rates
        self.weight_max = weight_max
        self.weight_min = weight_min

        # Neurons
        self.exc_neurons = AdaptiveLIF(neuron_count, dt = dt, refractory_period = 5e-3, 
                 membrane_time_constant = 1e-1, resting_potential = -65, reset_potential = -65, 
                 initial_threshold = -52, peak_potential = 20, threshold_increment = 0.05, 
                 max_threshold = 35, threshold_time_constant = 1e4, exc_potential = 0, 
                 inh_potential = -100)
        

        self.inh_neurons = ConductanceBasedLIF(neuron_count, dt = dt, refractory_period = 2e-3, 
                 membrane_time_constant = 1e-2, resting_potential = -60, reset_potential = -45, 
                 threshold_potential = -40, peak_potential = 20, exc_potential = 0, 
                 inh_potential = -85)

        # Synapses
        self.input_synapse = ExponentialSynapse(input_size, dt=dt, decay_time=1e-3)
        self.exc_synapse = ExponentialSynapse(neuron_count, dt=dt, decay_time=1e-3)
        self.inh_synapse = ExponentialSynapse(neuron_count, dt=dt, decay_time=1e-3)
        
        self.input_trace_synapse = ExponentialSynapse(input_size, dt=dt, decay_time=2e-2)
        self.exc_trace_synapse = ExponentialSynapse(neuron_count, dt=dt, decay_time=2e-2)
        
        # Connections
        initial_weights = 1e-3 * np.random.rand(neuron_count, input_size)
        self.input_connection = FullyConnected(input_size, neuron_count, initial_weights=initial_weights)
        self.exc_to_inh_weights = exc_weight * np.eye(neuron_count)
        self.inh_to_exc_weights = (inh_weight / (neuron_count - 1)) * (
            np.ones((neuron_count, neuron_count)) - np.eye(neuron_count))
        
        self.input_delay = DelayedConnection(num_neurons=neuron_count, delay_time=5e-3, dt=dt)
        self.exc_to_inh_delay = DelayedConnection(num_neurons=neuron_count, delay_time=2e-3, dt=dt)
        
        self.weight_norm_factor = 0.1
        self.inh_conductance = np.zeros(neuron_count)
        self.time_counter = 0
        self.update_interval = update_interval
        self.neuron_count = neuron_count
        self.input_size = input_size
        self.input_spike_buffer = np.zeros((self.update_interval, input_size)) 
        self.exc_spike_buffer = np.zeros((neuron_count, self.update_interval))
        self.input_trace_buffer = np.zeros((self.update_interval, input_size)) 
        self.exc_trace_buffer = np.zeros((neuron_count, self.update_interval))
        
    def reset_spike_traces(self):
        self.input_spike_buffer = np.zeros((self.update_interval, self.input_size)) 
        self.exc_spike_buffer = np.zeros((self.neuron_count, self.update_interval))
        self.input_trace_buffer = np.zeros((self.update_interval, self.input_size)) 
        self.exc_trace_buffer = np.zeros((self.neuron_count, self.update_interval))
        self.time_counter = 0
    
    def initialize_states(self):
        self.exc_neurons.reset_state()
        self.inh_neurons.reset_state()
        self.input_delay.reset()
        self.exc_to_inh_delay.reset()
        self.input_synapse.reset()
        self.exc_synapse.reset()
        self.inh_synapse.reset()
        
    def __call__(self, input_spikes, apply_stdp=True):
        # Input layer
        current_input = self.input_synapse(input_spikes)
        trace_input = self.input_trace_synapse(input_spikes)
        synaptic_input = self.input_connection(current_input)

        # exc neuron layer
        exc_spikes = self.exc_neurons(self.input_delay(synaptic_input), self.inh_conductance)
        current_exc = self.exc_synapse(exc_spikes)
        exc_conductance = np.dot(self.exc_to_inh_weights, current_exc)
        trace_exc = self.exc_trace_synapse(exc_spikes)

        # inh neuron layer        
        inh_spikes = self.inh_neurons(self.exc_to_inh_delay(exc_conductance), 0)
        current_inh = self.inh_synapse(inh_spikes)
        self.inh_conductance = np.dot(self.inh_to_exc_weights, current_inh)

        if apply_stdp:
            # Record spike traces
            self.input_spike_buffer[self.time_counter] = input_spikes
            self.exc_spike_buffer[:, self.time_counter] = exc_spikes
            self.input_trace_buffer[self.time_counter] = trace_input 
            self.exc_trace_buffer[:, self.time_counter] = trace_exc
            self.time_counter += 1

            # Online STDP learning rule
            if self.time_counter == self.update_interval:
                weight_matrix = np.copy(self.input_connection.weights)
                
                weight_norm = np.expand_dims(np.sum(np.abs(weight_matrix), axis=1), 1)
                weight_norm[weight_norm == 0] = 1.0
                weight_matrix *= self.weight_norm_factor / weight_norm
                
                weight_update = self.learning_rate_pre * (self.weight_max - weight_matrix) * np.dot(self.exc_spike_buffer, self.input_trace_buffer)
                weight_update -= self.learning_rate_post * weight_matrix * np.dot(self.exc_trace_buffer, self.input_spike_buffer)
                clipped_update = np.clip(weight_update / self.update_interval, -1e-3, 1e-3)
                self.input_connection.weights = np.clip(weight_matrix + clipped_update, self.weight_min, self.weight_max)
                self.reset_spike_traces()
        
        return exc_spikes

if __name__ == '__main__':

    # Simulation Parameters
    dt = 1e-3  # Time step (sec)
    stimulus_duration = 0.350  # Stimulus input duration (sec)
    rest_duration = 0.150  # rest input duration (sec)
    stimulus_steps = round(stimulus_duration / dt)
    rest_steps = round(rest_duration / dt)

    # Network Parameters
    num_neurons = 100  # Number of exc/inh neurons
    num_labels = 10  # Number of labels
    num_epochs = 30  # Number of training epochs
    num_training_samples = 5000  # Number of training samples
    weight_update_interval = stimulus_steps  # STDP weight update interval

    # Load MNIST dataset
    train_dataset, _ = chainer.datasets.get_mnist()
    labels = np.array([train_dataset[i][1] for i in range(num_training_samples)])  # Extract labels

    # Initialize Network
    network = DiehlAndCookNetwork(
        input_size=784, neuron_count=num_neurons, exc_weight=8, inh_weight=6,
                    dt=dt, weight_min=0.0, weight_max=5e-2, learning_rates=(1e-2, 1e-3),
                    update_interval=weight_update_interval
    )

    network.initialize_states()  # Reset network states
    neuron_spike_counts = np.zeros((num_training_samples, num_neurons), dtype=np.uint8)  # Track spikes
    training_accuracy = np.zeros(num_epochs)  # Store training accuracy
    rest_input = np.zeros(784)  # rest stimulus
    initial_max_firing_rate = 32  # Initial maximum firing rate for Poisson spikes

    # Directory to Save Results
    results_directory = "./results/"
    os.makedirs(results_directory, exist_ok=True)

    # Simulation Loop
    features_per_epoch = []
    for epoch in range(num_epochs):
        for sample_idx in tqdm(range(num_training_samples)):
            max_firing_rate = initial_max_firing_rate
            while True:
                # Generate input spikes dynamically
                input_spikes = encode_dataset_spikes(
                    train_dataset, sample_idx, dt, stimulus_steps, max_firing_rate
                )
                spike_record = []
                
                # Apply image stimulus
                for t in range(stimulus_steps):
                    exc_spikes = network(input_spikes[t], apply_stdp=True)
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
        
        features_per_epoch.append(copy.deepcopy(neuron_spike_counts))
        
        # Assign neurons to labels
        if epoch == 0:
            neuron_assignments, assignment_proportions, firing_rates = assign_neuron_labels(
                neuron_spike_counts, labels, num_labels
            )
        else:
            neuron_assignments, assignment_proportions, firing_rates = assign_neuron_labels(
                neuron_spike_counts, labels, num_labels, firing_rates
            )
        print("Neuron Assignments:\n", neuron_assignments)
        
        # Check firing rates
        total_spike_counts = np.sum(neuron_spike_counts, axis=1)
        avg_spikes = np.mean(total_spike_counts).astype(np.float16)
        print("Average spikes:", avg_spikes)
        print("Min spikes:", total_spike_counts.min())
        print("Max spikes:", total_spike_counts.max())
        
        # Predict labels from spike activity
        predicted_labels = predict_labels(neuron_spike_counts, neuron_assignments, num_labels)
        
        # Compute training accuracy
        accuracy = np.mean((labels == predicted_labels).astype(np.float16))
        print("Epoch:", epoch, "Accuracy:", accuracy)
        training_accuracy[epoch] = accuracy
        
        # Decay learning rates
        network.learning_rate_pre *= 0.9
        network.learning_rate_post *= 0.9

        # Save Weights and Assignments Per Epoch
        np.save(results_directory + f"weight_epoch{epoch}.npy", network.input_connection.weights)
        np.save(results_directory + f"assignments_epoch{epoch}.npy", neuron_assignments)
        np.save(results_directory + f"exc_neurons_theta_epoch{epoch}.npy", network.exc_neurons.adaptive_threshold_theta)
        np.save(results_directory + f"train_features{epoch}.npy", np.array(features_per_epoch[-1]))
        
        # Save Weight Visualization
        fig = plt.figure(figsize=(10, 10))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
        grid_size = int(np.sqrt(num_neurons))
        for i in tqdm(range(num_neurons)):
            ax = fig.add_subplot(grid_size, grid_size, i + 1, xticks=[], yticks=[])
            reshaped_weights = np.reshape(network.input_connection.weights, (num_neurons, 28, 28))
            ax.imshow(reshaped_weights[i], cmap="gray")
        plt.savefig(results_directory + f"weights_{epoch}_acc_{accuracy}.png")
        plt.close("all")
