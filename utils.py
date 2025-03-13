import numpy as np
from typing import Optional

class FullyConnected:
    def __init__(self, input_size, output_size, initial_weights=None):
        """
        FullyConnected: Implements a fully connected layer.
        """
        self.weights = initial_weights if initial_weights is not None else 0.1 * np.random.rand(output_size, input_size)
    
    def backward(self, input_vector):
        return np.dot(self.weights.T, input_vector)
    
    def __call__(self, input_vector):
        return np.dot(self.weights, input_vector)


class DelayedConnection:
    def __init__(self, num_neurons, delay_time, dt=1e-4):
        """
        Args:
            num_neurons (int): Number of neurons.
            delay_time (float): Delay duration.
            dt (float): Simulation time step.
        """
        self.num_neurons = num_neurons
        self.num_delay_steps = round(delay_time / dt)
        self.buffer = np.zeros((num_neurons, self.num_delay_steps))
    
    def reset(self):
        self.buffer = np.zeros((self.num_neurons, self.num_delay_steps))
    
    def __call__(self, input_vector):
        output_vector = self.buffer[:, -1]  # Retrieve the oldest stored value
        
        self.buffer[:, 1:] = self.buffer[:, :-1]  # Shift buffer values to the right
        self.buffer[:, 0] = input_vector  # Store the new input at the beginning
        
        return output_vector
    
class ExponentialSynapse:
    def __init__(self, num_neurons, dt=1e-4, decay_time=5e-3):
        """
        Args:
            num_neurons (int): Number of neurons.
            dt (float): Simulation time step.
            decay_time (float): Synaptic decay time.
        """
        self.num_neurons = num_neurons
        self.dt = dt
        self.decay_time = decay_time
        self.synaptic_response = np.zeros(num_neurons)

    def reset(self):
        self.synaptic_response = np.zeros(self.num_neurons)

    def __call__(self, spike_input):
        self.synaptic_response = self.synaptic_response * (1 - self.dt / self.decay_time) + spike_input / self.decay_time
        return self.synaptic_response


class ConductanceBasedLIF:
    def __init__(self, num_neurons, dt = 1e-4, refractory_period = 5e-3, 
                 membrane_time_constant = 1e-2, resting_potential = -60, reset_potential = -60, 
                 threshold_potential = -50, peak_potential = 20, exc_potential = 0, 
                 inh_potential = -100):
        """
        Conductance-based Leaky Integrate-and-Fire model.
        """
        self.num_neurons = num_neurons
        self.dt = dt
        self.refractory_period = refractory_period
        self.membrane_time_constant = membrane_time_constant
        self.resting_potential = resting_potential
        self.reset_potential = reset_potential
        self.threshold_potential = threshold_potential
        self.peak_potential = peak_potential
        self.exc_potential = exc_potential
        self.inh_potential = inh_potential
        
        self.membrane_potential = np.full(num_neurons, self.reset_potential)
        self.last_spike_time = 0
        self.time_counter = 0
    
    def reset_state(self, random_init: bool = False) -> None:
        if random_init:
            self.membrane_potential = self.reset_potential + np.random.rand(self.num_neurons) * (self.threshold_potential - self.reset_potential)
        else:
            self.membrane_potential.fill(self.reset_potential)
        self.last_spike_time = 0
        self.time_counter = 0
        
    def __call__(self, exc_conductance: np.ndarray, inh_conductance: np.ndarray) -> np.ndarray:
        exc_current = exc_conductance * (self.exc_potential - self.membrane_potential)
        inh_current = inh_conductance * (self.inh_potential - self.membrane_potential)
        
        voltage_change = (self.resting_potential - self.membrane_potential + exc_current + inh_current) / self.membrane_time_constant
        updated_potential = self.membrane_potential + ((self.dt * self.time_counter) > (self.last_spike_time + self.refractory_period)) * voltage_change * self.dt
        
        spikes = (updated_potential >= self.threshold_potential).astype(int)
        self.last_spike_time = self.last_spike_time * (1 - spikes) + self.dt * self.time_counter * spikes
        updated_potential = updated_potential * (1 - spikes) + self.peak_potential * spikes
        
        self.membrane_potential = updated_potential * (1 - spikes) + self.reset_potential * spikes
        self.time_counter += 1
        
        return spikes


class AdaptiveLIF:
    def __init__(self, num_neurons, dt = 1e-3, refractory_period = 5e-3, 
                 membrane_time_constant = 1e-1, resting_potential = -65, reset_potential = -65, 
                 initial_threshold = -52, peak_potential = 20, threshold_increment = 0.05, 
                 max_threshold = 35, threshold_time_constant = 1e4, exc_potential = 0, 
                 inh_potential = -100):
        """
        Adaptive Leaky Integrate-and-Fire model based on Diehl and Cook (2015).
        """
        self.num_neurons = num_neurons
        self.dt = dt
        self.refractory_period = refractory_period
        self.membrane_time_constant = membrane_time_constant
        self.resting_potential = resting_potential
        self.reset_potential = reset_potential
        self.initial_threshold = initial_threshold
        self.peak_potential = peak_potential
        self.threshold_increment = threshold_increment
        self.max_threshold = max_threshold
        self.threshold_time_constant = threshold_time_constant
        self.exc_potential = exc_potential
        self.inh_potential = inh_potential
        
        self.membrane_potential = np.full(num_neurons, self.reset_potential)
        self.adaptive_threshold_theta = np.zeros(num_neurons)
        self.dynamic_threshold = self.initial_threshold
        self.last_spike_time = 0
        self.time_counter = 0
        
    def reset_state(self, random_init = False):
        if random_init:
            self.membrane_potential = self.reset_potential + np.random.rand(self.num_neurons) * (self.dynamic_threshold - self.reset_potential)
        else:
            self.membrane_potential.fill(self.reset_potential)
        self.dynamic_threshold = self.initial_threshold
        self.adaptive_threshold_theta.fill(0)
        self.last_spike_time = 0
        self.time_counter = 0
        
    def __call__(self, exc_conductance, inh_conductance):
        exc_current = exc_conductance * (self.exc_potential - self.membrane_potential)
        inh_current = inh_conductance * (self.inh_potential - self.membrane_potential)
        
        voltage_change = (self.resting_potential - self.membrane_potential + exc_current + inh_current) / self.membrane_time_constant
        updated_potential = self.membrane_potential + ((self.dt * self.time_counter) > (self.last_spike_time + self.refractory_period)) * voltage_change * self.dt
        
        spikes = (updated_potential >= self.dynamic_threshold).astype(int)
        self.adaptive_threshold_theta = (1 - self.dt / self.threshold_time_constant) * self.adaptive_threshold_theta + self.threshold_increment * spikes
        self.adaptive_threshold_theta = np.clip(self.adaptive_threshold_theta, 0, self.max_threshold)
        self.dynamic_threshold = self.adaptive_threshold_theta + self.initial_threshold
        
        self.last_spike_time = self.last_spike_time * (1 - spikes) + self.dt * self.time_counter * spikes
        updated_potential = updated_potential * (1 - spikes) + self.peak_potential * spikes
        
        self.membrane_potential = updated_potential * (1 - spikes) + self.reset_potential * spikes
        self.time_counter += 1
        
        return spikes

def encode_dataset_spikes(dataset, sample_index, dt, num_dts, max_firing_rate=32, normalization_factor=140):
    """
    Encode a dataset sample into spike trains using Poisson encoding.
    
    Args:
        dataset (array-like): Dataset containing input samples.
        sample_index (int): Index of the sample to encode.
        dt (float): Simulation time step.
        num_dts (int): Number of time steps.
        max_firing_rate (float): Maximum firing rate.
        normalization_factor (float): Normalization constant.
    
    Returns:
        np.ndarray: Spike train representation of the input sample.
    """
    firing_rate = max_firing_rate * normalization_factor / np.sum(dataset[sample_index][0])
    firing_rate_matrix = firing_rate * np.repeat(np.expand_dims(dataset[sample_index][0], axis=0), num_dts, axis=0)
    spikes = np.where(np.random.rand(num_dts, 784) < firing_rate_matrix * dt, 1, 0)
    return spikes.astype(np.uint8)

def assign_neuron_labels(spike_activity, true_labels, num_classes, previous_rates=None, decay_factor=1.0):
    """
    Assign labels to neurons based on their highest average spiking activity.
    
    Args:
        spike_activity (np.ndarray): Spiking activity of neurons (n_samples, n_neurons).
        true_labels (np.ndarray): Ground truth labels (n_samples,).
        num_classes (int): Number of unique labels.
        previous_rates (np.ndarray, optional): Previous spike rates for incremental updates.
        decay_factor (float): Decay rate for label assignment.
    
    Returns:
        tuple: Neuron label assignments, per-class spike proportions, per-class firing rates.
    """
    num_neurons = spike_activity.shape[1] 
    
    if previous_rates is None:        
        previous_rates = np.zeros((num_neurons, num_classes), dtype=np.float32)
    
    for label in range(num_classes):
        num_samples_with_label = np.sum(true_labels == label).astype(np.int16)
    
        if num_samples_with_label > 0:
            indices = np.where(true_labels == label)[0]
            previous_rates[:, label] = decay_factor * previous_rates[:, label] + (np.sum(spike_activity[indices], axis=0) / num_samples_with_label)
    
    total_rates = np.sum(previous_rates, axis=1)
    total_rates[total_rates == 0] = 1  # Avoid division by zero
    
    proportions = previous_rates / np.expand_dims(total_rates, 1)  # (n_neurons, n_classes)
    proportions[np.isnan(proportions)] = 0  # Replace NaNs with 0
    
    neuron_assignments = np.argmax(proportions, axis=1).astype(np.uint8)  # (n_neurons,)
    
    return neuron_assignments, proportions, previous_rates

def predict_labels(spike_activity, neuron_assignments, num_classes):
    """
    Predict labels for input samples based on highest spiking activity.
    
    Args:
        spike_activity (np.ndarray): Spiking activity of neurons (n_samples, n_neurons).
        neuron_assignments (np.ndarray): Neuron label assignments (n_neurons,).
        num_classes (int): Number of unique labels.
    
    Returns:
        np.ndarray: Predicted labels (n_samples,).
    """
    num_samples = spike_activity.shape[0]
    firing_rates = np.zeros((num_samples, num_classes), dtype=np.float32)
    
    for label in range(num_classes):
        num_neurons_assigned = np.sum(neuron_assignments == label).astype(np.uint8)
    
        if num_neurons_assigned > 0:
            neuron_indices = np.where(neuron_assignments == label)[0]
            firing_rates[:, label] = np.sum(spike_activity[:, neuron_indices], axis=1) / num_neurons_assigned
    
    return np.argmax(firing_rates, axis=1).astype(np.uint8)