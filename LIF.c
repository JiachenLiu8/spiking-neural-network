#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Implementation of a Leaky Integrate-and-Fire (LIF) neuron model.

#define TOTAL_TIME 100 // Total time in ms
#define DT 0.1         // Timestep in ms

// Function to simulate the LIF neuron
void simulateLIF(double tau, double inputTimes[], int numSpikes, double weight, double threshold, double reset) {
    int steps = (int)(TOTAL_TIME / DT);
    double alpha = exp(-DT / tau);
    double V = 0.0; // Initial membrane potential

    int spikeIdx = 0; // Index to track input spikes
    double nextSpikeTime = inputTimes[spikeIdx];

    for (int i = 0; i < steps; i++) {
        double t = i * DT;

        V *= alpha; // Decay the current value of V

        if (spikeIdx < numSpikes && t > nextSpikeTime) {
            V += weight; // Add weight if there's an input spike
            spikeIdx++;
            if (spikeIdx < numSpikes) {
                nextSpikeTime = inputTimes[spikeIdx];
            }
        }

        if (V > threshold) {
            printf("Spike at time %f ms\n", t);
            V = reset; // Reset V if threshold is crossed
        }
    }
}

int main() {
    double tau = 10; // Time constant in ms
    double spikes[] = {20, 40, 60}; // Spike times in ms
    double weight = 0.1; // Input synapse weight
    double threshold = 1.0; // Threshold to produce a spike
    double reset = 0.0; // Reset value after a spike

    // Make sure to sort the input spike times if they are not in chronological order
    simulateLIF(tau, spikes, sizeof(spikes) / sizeof(spikes[0]), weight, threshold, reset);
    return 0;
}