#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Use spiking neural networks to solve a sound localisation task using coincidence detection.

#define PI 3.14159265358979323846
#define MAX_NEURONS 100
#define SIMULATION_TIME 1000  // 1000 ms or 1 second of simulation
#define DT 1  // time step in ms

typedef struct {
    double rate; // Firing rate
    double ipd; // Interaural phase difference
    double itd; // Interaural time difference
    double threshold;
    int spike_count;
} Neuron;

// Generate a sinusoidal input with phase difference
void generate_input(Neuron *neurons, double rate_max, double ipd, double freq, int n) {
    double theta, signal_on;
    for (int i = 0; i < n; i++) {
        theta = 2 * PI * freq * i * DT / 1000 + (i % 2) * ipd;  // Modulate input by ear
        signal_on = (sin(theta) + 1) / 2;  // Normalized from -1 to 1 to 0 to 1
        neurons[i].rate = rate_max * signal_on;
    }
}

// Run the simulation
void simulate(Neuron *neurons, int n) {
    double rand_val;
    for (int t = 0; t < SIMULATION_TIME; t += DT) {
        for (int i = 0; i < n; i++) {
            rand_val = (double)rand() / RAND_MAX;
            if (rand_val < neurons[i].rate * DT / 1000) {
                neurons[i].spike_count++;
                printf("Neuron %d spiked at time %d ms\n", i, t);
            }
        }
    }
}

int main() {
    srand(time(NULL)); // Seed the random number generator
    Neuron neurons[MAX_NEURONS];
    double rate_max = 100.0; // Max rate in Hz
    double ipd = PI / 2; // 90 degrees phase difference
    double freq = 3.0; // Frequency of the sine wave

    // Initialize neurons
    for (int i = 0; i < MAX_NEURONS; i++) {
        neurons[i].rate = 0;
        neurons[i].spike_count = 0;
        neurons[i].ipd = ipd;
        neurons[i].itd = ipd / (2 * PI * freq); // Calculate ITD based on IPD and frequency
        neurons[i].threshold = 0.5; // Example threshold
    }

    generate_input(neurons, rate_max, ipd, freq, MAX_NEURONS);
    simulate(neurons, MAX_NEURONS);

    return 0;
}
