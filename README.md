# Unsupervised learning with spikes on neuromorphic hardware

A small demo that showcases unsupervised learning with spiking networks on neuromorphic hardware.

A very rough outline:

1. The network consists of 784 excitatory and 157 inhibitory neurons. There is STDP between excitatory neurons, so that those which fire together will wire together. We will also need inhibitory plasticity to balance the excitatory plasticity. Connection probability still needs to be figured out.
3. The control script shall take an MNIST digit, encode it into 784 poisson spike trains, and expose the network to it.
4. Through STDP the network shall "learn" the digit in an associative way. 