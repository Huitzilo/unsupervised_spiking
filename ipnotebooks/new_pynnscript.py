#coding: utf-8
import socket
import spynnaker8 as p
from pyNN.random import NumpyRNG, RandomDistribution
from pyNN.utility import Timer

import os
import numpy as np
import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
#mpl.use('Agg')

# === Reading AER Data ===
directory = os.getcwd() + '/' 
fname = directory + 'simulate_events_11x11.txt'

print "Reading Data: {}".format(fname)

event_data = np.genfromtxt(fname,delimiter=' ')

print "Finished Reading Data"


#input_rect (min_x,max_x,min_y,max_y)
input_rect = (0,10,0,10)

'''
#get data within input_rect
x_rows = np.where(np.logical_and(event_data[:,1] >= input_rect[0], event_data[:,1] <= input_rect[1]))
y_rows = np.where(np.logical_and(event_data[:,2] >= input_rect[2], event_data[:,2] <= input_rect[3]))
rows = np.intersect1d(x_rows,y_rows)
event_data = event_data[rows]
'''

#remove off events
rows = np.where(event_data[:,3] == 0)
event_data = event_data[rows]

x_width = input_rect[1] - input_rect[0] + 1
y_width = input_rect[3] - input_rect[2] + 1
n_inj = x_width * y_width
print 'Number of inj {}'.format(n_inj)

print 'Length of event_data in s: {}'.format(max(event_data[:,0]))
print 'n_nj {}'.format(n_inj)

spike_times = [[] for i in range(n_inj)]

for i in range(event_data.shape[0]):
    time = int(event_data[i,0] * 1000)
    neuron_x = event_data[i,1] - input_rect[0]
    neuron_y = event_data[i,2] - input_rect[2]
    row_length = input_rect[1] - input_rect[0] + 1
    neuron_id = int( row_length * neuron_y + neuron_x)

    if(neuron_id < 0):
        print "Neuron Id is too Low"
    elif (neuron_id >= n_inj):
        print "Neuron Id is too High"
    try:
        spike_times[neuron_id].append(time)
    except Exception as e:
        print "n_id {} spike_times len {} n_x {} n_y {} row_length {}"(neuron_id, len(spike_times), neuron_x, neuron_y, row_length)
        raise e


sconn =0.1 #10% Injector
#iicon = 0.1 #10% Inter Inhib

t_input_end = int(event_data[event_data.shape[0] - 1][0]) * 1000
t_extra_time = 10000

# === Define parameters ===

threads = 1
rngseed = 98766987
parallel_safe = True

n = 1500          # number of cells
r_ei = 4.0        # number of excitatory cells:number of inhibitory cells
pconn = 0.01 #changed to 10%      # connection probability
stim_dur = 50.    # (ms) duration of random stimulation
rate = 100.       # (Hz) frequency of the random stimulation

dt = 1.0          # (ms) simulation timestep
tstop = t_input_end + t_extra_time      # (ms) simulaton duration
delay = 2

# Cell parameters
area = 20000.     # (µm²)
tau_m = 20.       # (ms)
cm = 1.           # (µF/cm²)
g_leak = 5e-5     # (S/cm²)

E_leak = -49.  # (mV)
v_thresh = -50.   # (mV)
v_reset = -60.    # (mV)
t_refrac = 5.     # (ms) (clamped at v_reset)
v_mean = -60.     # (mV) mean membrane potential, for calculating CUBA weights
tau_exc = 5.      # (ms)
tau_inh = 10.     # (ms)

# Synapse parameters
Gexc = 0.27   # (nS)
Ginh = 4.5    # (nS)
Erev_exc = 0.     # (mV)
Erev_inh = -80.   # (mV)

# === Calculate derived parameters ===
area = area * 1e-8                     # convert to cm²
cm = cm * area * 1000                  # convert to nF
Rm = 1e-6 / (g_leak * area)            # membrane resistance in MΩ
assert tau_m == cm * Rm                # just to check

n_exc = int(round((n * r_ei / (1 + r_ei))))  # number of excitatory cells
n_inh = n - n_exc                            # number of inhibitory cells

print n_exc, n_inh

celltype = p.IF_curr_exp
w_exc = 1e-3 * Gexc * (Erev_exc - v_mean)  # (nA) weight of exc synapses
w_inh = 1e-3 * Ginh * (Erev_inh - v_mean)  # (nA)
assert w_exc > 0
assert w_inh < 0

# ===  STDP Synapse parameters

##################################
# Parameters for the injector population.  This is the minimal set of
# parameters required, which is for a set of spikes where the key is not
# important.  Note that a virtual key *will* be assigned to the population,
# and that spikes sent which do not match this virtual key will be dropped;
# however, if spikes are sent using 16-bit keys, they will automatically be
# made to match the virtual key.  The virtual key assigned can be obtained
# from the database.
##################################
cell_params_spike_injector = {
    # The port on which the spiNNaker machine should listen for packets.
    # Packets to be injected should be sent to this port on the spiNNaker
    # machine
    'port': 12345,
}


##################################
# Parameters for the injector population.  Note that each injector needs to
# be given a different port.  The virtual key is assigned here, rather than
# being allocated later.  As with the above, spikes injected need to match
# this key, and this will be done automatically with 16-bit keys.
##################################
cell_params_spike_injector_with_key = {

    # The port on which the spiNNaker machine should listen for packets.
    # Packets to be injected should be sent to this port on the spiNNaker
    # machine
    'port': 12346,

    # This is the base key to be used for the injection, which is used to
    # allow the keys to be routed around the spiNNaker machine.  This
    # assignment means that 32-bit keys must have the high-order 16-bit
    # set to 0x7; This will automatically be prepended to 16-bit keys.
    'virtual_key': 0x70000,
}

# === Build the network ===

benchmark = 'blah'

extra = {'threads': threads,
         'filename': "va_%s.xml" % benchmark,
         'label': 'VA'}

node_id = p.setup(
    timestep=dt, min_delay=delay, max_delay=delay,
    db_name='va_benchmark.sqlite', **extra)

p.set_number_of_neurons_per_core(p.IF_curr_exp, 100)      # this will set
#  100 neurons per core
#np = 1

host_name = socket.gethostname()
print "Host #%d is on %s" % (1, host_name)

print "%s Initialising the simulator with %d thread(s)..." % (
    node_id, extra['threads'])

cell_params = {'tau_m': tau_m,
               'tau_syn_E': tau_exc,
               'tau_syn_I': tau_inh,
               'v_rest': E_leak,
               'v_reset': v_reset,
               'v_thresh': v_thresh,
               'cm': cm,
               'tau_refrac': t_refrac,
               'i_offset': 0
               }

print cell_params

print "%s Creating cell populations..." % node_id
exc_cells = p.Population(
    n_exc, celltype(**cell_params), label="Excitatory_Cells")
inh_cells = p.Population(
    n_inh, celltype(**cell_params), label="Inhibitory_Cells")

timing_rule = p.SpikePairRule(tau_plus=20.0, tau_minus=20.0, A_plus=0.5, A_minus=0.5)
weight_rule = p.AdditiveWeightDependence(w_max=2.0, w_min=0.0)
stdp_model = p.STDPMechanism(timing_dependence=timing_rule, weight_dependence=weight_rule, weight=0.0, delay=5.0)


pops = []
inj_cells = p.Population(n_inj, p.SpikeSourceArray, {'spike_times': spike_times}, label='spike_injector')

print "%s Initialising membrane potential to random values..." % node_id
rng = NumpyRNG(seed=rngseed, parallel_safe=parallel_safe)
uniformDistr = RandomDistribution('uniform', [v_reset, v_thresh], rng=rng)
exc_cells.set(v=uniformDistr)
inh_cells.set(v=uniformDistr)

pops.append(exc_cells)
pops.append(inh_cells)
pops.append(inj_cells)

#inj_cells.set(v=uniformDistr)

print "%s Connecting populations..." % node_id
exc_conn = p.FixedProbabilityConnector(pconn, rng=rng)
inh_conn = p.FixedProbabilityConnector(pconn, rng=rng)

#inter_inh_con = p.FixedProbabilityConnector(iiconn, rng=rng)
#inj_conn = p.FixedProbabilityConnector(sconn, rng=rng)

inj_exc_cons = int(sconn * n_exc) 
inj_exc = []
for inj_ind in range(n_inj):
    exc_list = random.sample(range(0, n_exc - 1), inj_exc_cons)
    for exc_ind in exc_list:
        inj_exc.append((inj_ind, exc_ind))
inj_conn = p.FromListConnector(inj_exc)

connections = {
    'e2e': p.Projection(
        exc_cells, exc_cells, exc_conn, receptor_type='excitatory',
        synapse_type=p.StaticSynapse(weight=w_exc, delay=delay)),
    'e2i': p.Projection(
        exc_cells, inh_cells, exc_conn, receptor_type='excitatory',
        synapse_type=p.StaticSynapse(weight=w_exc, delay=delay)),
    'i2e': p.Projection(
        inh_cells, exc_cells, inh_conn, receptor_type='inhibitory',
        synapse_type=p.StaticSynapse(weight=w_inh, delay=delay)),
    'i2i': p.Projection(
        inh_cells, inh_cells, inh_conn, receptor_type='inhibitory',
        synapse_type=p.StaticSynapse(weight=w_inh, delay=delay)),
    's2e': p.Projection(
        inj_cells, exc_cells, inj_conn, receptor_type='excitatory',
        synapse_type=stdp_model)}

'''
's2e': p.Projection(
        inj_cells, exc_cells, inj_conn, receptor_type='excitatory',
        synapse_type=stdp_model)
'''
# Set up the live connection for sending spikes
live_spikes_connection_send = \
    p.external_devices.SpynnakerLiveSpikesConnection(receive_labels=None, local_port=19999,send_labels=["spike_injector"])


# === Setup recording ===
print "%s Setting up recording..." % node_id
#exc_cells.record("spikes")
for pop in pops:
    pop.record("spikes")
# === Run simulation ===
print "%d Running simulation..." % node_id


p.run(tstop)


# === Print results to file ===
results_dir = os.getcwd() + '/results/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

pop_labels = ['exc', 'inh', 'inj']

for i in range(len(pops)):
    plt.figure()
    f, ax = plt.subplots()

    pop_spikes = pops[i].get_data("spikes")
    spiketrains = pop_spikes.segments[0].spiketrains

    neurons = np.concatenate(map(lambda x:
                                    np.repeat(x.annotations['source_index'],
                                                               len(x)),
                                                   spiketrains))
    spike_times = np.concatenate(spiketrains, axis=0)
    ax.set_title('Spikes for Pop ' + pop_labels[i])  
    ax.plot(spike_times, neurons, 'o', ms=0.1)
    ax.set_ylabel('Neuron id')
    ax.set_xlabel('Time/ms')

    plt.savefig(results_dir +
              datetime.datetime.now().strftime("%Y-%m-%d%H:%M:%S") + ' ' +
              pop_labels[i] + '.png')

    with open(results_dir + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") +
              ' ' + pop_labels[i] + '.csv', 'w') as f:
        f.write('Neuron Id,Time/ms' + '\n')
        for i in range(neurons.shape[0]):
            line = str(spike_times[i]) + ',' + str(neurons[i])
            f.write(line + '\n')


shape = (input_rect[1] - input_rect[0] + 1, input_rect[3] - input_rect[2] + 1)

s2e_weights = connections.get('s2e').getWeights()
with open(results_dir + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") +
          ' weights' + '.txt', 'w') as f:
    for w in s2e_weights:
        f.write(str(w) + '\n')

np_weights = np.array(s2e_weights)
np_inj_exc = np.array(inj_exc)
weights_with_input = []
for i in range(shape[0] * shape[1]):
    indices = np.where(np_inj_exc[:,0] == i)
    weights = np_weights[indices]
    avg = np.average(weights)
    weights_with_input.append(avg)

np_weights = np.array(weights_with_input).reshape(shape)

plt.figure()
plt.imshow(np_weights, cmap='Reds', interpolation='nearest')
plt.title('')
plt.ylabel('Y')
plt.xlabel('X')
plt.colorbar()
plt.savefig(results_dir + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' weights' + '.png')



'''
if node_id == 0:
    print "\n--- Vogels-Abbott Network Simulation ---"
    print "Nodes                  : %d" % np
    print "Number of Neurons      : %d" % n
    print "Number of Synapses     : %s" % connections
    print "Excitatory conductance : %g nS" % Gexc
    print "Inhibitory conductance : %g nS" % Ginh

'''
# === Finished with simulator ===

p.end()
