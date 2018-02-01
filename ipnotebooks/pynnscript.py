
import os
import urllib
import zipfile

download_data = True
remove_data = False
directory = os.getcwd() + '/data/'

zipfn = 'data.zip'

if download_data:
    if not os.path.exists(directory):
        os.makedirs(directory)

    url = 'http://rpg.ifi.uzh.ch/datasets/davis/office_zigzag.zip'

    urllib.urlretrieve(url, zipfn)
    with zipfile.ZipFile(os.getcwd() + '/' + zipfn, "r") as z:
        z.extractall(directory)

import numpy as np
fname = directory + 'events.txt'

print "Reading Data: {}".format(fname)

event_data = np.genfromtxt(fname,delimiter=' ')

print "Finished Reading Data"

if remove_data:
    import shutil
    shutil.rmtree(directory)
    os.remove(zipfn)
#input_rect (min_x,max_x,min_y,max_y)
input_rect = (0,9,0,9)

#get data within input_rect
x_rows = np.where(np.logical_and(event_data[:,1] >= input_rect[0], event_data[:,1] <= input_rect[1]))
y_rows = np.where(np.logical_and(event_data[:,2] >= input_rect[2], event_data[:,2] <= input_rect[3]))
rows = np.intersect1d(x_rows,y_rows)
event_data = event_data[rows]

#remove on events
rows = np.where(event_data[:,3] == 0)
event_data = event_data[rows]

n_neurons = (input_rect[1] - input_rect[0] + 1) * (input_rect[3] - input_rect[2] + 1)

x = event_data[:,1]
y = event_data[:,2]

print 'n_neurons {} max event x {} y {} min event x {} y {} '.format(n_neurons, max(x), max(y), min(x), min(y))
spike_times = [[] for i in range(n_neurons)]

for i in range(event_data.shape[0]):
    time = int(event_data[i,0] * 1000)
    neuron_x = event_data[i,1] - input_rect[0]
    neuron_y = event_data[i,2] - input_rect[2]
    row_length = input_rect[1] - input_rect[0]
    neuron_id = int( row_length * neuron_y + neuron_x)

    if(neuron_id < 0):
        print "Neuron Id is too Low"
    elif (neuron_id >= n_neurons):
        print "Neuron Id is too High"
    try:
        spike_times[neuron_id].append(time)
    except Exception as e:
        print "n_id {} spike_times len {} n_x {} n_y {} row_length {}"(neuron_id, len(spike_times), neuron_x, neuron_y, row_length)
        raise e
#import pyNN.spiNNaker as p
import spynnaker7.pyNN as p
#import spynnaker8.pyNN as p
import time
from threading import Condition

# initial call to set up the front end (pynn requirement)
p.setup(timestep=1.0, min_delay=1.0, max_delay=144.0)
# neurons per population and the length of runtime in ms for the simulation,
# as well as the expected weight each spike will contain

n_pops = 3

input_runtime = np.max(event_data[:,0]) * 1000
extra_time = 5000
run_time = input_runtime + extra_time

weight_to_spike = 0.5
# neural parameters of the ifcur model used to respond to injected spikes.
# (cell params for a synfire chain)
cell_params_lif = {'cm': 0.2,
                   'i_offset': 0.0,
                   'tau_m': 20.0,
                   'tau_refrac': 5.0,
                   'tau_syn_E': 5.0,
                   'tau_syn_I': 10.0,
                   'v_reset': -60.0,
                   'v_rest': -60.0,
                   'v_thresh': -50.0
                   }

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
#timing_rule = p.SpikePairRule(tau_plus=20.0, tau_minus=20.0, A_plus=0.5, A_minus=0.5)
timing_rule = p.SpikePairRule(tau_plus=20.0, tau_minus=20.0)
#weight_rule = p.AdditiveWeightDependence(w_max=5.0, w_min=0.0)
weight_rule = p.AdditiveWeightDependence(w_max=5.0, w_min=0.0, A_plus=0.5, A_minus=0.5)

#stdp_model = p.STDPMechanism(timing_dependence=timing_rule, weight_dependence=weight_rule, weight=0.0, delay=5.0)
stdp_model = p.SynapseDynamics(slow=p.STDPMechanism(timing_dependence=timing_rule, weight_dependence=weight_rule))
injector = p.Population(n_neurons, p.SpikeSourceArray, {'spike_times': spike_times}, label='spike_injector')

pops = []
projs = []
for i in range(n_pops):
    # create populations (if cur exp)
    pops.append(p.Population(n_neurons, p.IF_curr_exp, cell_params_lif, label='pop_{}'.format(i)))

    # Create a connection from the injector into the populations
    #p.Projection(injector, pops[i], p.OneToOneConnector(weights=weight_to_spike), synapse_type=stdp_model)
    projs.append(p.Projection(injector, pops[i], p.OneToOneConnector(weights=weight_to_spike), synapse_dynamics=stdp_model))

    # record output for this population
    pops[i].record()

    # Activate the sending of live spikes
    p.external_devices.activate_live_output_for(pops[i], database_notify_host="localhost", database_notify_port_num=19996)
# Create a connection from the injector into the populations

for i in range(n_pops):
    for j in range(n_pops):
        if(i != j):
            #p.Projection(pops[i], pops[j], p.OneToOneConnector(weights=weight_to_spike), synapse_type=p.StaticSynapse(weight=-0.75,delay=1.0))# Create a condition to avoid overlapping prints
            p.Projection(pops[i], pops[j], p.OneToOneConnector(weights=weight_to_spike))
# Create a condition to avoid overlapping prints
print_condition = Condition()
# Create an initialisation method
def init_pop(label, n_neurons, run_time_ms, machine_timestep_ms):
    print "{} has {} neurons".format(label, n_neurons)
    print "Simulation will run for {}ms at {}ms timesteps".format(run_time_ms, machine_timestep_ms)
 
# Create a receiver of live spikes
def receive_spikes(label, time, neuron_ids):
    for neuron_id in neuron_ids:
        print_condition.acquire()
        print "Received spike at time", time, "from", label, "-", neuron_id
        print_condition.release()
        # Set up the live connection for sending spikes
        
# Set up the live connection for sending spikes
live_spikes_connection_send = \
    p.external_devices.SpynnakerLiveSpikesConnection(receive_labels=None, local_port=19999,send_labels=["spike_injector"])

# Set up callbacks to occur at initialisation
live_spikes_connection_send.add_init_callback("spike_injector", init_pop)
# if not using the c visualiser, then a new spynnaker live spikes
# connection is created to define that there is a python function which
# receives the spikes.

pop_labels = ['pop_{}'.format(i) for i in range(n_pops)]

live_spikes_connection_receive = \
    p.external_devices.SpynnakerLiveSpikesConnection(receive_labels=pop_labels,local_port=19996, send_labels=None)

# Set up callbacks to occur when spikes are received
for i in range(n_pops):
    live_spikes_connection_receive.add_receive_callback(pop_labels[i], receive_spikes)
# Run the simulation on spiNNaker
p.run(run_time)
# Retrieve spikes from the synfire chain population
spikes = [pops[i].getSpikes() for i in range(n_pops)]

results_dir = os.getcwd() + '/results/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# If there are spikes, plot using matplotlib
number_of_spikes = sum([len(pops[i]) for i in range(n_pops)])
if number_of_spikes != 0:
    plt.figure(figsize=(20,10))
    f, axarr = plt.subplots(n_pops, sharex=True)
    
    colourarr = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for i in range(n_pops):
        axarr[i].set_title('Spikes Pop ' + str(i))
        if len(spikes[i]) != 0:
            axarr[i].plot([j[1] for j in spikes[i]],
                    [j[0] for j in spikes[i]],'o', ms=0.1, c=colourarr[i % len(colourarr)])
        axarr[i].set_ylabel('Neuron id')
        axarr[i].set_xlabel('Time/ms')

    import datetime
    f.savefig(results_dir + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '.pdf')
else:
    print "No spikes received"

with open(results_dir + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '.csv', 'w') as f:
    for i in range(n_pops):
        f.write('Neuron Id,Time/ms' + '\n')
        f.write('Population ' + str(i) + '\n')
        for row in spikes[i]:
        	line = str(row[1]) + ',' + str(row[0])
        	f.write(line + '\n')
weights = []
#delays = []
for i in range(n_pops):
    weights.append(projs[i].getWeights())
    #delays.append(projs[i].getDelays())

plt.figure(figsize=(20,10))
f, axarr = plt.subplots(n_pops, sharex=True)


for i in range(n_pops):
    axarr[i].set_title('Weights for Pop ' + str(i))
    np_weights = np.array(weights[i]).reshape(input_rect[1] - input_rect[0] + 1, input_rect[3] - input_rect[2] + 1)
    axarr[i].imshow(np_weights, cmap='hot', interpolation='nearest')
f.savefig(results_dir + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' weights' + '.pdf')

with open(results_dir + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' weights' + '.csv', 'w') as f:
    for i in range(n_pops):
        f.write('Population ' + str(i) + '\n')
        for row in weights[i]:
		line = str(row)
		f.write(line + '\n')

# Clear data structures on spiNNaker to leave the machine in a clean state for
# future executions
p.end()
