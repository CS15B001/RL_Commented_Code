import threading

import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_shapes, size_in_transitions, T, sample_transitions):
        """Creates a replay buffer.

        Args:
            buffer_shapes (dict of ints): the shape for all buffers that are used in the replay
                buffer
            size_in_transitions (int): the size of the buffer, measured in transitions
            T (int): the time horizon for episodes
            sample_transitions (function): a function that samples from the replay buffer
        """
        self.buffer_shapes = buffer_shapes
        # Integer division
        self.size = size_in_transitions // T
        # T represents the number of timesteps in each episode
        self.T = T
        self.sample_transitions = sample_transitions

        # self.buffers is {key: array(size_in_episodes x T or T+1 x dim_key)} - OpenAI comment
        # The keys are 'o', 'g', 'ag' etc. dim_key represents the dimensionality of the 
        # vector corresponding to 'o', 'g' etc. * (starred expression) is used to unpack
        # the elements of the list

        # self.buffers now contains static memory to store all the transitions
        self.buffers = {key: np.empty([self.size, *shape])
                        for key, shape in buffer_shapes.items()}

        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0

        self.lock = threading.Lock()

    @property
    def full(self):
        with self.lock:
            return self.current_size == self.size

    def sample(self, batch_size):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """
        buffers = {}

        with self.lock:
            assert self.current_size > 0
            for key in self.buffers.keys():
                buffers[key] = self.buffers[key][:self.current_size]

        # Doubt: 1: represents Time step 1 onwards?
        buffers['o_2'] = buffers['o'][:, 1:, :]
        buffers['ag_2'] = buffers['ag'][:, 1:, :]

        transitions = self.sample_transitions(buffers, batch_size)

        for key in (['r', 'o_2', 'ag_2'] + list(self.buffers.keys())):
            assert key in transitions, "key %s missing from transitions" % key

        return transitions

    def store_episode(self, episode_batch):
        """episode_batch: array(batch_size x (T or T+1) x dim_key)
        """
        batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]

        # Assert that sizes of all the keys are the same, i,e., a, g and o should
        # all have equal number of entries
        assert np.all(np.array(batch_sizes) == batch_sizes[0])
        batch_size = batch_sizes[0]

        with self.lock:
            idxs = self._get_storage_idx(batch_size)

            # load inputs into buffers
            # The keys are 'o', 'a', 'ag' etc. self.buffers[key] is a numpy array and
            # idxs are a list of numbers. Thus only a few of the experiences will be replaced
            for key in self.buffers.keys():
                self.buffers[key][idxs] = episode_batch[key]

            # Total number of transitions includes each time step
            self.n_transitions_stored += batch_size * self.T

    # Size in terms of number of episodes stored in the buffer
    def get_current_episode_size(self):
        with self.lock:
            return self.current_size

    # T represents the number of time steps in each episode
    # This represents the total number of time steps of
    # interation with the environment. (Previous function)*T
    def get_current_size(self):
        with self.lock:
            return self.current_size * self.T

    # The number of transitions stored in the buffer
    def get_transitions_stored(self):
        with self.lock:
            return self.n_transitions_stored

    # Empty the buffer
    def clear_buffer(self):
        with self.lock:
            self.current_size = 0

    # Get the indices to store the episodes
    # To remind, buffer is characterized by ['key_of_interest', episode_index, time_inside_episode, vector_of_interest]
    # If the replay buffer is full, randomly remove episodes till there is space
    def _get_storage_idx(self, inc=None):
        inc = inc or 1   # size increment
        assert inc <= self.size, "Batch committed to replay is too large!"
        # go consecutively until you hit the end, and then go randomly.
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        # If there isn't enough space, evict random episodes
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)

        # update replay size
        self.current_size = min(self.size, self.current_size+inc)

        # If it is only one episode, return the index, else a list
        if inc == 1:
            idx = idx[0]
        return idx