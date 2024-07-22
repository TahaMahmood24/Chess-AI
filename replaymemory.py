"""
Module for Replay Memory used in reinforcement learning.

This module provides the ReplayMemory class for storing and sampling transitions in a deque.

Classes:
    ReplayMemory: A class to manage the storage and sampling of transitions for reinforcement learning.

Usage Example:
    memory = ReplayMemory(maxlen=10000)
    memory.append((state, action, reward, next_state, done))
    batch = memory.sample(sample_size=32)
"""

import random
from collections import deque

class ReplayMemory:
    """
    A class to manage the storage and sampling of transitions for reinforcement learning.

    Attributes:
        memory (deque): A deque to store transitions with a fixed maximum length.

    Methods:
        append(transition): Add a transition to the memory.
        sample(sample_size): Sample a batch of transitions from the memory.
        __len__(): Get the current size of the memory.
    """

    def __init__(self, maxlen):
        """
        Initialize the ReplayMemory with a maximum length.

        Parameters:
            maxlen (int): The maximum number of transitions the memory can hold.
        """
        self.memory = deque([], maxlen=maxlen)
    
    def append(self, transition):
        """
        Add a transition to the memory.

        Parameters:
            transition (tuple): A tuple representing a transition (state, action, reward, next_state, done).
        """
        self.memory.append(transition)

    def sample(self, sample_size):
        """
        Sample a batch of transitions from the memory.

        Parameters:
            sample_size (int): The number of transitions to sample.

        Returns:
            list: A list of sampled transitions.
        """
        return random.sample(self.memory, sample_size)

    def __len__(self):
        """
        Get the current size of the memory.

        Returns:
            int: The number of transitions currently stored in the memory.
        """
        return len(self.memory)
