__author__ = "nasim.rahaman (at) iwr.uni-heidelberg.de"

import random


class ExperienceDatabase(object):
    """A simple list like class with a few bells and whistles to implement experience replay for Deep Q-Learning."""
    def __init__(self, maxsize=None):
        """
        :type maxsize: int
        :param maxsize: Maximum size of the database.
        """
        # Meta
        self.maxsize = maxsize
        # Database, implemented as a simple list
        self.db = []

    def __len__(self):
        return len(self.db)

    def sample(self, idx=None):
        """Sample experience from database."""
        # Get index of the experience to sample
        if idx is None:
            idx = random.randint(0, len(self) - 1)
        # Sample
        experience = self.db[idx]
        # Return
        return experience

    def record(self, *experience):
        """Record an experience to the database."""
        # Pop if full.
        if len(self) > self.maxsize:
            self.db.pop(0)
        # Record experience
        self.db.append(experience)

    def save(self):
        """Saves a copy of itself to disk."""
        raise NotImplementedError
