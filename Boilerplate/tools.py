__author__ = "nasim.rahaman (at) iwr.uni-heidelberg.de"


class ExperienceDatabase(object):
    """A simple list like class with a few bells and whistles to implement experience replay for Deep Q-Learning."""
    def __init__(self):
        pass

    def sample(self):
        """Sample experience from database."""
        pass

    def record(self, *experience):
        """Record an experience to the database."""
        # Remember to pop if necessarily.
        pass

    def save(self):
        """Saves a copy of itself to disk."""
        pass
