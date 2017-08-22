import os


class Dir(object):
    @staticmethod
    def make(parent_path, sub_path):
        full_path = os.path.join(parent_path, sub_path)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        return full_path

class JobDir(Dir):
    def __init__(self, parent_path):
        self.parent_path = parent_path

    def join(self, sub_path):
        # type: (object) -> object
        return self.make(self.parent_path, sub_path)

    def join_path(self, parent_path, sub_path):
        return self.make(parent_path, sub_path)

    @property
    def checkpoint_path(self):
        return self.join("checkpoints")

    @property
    def log_dir(self):
        return self.join("logs")