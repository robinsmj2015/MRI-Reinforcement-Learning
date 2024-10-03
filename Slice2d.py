# MRI slice
class Slice2d:
    def __init__(self, name, box):
        self.name = name
        self.box = box
        self.proposed_start = None  # where box starts next time in scaling mode
