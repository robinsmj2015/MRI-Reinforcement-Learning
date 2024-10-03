# bounding boxes
class Box:
    def __init__(self, tl, br):
        self.tl = tl  # top left
        self.br = br  # bottom right
        self.tr = [self.br[0], self.tl[1]]  # top right
        self.bl = [self.tl[0], self.br[1]]  # bottom left
        # sides of box...
        self.top_most = self.tl[1]
        self.left_most = self.tl[0]
        self.right_most = self.br[0]
        self.bottom_most = self.br[1]
        # box center
        self.center = [round(self.tl[0] + (self.br[0] - self.tl[0]) / 2),
                       round(self.tl[1] + (self.br[1] - self.tl[1]) / 2)]
