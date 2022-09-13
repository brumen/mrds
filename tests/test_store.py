""" Testing the stored objects to the database.
"""
import sys
if '/home/brumen/work' not in sys.path:
    sys.path.append('/home/brumen/work')

from mrds.store_dec import Store, stored


class TestStore2(Store):

    def __init__(self, a, b):
        self.a = a
        self.b = b

    @stored()
    def k1(self):
        return self.a + 1

    @stored()
    def k2(self):
        return self.b **2

    def k3(self):
        return 222


# TODO: HOW TO LOAD, WHAT TO DO W/ PROPERTIES...
#ts = TestStore2(1,2)
#ts.write()

k2 = TestStore2.load('object3', 10, 10)
print(k2.k2())
print(k2.k1())
