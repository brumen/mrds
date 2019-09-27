#
# display for the freight movement, TO BE WORKED MORE ONE -
# see test_freight.py
#

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from freight import Freight


class FreightDisplay(Freight):
    """ Display class of the freight model.
    """

    def __updateDisplayMovement(self, timeStep : int):
        """
        Updates the display for freight movement for every timeStep in all the timesteps in the freight model.

        """

        print ("Times:" + str(self._timeGrid[timeStep]))
        self.__ax.clear()
        fg, pos = self.__freightGraph, self.__freightGraphLayout  # abbreviations


        condMovesActive = [(self._nbsToLocations[i], self._nbsToLocations[j])
                           for i in range(self._nbLocations)
                           for j in range(self._nbLocations)
                           for u in range(timeStep+1, self._nbTimePeriods)
                           if self.freight_hedge_x(i, j, timeStep, u) != 0.]
        uncondMovesActive = [(self._nbsToLocations[i], self._nbsToLocations[j])
                             for i in range(self._nbLocations)
                             for j in range(self._nbLocations)
                             for u in range(timeStep+1, self._nbTimePeriods)
                             if self.freight_hedge_y(i, j, timeStep, u) != 0.]

        nx.draw_networkx_labels(fg, pos=pos, labels=dict(zip(self.__tankerLocations, self.__tankerLocations)), font_size=16)
        nx.draw_networkx_nodes(fg, pos=pos, ax = self.__ax, node_color= 'black', node_size=50)
        nx.draw_networkx_edges(fg, pos=pos, edgelist = condMovesActive  , ax=self.__ax, edge_color="blue", arrows=True)
        nx.draw_networkx_edges(fg, pos=pos, edgelist = uncondMovesActive, ax=self.__ax, edge_color="red")

        # Scale plot ax
        self.__ax.set_title('Time period {0}.'.format(timeStep))
        self.__ax.set_xticks([])
        self.__ax.set_yticks([])

    def display_movement(self):
        """ Displays the movement of tankers.
        """

        self.__fig, self.__ax  = plt.subplots(figsize=(6, 4))
        self.__tankerLocations = self._locations
        self.__freightGraph    = nx.Graph()
        [self.__freightGraph.add_node(location) for location in self._locations]  # add locations
        self.__freightGraphLayout = nx.spring_layout(self.__freightGraph)

        ani = FuncAnimation( self.__fig
                           , self.__updateDisplayMovement
                           , frames    = self._nbTimePeriods
                           , init_func = None
                           , interval  = 1000
                           , repeat    = True )
        plt.show()
