#
# display for the freight movement, TO BE WORKED MORE ONE -
# see test_freight.py
#

import logging

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from freight import Freight


logger = logging.getLogger(__name__)


class FreightDisplay(Freight):
    """ Display class of the freight model.
    """

    def __update_display_movement(self, time_step : int, ax, tanker_locations, freight_graph, freight_graph_locations):
        """ Updates the display for freight movement for every time_step in all the timesteps in the freight model.

        :param time_step: time step
        """

        logger.info("Times: {0}".format(self._time_grid[time_step]))

        ax.clear()
        fg, pos = freight_graph, freight_graph_locations

        cond_moves_active = [(self.__nbs_to_locations[i], self.__nbs_to_locations[j])
                             for i in range(self.__nb_locations)
                             for j in range(self.__nb_locations)
                             for u in range(time_step + 1, self.__nb_time_periods)
                             if self.__freight_hedge_x(i, j, time_step, u) != 0.]
        uncond_moves_active = [(self.__nbs_to_locations[i], self.__nbs_to_locations[j])
                               for i in range(self.__nb_locations)
                               for j in range(self.__nb_locations)
                               for u in range(time_step + 1, self.__nb_time_periods)
                               if self.__freight_hedge_y(i, j, time_step, u) != 0.]

        nx.draw_networkx_labels(fg, pos=pos, labels=dict(zip(tanker_locations, tanker_locations)), font_size=16)
        nx.draw_networkx_nodes(fg, pos=pos, ax = ax, node_color= 'black', node_size=50)
        nx.draw_networkx_edges(fg, pos=pos, edgelist = cond_moves_active  , ax=ax, edge_color="blue", arrows=True)
        nx.draw_networkx_edges(fg, pos=pos, edgelist = uncond_moves_active, ax=ax, edge_color="red")

        # Scale plot ax
        ax.set_title('Time period {0}.'.format(time_step))
        ax.set_xticks([])
        ax.set_yticks([])

    def display_movement(self):
        """ Displays the movement of tankers.
        """

        figure, ax = plt.subplots(figsize=(6, 4))

        # create the graph and fill it w/ nodes.
        freight_graph    = nx.Graph()
        [freight_graph.add_node(location) for location in self.__locations]  # add locations

        # animation
        ani = FuncAnimation( figure
                           , lambda time_step: self.__update_display_movement( time_step
                                                                             , ax
                                                                             , self.__locations
                                                                             , freight_graph
                                                                             , nx.spring_layout(freight_graph))
                           , frames    = self.__nb_time_periods
                           , init_func = None
                           , interval  = 1000
                           , repeat    = True )
        plt.show()
