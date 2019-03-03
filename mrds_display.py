#
# Display class for the Commodity skew model.
#

import matplotlib as mpl
mpl.use('TkAgg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk


from mrds import ComSkew


class ComSkewDisplay(ComSkew):
    """
    ComSkew model w/ display features for volatility.

    """

    def __refresh_model_vols(self
                             , asset
                             , fwd
                             , c
                             , a
                             , canvas
                             , li1
                             , li2):
        deltaLabels = self.deltas_to_strikes(asset, fwd)
        li1.set_xdata(deltaLabels)
        li1.set_ydata(self.model_vol_surface(asset, c, fwd))
        li2.set_xdata(deltaLabels)
        li2.set_ydata(self.volCurve(asset)[fwd])  # TODO: FIX THIS PART HERE

        canvas.draw()

    def disp_model_vols(self, asset, fwd):
        """
        Plotting the model vols as you click the button.

        """

        canvas = tk.Tk()  # main canvas
        # plot market vols as initial
        delta_x = self.deltas_to_strikes(asset, fwd)
        mv_y = self.vol_surface_list[asset][fwd]
        f = Figure(figsize=(5, 4), dpi=100)
        a = f.add_subplot(111)
        line1, = a.plot(delta_x, mv_y)
        line2, = a.plot(delta_x, mv_y)  # 2 are needed
        # plot the graph
        dataPlot_canvas = FigureCanvasTkAgg(f, master=canvas)
        dataPlot_canvas.get_tk_widget().grid(row=0, column=0, rowspan=2)

        fct_update = lambda cc: self.__refresh_model_vols(asset
                                                          , fwd
                                                          , [c1.get(), c2.get(), c3.get()]
                                                          , a
                                                          , dataPlot_canvas
                                                          , line1
                                                          , line2)

        c1 = tk.Scale(canvas, from_=-5., to=5., resolution=0.1, label='c0', command=fct_update)
        c2 = tk.Scale(canvas, from_=-25., to=25., resolution=0.2, label='c1', command=fct_update)
        c3 = tk.Scale(canvas, from_=-15., to=5., resolution=0.2, label='c2', command=fct_update)
        c1.grid(row=0, column=1)
        c2.grid(row=0, column=2)
        c3.grid(row=0, column=3)
        c1.set(self._CVecList[asset][fwd][0])
        c2.set(self._CVecList[asset][fwd][1])
        c3.set(self._CVecList[asset][fwd][2])
        dataPlot_canvas.show()
        canvas.mainloop()

    def disp_model_surf(self, asset, fwd):
        """
        Display the model surface.

        """

        root = tk.Tk()

        c1 = tk.Scale(root, from_=-2.0, to=2.0, resolution=0.1)
        c2 = tk.Scale(root, from_=-5.0, to=5.0, resolution=0.2)
        c3 = tk.Scale(root, from_=-5.0, to=5.0, resolution=0.2)

        c1.grid(row=0, column=1)
        c2.grid(row=0, column=2)
        c3.grid(row=0, column=3)

        # plot market vols as initial
        f = Figure(figsize=(5, 4), dpi=100)
        a = f.add_subplot(111)
        a.plot(self.deltas_to_strikes(asset, fwd), self.vol_surface_list[asset][fwd])

        # plot the graph
        dataPlot_canvas = FigureCanvasTkAgg(f, master=root)
        dataPlot_canvas.show()
        dataPlot_canvas.get_tk_widget().grid(row=0, column=0, rowspan=2)

        # replot button
        b1 = tk.Button(root
                       , text="replot"
                       , command=lambda: self.refresh_model_vols(asset
                                                                 , fwd
                                                                 , [c1.get(), c2.get(), c3.get()]
                                                                 , a
                                                                 , dataPlot_canvas)).grid(row=1, column=1, columnspan=3)
        root.mainloop()
