import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class PlotWindow:
    def __init__(self, title, figure, algorithm_name=""):
        self.figure = figure
        self.algorithm_name = algorithm_name
        self.grid_on = True
        self.legend_on = True

        self.win = tk.Toplevel()
        self.win.title(title)
        self.win.geometry("1000x750")

        menubar = tk.Menu(self.win)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Save as PNG", command=lambda: self.save("png"))
        file_menu.add_command(label="Save as JPG", command=lambda: self.save("jpg"))
        file_menu.add_command(label="Save as PDF", command=lambda: self.save("pdf"))
        file_menu.add_separator()
        file_menu.add_command(label="Close", command=self.win.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="Toggle Grid", command=self.toggle_grid)
        view_menu.add_command(label="Toggle Legend", command=self.toggle_legend)
        menubar.add_cascade(label="View", menu=view_menu)

        self.win.config(menu=menubar)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.win)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, self.win)
        toolbar.update()

        for child in toolbar.winfo_children():
            if isinstance(child, tk.Label):
                child.pack_forget()

    def toggle_grid(self):
        self.grid_on = not self.grid_on
        for ax in self.figure.axes:
            ax.grid(self.grid_on)
        self.canvas.draw()

    def toggle_legend(self):
        self.legend_on = not self.legend_on
        for ax in self.figure.axes:
            leg = ax.get_legend()
            if leg:
                leg.set_visible(self.legend_on)
        self.canvas.draw()

    def save(self, fmt):
        file = filedialog.asksaveasfilename(
            defaultextension=f".{fmt}",
            filetypes=[(fmt.upper(), f"*.{fmt}")]
        )
        if file:
            self.figure.savefig(file, dpi=300)