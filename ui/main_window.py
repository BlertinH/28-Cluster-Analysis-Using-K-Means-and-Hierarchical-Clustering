import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import random
import os

from core.dataloader import load_data
from core.kmeans import KMeans
from core.hierarchical import hierarchical_clustering

from visualization.plot_clusters import plot_clusters
from visualization.plot_voronoi import plot_voronoi
from visualization.popup import PlotWindow


class ClusteringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Clustering Tool")
        self.root.geometry("600x700")

        self.colors = {
            "bg": "#f5f5f5",
            "fg": "#333",
            "accent": "#4a6fa5",
            "secondary": "#d9d9d9",
            "card_bg": "white",
            "entry_bg": "white",
            "border": "#cccccc"
        }

        self.root.configure(bg=self.colors["bg"])
        self.file_path = "sampledata.csv"

        self.create_widgets()

    def create_section(self, parent, title, pady_top):
        tk.Label(parent, text=title, bg=self.colors["bg"],
                 fg=self.colors["accent"], font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(pady_top, 10))
        tk.Frame(parent, height=1, bg=self.colors["border"]).pack(fill=tk.X, pady=(0, 10))


    def create_widgets(self):
        main = tk.Frame(self.root, bg=self.colors["bg"])
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        tk.Label(main, text="Clustering Analysis", font=('Arial', 18, 'bold'),
                 bg=self.colors["bg"], fg=self.colors["accent"]).pack(pady=(0, 20))

        self.create_section(main, "Algorithm Settings", 0)

        tk.Label(main, text="Algorithm:", bg=self.colors["bg"], fg=self.colors["fg"]).pack(anchor=tk.W)
        self.method = ttk.Combobox(main, values=["K-Means", "Hierarchical"], state="readonly", width=25)
        self.method.current(0)
        self.method.pack(anchor=tk.W, pady=(0, 15))

        tk.Label(main, text="Clusters:", bg=self.colors["bg"], fg=self.colors["fg"]).pack(anchor=tk.W)
        self.k_entry = tk.Entry(main, width=10, bg=self.colors["entry_bg"])
        self.k_entry.insert(0, "3")
        self.k_entry.pack(anchor=tk.W, pady=(0, 15))

        tk.Label(main, text="Distance:", bg=self.colors["bg"], fg=self.colors["fg"]).pack(anchor=tk.W)
        self.metric = ttk.Combobox(main, values=["euclidean", "cityblock"],
                                   state="readonly", width=20)
        self.metric.current(0)
        self.metric.pack(anchor=tk.W, pady=(0, 20))

        self.create_section(main, "Data Source", 1)

        self.data_source = tk.StringVar(value="csv")

        source_frame = tk.Frame(main, bg=self.colors["bg"])
        source_frame.pack(fill=tk.X)

        tk.Radiobutton(source_frame, text="Load CSV", variable=self.data_source, value="csv",
                       bg=self.colors["bg"], fg=self.colors["fg"],
                       command=self.toggle_csv_button).pack(side=tk.LEFT, padx=10)

        tk.Radiobutton(source_frame, text="Manual Input", variable=self.data_source, value="manual",
                       bg=self.colors["bg"], fg=self.colors["fg"]).pack(side=tk.LEFT, padx=10)


        self.csv_button = tk.Button(source_frame, text="Browse CSV...",
                                    command=self.browse_csv,
                                    bg=self.colors["accent"], fg="white")
        self.csv_button.pack(side=tk.RIGHT)

        self.create_section(main, "Data Points (x,y)", 2)

        text_frame = tk.Frame(main, bg=self.colors["card_bg"], bd=1, relief=tk.SUNKEN)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.text_box = tk.Text(text_frame, height=10, font=('Consolas', 10),
                                bg=self.colors["entry_bg"])
        self.text_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scroll = tk.Scrollbar(text_frame, command=self.text_box.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_box.config(yscrollcommand=scroll.set)

        self.text_box.insert("1.0", "1,2\n3,4\n5,6\n8,3\n")

        auto_frame = tk.Frame(main, bg=self.colors["bg"])
        auto_frame.pack(pady=10)

        tk.Label(auto_frame, text="Auto-generate:", bg=self.colors["bg"]).pack(side=tk.LEFT)
        self.auto_points = ttk.Combobox(auto_frame, values=["50", "100", "200", "500", "1000", "5000", "10000", "100000"], width=10, state="readonly")
        self.auto_points.current(1)
        self.auto_points.pack(side=tk.LEFT, padx=10)
        tk.Button(auto_frame, text="Generate", bg="#5cb85c", fg="white",
                  command=self.generate_data).pack(side=tk.LEFT)

        btn_frame = tk.Frame(main, bg=self.colors["bg"])
        btn_frame.pack(pady=10)

        self.run_button = tk.Button(
            btn_frame, text="Run Clustering", bg=self.colors["accent"], fg="white",
            font=('Arial', 11, 'bold'), padx=30, pady=10,
            command=self.run
        )
        self.run_button.pack(side=tk.LEFT, padx=10)

        tk.Button(btn_frame, text="Clear", bg=self.colors["secondary"], fg="black",
                  padx=20, pady=10, command=self.clear_data).pack(side=tk.LEFT)

        self.status_label = tk.Label(main, text="Ready",
                                     bg=self.colors["bg"], anchor="w")
        self.status_label.pack(fill=tk.X)


    def toggle_csv_button(self):
        if self.data_source.get() == "csv":
            self.csv_button.config(state=tk.NORMAL, bg=self.colors["accent"])
        else:
            self.csv_button.config(state=tk.DISABLED, bg=self.colors["secondary"])

    def browse_csv(self):
        filename = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv")]
        )
        if filename:
            self.file_path = filename
            self.status_label.config(text=f"Selected: {os.path.basename(filename)}")

    def generate_data(self):
        n = int(self.auto_points.get())
        clusters = 3
        data = []

        centers = [[random.uniform(0, 10), random.uniform(0, 10)] for _ in range(clusters)]

        pts_per_cluster = n // clusters
        for cx, cy in centers:
            for _ in range(pts_per_cluster):
                x = cx + random.uniform(-2, 2)
                y = cy + random.uniform(-2, 2)
                data.append([x, y])

        random.shuffle(data)

        self.text_box.delete("1.0", tk.END)
        for x, y in data:
            self.text_box.insert(tk.END, f"{x:.2f},{y:.2f}\n")

        self.data_source.set("manual")
        self.k_entry.delete(0, tk.END)
        self.k_entry.insert(0, str(clusters))

        self.status_label.config(text=f"Generated {n} points")

    def clear_data(self):
        self.text_box.delete("1.0", tk.END)
        self.status_label.config(text="Data cleared")

    def parse_manual_input(self):
        lines = self.text_box.get("1.0", tk.END).strip().split("\n")
        data = []
        for line in lines:
            if not line.strip():
                continue
            try:
                x, y = map(float, line.split(","))
            except:
                raise ValueError("Invalid manual input. Use 'x,y'")
            data.append([x, y])
        return np.array(data)

    def run(self):
        try:
            self.status_label.config(text="Processing...")
            self.run_button.config(state="disabled")
            self.root.update()

            k = int(self.k_entry.get())

            if k < 1:
                raise ValueError("Number of clusters must be at least 1.")

            metric = self.metric.get()
            method = self.method.get()

            if self.data_source.get() == "csv":
                data = load_data(self.file_path)

                if data.size == 0:
                    raise ValueError("CSV file is empty.")

            else:
                data = self.parse_manual_input()

                if len(data) == 0:
                    raise ValueError("No data points found. Please enter or generate points.")

            if len(data) < k:
                raise ValueError(f"Cluster count ({k}) cannot exceed number of points ({len(data)}).")

            if method == "K-Means":
                model = KMeans(k=k, distance_metric=metric)
                labels = model.fit(data)

                fig_clusters = plot_clusters(
                    data,
                    labels,
                    model.centroids,
                    title=f"K Means Clusters | Distance: {metric}"
                )

                PlotWindow("K-Means Results", fig_clusters, "kmeans")

                if metric != "euclidean":
                    messagebox.showinfo(
                        "Voronoi Disabled",
                        "Voronoi boundaries can only be generated using Euclidean distance."
                    )
                else:
                    fig_boundary = plot_voronoi(data, model.centroids, labels)
                    PlotWindow("K-Means Decision Boundary", fig_boundary, "kmeans")

                self.status_label.config(text="K-Means completed successfully.")


            else:
                labels, dendro_fig = hierarchical_clustering(data, metric, k)
                fig_clusters = plot_clusters(data, labels, title=f"Hierarchical Clusters | Distance: {metric}"
)

                if dendro_fig is not None:
                    PlotWindow("Hierarchical Dendrogram", dendro_fig, "hierarchical")
                else:
                    messagebox.showinfo(
                        "Dendrogram Skipped",
                        "The dataset is too large. The dendrogram was not generated for performance reasons."
                    )

                PlotWindow("Hierarchical Clusters", fig_clusters, "hierarchical")

                self.status_label.config(text=f"Hierarchical clustering completed successfully.")

        except ValueError as ve:
            messagebox.showerror("Invalid Input", str(ve))
            self.status_label.config(text="Input error occurred.")

        except FileNotFoundError:
            messagebox.showerror("File Error", "CSV file not found. Please select a valid file.")
            self.status_label.config(text="File error.")

        except Exception as e:
            messagebox.showerror("Unexpected Error", f"An unknown error occurred:\n{str(e)}")
            self.status_label.config(text="Unexpected error occurred.")

        finally:
            self.run_button.config(state="normal")