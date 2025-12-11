import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import random
import os

from core.dataloader import load_data
from core.kmeans import KMeans
from core.hierarchical import hierarchical_clustering


class ClusteringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Clustering Tool")
        self.root.geometry("580x680")

        self.colors = {
            "bg": "#f5f5f5",
            "fg": "#222",
            "accent": "#4a6fa5",
            "card": "#ffffff",
            "disabled": "#cccccc"
        }

        self.root.configure(bg=self.colors["bg"])
        self.file_path = "sampledata.csv"

        self.create_ui()

    def section(self, parent, label):
        tk.Label(parent, text=label, fg=self.colors["accent"], bg=self.colors["bg"],
                 font=("Arial", 12, "bold")).pack(anchor="w", pady=(20, 5))
        tk.Frame(parent, height=1, bg=self.colors["disabled"]).pack(fill="x", pady=(0, 10))

    def create_ui(self):
        main = tk.Frame(self.root, bg=self.colors["bg"])
        main.pack(fill="both", expand=True, padx=20, pady=20)

        tk.Label(main, text="Clustering Analysis", fg=self.colors["accent"],
                 bg=self.colors["bg"], font=("Arial", 18, "bold")).pack(pady=10)

        self.section(main, "Algorithm Settings")

        tk.Label(main, text="Algorithm:", bg=self.colors["bg"]).pack(anchor="w")
        self.method = ttk.Combobox(main, values=["K-Means", "Hierarchical"], state="readonly", width=25)
        self.method.current(0)
        self.method.pack(anchor="w", pady=5)

        tk.Label(main, text="Clusters:", bg=self.colors["bg"]).pack(anchor="w")
        self.k_entry = tk.Entry(main, width=10)
        self.k_entry.insert(0, "3")
        self.k_entry.pack(anchor="w", pady=5)

        tk.Label(main, text="Distance Metric:", bg=self.colors["bg"]).pack(anchor="w")
        self.metric = ttk.Combobox(main, values=["euclidean", "cityblock"], state="readonly", width=20)
        self.metric.current(0)
        self.metric.pack(anchor="w", pady=10)

        self.section(main, "Data Source")

        self.data_mode = tk.StringVar(value="csv")

        tk.Radiobutton(main, text="Load CSV", variable=self.data_mode, value="csv",
                       bg=self.colors["bg"]).pack(anchor="w")
        tk.Radiobutton(main, text="Manual Input", variable=self.data_mode, value="manual",
                       bg=self.colors["bg"]).pack(anchor="w")

        self.csv_button = tk.Button(main, text="Browse CSV", bg=self.colors["accent"],
                                    fg="white", command=self.browse_csv)
        self.csv_button.pack(pady=5)

        self.section(main, "Data Points (x,y)")

        frame = tk.Frame(main, bg=self.colors["card"], bd=1, relief="sunken")
        frame.pack(fill="both", expand=True)

        self.text_box = tk.Text(frame, height=12)
        self.text_box.pack(side="left", fill="both", expand=True)

        scroll = tk.Scrollbar(frame, command=self.text_box.yview)
        scroll.pack(side="right", fill="y")
        self.text_box.configure(yscrollcommand=scroll.set)

        self.text_box.insert("1.0", "1,2\n3,4\n5,6\n8,3")

        tk.Button(main, text="Run Clustering", bg=self.colors["accent"], fg="white",
                  font=("Arial", 11, "bold"), command=self.run).pack(pady=15)

        self.status = tk.Label(main, text="Ready", bg=self.colors["bg"], anchor="w")
        self.status.pack(fill="x")

    def browse_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path:
            self.file_path = path
            self.status.config(text=os.path.basename(path))

    def parse_manual(self):
        data = []
        for line in self.text_box.get("1.0", "end").strip().split("\n"):
            x, y = map(float, line.split(","))
            data.append([x, y])
        return np.array(data)

    def run(self):
        try:
            k = int(self.k_entry.get())
            metric = self.metric.get()
            method = self.method.get()

            if self.data_mode.get() == "csv":
                data = load_data(self.file_path)
            else:
                data = self.parse_manual()

            if len(data) < k:
                raise ValueError("Clusters cannot exceed number of points.")

            if method == "K-Means":
                model = KMeans(k=k, distance_metric=metric)
                labels = model.fit(data)
                msg = f"K-Means Done.\nCentroids:\n{model.centroids}\nLabels:\n{labels}"
            else:
                labels, _ = hierarchical_clustering(data, metric, k)
                msg = f"Hierarchical Done.\nLabels:\n{labels}"

            messagebox.showinfo("Results", msg)
            self.status.config(text="Completed")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status.config(text="Error")
