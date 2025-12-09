import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import os

from core.dataloader import load_data
from core.kmeans import KMeans
from core.hierarchical import hierarchical_clustering


class ClusteringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Clustering Tool")
        self.root.geometry("450x550")

        self.file_path = "sampledata.csv"

        self.setup_ui()

    def setup_ui(self):
        main = tk.Frame(self.root, padx=20, pady=20)
        main.pack(fill=tk.BOTH, expand=True)

        tk.Label(main, text="Clustering Tool", font=("Arial", 16, "bold")).pack(pady=10)

        tk.Label(main, text="Algorithm:").pack(anchor="w")
        self.method = ttk.Combobox(main, values=["K-Means", "Hierarchical"], state="readonly")
        self.method.current(0)
        self.method.pack(fill="x", pady=5)

        tk.Label(main, text="Clusters:").pack(anchor="w")
        self.k_entry = tk.Entry(main)
        self.k_entry.insert(0, "3")
        self.k_entry.pack(fill="x", pady=5)

        tk.Label(main, text="Distance Metric:").pack(anchor="w")
        self.metric = ttk.Combobox(main, values=["euclidean", "cityblock"], state="readonly")
        self.metric.current(0)
        self.metric.pack(fill="x", pady=5)

        self.data_src = tk.StringVar(value="csv")
        tk.Radiobutton(main, text="CSV File", variable=self.data_src, value="csv").pack(anchor="w")
        tk.Radiobutton(main, text="Manual Input", variable=self.data_src, value="manual").pack(anchor="w")

        tk.Button(main, text="Browse CSV", command=self.browse_csv).pack(pady=5)

        tk.Label(main, text="Data (x,y):").pack(anchor="w")
        self.text_box = tk.Text(main, height=8)
        self.text_box.pack(fill="both", expand=True)
        self.text_box.insert("1.0", "1,2\n3,4\n5,6")

        tk.Button(main, text="Run Clustering", bg="#4a6fa5", fg="white",
                  command=self.run_clustering).pack(pady=10)

        self.output = tk.Text(main, height=6, state="disabled", bg="#eeeeee")
        self.output.pack(fill="both", expand=True)

    def browse_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path:
            self.file_path = path
            messagebox.showinfo("Selected", f"Loaded: {os.path.basename(path)}")

    def parse_input(self):
        lines = self.text_box.get("1.0", "end").strip().split("\n")
        data = []
        for line in lines:
            if line.strip():
                x, y = map(float, line.split(","))
                data.append([x, y])
        return np.array(data)

    def display(self, text):
        self.output.config(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("end", text)
        self.output.config(state="disabled")

    def run_clustering(self):
        try:
            k = int(self.k_entry.get())
            metric = self.metric.get()
            method = self.method.get()

            if self.data_src.get() == "csv":
                data = load_data(self.file_path)
            else:
                data = self.parse_input()

            if len(data) < k:
                raise ValueError("Number of clusters cannot exceed number of points.")

            if method == "K-Means":
                model = KMeans(k=k, distance_metric=metric)
                labels = model.fit(data)
                result = f"K-Means Completed\nCentroids:\n{model.centroids}\nLabels:\n{labels}"
            else:
                labels, _ = hierarchical_clustering(data, metric, k)
                result = f"Hierarchical Completed\nLabels:\n{labels}"

            self.display(result)

        except Exception as e:
            messagebox.showerror("Error", str(e))



