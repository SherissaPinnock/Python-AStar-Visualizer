import tkinter as tk
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tkinter import messagebox

class AStarVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("A* Algorithm Visualizer")

        # Create adjacency matrix entry
        self.adj_matrix_label = tk.Label(root, text="Enter Adjacency Matrix (comma-separated):")
        self.adj_matrix_label.pack()

        self.adj_matrix_entry = tk.Entry(root, width=50)
        self.adj_matrix_entry.pack()

        # Create Heuristics entry
        self.heuristics_label = tk.Label(root, text="Enter Heuristics (comma-separated):")
        self.heuristics_label.pack()

        self.heuristics_entry = tk.Entry(root, width=50)
        self.heuristics_entry.pack()

        # Buttons for generating the graph
        self.gen_button = tk.Button(root, text="Generate Graph", command=self.generate_graph)
        self.gen_button.pack()

        self.random_button = tk.Button(root, text="Generate Random Graph", command=self.generate_random_graph)
        self.random_button.pack()

        # Canvas for displaying graph
        self.fig, self.ax = plt.subplots()
        
    def generate_graph(self):
        adj_matrix = self.get_matrix(self.adj_matrix_entry.get())
        heuristics = list(map(int, self.heuristics_entry.get().split(',')))

        if adj_matrix is not None and heuristics is not None:
            self.draw_graph(adj_matrix, heuristics)

    def generate_random_graph(self):
        num_nodes = 5
        adj_matrix = np.random.randint(0, 10, size=(num_nodes, num_nodes))
        adj_matrix[adj_matrix < 3] = 0  # Random sparsity
        heuristics = np.random.randint(1, 10, size=num_nodes)
        self.draw_graph(adj_matrix, heuristics)

    def get_matrix(self, matrix_str):
        try:
            matrix = np.array([list(map(int, row.split(','))) for row in matrix_str.split(';')])
            return matrix
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid matrix.")
            return None

    def draw_graph(self, adj_matrix, heuristics):
        graph = nx.DiGraph()
        num_nodes = len(adj_matrix)

        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_matrix[i][j] != 0:
                    graph.add_edge(i, j, weight=adj_matrix[i][j])

        pos = nx.spring_layout(graph)
        labels = nx.get_edge_attributes(graph, 'weight')

        nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=15)
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
        
        # Draw heuristics on each node
        for i, heuristic in enumerate(heuristics):
            x, y = pos[i]
            plt.text(x, y+0.1, f'H: {heuristic}', fontsize=12, ha='center')

        plt.show()

    def run_a_star(self):
        # Implement the A* algorithm here
        pass

if __name__ == "__main__":
    root = tk.Tk()
    app = AStarVisualizer(root)
    root.mainloop()
