import tkinter as tk
from tkinter import ttk
import networkx as nx
import matplotlib
matplotlib.use('TkAgg',force=True)
from matplotlib import pyplot as plt
import numpy as np
import time
import heapq
from tkinter import messagebox


def a_star_algorithm(adj_matrix, heuristics, start_node, goal_node):
    num_nodes = len(adj_matrix)
    open_list = []
    heapq.heappush(open_list, (0, start_node))  # Priority queue: (f_score, node)
    came_from = {}
    g_score = {node: float('inf') for node in range(num_nodes)}
    g_score[start_node] = 0
    
    f_score = {node: float('inf') for node in range(num_nodes)}
    f_score[start_node] = heuristics[start_node]
    
    while open_list:
        current_f, current_node = heapq.heappop(open_list)
        
        if current_node == goal_node:
            return reconstruct_path(came_from, current_node), g_score[goal_node]  # Path and total cost
        
        for neighbor, weight in enumerate(adj_matrix[current_node]):
            if weight > 0:  # If there's a connection
                tentative_g_score = g_score[current_node] + weight
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current_node
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristics[neighbor]
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))
                    
    return None, float('inf')  # Return no path and infinite cost if no solution

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.insert(0, current)
    return path


class AStarVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("A* Algorithm Visualizer")

        # Use a frame to better organize and group widgets
        input_frame = ttk.Frame(root, padding="10 10 10 10")
        input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Instructions for adjacency matrix format
        self.instructions = ttk.Label(input_frame, text="Adjacency Matrix Format (e.g. 0,1,0; 1,0,1; 0,1,0):", font=('Arial', 10, 'bold'))
        self.instructions.grid(row=0, column=0, columnspan=2, sticky=tk.W)

        # Adjacency matrix entry
        self.adj_matrix_label = ttk.Label(input_frame, text="Adjacency Matrix:", font=('Arial', 10))
        self.adj_matrix_label.grid(row=1, column=0, sticky=tk.W)
        
        self.adj_matrix_entry = ttk.Entry(input_frame, width=50)
        self.adj_matrix_entry.grid(row=1, column=1, pady=5)

        # Heuristics entry
        self.heuristics_label = ttk.Label(input_frame, text="Heuristics (comma-separated):", font=('Arial', 10))
        self.heuristics_label.grid(row=2, column=0, sticky=tk.W)

        self.heuristics_entry = ttk.Entry(input_frame, width=50)
        self.heuristics_entry.grid(row=2, column=1, pady=5)

        # Start and Goal Node entries
        self.start_label = ttk.Label(input_frame, text="Start Node:", font=('Arial', 10))
        self.start_label.grid(row=3, column=0, sticky=tk.W)

        self.start_entry = ttk.Entry(input_frame, width=10)
        self.start_entry.grid(row=3, column=1, pady=5, sticky=tk.W)

        self.goal_label = ttk.Label(input_frame, text="Goal Node:", font=('Arial', 10))
        self.goal_label.grid(row=4, column=0, sticky=tk.W)

        self.goal_entry = ttk.Entry(input_frame, width=10)
        self.goal_entry.grid(row=4, column=1, pady=5, sticky=tk.W)

        # Buttons for A* and graph generation
        button_frame = ttk.Frame(root, padding="10 10 10 10")
        button_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))

        self.run_button = ttk.Button(button_frame, text="Run A*", command=self.run_a_star)
        self.run_button.grid(row=0, column=0, padx=5, pady=10)

        self.gen_button = ttk.Button(button_frame, text="Generate Graph", command=self.generate_graph)
        self.gen_button.grid(row=0, column=1, padx=5)

        self.random_button = ttk.Button(button_frame, text="Generate Random Graph", command=self.generate_random_graph)
        self.random_button.grid(row=0, column=2, padx=5)

        # Canvas for graph display
        graph_frame = ttk.Frame(root, padding="10 10 10 10")
        graph_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.fig, self.ax = plt.subplots(figsize=(7, 5))  # Increase canvas size
        plt.tight_layout()

        # Add an area to display errors or results below
        self.result_label = ttk.Label(root, text="", font=('Arial', 12, 'italic'), foreground="green")
        self.result_label.grid(row=3, column=0, pady=10)

    
    def run_a_star(self):
        if self.adj_matrix_entry.get() and self.heuristics_entry.get():
            adj_matrix = self.get_matrix(self.adj_matrix_entry.get())
            heuristics = list(map(int, self.heuristics_entry.get().split(',')))
        else:
            # If no input, use the randomly generated graph
            adj_matrix = self.adj_matrix
            heuristics = self.heuristics

        try:
            start_node = int(self.start_entry.get())
            goal_node = int(self.goal_entry.get())

            if adj_matrix is None or heuristics is None:
                raise ValueError("Invalid input for adjacency matrix or heuristics.")

            path, total_cost = a_star_algorithm(adj_matrix, heuristics, start_node, goal_node)

            if path is not None:
                self.animate_path(path, adj_matrix, heuristics, total_cost)
            else:
                messagebox.showinfo("Result", "No path found.")
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input: {str(e)}")



    def animate_path(self, path, adj_matrix, heuristics, total_cost):
        # Clear the previous plot
        self.ax.clear()

        # Create a directed graph from the adjacency matrix
        G = nx.DiGraph()
        num_nodes = len(adj_matrix)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_matrix[i][j] != 0:
                    G.add_edge(i, j, weight=adj_matrix[i][j])

        # Layout for the graph
        pos = nx.spring_layout(G)

        # Path animation step-by-step
        for i, node in enumerate(path):
            plt.clf()  # Clear the previous frame
            plt.title(f"Step {i+1}: Node {node}")

            # Draw the graph with the current node highlighted
            nx.draw(G, pos, with_labels=True, node_color=['r' if n == node else 'b' for n in G.nodes], node_size=3000, font_size=15)

            # Draw edge labels with weights
            edge_labels = nx.get_edge_attributes(G, 'weight')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

            # Show heuristics on each node
            for n, heuristic in enumerate(heuristics):
                x, y = pos[n]
                plt.text(x, y + 0.1, f'H: {heuristic}', fontsize=12, ha='center')

            plt.draw()  # Redraw the plot with updates
            plt.pause(1)  # Pause to visualize the current step (1 second per step)

        # Show the final path with all steps
        plt.title(f"Optimal Path: {path}\nTotal Cost: {total_cost}")
        plt.show()

        # Display total cost in a message box
        messagebox.showinfo("A* Result", f"Optimal Path: {path}\nTotal Cost: {total_cost}")

        
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



if __name__ == "__main__":
    root = tk.Tk()
    app = AStarVisualizer(root)
    root.mainloop()
