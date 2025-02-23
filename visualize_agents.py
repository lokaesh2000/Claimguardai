import networkx as nx
import matplotlib.pyplot as plt
import random
import time

# Define the agents
agents = ["Text Agent", "Video Agent", "Document Agent", "Speech Agent"]
result_node = "Result"

# Create a directed graph
G = nx.DiGraph()

# Add nodes (agents)
for agent in agents:
    G.add_node(agent)

# Add result node
G.add_node(result_node)

# Connect agents to each other
for agent in agents:
    for other_agent in agents:
        if agent != other_agent:
            G.add_edge(agent, other_agent)

# Connect all agents to the result node
for agent in agents:
    G.add_edge(agent, result_node)

# Define positions for visualization
pos = {
    "Text Agent": (-1, 1),
    "Video Agent": (1, 1),
    "Document Agent": (-1, -1),
    "Speech Agent": (1, -1),
    "Result": (0, 0)
}

# Function to update visualization
def update_graph(active_agents):
    plt.clf()  # Clear previous plot
    plt.title("Agent Interaction Visualization")

    # Define node colors: active agents blink in red
    node_colors = ["red" if node in active_agents else "lightblue" for node in G.nodes]

    # Draw graph
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color=node_colors, edge_color="gray", font_size=10)
    plt.pause(1)  # Pause to visualize changes

# Visualization loop
plt.ion()  # Enable interactive mode

for _ in range(10):  # Run for 10 seconds
    active_agents = random.sample(agents, 2)  # Pick 2 random agents
    update_graph(active_agents)

# Final step: highlight result node
plt.clf()
plt.title("Final Result")

node_colors = ["green" if node == result_node else "lightblue" for node in G.nodes]
nx.draw(G, pos, with_labels=True, node_size=3000, node_color=node_colors, edge_color="gray", font_size=10)

plt.ioff()  # Disable interactive mode
plt.show()
