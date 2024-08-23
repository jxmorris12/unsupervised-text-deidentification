import matplotlib.pyplot as plt

# Updated data
data1 = [
    [0.14, 0.9093], [3.11, 0.4076], [5.15, 0.3545], [7.40, 0.3579], 
    [8.69, 0.352], [10.73, 0.2654], [12.69, 0.2107], [14.84, 0.1854], 
    [15.99, 0.1579], [19.02, 0.1038]
]

data2 = [
    [1.29, 0.819], [1.71, 0.7971], [2.12, 0.7958], [2.49, 0.7504], 
    [2.92, 0.7675], [3.80, 0.7298], [5.29, 0.7214], [7.47, 0.5472], 
    [8.61, 0.5185], [11.28, 0.3752]
]

# Extracting x and y values
x1, y1 = zip(*data1)
x2, y2 = zip(*data2)

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(x1, y1, marker='o', label='Data 1', linestyle='-', color='red')
plt.plot(x2, y2, marker='s', label='Data 2', linestyle='--', color='blue')

# Adding titles and labels
plt.title('Comparative Graph of Data 1 and Data 2')
plt.xlabel('% Words masked')
plt.ylabel('Reidentification % (Ensemble)')
plt.legend()

# Display the plot
plt.grid(True)
plt.savefig("plot_masking_percentage_vs_val_acc_graph.png")
