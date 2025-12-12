import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Load the training data
train_inputs = np.load("data/kis_model/train_inputs.npy")
train_outputs = np.load("data/kis_model/train_outputs.npy")

# Global variable to track current index
current_idx = 0

def visualize_sample(idx):
    """
    Visualize a single input-output pair.
    """
    # Get the input and output windows
    input_window = train_inputs[idx].reshape(3, 3)
    output_window = train_outputs[idx].reshape(3, 3)
    
    # Clear the figure
    plt.clf()
    
    # Create subplots
    fig = plt.gcf()
    
    # Plot input window
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(input_window, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    ax1.set_title(f'Input Window (3x3)\nSample {idx + 1}/{len(train_inputs)}')
    ax1.axis('off')
    
    # Add pixel values as text
    for i in range(3):
        for j in range(3):
            text_color = 'white' if input_window[i, j] < 128 else 'black'
            ax1.text(j, i, f'{int(input_window[i, j])}', 
                    ha='center', va='center', color=text_color, fontsize=8)
    
    # Plot output window
    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(output_window, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    ax2.set_title('Output Window (3x3)')
    ax2.axis('off')
    
    # Add pixel values as text
    for i in range(3):
        for j in range(3):
            text_color = 'white' if output_window[i, j] < 128 else 'black'
            ax2.text(j, i, f'{int(output_window[i, j])}', 
                    ha='center', va='center', color=text_color, fontsize=8)
    
    plt.tight_layout()
    plt.draw()

def next_sample(event):
    """Callback for Next button"""
    global current_idx
    current_idx = (current_idx + 1) % len(train_inputs)
    visualize_sample(current_idx)

def prev_sample(event):
    """Callback for Previous button"""
    global current_idx
    current_idx = (current_idx - 1) % len(train_inputs)
    visualize_sample(current_idx)

def random_sample(event):
    """Callback for Random button"""
    global current_idx
    current_idx = np.random.randint(0, len(train_inputs))
    visualize_sample(current_idx)

# Create the figure
fig = plt.figure(figsize=(10, 5))

# Add buttons
ax_prev = plt.axes([0.2, 0.05, 0.1, 0.05])
ax_next = plt.axes([0.7, 0.05, 0.1, 0.05])
ax_random = plt.axes([0.45, 0.05, 0.1, 0.05])

btn_prev = Button(ax_prev, 'Previous')
btn_next = Button(ax_next, 'Next')
btn_random = Button(ax_random, 'Random')

btn_prev.on_clicked(prev_sample)
btn_next.on_clicked(next_sample)
btn_random.on_clicked(random_sample)

# Show the first sample
visualize_sample(current_idx)

plt.show()