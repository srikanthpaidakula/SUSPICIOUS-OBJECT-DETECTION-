# Load the trained model weights
weights_n = torch.load("path_to_yolov8n_weights.pt")
weights_l = torch.load("path_to_yolov8l_weights.pt")

# Ensure both models have the same keys
assert weights_n.keys() == weights_l.keys()

# Create a new model for fusion
model_fusion = YOLO("yolov8n.yaml")  # Use a base model to start with

# Average the weights
fused_weights = {}
for key in weights_n.keys():
    fused_weights[key] = (weights_n[key] + weights_l[key]) / 2

# Load the fused weights into the new model
model_fusion.model.load_state_dict(fused_weights)

# Save the fused model
torch.save(model_fusion.model.state_dict(), "path_to_fused_model_weights.pt")

# Train the fused model and get its history
history_fusion = train_model(model_fusion, "config.yaml", 1)

# Plot training history
def plot_history(history, title):
    plt.figure(figsize=(10, 5))

    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train']['loss'], label='Train Loss')
    plt.plot(history['val']['loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{title} - Loss')
    plt.legend()

    # Plot training accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train']['accuracy'], label='Train Accuracy')
    plt.plot(history['val']['accuracy'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Plot histories
plot_history(history_n, "YOLOv8n")
plot_history(history_l, "YOLOv8l")
plot_history(history_fusion, "Fused Model")