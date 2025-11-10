import matplotlib.pyplot as plt
# import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


class TrainingVisualizer:
    def __init__(self, output_dir=None):
        """
        Initialize the TrainingVisualizer class.

        Args:
            output_dir (str): Directory to save plots. If None, plots will only be shown.
        """
        self.output_dir = output_dir

    def plot_training_loss(self, train_losses):
        plt.plot(range(len(train_losses)), train_losses, label="Training Loss", color="blue")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Epochs")
        plt.legend()
        if self.output_dir:
            plt.savefig(f"{self.output_dir}/training_loss.png")
        plt.show()

    def plot_validation_loss(self, val_losses):
        plt.plot(range(len(val_losses)), val_losses, label="Validation Loss", color="orange")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Validation Loss Over Epochs")
        plt.legend()
        if self.output_dir:
            plt.savefig(f"{self.output_dir}/validation_loss.png")
        plt.show()

    def plot_mse(self, mse_values):
        plt.plot(range(len(mse_values)), mse_values, label="Validation MSE", color="green")
        plt.xlabel("Epochs")
        plt.ylabel("MSE")
        plt.title("Validation MSE Over Epochs")
        plt.legend()
        if self.output_dir:
            plt.savefig(f"{self.output_dir}/validation_mse.png")
        plt.show()

    def plot_action_comparison(self, gt_action, pred_action, dim):
        plt.plot(gt_action[:, dim], label="Ground Truth", linestyle="--", marker="o")
        plt.plot(pred_action[:, dim], label="Predicted", linestyle="-", marker="x")
        plt.xlabel("Time Steps")
        plt.ylabel("Action Value")
        plt.title(f"Ground Truth vs Predicted Actions (Dimension {dim})")
        plt.legend()
        if self.output_dir:
            plt.savefig(f"{self.output_dir}/action_comparison_dim{dim}.png")
        plt.show()

    def plot_gripper_states(self, gt_action, pred_action):
        plt.plot(gt_action[:, -2], label="GT - Left Gripper", linestyle="--", color="blue")
        plt.plot(pred_action[:, -2], label="Pred - Left Gripper", linestyle="-", color="cyan")
        plt.plot(gt_action[:, -1], label="GT - Right Gripper", linestyle="--", color="green")
        plt.plot(pred_action[:, -1], label="Pred - Right Gripper", linestyle="-", color="lime")
        plt.xlabel("Time Steps")
        plt.ylabel("Gripper State")
        plt.title("Gripper States Over Time")
        plt.legend()
        if self.output_dir:
            plt.savefig(f"{self.output_dir}/gripper_states.png")
        plt.show()

    def plot_learning_rate(self, lrs):
        plt.plot(range(len(lrs)), lrs, label="Learning Rate", color="purple")
        plt.xlabel("Steps")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Over Steps")
        plt.legend()
        if self.output_dir:
            plt.savefig(f"{self.output_dir}/learning_rate.png")
        plt.show()

    def plot_action_distribution(self, gt_action, pred_action):
        plt.hist(gt_action.flatten(), bins=50, alpha=0.5, label="Ground Truth", color="blue")
        plt.hist(pred_action.flatten(), bins=50, alpha=0.5, label="Predicted", color="orange")
        plt.xlabel("Action Value")
        plt.ylabel("Frequency")
        plt.title("Action Value Distribution")
        plt.legend()
        if self.output_dir:
            plt.savefig(f"{self.output_dir}/action_distribution.png")
        plt.show()

    # def plot_error_heatmap(self, gt_action, pred_action):
    #     differences = gt_action - pred_action
    #     sns.heatmap(differences, cmap="coolwarm", annot=False)
    #     plt.title("Prediction Error Heatmap")
    #     plt.xlabel("Action Dimensions")
    #     plt.ylabel("Time Steps")
    #     if self.output_dir:
    #         plt.savefig(f"{self.output_dir}/error_heatmap.png")
    #     plt.show()

    def plot_trajectory(self, gt_trajectory, pred_trajectory):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], gt_trajectory[:, 2], label="Ground Truth", color="blue")
        ax.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], pred_trajectory[:, 2], label="Predicted", color="orange")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.title("3D Trajectory Comparison")
        plt.legend()
        if self.output_dir:
            plt.savefig(f"{self.output_dir}/trajectory_comparison.png")
        plt.show()
