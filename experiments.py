import numpy as np
import matplotlib.pyplot as plt
from main import MLP, Parse_Data, calculate_mean, calculate_SD, standardize, BCE_loss, calculate_F1

def run_all_separate_experiments():
    print("Loading Raw Data...")
    X_train_raw, y_train = Parse_Data('magic/magic_train.txt')
    X_dev_raw, y_dev = Parse_Data('magic/magic_dev.txt')
    X_test_raw, y_test = Parse_Data('magic/magic_test.txt')

    print("Standardizing Datasets...")
    # 1. Standardize Train independently
    m_train = calculate_mean(X_train_raw)
    s_train = calculate_SD(X_train_raw)
    X_train = standardize(X_train_raw, m_train, s_train)
    
    # 2. Standardize Dev independently (as a separate training task)
    m_dev = calculate_mean(X_dev_raw)
    s_dev = calculate_SD(X_dev_raw)
    X_dev = standardize(X_dev_raw, m_dev, s_dev)

    # 3. Standardize Test using Train parameters 
    X_test = standardize(X_test_raw, m_train, s_train)

    epochs_to_run = 500 # Adjust to 1000 if your computer runs it fast enough!
    epochs_list = range(1, epochs_to_run + 1)
    
    final_model = None

    # ==========================================
    # EXPERIMENT 1: NETWORK SIZES (CAPACITY)
    # ==========================================
    print("\n" + "="*40)
    print("EXPERIMENT 1: NETWORK CAPACITIES")
    print("="*40)
    
    # YOUR UPDATED CAPACITIES
    network_sizes = [5, 25, 50, 100] 
    
    for size in network_sizes:
        print(f"\n--- Configuration: {size} Hidden Units (LR = 0.1) ---")
        
        # Train MLP 1 (Train Data)
        print("Training MLP on magic_train.txt...")
        mlp_train = MLP(input_size=X_train.shape[1], hidden_size=size, output_size=1)
        history_train = mlp_train.train(X_train, y_train, epochs=epochs_to_run, learning_rate=0.1)

        # Train MLP 2 (Dev Data)
        print("Training MLP on magic_dev.txt...")
        mlp_dev = MLP(input_size=X_dev.shape[1], hidden_size=size, output_size=1)
        history_dev = mlp_dev.train(X_dev, y_dev, epochs=epochs_to_run, learning_rate=0.1)

        # Plot and Save
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs_list, history_train['train_loss'], color='blue', label='Trial: Train Data')
        plt.plot(epochs_list, history_dev['train_loss'], color='orange', label='Trial: Dev Data')
        plt.title(f'BCE Loss (Capacity={size})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.subplot(1, 2, 2)
        plt.plot(epochs_list, history_train['train_f1'], color='blue', label='Trial: Train Data')
        plt.plot(epochs_list, history_dev['train_f1'], color='orange', label='Trial: Dev Data')
        plt.title(f'F1 Score (Capacity={size})')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        filename = f'exp_capacity_{size}.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {filename}")

    # ==========================================
    # EXPERIMENT 2: LEARNING RATES
    # ==========================================
    print("\n" + "="*40)
    print("EXPERIMENT 2: LEARNING RATES")
    print("="*40)
    
    # YOUR UPDATED LEARNING RATES
    learning_rates = [1.0, 0.1, 0.01, 0.001] 
    
    for lr in learning_rates:
        print(f"\n--- Configuration: Learning Rate = {lr} (Capacity = 50) ---")
        
        # Train MLP 1 (Train Data) - Using 50 units as the baseline for this experiment
        print("Training MLP on magic_train.txt...")
        mlp_train = MLP(input_size=X_train.shape[1], hidden_size=50, output_size=1)
        history_train = mlp_train.train(X_train, y_train, epochs=epochs_to_run, learning_rate=lr)

        # Save the model trained on LR=0.1 and Size=50 to use for the final test
        if lr == 0.1:
            final_model = mlp_train

        # Train MLP 2 (Dev Data)
        print("Training MLP on magic_dev.txt...")
        mlp_dev = MLP(input_size=X_dev.shape[1], hidden_size=50, output_size=1)
        history_dev = mlp_dev.train(X_dev, y_dev, epochs=epochs_to_run, learning_rate=lr)

        # Plot and Save
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs_list, history_train['train_loss'], color='blue', label='Trial: Train Data')
        plt.plot(epochs_list, history_dev['train_loss'], color='orange', label='Trial: Dev Data')
        plt.title(f'BCE Loss (LR={lr})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.subplot(1, 2, 2)
        plt.plot(epochs_list, history_train['train_f1'], color='blue', label='Trial: Train Data')
        plt.plot(epochs_list, history_dev['train_f1'], color='orange', label='Trial: Dev Data')
        plt.title(f'F1 Score (LR={lr})')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        filename = f'exp_lr_{lr}.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {filename}")

    # ==========================================
    # FINAL EVALUATION ON TEST SET
    # ==========================================
    print("\n" + "="*40)
    print("FINAL TEST EVALUATION")
    print("="*40)
    
    # We use the final_model (trained purely on train.txt with Capacity=50 and LR=0.1)
    y_test_col = y_test.reshape(-1, 1)
    
    test_pred = final_model.predict(X_test, None) 
    
    final_loss = BCE_loss(y_test_col, test_pred)
    final_f1 = calculate_F1(y_test_col, test_pred)

    print(f"Test Loss: {final_loss:.4f}")
    print(f"Test F1 Score: {final_f1:.4f}")
    print("\nAll experiments complete! 8 image files have been saved to your directory.")

if __name__ == "__main__":
    run_all_separate_experiments()