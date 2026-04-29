import numpy as np
import sys

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        # in this project, the output size will always be 1
            self.input_hidden_weights = np.random.randn(input_size, hidden_size) * 0.2
            self.hidden_output_weights = np.random.randn(hidden_size, output_size) * 0.2
            self.bias_hidden = np.zeros((1, hidden_size))
            self.bias_output = np.zeros((1, output_size))
    
    def forward_pass(self, X):
        #apply weights on inputs and sum
        self.hidden_input = np.dot(X, self.input_hidden_weights) + self.bias_hidden

        #apply the activation function of the hidden layer (tanh)
        self.hidden_output = np.tanh(self.hidden_input)

        #apply weights on hidden outputs and sum
        self.final_input = np.dot(self.hidden_output, self.hidden_output_weights) + self.bias_output

        #apply activation function of final layer (sigmoid)
        self.final_output = self.sigmoid(self.final_input)
        return self.final_output
    
    def backprop(self, X, y, predictions):
        # numpy can compute the forward pass for all 128 examples fast
        # by feeding the mini-bath as a 128 x D matrix.

        # note that y is considered an array rather than a matrix column, so we change it to a column
        batch_size = X.shape[0]
        y1 = y.reshape(-1,1)

        output_error = predictions - y1

        
        self.gradient_hidden_output_weights = np.dot(self.hidden_output.T, output_error) / batch_size
        self.gradient_output_bias = np.sum(output_error, axis = 0) / batch_size
        
        # the hidden error is the product of the propagated error and the derivative of the activation function
        # because the tanh derivative is 1-tanh^2(x), we can just use the hidden outputs
        hidden_error = np.dot(output_error, self.hidden_output_weights.T) * (1.0 -(self.hidden_output ** 2))
        
        self.gradient_input_hidden_weights = np.dot(X.T, hidden_error) / batch_size
        self.gradient_hidden_bias = np.sum(hidden_error, axis = 0) / batch_size
        return None
    
    #the gradient descent is separate to maintain program flexibility
    def apply_GD(self, learning_rate):
        self.input_hidden_weights -= learning_rate * self.gradient_input_hidden_weights
        self.bias_hidden -= learning_rate * self.gradient_hidden_bias
        self.hidden_output_weights -= learning_rate * self.gradient_hidden_output_weights
        self.bias_output -= learning_rate * self.gradient_output_bias
        return None
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, X, y, epochs, learning_rate):
        history = {
            'train_loss': [],
            'train_f1': []
        }
        num_samples = X.shape[0]
        
        for epoch in range(epochs):
            permutation = np.random.permutation(num_samples)
            X_shuffled = X[permutation] # shuffle X
            y_shuffled = y[permutation] # shuffle y
            for i in range(0,num_samples, 128):
                X_batch = X_shuffled[i : i+128] #collect 128 rows of X
                y_batch = y_shuffled[i : i+128] #collect 128 rows of y
                predictions = self.forward_pass(X_batch)
                self.backprop(X_batch, y_batch, predictions)
                self.apply_GD(learning_rate)

            
            epoch_pred = self.forward_pass(X)

            

            #at the end of the epoch we print the BCE loss and F1 score
            y1 = y.reshape(-1,1)
            loss = BCE_loss(y1, epoch_pred)
            f1 = calculate_F1(y1, epoch_pred)

            #prevent stochastic noise in the case of easy/linear problems by stopping when F1 reaches 1
            if f1 == 1.0:
                print(f"Perfect F1 Score reached at Epoch {epoch+1}")
                history['train_loss'].append(loss)
                history['train_f1'].append(f1)
                break

            history['train_loss'].append(loss)
            history['train_f1'].append(f1)

            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - F1: {f1:.4f}")

        return history
            
    
    def predict(self, X):
        predictions = self.forward_pass(X)
        return predictions


    
def BCE_loss(y, predictions):
    epsilon = 1e-15
    clipped_predictions = np.clip(predictions, epsilon, 1.0-epsilon)
    loss = -np.mean(y*np.log(clipped_predictions)+(1.0-y)*np.log(1.0-clipped_predictions))
    return loss

def calculate_F1(y, predictions):
    # we can convert the predictions into a boolean array
    bool_pred = (predictions >= 0.5).astype(int)

    TP = np.sum((bool_pred == 1) & (y == 1))
    FP = np.sum((bool_pred == 1) & (y == 0))
    FN = np.sum((bool_pred == 0) & (y == 1))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1
# we need to standardize the MAGIC data, so we define 2 helper functions for finding the mean,
# and the standard deviation, and 
def calculate_mean(X):
    return np.mean(X, axis=0)

def calculate_SD(X):
    return np.std(X, axis=0)

def standardize(X, mean, SD):
    epsilon = 1e-15
    return (X-mean) / (SD + epsilon)

def Parse_Data(filename):
    X_list = []
    y_list = []
    with open(filename, 'r') as file:
        for line in file:
            l = line.strip().split(' ')
            X_list.append(l[:-1])
            y_list.append(l[-1])

    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=int)
    return X, y

              
def main():
    if len(sys.argv) != 5:
        print("Incorrect arguments")
        sys.exit(1)
    dataset_path = sys.argv[1]         
    hidden_units = int(sys.argv[2])    
    epochs = int(sys.argv[3])          
    learning_rate = float(sys.argv[4]) 

    print(f"Loading data from: {dataset_path}")
    print(f"Configuration: {hidden_units} hidden units, {epochs} epochs, LR: {learning_rate}")
    
    X_raw, y_raw = Parse_Data(dataset_path)
    means = calculate_mean(X_raw)
    stds = calculate_SD(X_raw)
    X = standardize(X_raw, means, stds)

    y = y_raw.reshape(-1,1)
    myMLP = MLP(input_size=X.shape[1], hidden_size=hidden_units, output_size=1)
    myMLP.train(X, y, epochs=epochs, learning_rate=learning_rate)


    predictions = myMLP.predict(X) 
    final_loss = BCE_loss(y, predictions)
    final_f1 = calculate_F1(y, predictions)

    print(f"\nFinal Loss: {final_loss:.4f}")
    print(f"Final F1 Score: {final_f1:.4f}")

    

if __name__=="__main__":
    main()