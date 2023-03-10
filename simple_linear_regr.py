import numpy as np
from simple_linear_regr_utils import generate_data, evaluate
import time
from datetime import datetime
import pickle

class SimpleLinearRegression:
    def __init__(self, iterations=15000, lr=0.1):
        self.iterations = iterations # number of iterations the fit method will be called
        self.lr = lr # The learning rate
        self.losses = [] # A list to hold the history of the calculated losses
        self.W, self.b = None, None # the slope and the intercept of the model

    def __loss(self, y, y_hat):
        """

        :param y: the actual output on the training set
        :param y_hat: the predicted output on the training set
        :return:
            loss: the sum of squared error

        """
        #ToDO calculate the loss. use the sum of squared error formula for simplicity
        loss = 1 / (2 * y.shape[0]) * np.sum(np.square(y_hat - y))
        #loss = np.sum(y_hat.T - y)

        self.losses.append(loss)
        return loss

    def __init_weights(self, X):
        """

        :param X: The training set
        """
        weights = np.random.normal(size=X.shape[1] + 1)
        self.W = weights[:X.shape[1]].reshape(-1, X.shape[1])
        self.b = weights[-1]

    def __sgd(self, X, y, y_hat):
        """

        :param X: The training set
        :param y: The actual output on the training set
        :param y_hat: The predicted output on the training set
        :return:
            sets updated W and b to the instance Object (self)
        """
        # ToDo calculate dW & db.
        dW = (2/X.shape[0])*(np.dot(X.T, (y_hat-y)))
        db = (2/X.shape[0])*(np.sum(y_hat-y))
        #  ToDO update the self.W and self.b using the learning rate and the values for dW and db
        self.W = self.W - self.lr * dW
        self.b = self.b - self.lr * db


    def fit(self, X, y):
        """

        :param X: The training set
        :param y: The true output of the training set
        :return:
        """
        self.__init_weights(X)
        y_hat = self.predict(X)
        loss = self.__loss(y, y_hat)
        print(f"Initial Loss: {loss}")
        for i in range(self.iterations + 1):
            self.__sgd(X, y, y_hat)
            y_hat = self.predict(X)
            loss = self.__loss(y, y_hat)
            if not i % 100:
                print(f"Iteration {i}, Loss: {loss}")

    def predict(self, X):
        """

        :param X: The training dataset
        :return:
            y_hat: the predicted output
        """
        #ToDO calculate the predicted output y_hat. remember the function of a line is defined as y = WX + b
        y_hat = (np.dot(self.W, X.T) + self.b).T
        return y_hat


    def save_model(self):
        """
        output model Weight, loss, and build timestamp in pickle file
        should be save into a gcs bucket, but we save to local for now.

        """
        output_dict = {
            "W": self.W,
            "b": self.b,
            "loss": self.losses[-1],
            "save_timestamp": time.time()
            }
        
        date_time = datetime.utcfromtimestamp(output_dict['save_timestamp']).strftime('%Y-%m-%d_%H_%M_%S')

        with open(f'model/model_weight_{date_time}.pickle', 'wb') as fp:
            pickle.dump(output_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
        
        model_file_name = f'model_weight_{date_time}'
        print("model saved: ", f'model/{model_file_name}.pickle')
        return model_file_name

        
    def load_model(self, saved_file):
        """
        :param output_dict: save_model output file name
        load W, b with saved weights
        """
        with open(f'model/{saved_file}', 'rb') as fp:
            output_dict = pickle.load(fp)
            
        self.W = output_dict['W']
        self.b = output_dict['b']




if __name__ == "__main__":
    X_train, y_train, X_test, y_test = generate_data()
    model = SimpleLinearRegression()
    model.fit(X_train,y_train)
    predicted = model.predict(X_test)
    evaluate(model, X_test, y_test, predicted)
