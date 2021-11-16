

"""this module contains the libraries that I have created
1- Linear regression gradient descent
2- logistic regression
3- KNN classifier

by : Ahmed Nabil Awaad
linkedin: https://www.linkedin.com/in/ahmed-n-awaad/
"""

# ----------------------------------------------------------------
# ---------------------Importing libraries -----------------------
# ----------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from numpy.linalg import norm


# ----------------------------------------------------------------
# ------------------------logistic Regression class --------------
# ----------------------------------------------------------------

class Logistic_Regression_Batch_GD:

    def __init__(self,alpha=.01, iterations=1000, threshold = 0.000001):
        """Logistic regression

        Parameters
        ----------
        alpha : float, optional
            [description], by default .01
        iterations : int, optional
            [description], by default 1000
        threshold : float, optional
            [description], by default 0.000001
        """
        self.alpha = alpha
        self.iterations = iterations
        self.threshold = threshold
        self.features_matrix = None
        self.y_vec =None

        # make temp lists Just for visualization
        self.cost_lst = []
        self.GD_list = []
        self.all_theta_vecs = []
        self.h_theta_lst =[]


    def prepare_features_matrix(self):
        if self.features_matrix.ndim == 1:
            self.features_matrix = self.features_matrix[:,np.newaxis]
        self.features_matrix = np.insert(self.features_matrix, 0 , np.ones(self.features_matrix.shape[0]), axis = 1)
        return self.features_matrix


    def fit(self, features_matrix, y_vec):
        """fit x vs. y

        Parameters
        ----------
        features_matrix : [type]
            [description]
        y_vec : [type]
            [description]

        Returns
        -------
        numpy array
            [description]
        """
        self.features_matrix = features_matrix
        self.y_vec = y_vec
        self.features_matrix = self.prepare_features_matrix()
        # initial parameters
        self.theta_vec = np.zeros(self.features_matrix.shape[1])

        for _ in range(self.iterations):
            # appending theta vector to list
            self.all_theta_vecs.append(self.theta_vec)
            # Calculate the predicted output
            z = self.features_matrix @ self.theta_vec
            # Calculate the P(y=1|X;0)
            self.h_theta = (1 / (1+ np.exp(-z)))
            self.h_theta_lst.append(self.h_theta)
            # # Calculate loss
            error_vec = self.h_theta - self.y_vec
            cost = -np.mean((self.y_vec * np.log(self.h_theta)) + ((1- self.y_vec)*( 1- np.log(self.h_theta)) ) )
            self.cost_lst.append(cost)
            # Calculate gradient descent
            GD = np.array((1/self.y_vec.shape[0])*(self.features_matrix.T @ error_vec))
            self.GD_list.append(round(np.linalg.norm(GD),8))
            # Update theta vector
            self.theta_vec = self.theta_vec - self.alpha * GD

            # # # Stop conditions
            if len(self.cost_lst) > 2:
                    # Check minimum gradient
                if np.linalg.norm(self.GD_list[-1]) < self.threshold or np.linalg.norm(self.GD_list[-1]) > 1e100 :
                    break
                # Check cost
                elif abs(self.cost_lst[-1] - self.cost_lst[-2]) < self.threshold :
                    break
                # Check theta
                elif abs(np.linalg.norm(self.theta_vec) - np.linalg.norm(self.all_theta_vecs[-1])) < self.threshold:
                    break
        return self.theta_vec


    def plot_GD(self):
        plt.figure(figsize=(10,7))
        plt.plot(self.GD_list,
                            color='blue',linewidth='1',linestyle='--',marker='o',markersize='2',alpha=0.7)
        plt.xlabel("Number of iterations")
        plt.ylabel("Gradient")
        plt.show()

    def plot_loss(self):
        plt.figure(figsize=(10,7))
        plt.plot(self.cost_lst,
                            color='blue',linewidth='1',linestyle='--',marker='o',markersize='2',alpha=0.7)
        plt.xlabel("Number of iterations")
        plt.ylabel("Loss")
        plt.show()


    def predict(self, X):
        h_theta_final = X @ self.theta_vec[1:] + self.theta_vec[0]
        predictions = np.where(h_theta_final <= 0, 0, 1)
        return predictions


    def plot_prediction_prob_hist(self,X, bins=20):
        h_theta_final = X @ self.theta_vec[1:] + self.theta_vec[0]
        prob_y = 1/(1 + np.exp(1 - h_theta_final))
        plt.hist(prob_y, bins = bins)
        plt.show()

    def accuracy(self, X,y):
        predictions = self.predict(X)
        acc = np.mean(predictions == y)
        return acc



# ----------------------------------------------------------------
# ------------------------linear Regression class --------------
# ----------------------------------------------------------------




class Linear_Regression_Batch_GD:

    def __init__(self,alpha=.01, iterations=1000, threshold = 0.000001):
        """Linear regression

        Parameters
        ----------
        alpha : float, optional
            [description], by default .01
        iterations : int, optional
            [description], by default 1000
        threshold : float, optional
            [description], by default 0.000001
        """
        self.alpha = alpha
        self.iterations = iterations
        self.threshold = threshold
        self.features_matrix = None
        self.y_vec =None

        # make temp lists Just for visualization
        self.cost_lst = []
        self.GD_list = []
        self.all_theta_vecs = []
        self.h_theta_lst =[]


    def prepare_features_matrix(self):
        if self.features_matrix.ndim == 1:
            self.features_matrix = self.features_matrix[:,np.newaxis]
        self.features_matrix = np.insert(self.features_matrix, 0 , np.ones(self.features_matrix.shape[0]), axis = 1)
        return self.features_matrix


    def fit(self, features_matrix, y_vec):
        """fit x vs. y

        Parameters
        ----------
        features_matrix : [type]
            [description]
        y_vec : [type]
            [description]

        Returns
        -------
        numpy array
            [description]
        """
        self.features_matrix = features_matrix
        self.y_vec = y_vec
        self.features_matrix = self.prepare_features_matrix()
        # initial parameters
        self.theta_vec = np.zeros(self.features_matrix.shape[1])

        for _ in range(self.iterations):

            # appending theta vector to list
            self.all_theta_vecs.append(self.theta_vec)

            # Calculate the predicted output
            self.h_theta = self.features_matrix @ self.theta_vec
            self.h_theta_lst.append(self.h_theta)

            # Calculate loss
            error_vec = self.h_theta - self.y_vec
            cost = ((np.linalg.norm(error_vec)) ** 2) / y_vec.shape[0]
            self.cost_lst.append(cost)

            # Calculate gradient descent
            GD = np.array((1/self.y_vec.shape[0])*(self.features_matrix.T @ error_vec))
            self.GD_list.append(round(np.linalg.norm(GD),8))

            # Update theta vector
            self.theta_vec = self.theta_vec - self.alpha * GD


            # Stop conditions
            if len(self.cost_lst) > 2:
                # Check minimum gradient
                if np.linalg.norm(self.GD_list[-1]) < self.threshold or np.linalg.norm(self.GD_list[-1]) > 1e100 :
                    break
                # Check cost
                elif abs(self.cost_lst[-1] - self.cost_lst[-2]) < self.threshold :
                    break
                # Check theta
                elif abs(np.linalg.norm(self.theta_vec) - np.linalg.norm(self.all_theta_vecs[-1])) < self.threshold:
                    break

        return self.theta_vec


    def plot_GD(self):
        plt.figure(figsize=(10,7))
        plt.plot(self.GD_list,
                            color='blue',linewidth='1',linestyle='--',marker='o',markersize='2',alpha=0.7)
        plt.xlabel("Number of iterations")
        plt.ylabel("Gradient")
        plt.show()

    def plot_loss(self):
        plt.figure(figsize=(10,7))
        plt.plot(self.cost_lst,
                            color='blue',linewidth='1',linestyle='--',marker='o',markersize='2',alpha=0.7)
        plt.xlabel("Number of iterations")
        plt.ylabel("Loss")
        plt.show()


    def predict(self, X):
        predictions = X @ self.theta_vec[1:] + self.theta_vec[0]
        return predictions


    def plot_prediction_prob_hist(self,X, bins=20):
        h_theta_final = X @ self.theta_vec[1:] + self.theta_vec[0]
        prob_y = 1/(1 + np.exp(1 - h_theta_final))
        plt.hist(prob_y, bins = bins)
        plt.show()

    def r2_score(self, X,y):
        predictions = self.predict(X)
        r2 =r2_score(y, predictions)
        return r2

    def plot_all(self, title ="Title", size = (22,5)):
        fig,axes = plt.subplots(1,4,figsize=size)
        plt.suptitle(title,fontsize=20);
    #-----------------------------------------------------------------------#
        axes[0].plot(np.arange(len(self.GD_list)),self.GD_list, label ="alpha = {}".format(self.alpha),
                                color='blue',linewidth='1',linestyle='--',marker='o',markersize='2',alpha=0.7)
        axes[0].set_xlabel("Number of iterations")
        axes[0].set_ylabel("Gradient")
        axes[0].legend()
    #-----------------------------------------------------------------------#
        axes[1].plot(np.array(self.all_theta_vecs)[:,0],self.cost_lst, label ="alpha = {}".format(self.alpha),
                            color='red',linewidth='1',linestyle='--',marker='o',markersize='2',alpha=0.7)
        axes[1].set_xlabel("theta0")
        axes[1].set_ylabel("Loss")
        axes[1].legend()
    #-----------------------------------------------------------------------#
        axes[2].plot(np.array(self.all_theta_vecs)[:,1],self.cost_lst, label ="alpha = {}".format(self.alpha),
                            color='red',linewidth='1',linestyle='--',marker='o',markersize='2',alpha=0.7)
        axes[2].set_xlabel("theta1")
        axes[2].set_ylabel("Loss")
        axes[2].legend()
    #-----------------------------------------------------------------------#
        axes[3].plot(np.arange(len(self.cost_lst)), self.cost_lst, label ="alpha = {}".format(self.alpha),
                            color='g',linewidth='1',linestyle='--',marker='o',markersize='2',alpha=0.7)
        axes[3].set_xlabel("Number of iterations")
        axes[3].set_ylabel("Loss")
        axes[3].legend()





# ----------------------------------------------------------------
# ------------------------KNN Classifier class -------------------
# ----------------------------------------------------------------



class KNNClassifier:
    def __init__(self, p=2, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.p = p
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    # Get the most similar neighbors
    def get_neighbors(self, test_row):
        distances = np.power(np.power(test_row - self.X_train, self.p).sum(axis=1), 1/self.p)
        dist_and_y = np.concatenate((distances.reshape(-1,1), self.y_train.reshape(-1,1)), axis=1)
        dist_and_y = dist_and_y[dist_and_y[:, 0].argsort()]
        neighbors = dist_and_y[:, 1][:self.n_neighbors]
        return neighbors

    # Make a classification prediction with neighbors
    def predict(self, X_test):
        predictions = np.empty(X_test.shape[0])
        for i, test_row in enumerate(X_test):
            neighbors = self.get_neighbors(test_row)
            unique, counts = np.unique(neighbors, return_counts=True)
            prediction = unique[np.argmax(counts)]
            predictions[i] =prediction
        return np.array(predictions)








# ----------------------------------------------------------------
# ------------------------Testing --------------------------------
# ----------------------------------------------------------------
def main():
    x = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
    y = np.array([0,0,0,0,0,0,1,1,1,1,1,1,])
    my_model = Logistic_Regression_Batch_GD(alpha=.5)
    my_model.fit(x, y)
    my_model.plot_GD()
    my_model.plot_loss()

if __name__ == '__main__':
    main()