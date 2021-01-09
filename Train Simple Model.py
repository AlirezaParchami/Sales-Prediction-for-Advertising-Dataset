import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

def read_file():
    data = pd.read_csv("./Dataset/advertising.csv")
    return data

def data_cleaning():
    global df
    print("=====>> Dataframe information")
    print(df.info(), end="\n\n")
    print("\n=====>> Dataframe head")
    print(df.head(n=5), end="\n\n")
    print("\n=====>> Boolean series if not null")
    bool_series = pd.notnull(df)
    print(bool_series, end="\n\n")
    df = df[bool_series]  # remove null elements
    print("%%%%%%%%%%% Data Cleaning Stage Finished Successfully", end="\n\n\n")

class feature_selection_methods:

    @staticmethod
    def SelectKBest_universal_selection(X,Y):
        test = SelectKBest(score_func=f_classif, k=2)
        fit = test.fit(X,Y)
        print("=====>> Fit Scores:", fit.scores_)
        return

    @staticmethod
    def PCA(X_train):
        pca = PCA(n_components=3)
        fit = pca.fit(X_train)
        ## show results
        print("=====>> Explained variances: ", fit.explained_variance_ratio_)
        print("=====>> Fit components: ", fit.components_)
        return fit.components_

def feature_selection(X_train):
    ## Feature Selection
    #feature_selection_methods.SelectKBest_universal_selection(X_train,Y)
    #X_train = feature_selection_methods.PCA(X_train)

    ## Data Transform
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    print("&&&&&&&&&&&&&&&&&&&", X_test.shape)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
    print("%%%%%%%%%%% Feature Selection Finished Successfully", end="\n\n\n")


def plot_figures():
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True)


def apply_model():
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    print("=====> intercept of model:", regressor.intercept_)
    print("=====> coefficients of model:", regressor.coef_)
    y_pred = regressor.predict(X_test)
    print("=====> MSE of model:", mean_squared_error(y_pred, y_test))
    results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    print(results)


df = read_file()
data_cleaning()

X = df.drop('Sales', axis=1)
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
## Since the size of dataset is not big enough, I prevent to define X_validation and Y_validation
X_train, X_test = feature_selection(X_train)

plot_figures()
apply_model()

plt.show()