import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sk_pca
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap

from scipy.signal import savgol_filter, medfilt
import seaborn as sns


data = pd.read_csv('./data/pbs.csv')

# The last column of the Data Frame contains the labels
lab = data.values[1:-1,-1].astype('uint8') 
# потому что для теста ниже взяты первый и последний спектры test = nfeat1[[0,-1],:]
        
# Read the features (scans) 
feat = data.values[1:-1,1:-1]

# Calculate first derivative applying a Savitzky-Golay and median filters
feat = medfilt(feat, kernel_size=5)
feat = savgol_filter(feat, window_length=25, polyorder=3, deriv=0)
dfeat = savgol_filter(feat, window_length=25, polyorder=3, deriv=1)


# Исходные спектры
x = range(800, 1901, 2)
na = data.to_numpy()
for i in range(len(feat)):
    plt.plot(x, na[i,1:-1])
plt.title('Normalized originals')
plt.show()

for i in range(len(feat)):
    plt.plot(x, feat[i,:])
plt.title('Filtered originals')
plt.show()

for i in range(len(feat)):
    plt.plot(x, dfeat[i,:])
plt.title('Filtered derivatives')
plt.show()




# Scale the features to have zero mean and standard devisation of 1
# This is important when correlating data with very different variances
nfeat1 = StandardScaler().fit_transform(feat)
test1 = nfeat1[[0,-1],:] # для теста - первый и последний, точно разные вещества


nfeat2 = StandardScaler().fit_transform(dfeat)
test2 = nfeat2[[0,-1],:]

 # сколько компонент имеет смысл?

# Initialise
skpca1 = sk_pca(n_components=10)
skpca2 = sk_pca(n_components=10)

# Fit the spectral data and extract the explained variance ratio
X1 = skpca1.fit(nfeat1)
expl_var_1 = X1.explained_variance_ratio_
 
# Fit the first data and extract the explained variance ratio
X2 = skpca2.fit(nfeat2)
expl_var_2 = X2.explained_variance_ratio_

# Plot data
with plt.style.context(('ggplot')):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9,6))
    fig.set_tight_layout(True)
 
    ax1.plot(expl_var_1,'-o', label="Explained Variance %")
    ax1.plot(np.cumsum(expl_var_1),'-o', label = 'Cumulative variance %')
    ax1.set_xlabel("PC number")
    ax1.set_title('Luminescence data')
 
    ax2.plot(expl_var_2,'-o', label="Explained Variance %")
    ax2.plot(np.cumsum(expl_var_2),'-o', label = 'Cumulative variance %')
    ax2.set_xlabel("PC number")
    ax2.set_title('First derivative data')
 
    plt.legend()
    plt.show()
    

# Running the Classification of NIR spectra using Principal Component Analysis

#подставить смотря что считается - спектр nfeat1 или производная nfeat2 и соответствующий тест:
nfeat = nfeat2
test = test2


skpca2 = sk_pca(n_components=2) #3
 
# Transform on the scaled features
Xt2 = skpca2.fit_transform(nfeat)

# Define the labels for the plot legend
labplot = ["PbS1060","PbS1640"]
 
# Scatter plot
unique = list(set(lab))
colors = [plt.cm.jet(float(i)/max(unique)) for i in unique]
with plt.style.context(('ggplot')):
    for i, u in enumerate(unique):
        col = np.expand_dims(np.array(colors[i]), axis=0)
        xi = [Xt2[j,0] for j in range(len(Xt2[:,0])) if lab[j] == u]
        yi = [Xt2[j,1] for j in range(len(Xt2[:,1])) if lab[j] == u]
        plt.scatter(xi, yi, c=col, s=60, edgecolors='k',label=str(u))
 
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(labplot,loc='lower right')
    plt.title('Principal Component Analysis')
    plt.show()
    
# layered kernel density estimate (KDE)  
# https://nirpyresearch.com/nir-data-correlograms-seaborn-python/     
df = pd.DataFrame(Xt2, columns=['PC1', 'PC2'])
#df = pd.DataFrame(Xt2, columns=['PC1', 'PC2', 'PC3'])

y_train = lab.T.astype("int")

df["QD"] = y_train
df['QD'] = df['QD'].replace([0], 'PbS1060')
df['QD'] = df['QD'].replace([1], 'PbS1640')
sns.pairplot(df, hue="QD", palette='OrRd')
plt.show()


# лолитическая регрессия - классификация и предсказание
#  https://www.geeksforgeeks.org/principal-component-analysis-with-python/


# Fitting Logistic Regression To the training set
classifier = LogisticRegression(random_state = 0)
classifier.fit(Xt2, y_train)


# Predicting the test set result 
x_test = skpca2.transform(test)
y_pred = classifier.predict(x_test)
y_test = lab[[0, -1]]

# Making the confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
 
cm = confusion_matrix(y_test, y_pred)
print('Оценка совпадения предсказаний: ', cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
disp.plot()
plt.title('ConfusionMatrixDisplay (Test set)')
plt.show()

# Predicting the training set result

x_set, y_set = Xt2, y_train

X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1,
                     stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1,
                     stop = x_set[:, 1].max() + 1, step = 0.01))
  
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),
             X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,
             cmap = ListedColormap(('yellow', 'white', 'aquamarine')))
  
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
  
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = 'Train ' + str(j))
                
#  добавим "проверочные" точки на график
for i, j in enumerate(np.unique(y_test)):
    plt.scatter(x_test[y_test == j, 0], x_test[y_test == j, 1],
                c = ListedColormap(('lightcoral', 'lime'))(i), label = 'Test ' + str(j))

  
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1') # for Xlabel
plt.ylabel('PC2') # for Ylabel
plt.legend() # to show legend
  
# show scatter plot
plt.show()


# еще можно выделить какая именно часть вносит вклад в компоненты
# https://nirpyresearch.com/pca-correlation-circle/