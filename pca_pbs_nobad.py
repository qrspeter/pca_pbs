import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sk_pca
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap

from scipy.signal import savgol_filter
import seaborn as sns


data = pd.read_csv('./data/pbs.csv')

# The last column of the Data Frame contains the labels
lab = data.values[[0,1,2,3,4,6,7,8,9,10,11],-1].astype('uint8') 
        
# Read the features (scans) 
feat = data.values[[0,1,2,3,4,6,7,8,9,10,11],1:-1]

# Calculate first derivative applying a Savitzky-Golay filter
feat = savgol_filter(feat, window_length=25, polyorder=3, deriv=0)
dfeat = savgol_filter(feat, window_length=25, polyorder=3, deriv=1)



#x = range(800, 1901, 2)
#for i in range(12):
#    plt.plot(x, feat[i,:])
#plt.show()
#
#x = range(800, 1901, 2)
#for i in range(12):
#    plt.plot(x, dfeat[i,:])
#plt.show()

#na = data.to_numpy()
#x = range(800, 1901, 2)
#for i in range(12):
#    plt.plot(x, na[i,1:-1])
#plt.show()

# Initialise
skpca1 = sk_pca(n_components=10)
skpca2 = sk_pca(n_components=10)

# Scale the features to have zero mean and standard devisation of 1
# This is important when correlating data with very different variances
nfeat1 = StandardScaler().fit_transform(feat)
test = nfeat1[3:7,:] # samples 6 & 7, некорректно из тренировочной базы брать, ну да ладно.
# можно взять для теста первую и последнюю, типа так - [[0,2], :]

#print(nfeat1)
#print(test)
#exit()

nfeat2 = StandardScaler().fit_transform(dfeat)

# Fit the spectral data and extract the explained variance ratio
X1 = skpca1.fit(nfeat1)
expl_var_1 = X1.explained_variance_ratio_
 
# Fit the first data and extract the explained variance ratio
X2 = skpca2.fit(nfeat2)
expl_var_2 = X2.explained_variance_ratio_

'''
# Plot data
with plt.style.context(('ggplot')):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9,6))
    fig.set_tight_layout(True)
 
    ax1.plot(expl_var_1,'-o', label="Explained Variance %")
    ax1.plot(np.cumsum(expl_var_1),'-o', label = 'Cumulative variance %')
    ax1.set_xlabel("PC number")
    ax1.set_title('Absorbance data')
 
    ax2.plot(expl_var_2,'-o', label="Explained Variance %")
    ax2.plot(np.cumsum(expl_var_2),'-o', label = 'Cumulative variance %')
    ax2.set_xlabel("PC number")
    ax2.set_title('First derivative data')
 
    plt.legend()
    plt.show()
'''    

# Running the Classification of NIR spectra using Principal Component Analysis

skpca2 = sk_pca(n_components=2) #3
 
# Transform on the scaled features
Xt2 = skpca2.fit_transform(nfeat1) # можно подставить nfeat2 и  получить то же самое для производных спектров

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

df["QD"] = lab.T.astype("int")
df['QD'] = df['QD'].replace([0], 'PbS1060')
df['QD'] = df['QD'].replace([1], 'PbS1640')
sns.pairplot(df, hue="QD", palette='OrRd')
plt.show()

# так, у нас есть преобразованный фрейм данных, теперь его по идее можно "обучать и разделять", хотя размер выборки преступно мал.


# после п.6 - https://www.geeksforgeeks.org/principal-component-analysis-with-python/
# добавить логистическую регрессию, и потом проверить на одном из спектров
# в п.6 обучение на всем массиве, а в п.7 можно давать тестовое значение отдельно

# Fitting Logistic Regression To the training set
classifier = LogisticRegression(random_state = 0)
classifier.fit(Xt2, lab.T.astype("int"))


# Predicting the test set result 
X_test = skpca2.transform(test)
y_pred = classifier.predict(X_test)
print('(псевдо)предсказание типов спектров 4-8: ', y_pred)

predicts = []
for i in y_pred:
    if i > 0.5:
        predicts.append('PbS1640')
    else:
        predicts.append('PbS1060')
print('(псевдо)предсказание типов спектров 4-8: ', predicts)

#(псевдо)предсказание типов спектров 4-8:  [0 0 1 1]
#(псевдо)предсказание типов спектров 4-8:  ['PbS1060', 'PbS1060', 'PbS1640', 'PbS1640']


# Predicting the training set result
X_set, y_set = Xt2, lab.T.astype("int")

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
                     stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1,
                     stop = X_set[:, 1].max() + 1, step = 0.01))
  
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),
             X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,
             cmap = ListedColormap(('yellow', 'white', 'aquamarine')))
  
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
  
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)

  
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1') # for Xlabel
plt.ylabel('PC2') # for Ylabel
plt.legend() # to show legend
  
# show scatter plot
plt.show()



# еще можно выделить какая именно часть вносит вклад в компоненты
# https://nirpyresearch.com/pca-correlation-circle/