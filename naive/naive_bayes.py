
# coding: utf-8

# # Naive Bayes
# 
# Os metodos de Naive Bayes são uma coleção de algoritmos que se baseam no teorema de Baye, com a suposição "naive".
# 
# <img src="Bayes-Theorem-Formula-Defined.jpeg">
# 
# 
# 

# ## Definindo lista de variáveis.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# get_ipython().run_line_magic('matplotlib', 'inline')
%matplotlib tk

# In[2]:


columns = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health', 'classification']
df = pd.read_csv('../data/nursery.data.csv', header=None, names=columns)


# In[3]:


df.describe()


# In[4]:


df.head()


# ## Separando a variável dependente.
# 

# In[5]:


X = pd.get_dummies(df.drop('classification', axis=1))


# ## Atribuição da variável dependente.

# In[6]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(df['classification'])


# ## Separando dados de teste e treino.

# In[7]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# ## Normalizando variáveis. Colocando todas em uma mesma escala para que a diferenças de grandezas não pese no resultado.

# In[8]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Instanciando o algoritmo "Gaussian Naive Bayes"

# In[9]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# In[10]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Printando o score.

# In[11]:


print(classifier.score(X_test, y_test))


# In[12]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[13]:


print(cm)

