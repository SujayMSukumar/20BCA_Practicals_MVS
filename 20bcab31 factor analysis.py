import os
import pandas as pd
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo

os.chdir("C:\Users\Sujay M S\Desktop\DATASETS")
data1=pd.read_csv('airlinedata.csv')
data1.drop(['Arrival Delay in Minutes'], axis=1, inplace=True)
data1.info()
data1.head()

x =data1[data1.columns[7:20]] 
fa = FactorAnalyzer()
fa.fit(x, 10)

chi_square_value,p_value=calculate_bartlett_sphericity(x)
print(chi_square_value, p_value)

kmo_all,kmo_model=calculate_kmo(x)
print(kmo_model)

ev, v = fa.get_eigenvalues()
print(ev)
plt.scatter(range(1,x.shape[1]+1),ev)
plt.plot(range(1,x.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()

fa = FactorAnalyzer(4, rotation='varimax')
fa.fit(x)
loads = fa.loadings_
print(loads)

print(fa.get_factor_variance())