from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
import pandas as pd

def save_decision_trees_as_dot(clf, iteration, feature_name, target_name):
    file_name = open("emirhan_project_planet" + str(iteration) + ".dot",'w')
    dot_data = export_graphviz(
        clf,
        out_file=file_name,
        feature_names=feature_name,
        class_names=target_name,
        rounded=True,
        proportion=False,
        precision=2,
        filled=True,)
    file_name.close()
    print("Decision Tree in forest :) {} saved as dot file".format(iteration + 1))


df = pd.read_csv('planets_large_data.csv')

X= df.drop(['Planet'], axis = 'columns')
#print(X)
y= df.drop(['Distance From The Sun','Confirmed Moons','Provisional Moons','Total Moons','(Volume/1000000000-cubic km)','Diameter of Planet(km)'], axis= 'columns')
#print(y)

y_data = LabelEncoder()
#LabelEncoder() function :))

y['Planet_Data'] = y_data.fit_transform(y['Planet'])
# Planet Columns value change to Planet_Data with fit_transform function

#print(connects)

y_n = y.drop(['Planet'],axis='columns')
#New Columns of Target :))

# In additionnn: print(y_n)


feature_names = X.columns
#a few fetaure names..

target_names = y_n.columns
# one of the columns is target name :)

model = RandomForestClassifier(n_estimators=1)

# our model like to above :)

model.fit(X,y_n)
#our model training to the above...

#print(model.estimators_[2])

#The collection of fitted sub-estimators = estimators_

for i in range(len(model.estimators_)):
    save_decision_trees_as_dot(model.estimators_[i], i, feature_names, target_names)
    print(i)


#prediction is the PLANET!


Distance_From_The_Sun = int(input("Enter to Distance From The Sun: "))
Confirmed_Moons = int(input("Enter to Confirmed Moons: "))
Provisional_Moons = int(input("Enter to Provisional Moons: "))
Total_Moons = int(input("Enter to Total Moons: "))
Volume_1000000000_cubic_km = int(input("Enter to Volume (Enter the state / 1.000.000.000) - Cubic (km) : "))
Diameter_of_Planet_km = int(input("Enter to Diameter Of Planet (km): "))

try:
    while True:
        model_run = model.predict([[Distance_From_The_Sun,Confirmed_Moons, Provisional_Moons, Total_Moons, Volume_1000000000_cubic_km, Diameter_of_Planet_km]])
        planets = pd.read_csv('planets_name.csv',index_col=None, na_values=None)
        planet_detect_algorithm = planets.columns.values[model_run]
        print("Predicted Planet: {}".format(planet_detect_algorithm))
        break

except:
    print("Try again!")
#print(model.predict([[predict_2014,predict_2020,predict_population]]))
