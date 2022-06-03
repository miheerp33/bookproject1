import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from pandas.plotting import scatter_matrix

HOUSING_PATH = 'https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv'
def load_housing_data(housing_path=HOUSING_PATH):
    return pd.read_csv(housing_path)

housing = load_housing_data()
# housing['income_cat'] = np.ceil(housing['median_income']/1.5)
# housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)
# split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# for train_index, test_index in split.split(housing, housing['income_cat']):
#     strat_train_set = housing.loc(train_index)
#     strat_test_set = housing.loc(test_index)
# housing['income_cat'].value_counts() / len(housing)
strat_train_set, strat_test_set = train_test_split(housing,test_size=0.20, random_state=1)

housing = strat_train_set.copy()
housing.plot(kind='scatter', x="longitude", y='latitude', alpha=0.1,s=housing['population']/100,
             label='population', c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True


             )
plt.legend()
#plt.show()
attributes = ["median_house_value", "median_income", "bedrooms_per_room", "housing_median_age"]
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
scatter_matrix(housing[attributes], figsize=(12,8))
plt.show()