from algorithm import GBDT
from data import DataSet


model = GBDT(tree_depth=3, learning_rate=0.01, max_iter=2000)
dataset = DataSet('data/haberman.csv', 'Survival')
model.fit(dataset)
x = {'Age': 30, 'Year': 60, 'Nodes': 2}
print(model.predict(x))
