from sklearn.ensemble import RandomForestRegressor



model = RandomForestRegressor()

params = {"criterion" : "mse"}

model.set_params(**params)

print model.get_params()

params = {"criterion" : "mae"}

model.set_params(**params)

print model.get_params()