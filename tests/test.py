import pydactim as pyd

pyd.init(model_path="folder/to/model/folds", force=True)
print(pyd.get_model())
print(pyd.get_force())