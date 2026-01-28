from model import qmodel

qmodel = qmodel.QModel(lambda x: 0, [418, 418, 418, 418])
qmodel.load_model()
qmodel.save_model_hdf()