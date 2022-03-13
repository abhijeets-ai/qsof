# https://medium.com/qiskit/introducing-qiskit-machine-learning-5f06b6597526
# https://github.com/Qiskit/qiskit-machine-learning
# https://qiskit.org/documentation/machine-learning/tutorials/01_neural_networks.html
# https://qiskit.org/documentation/machine-learning/tutorials/02_neural_network_classifier_and_regressor.html

# import the feature map and ansatz circuits
from qiskit.algorithms.optimizers import L_BFGS_B, COBYLA
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, TwoLocal
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.neural_networks import TwoLayerQNN
from qiskit.opflow import StateFn, PauliSumOp, AerPauliExpectation, ListOp, Gradient
from qiskit.providers.aer import QasmSimulator
from qiskit.utils import QuantumInstance
from qiskit import Aer
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


optim = L_BFGS_B()
device = QasmSimulator()
qi_sv = QuantumInstance(Aer.get_backend("aer_simulator_statevector"))


# X_train, y_train, X_test, y_test = ad_hoc_data(20, 10, 2, 0.1)
train = pd.read_csv('mock_train_set.csv')
X_train, y_train = train.iloc[:,:-1].values, train.iloc[:,-1].values.astype(int)
test = pd.read_csv('mock_test_set.csv')
X_test, y_test = test.iloc[:,:-1].values, test.iloc[:,-1].values.astype(int)

num_qubits = len(train.columns) - 1
print(f"Number of Features\t:\t{num_qubits}")
# Variational Quatum Classifier 2
aer = Aer.get_backend("aer_simulator")
aer.set_options(device='GPU')
sim = QuantumInstance(aer, shots=1024)
vqc = VQC(feature_map=ZZFeatureMap(num_qubits, reps=2),  # encoder circuit
          ansatz=RealAmplitudes(num_qubits, reps=5), # learnable ansatz for classification
          loss='cross_entropy',
          optimizer=COBYLA(),#L_BFGS_B(),
          quantum_instance=sim,
          callback=lambda weights, obj_val:  objective_func_vals.append(obj_val)

)
# one hot encode labels
ntr = len(X_train)
labels = np.zeros((ntr,2))
labels[range(ntr),y_train] = 1
nts = len(X_test)
y_labels = np.zeros((nts, 2))
y_labels[range(nts),y_test] = 1

# exp tracking
objective_func_vals = []
# training
vqc.fit(X_train, labels)
final_score = vqc.score(X_test, y_labels)
print(f'{final_score=}')
plt.xlabel("Steps")
plt.ylabel("obj_val")
plt.plot(range(len(objective_func_vals)), objective_func_vals)
plt.show()


# predictions
_, preds = np.where(vqc.predict(X_test)==1)
print(preds)