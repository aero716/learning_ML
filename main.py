import matplotlib.pylab as plt
from pybrain3.tools.shortcuts import buildNetwork
from pybrain3.datasets import SupervisedDataSet
from pybrain3.supervised.trainers import BackpropTrainer

net = buildNetwork(2, 3, 1)
y = net.activate([2, 1])

ds = SupervisedDataSet(2,1)
ds.addSample((0, 0), (0,))
ds.addSample((0, 1), (1,))
ds.addSample((1, 0), (1,))
ds.addSample((1, 1), (0,))
print(ds)

trainer = BackpropTrainer(net)
trner, valerr = trainer.trainUntilConvergence(dataset=ds, maxEpochs=1000)
plt.plot(trner, 'b', valerr, 'r')
plt.show()
print('test')