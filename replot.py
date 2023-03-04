import pickle
import sys
import matplotlib.pyplot as plt
print(sys.argv[1])
fig = pickle.load(open(sys.argv[1], 'rb'))
plt.show()
