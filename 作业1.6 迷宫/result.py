import json
import matplotlib.pyplot as plt
import argparse
import math

DynaQ = []
with open('DynaQBlocking.out','r') as f:
    lines = f.readlines()
    for i in range(3000):
        line = lines[i]
        a,b,c = line.split(' ')
        r = float(b)
        s = int(c)
        DynaQ.append(r)

DynaQplus = []
with open('DynaQplusBlocking.out','r') as f:
    lines = f.readlines()
    for i in range(3000):
        line = lines[i]
        a,b,c = line.split(' ')
        r = float(b)
        s = int(c)
        DynaQplus.append(r)

indexes = ["DynaQ","DynaQ+"]

fig,ax = plt.subplots()

plt.xlabel('Episodes')
plt.ylabel('Cumulative reward')

ax.plot(range(3000),DynaQ)
ax.plot(range(3000),DynaQplus)

fig.set_size_inches(14,8)
ax.legend(indexes,loc='upper right')
plt.savefig("BlockingReward.png")
plt.show()
