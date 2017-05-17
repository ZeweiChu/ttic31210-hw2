import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib

# plt.show()
# matplotlib.use("gtk")



data = pd.read_csv("data/1_5_c.txt", sep="\t")

x = data["log loss time"]
y = data["log loss acc"]
# plt.xticks(loglosstime)
# plt.yticks(loglossacc)
plt.plot(x, y, label='log loss')
x = data["hinge loss entire vocab time"]
y = data["hinge loss entire vocab acc"]
plt.plot(x, y, label="hinge loss entire vocab")
x = data["hinge loss r=10 time"]
y = data["hinge loss r=10 acc"]
plt.plot(x, y, label="hinge loss r=10")
plt.legend()
plt.xlabel("time")
plt.ylabel("accuracy")
plt.show()


