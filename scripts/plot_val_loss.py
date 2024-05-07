import numpy as np
import matplotlib.pyplot as plt

plt.style.use("slides")


iters, ppl = np.loadtxt("val_ppl.csv", delimiter=",")

plt.figure()
plt.title("Validation PPL Over Training")
plt.plot(iters, ppl, "o")
plt.xlabel("Training steps (log scale)")
plt.ylabel("Sliding-window Perplexity (log scale)")
plt.loglog()
plt.savefig("training_val.png")
