import numpy as np

aa = np.load("../predict_results_new/GCN-test.npy")
print(aa)
aa2 = np.float32(aa)
print(aa2)

new_test = np.delete(aa2, 5735)
print(new_test.shape)


bb = np.load("../predict_results_new/GCN-pred.npy")
bb2 = np.delete(bb, 5735)
print(bb2.shape)

np.save("../predict_results_new/GCN-pred_1.npy", bb2)
np.save("../predict_results_new/GCN-test_1.npy", new_test)

