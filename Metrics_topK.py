import numpy as np

def topK(model, test_vec, k=10):
	inv_lst = np.unique(test_vec[:, 0])
	pred = {}
	for inv in inv_lst:
		if pred.get(inv, None) is None:
			pred[inv] = np.argsort(model.predict(inv))[-k:]

	intersection_cnt = {}
	for i in range(test_vec.shape[0]):
		if test_vec[i, 1] in pred[test_vec[i, 0]]:
			intersection_cnt[test_vec[i, 0]] = intersection_cnt.get(test_vec[i, 0], 0) + 1
	invPairs_cnt = np.bincount(np.array(test_vec[:, 0], dtype='int32'))

	precision_acc = 0.0
	recall_acc = 0.0
	for inv in inv_lst:
		precision_acc += intersection_cnt.get(inv, 0)/float(k)
		recall_acc += intersection_cnt.get(inv, 0)/float(invPairs_cnt[inv])

	return precision_acc/len(inv_lst), recall_acc/len(inv_lst)