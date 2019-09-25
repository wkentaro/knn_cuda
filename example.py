import numpy as np
import knn

from sklearn.neighbors import NearestNeighbors


np.random.seed(0)

c = 128

for n in range(4):
    query = np.random.rand(c, 1000).astype(np.float32)

    reference = np.random.rand(c, 4000).astype(np.float32)

    # Index is 1-based
    dist, ind = knn.knn(query.reshape(c, -1),
                        reference.reshape(c, -1), 2)

    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(ind - 1)

    knn2 = NearestNeighbors(n_neighbors=2)
    knn2.fit(reference.transpose(1, 0))
    dist2, ind2 = knn2.kneighbors(query.transpose(1, 0))

    print(ind2.transpose(1, 0))
    print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
