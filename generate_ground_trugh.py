from __future__ import absolute_import
import multiprocessing
import tqdm
import sklearn.model_selection
from multiprocessing.pool import ThreadPool
import psutil

from scipy.spatial.distance import pdist as scipy_pdist
from datasets import load_dataset

import numpy
import sklearn.neighbors


def pdist(a, b, metric):
    return scipy_pdist([a, b], metric=metric)[0]


def jaccard(a, b):
    if len(a) == 0 or len(b) == 0:
        return 0
    intersect = len(set(a) & set(b))
    return intersect / (float)(len(a) + len(b) - intersect)


metrics = {
    "hamming": {
        "distance": lambda a, b: pdist(a, b, "hamming"),
        "distance_valid": lambda a: True,
    },
    # return 1 - jaccard similarity, because smaller distances are better.
    "jaccard": {
        "distance": lambda a, b: 1 - jaccard(a, b),
        "distance_valid": lambda a: a < 1 - 1e-5,
    },
    "euclidean": {
        "distance": lambda a, b: pdist(a, b, "euclidean"),
        "distance_valid": lambda a: True,
    },
    "angular": {
        "distance": lambda a, b: pdist(a, b, "cosine"),
        "distance_valid": lambda a: True,
    },
}


class BaseANN(object):
    def done(self):
        pass

    def get_memory_usage(self):
        """Return the current memory usage of this algorithm instance
        (in kilobytes), or None if this information is not available."""
        # return in kB for backwards compatibility
        return psutil.Process().memory_info().rss / 1024

    def fit(self, X):
        pass

    def query(self, q, n):
        return []  # array of candidate indices

    def batch_query(self, X, n):
        """Provide all queries at once and let algorithm figure out
        how to handle it. Default implementation uses a ThreadPool
        to parallelize query processing."""
        pool = ThreadPool()
        self.res = pool.map(lambda q: self.query(q, n), X)

    def get_batch_results(self):
        return self.res

    def get_additional(self):
        return {}

    def __str__(self):
        return self.name


class BruteForce(BaseANN):
    def __init__(self, metric):
        if metric not in ("angular", "euclidean", "hamming"):
            raise NotImplementedError("BruteForce doesn't support metric %s" % metric)
        self._metric = metric
        self.name = "BruteForce()"

    def fit(self, X):
        metric = {"angular": "cosine", "euclidean": "l2", "hamming": "hamming"}[
            self._metric
        ]
        self._nbrs = sklearn.neighbors.NearestNeighbors(
            algorithm="brute", metric=metric
        )
        self._nbrs.fit(X)

    def query(self, v, n):
        return list(self._nbrs.kneighbors([v], return_distance=False, n_neighbors=n)[0])

    def query_with_distances(self, v, n):
        (distances, positions) = self._nbrs.kneighbors(
            [v], return_distance=True, n_neighbors=n
        )
        return zip(list(positions[0]), list(distances[0]))


class BruteForceBLAS(BaseANN):
    """kNN search that uses a linear scan = brute force."""

    def __init__(self, metric, precision=numpy.float32):
        if metric not in ("angular", "euclidean", "hamming", "jaccard"):
            raise NotImplementedError(
                "BruteForceBLAS doesn't support metric %s" % metric
            )
        elif metric == "hamming" and precision != numpy.bool:
            raise NotImplementedError(
                "BruteForceBLAS doesn't support precision"
                " %s with Hamming distances" % precision
            )
        self._metric = metric
        self._precision = precision
        self.name = "BruteForceBLAS()"

    def fit(self, X):
        """Initialize the search index."""
        if self._metric == "angular":
            # precompute (squared) length of each vector
            lens = (X**2).sum(-1)
            # normalize index vectors to unit length
            X /= numpy.sqrt(lens)[..., numpy.newaxis]
            self.index = numpy.ascontiguousarray(X, dtype=self._precision)
        elif self._metric == "hamming":
            # Regarding bitvectors as vectors in l_2 is faster for blas
            X = X.astype(numpy.float32)
            # precompute (squared) length of each vector
            lens = (X**2).sum(-1)
            self.index = numpy.ascontiguousarray(X, dtype=numpy.float32)
            self.lengths = numpy.ascontiguousarray(lens, dtype=numpy.float32)
        elif self._metric == "euclidean":
            # precompute (squared) length of each vector
            lens = (X**2).sum(-1)
            self.index = numpy.ascontiguousarray(X, dtype=self._precision)
            self.lengths = numpy.ascontiguousarray(lens, dtype=self._precision)
        elif self._metric == "jaccard":
            self.index = X
        else:
            # shouldn't get past the constructor!
            assert False, "invalid metric"

    def query(self, v, n):
        return [index for index, _ in self.query_with_distances(v, n)]

    def query_with_distances(self, v, n):
        """Find indices of `n` most similar vectors from the index to query
        vector `v`."""

        if self._metric != "jaccard":
            # use same precision for query as for index
            v = numpy.ascontiguousarray(v, dtype=self.index.dtype)

        # HACK we ignore query length as that's a constant
        # not affecting the final ordering
        if self._metric == "angular":
            # argmax_a cossim(a, b) = argmax_a dot(a, b) / |a||b| = argmin_a -dot(a, b)  # noqa
            dists = -numpy.dot(self.index, v)
        elif self._metric == "euclidean":
            # argmin_a (a - b)^2 = argmin_a a^2 - 2ab + b^2 = argmin_a a^2 - 2ab  # noqa
            dists = self.lengths - 2 * numpy.dot(self.index, v)
        elif self._metric == "hamming":
            # Just compute hamming distance using euclidean distance
            dists = self.lengths - 2 * numpy.dot(self.index, v)
        elif self._metric == "jaccard":
            dists = [pd[self._metric]["distance"](v, e) for e in self.index]
        else:
            # shouldn't get past the constructor!
            assert False, "invalid metric"
        # partition-sort by distance, get `n` closest
        nearest_indices = numpy.argpartition(dists, n)[:n]
        indices = [
            idx
            for idx in nearest_indices
            if pd[self._metric]["distance_valid"](dists[idx])
        ]

        def fix(index):
            ep = self.index[index]
            ev = v
            return (index, pd[self._metric]["distance"](ep, ev))

        return map(fix, indices)


num_cpus = multiprocessing.cpu_count()
print("Using {} CPUs".format(num_cpus))
dataset = load_dataset(
    "filipecosta90/dbpedia-openai-1M-text-embedding-3-large-3072d",
    # streaming=True,
    keep_in_memory=False,
    split="train"
    # , num_proc=num_cpus
)
print("finished streaming...")

_id = dataset["corpus"]["_id"][:1_000_000]
title = dataset["corpus"]["title"][:1_000_000]
text = dataset["corpus"]["text"][:1_000_000]
embedding = dataset["corpus"]["embedding"][:1_000_000]

print("finished downloading dataset")
#
# def download(src, dst):
#     if not os.path.exists(dst):
#         # TODO: should be atomic
#         print("downloading %s -> %s..." % (src, dst))
#         wget.download(src, dst)
#
#
# def calc_i(i, x, bf, test, neighbors, distances, count):
#     if i % 1000 == 0:
#         print("%d/%d..." % (i, len(test)))
#     res = list(bf.query_with_distances(x, count))
#     res.sort(key=lambda t: t[-1])
#     neighbors[i] = [j for j, _ in res]
#     distances[i] = [d for _, d in res]
#
#
# def calc(bf, test, neighbors, distances, count):
#     Parallel(n_jobs=multiprocessing.cpu_count(), require="sharedmem")(
#         delayed(calc_i)(i, x, bf, test, neighbors, distances, count)
#         for i, x in enumerate(test)
#     )
#
#
# def human_format(num):
#     magnitude = 0
#     while abs(num) >= 1000:
#         magnitude += 1
#         num /= 1000.0
#     # add more suffixes if you need them
#     return "%.0f%s" % (num, ["", "K", "M", "G", "T", "P"][magnitude])
#
#
# def write_output(train, test, fn, distance, point_type="float", count=100):
#     f = h5py.File(fn, "w")
#     f.attrs["type"] = "dense"
#     f.attrs["distance"] = distance
#     f.attrs["dimension"] = len(train[0])
#     f.attrs["point_type"] = point_type
#     print("train size: %9d * %4d" % train.shape)
#     print("test size:  %9d * %4d" % test.shape)
#     f.create_dataset("train", (len(train), len(train[0])), dtype=train.dtype)[:] = train
#     f.create_dataset("test", (len(test), len(test[0])), dtype=test.dtype)[:] = test
#     neighbors = f.create_dataset("neighbors", (len(test), count), dtype="i")
#     distances = f.create_dataset("distances", (len(test), count), dtype="f")
#     bf = BruteForceBLAS(distance, precision=train.dtype)
#
#     bf.fit(train)
#     calc(bf, test, neighbors, distances, count)
#     f.close()
#
#
# @click.command()
# @click.option("--train_size", default=1000000, help="Train size.")
# @click.option("--test_size", default=10000, help="Test size.")
# @click.option("--distance", default="cosine", help="distance metric.")
# def create_ds(train_size, test_size, distance):
#     dim = 512
#     total_vecs = train_size + test_size
#     file_limit = 409
#     vector_limit = 400 * 1000000
#     if total_vecs > vector_limit:
#         print("vector limit is larger than the dataset")
#         sys.exit(1)
#     pos = 0
#     print(
#         f"generating train set of size {train_size} and test set of size {test_size}. Fetching {total_vecs} embeddings."
#     )
#     X = np.zeros((total_vecs, dim), dtype=np.float32)
#
#     pbar = tqdm.tqdm(total=total_vecs)
#     file_n = 0
#     while pos < total_vecs:
#         filename = f"img_emb_{file_n}.npy"
#         url = f"https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-embeddings/images/{filename}"
#         download(url, filename)
#         img_emb = np.load(filename)
#         for row in img_emb:
#             X[pos] = row.astype(np.float32)
#             pbar.update(1)
#             pos = pos + 1
#             if pos >= total_vecs:
#                 break
#         file_n = file_n + 1
#         if file_n > file_limit:
#             print("vector limit is larger than the dataset")
#             sys.exit(1)
#
#     print("Splitting %d*%d into train/test" % (X.shape[0], X.shape[1]))
#     X_train, X_test = sklearn.model_selection.train_test_split(
#         X, test_size=test_size, random_state=1
#     )
#
#     human_size = human_format(train_size)
#     write_output(
#         train=np.array(X_train),
#         test=np.array(X_test),
#         fn=f"laion-img-emb-{dim}-{human_size}-{distance}.hdf5",
#         distance=distance,
#         point_type="float",
#         count=100,
#     )
#
#
# if __name__ == "__main__":
#     create_ds()
