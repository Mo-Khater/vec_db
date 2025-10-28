from typing import Dict, List, Annotated
import numpy as np
import os
import faiss
import math

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70

class VecDB:
    def __init__(self, database_file_path = "saved_db.dat", index_file_path = "index.dat", new_db = True, db_size = None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # delete the old DB file if exists
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database(db_size)
    
    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)
        self._build_index()

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, 70)]):
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        #TODO: might change to call insert in the index, if you need
        self._build_index()

    def get_one_row(self, row_num: int) -> np.ndarray:
        # This function is only load one row in memory
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            return f"An error occurred: {e}"

    def get_all_rows(self) -> np.ndarray:
        # Take care this load all the data in memory
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)
    
    def retrieve(self, query, top_k=5):
        index = faiss.read_index(self.index_path)
        query = query.astype(np.float32)
        faiss.normalize_L2(query)
        _, indices = index.search(query, top_k)
        return indices[0].tolist()
        
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity
    def _determine_n_clusters(self, n_vectors):
            # Resources:
            # https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
            # https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
            # https://github.com/facebookresearch/faiss/wiki/The-index-factory#ivf-indexes
            # https://www-users.cse.umn.edu/~kumar001/papers/high_dim_clustering_19.pdf
            # Heuristic: n_clusters = 2 ^ ceil(log2(sqrt(n_vectors))) bounded between 256 and 4096
            if n_vectors < 256:
                n_clusters = max(1, n_vectors // 4)
                n_clusters = 2 ** math.ceil(math.log2(n_clusters))
                return max(1, n_clusters)
            else:
                base = math.sqrt(n_vectors)
                n_clusters = int(base)  
                n_clusters = max(256, min(n_clusters, 4096))
                n_clusters = 2 ** math.ceil(math.log2(n_clusters))
                return min(n_clusters, n_vectors)
            
    def _build_index(self):
        quantizer = faiss.IndexFlatL2(DIMENSION)
        num_records = self._get_num_records()
        # nlist = max(1, int(np.sqrt(num_records)))  # instead of num_records/100
        nlist = self._determine_n_clusters(num_records)
        index = faiss.IndexIVFFlat(quantizer, DIMENSION, nlist, faiss.METRIC_L2)

        vectors = self.get_all_rows().astype(np.float32)
        faiss.normalize_L2(vectors)  # normalize for cosine-like search

        index.train(vectors)
        index.add(vectors)
        # index.nprobe = 64  # increase probe count
        n_probes = max(4, min(32, nlist // 16))
        index.nprobe = n_probes
        faiss.write_index(index, self.index_path)


