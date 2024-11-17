import numpy as np
import pandas as pd
from pyspark import keyword_only
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
import faiss
from sklearn.metrics import euclidean_distances

from src.params.sampling._smogn import _KMeansParams, _SMOGNParams
from src.sampling.mixed_sampling.base import BaseMixedSampler
from src.utils.dataframe import get_num_cols, get_cat_cols


class DistributedSMOGN(BaseMixedSampler, _KMeansParams, _SMOGNParams):
    @keyword_only
    def __init__(self, label_col=None, sampling_strategy="balance", k_partitions=2, threshold=0.8, method="auto",
                 xtrm_type="both", coef=1.5, ctrl_pts_region=None, init_steps=2, tol=1e-4, max_iter=20, k_neighbours=5,
                 perturbation=0.02):
        super(DistributedSMOGN, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, label_col=None, sampling_strategy="balance", k_partitions=2, threshold=0.8, method="auto",
                  xtrm_type="both", coef=1.5, ctrl_pts_region=None, init_steps=2, tol=1e-4, max_iter=20, k_neighbours=5,
                  perturbation=0.02):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _partition(self, df, partition_col):
        feature_vector_col = "feature_vector"
        feature_vector_cols = get_num_cols(df)

        df = VectorAssembler(inputCols=feature_vector_cols, outputCol=feature_vector_col).transform(df)
        df = KMeans(featuresCol=feature_vector_col, predictionCol=partition_col, k=self.getKPartitions(),
                    initSteps=self.getInitSteps(), tol=self.getTol(), maxIter=self.getMaxIter()).fit(df).transform(df)

        df = df.drop(feature_vector_col)

        return df.repartition(self.getKPartitions(), partition_col)

    def _oversample(self, bump):
        schema = bump.samples.schema

        # Broadcast shared variables to all partitions
        sc = bump.samples.rdd.context

        cat_feature_cols = sc.broadcast(get_cat_cols(bump.samples.drop(self.getLabelCol())))
        num_feature_cols = sc.broadcast(get_num_cols(bump.samples.drop(self.getLabelCol())))
        label_col = sc.broadcast(self.getLabelCol())
        n_synth_samples = sc.broadcast(round(bump.sampling_percentage))
        k_neighbours = sc.broadcast(self.getKNeighbours())
        perturbation = sc.broadcast(self.getPerturbation())

        partition_col = "partition"
        bump.samples = self._partition(bump.samples, partition_col)

        def create_synth_samples(partition):
            synth_samples = self._create_synth_samples(
                partition=partition,
                cat_feature_cols=cat_feature_cols.value,
                num_feature_cols=num_feature_cols.value,
                label_col=label_col.value,
                n_synth_samples=n_synth_samples.value,
                k=k_neighbours.value,
                perturbation=perturbation.value
            )

            return pd.DataFrame(data=synth_samples)

        return bump.samples.groupby(partition_col).applyInPandas(create_synth_samples, schema=schema)

    def _undersample(self, bump):
        return super()._partition(bump.samples).sample(withReplacement=False, fraction=bump.sampling_percentage)

    def _create_synth_sample_SMOTE(self, base_sample, neighbour_sample, cat_feature_cols, num_feature_cols, label_col,
                                   base_sample_feature_vector, neighbour_sample_feature_vector):
        synth_sample_cat_features = {
            cat_feature_col: np.random.choice([base_sample[cat_feature_col], neighbour_sample[cat_feature_col]])
            for cat_feature_col in cat_feature_cols
        }

        synth_sample_num_features = {
            num_feature_col: base_sample[num_feature_col] + abs(
                (neighbour_sample[num_feature_col] - base_sample[num_feature_col])) * np.random.uniform(0, 1)
            for num_feature_col in num_feature_cols
        }

        synth_sample_feature_vector = np.asarray(list(synth_sample_num_features.values()))

        base_sample_dist = np.linalg.norm(synth_sample_feature_vector - base_sample_feature_vector)
        neighbour_sample_dist = np.linalg.norm(synth_sample_feature_vector - neighbour_sample_feature_vector)

        synth_sample_label = {
            label_col: (base_sample[label_col] + neighbour_sample[label_col]) / 2
            if base_sample_dist == neighbour_sample_dist
            else (neighbour_sample_dist * base_sample[label_col] + base_sample_dist * neighbour_sample[label_col]) / (
                    base_sample_dist + neighbour_sample_dist)
        }

        return {**synth_sample_cat_features, **synth_sample_num_features, **synth_sample_label}

    def _create_synth_sample_GN(self, base_sample, cat_feature_cols, num_feature_cols, label_col, cat_feature_probs,
                                num_feature_stds, label_std, perturbation):
        synth_sample_cat_features = {
            cat_feature_col: np.random.choice(list(cat_feature_probs[cat_feature_col].keys()),
                                              p=list(cat_feature_probs[cat_feature_col].values()))
            for cat_feature_col in cat_feature_cols
        }

        synth_sample_num_features = {
            num_feature_col: base_sample[num_feature_col] + np.random.normal(0, num_feature_stds[
                num_feature_col] * perturbation)
            for num_feature_col in num_feature_cols
        }

        synth_sample_label = {
            label_col: base_sample[label_col] + np.random.normal(0, label_std * perturbation)
        }

        return {**synth_sample_cat_features, **synth_sample_num_features, **synth_sample_label}

    def _create_synth_samples(self, partition, cat_feature_cols, num_feature_cols, label_col, n_synth_samples, k, perturbation):
        n_rows = len(partition.index)
        k = min(k, n_rows)
        feature_vectors = partition[[*num_feature_cols]].to_numpy()
        dist_matrix = euclidean_distances(feature_vectors, feature_vectors)
        neighbour_sample_index_matrix = np.delete(np.argsort(dist_matrix, axis=1), np.s_[k + 1:], axis=1)

        cat_feature_probs = {
            cat_feature_col: partition[cat_feature_col].value_counts(normalize=True).to_dict()
            for cat_feature_col in cat_feature_cols
        }
        num_feature_stds = partition[[*num_feature_cols]].std()
        label_std = partition[label_col].std()

        synth_samples = [None for _ in range(n_rows * n_synth_samples)]

        for base_sample_index, base_sample in partition.iterrows():
            dists = dist_matrix[base_sample_index]
            neighbour_sample_indices = neighbour_sample_index_matrix[base_sample_index]
            effective_k = min(k, len(neighbour_sample_indices))

            for n_synth_sample in range(n_synth_samples):
                neighbour_sample_index = np.random.choice(neighbour_sample_index_matrix[base_sample_index])
                neighbour_sample = partition.iloc[neighbour_sample_index]

                dist = dists[neighbour_sample_index]
                if effective_k > 1:
                    safe_dist = dists[neighbour_sample_indices[(effective_k + 1) // 2]] / 2
                else:
                    safe_dist = np.inf

                if dist < safe_dist:
                    synth_sample = self._create_synth_sample_SMOTE(
                        base_sample=base_sample,
                        neighbour_sample=neighbour_sample,
                        cat_feature_cols=cat_feature_cols,
                        num_feature_cols=num_feature_cols,
                        label_col=label_col,
                        base_sample_feature_vector=feature_vectors[base_sample_index],
                        neighbour_sample_feature_vector=feature_vectors[neighbour_sample_index]
                    )

                else:
                    synth_sample = self._create_synth_sample_GN(
                        base_sample=base_sample,
                        cat_feature_cols=cat_feature_cols,
                        num_feature_cols=num_feature_cols,
                        label_col=label_col,
                        cat_feature_probs=cat_feature_probs,
                        num_feature_stds=num_feature_stds,
                        label_std=label_std,
                        perturbation=min(safe_dist, perturbation)
                    )

                synth_samples[base_sample_index * n_synth_samples + n_synth_sample] = synth_sample

        return synth_samples
    
    #NEW
    # def _create_synth_samples(self, partition, cat_feature_cols, num_feature_cols, label_col, n_synth_samples, k, perturbation):
    #     n_rows = len(partition.index)
    #     k = min(k, n_rows)
    #     feature_vectors = partition[[*num_feature_cols]].to_numpy().astype('float32')
    #     nlist = int(np.sqrt(n_rows))
    #     quantizer = faiss.IndexFlatL2(feature_vectors.shape[1])
    #     index = faiss.IndexIVFFlat(quantizer, feature_vectors.shape[1], nlist, faiss.METRIC_L2)
    #     index.train(feature_vectors)
    #     index.add(feature_vectors)
    #     distances, indices = index.search(feature_vectors, k + 1)
    #     distances = distances[:, 1:]
    #     neighbour_sample_index_matrix = indices[:, 1:]

    #     cat_feature_probs = {
    #         cat_feature_col: partition[cat_feature_col].value_counts(normalize=True).to_dict()
    #         for cat_feature_col in cat_feature_cols
    #     }
    #     num_feature_stds = partition[[*num_feature_cols]].std()
    #     label_std = partition[label_col].std()

    #     synth_samples = [None for _ in range(n_rows * n_synth_samples)]

    #     for base_sample_index, base_sample in partition.iterrows():
    #         dists = distances[base_sample_index]
    #         neighbour_sample_indices = neighbour_sample_index_matrix[base_sample_index]
    #         effective_k = min(k, len(neighbour_sample_indices))

    #         for n_synth_sample in range(n_synth_samples):
    #             neighbour_idx_in_dists = np.random.randint(effective_k)  
    #             neighbour_sample_index = neighbour_sample_indices[neighbour_idx_in_dists]  
    #             neighbour_sample = partition.iloc[neighbour_sample_index]

    #             dist = dists[neighbour_idx_in_dists]
    #             if effective_k > 1:
    #                 safe_dist = dists[(effective_k + 1) // 2] / 2
    #             else:
    #                 safe_dist = np.inf

    #             if dist < safe_dist:
    #                 synth_sample = self._create_synth_sample_SMOTE(
    #                     base_sample=base_sample,
    #                     neighbour_sample=neighbour_sample,
    #                     cat_feature_cols=cat_feature_cols,
    #                     num_feature_cols=num_feature_cols,
    #                     label_col=label_col,
    #                     base_sample_feature_vector=feature_vectors[base_sample_index],
    #                     neighbour_sample_feature_vector=feature_vectors[neighbour_sample_index]
    #                 )

    #             else:
    #                 synth_sample = self._create_synth_sample_GN(
    #                     base_sample=base_sample,
    #                     cat_feature_cols=cat_feature_cols,
    #                     num_feature_cols=num_feature_cols,
    #                     label_col=label_col,
    #                     cat_feature_probs=cat_feature_probs,
    #                     num_feature_stds=num_feature_stds,
    #                     label_std=label_std,
    #                     perturbation=min(safe_dist, perturbation)
    #                 )

    #             synth_samples[base_sample_index * n_synth_samples + n_synth_sample] = synth_sample

    #     return synth_samples