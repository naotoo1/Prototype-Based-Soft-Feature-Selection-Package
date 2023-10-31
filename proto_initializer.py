"""Mean prototype-based iniatializer for sofes"""

from enum import Enum
from dataclasses import dataclass
import numpy as np
from sklearn.cluster import KMeans

class Initializer(str,Enum):
    KMEANS = 'Kmeans'


@dataclass
class MeanPrototypeInitializer:
    Prototypes:np.ndarray


def get_Kmeans_prototypes(
        input_data:np.ndarray,
        num_cluster:int,
        )->MeanPrototypeInitializer:
    self_supervised_model= KMeans(n_clusters=num_cluster)
    self_supervised_model.fit(input_data)
    prototypes= [
        np.append(v,i) for i,v in enumerate(
            self_supervised_model.cluster_centers_
        )
    ]
    return MeanPrototypeInitializer(
        Prototypes=prototypes
    )
