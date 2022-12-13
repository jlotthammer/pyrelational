"""
This file contains the implementation of a generic oracle interface for PyRelationAL
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pyrelational.data.data_manager import GenericDataManager


class GenericOracle(ABC):
    """An abstract class acting as an interface for implementing concrete oracles
    that can interact with a pyrelational pipeline"""

    def __init__(self):
        super(GenericOracle, self).__init__()

    def update_target_value(self, data_manager: GenericDataManager, idx: int, value: Any) -> None:
        """Update the target value for the observation denoted by the index

        :param data_manager: reference to the data_manager whose dataset we want to update
        :param idx: index to the observation we want to update
        :param value: value to update the observation with
        """
        data_manager.set_target_value(idx=idx, value=value)

    def update_target_values(self, data_manager: GenericDataManager, indices: List[int], values: List[Any]) -> None:
        """Updates the target values of the observations at the supplied indices

        :param data_manager: reference to the data_manager whose dataset we want to update
        :param indices: list of indices to observations whose target values we want to update
        :param values: list of values which we want to assign to the corresponding observations in indices
        """
        for idx, val in zip(indices, values):
            data_manager.set_target_value(idx=idx, value=val)

    @abstractmethod
    def update_dataset(self, data_manager: GenericDataManager, indices: List[int]) -> List[Any]:
        """
        This method serves to obtain labels for the supplied indices and update the
        target values in the corresponding observations of the data manager
        """
        pass
