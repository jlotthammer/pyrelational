"""
Active learning using marginal confidence uncertainty measure
between classes in the posterior predictive distribution to
choose which observations to propose to the oracle
"""
from pyrelational.informativeness import classification_margin_confidence
from pyrelational.strategies.classification.abstract_classification_strategy import (
    ClassificationStrategy,
)


class MarginalConfidenceStrategy(ClassificationStrategy):
    """Implements Marginal Confidence Strategy whereby unlabelled samples are scored and queried based on
    the marginal confidence for classification scorer"""

    def __init__(self):
        super(MarginalConfidenceStrategy, self).__init__()
        self.scoring_fn = classification_margin_confidence
