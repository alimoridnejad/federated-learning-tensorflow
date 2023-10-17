import tensorflow as tf


def log2(x):
    """
    use logarithmic properties: log2(x) = ln(x) / ln(2)
    e = sum 1 / n!
    :param x:
    :return:
    """
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator


class DcgMetric:
    """
    This class computes Discounted Cumulative Gain (DCG) which is a ranking
    quality measure. The premise of DCG is that relevance score is reduced
    logarithmically proportional to the position of the result.

    industry wide formula:
                    DCG@k = sum (2^rel_i - 1) / log2(i + 1)
    alternative:
                    DCG@k = sum rel_i / log2(i + 1)

    Note: Two formula give same results when relevance judgement is binary (0,1)

    CG: is sum of the relevance scores(usefulness) of all results in search list
    DCG: penalized highly relevant documents appearing lower in a search list
    Ref: https://en.wikipedia.org/wiki/Discounted_cumulative_gain

    limitations: does not penalize for bad documents and missing docs

    Args:
        k (int): truncation level (the first k document in the list)
        gain_type (str): 'exp2' or 'identity'

    Attributes:
        k (int):
        gain_type (str):
    """
    def __init__(self, k=tf.constant(10.), gain_type: str = 'exp2'):
        self.k = k
        if gain_type in ['exp2', 'identity']:
            self.gain_type = gain_type
        else:
            raise ValueError('gain type not equal to exp2 or identity')

    def evaluate(self, rels):
        """
        This method computes discounted cumulative gain (DCG) for a list of
        relevance scores ranked by a model.
        :param rels:
        :return:
        """
        gain = self._compute_gain(rels)

        # if k is larger than size of returned result, reduce k to that size
        discount = self._compute_discount(tf.cast(tf.minimum(self.k, tf.shape(gain)[-1]), tf.float32))

        return tf.reduce_sum(tf.math.divide(gain, discount))

    def _compute_gain(self, rels):
        """
        This method computes gains for a list of relevance scores ranked by a
        model.
        :param rels:
        :return:
        """

        # truncate up to k level
        rels_k = rels[:self.k]
        if self.gain_type == 'exp2':
            return tf.pow(2.0, rels_k) - 1.0
        else:
            return rels_k

    @staticmethod
    def _compute_discount(k):
        """
        This method computes discount factors for the k documents.
        :param k:
        :return:
        """

        i = tf.range(1, k + 1)
        discount = log2(i + 1.)

        return discount


class NdcgMetric(DcgMetric):
    """
    This class computes Normalized Discounted Cumulative Gain (NDCG) which is a
    ranking quality measure.

                        NDCG = DCG / IDCG

    where IDCG (also called maxDCG) is ideal discounted cumulative gain, and it
    is DCG for list of relevant documents (ordered by their relevance) in corpus
    up to position k.
    Ref: https://en.wikipedia.org/wiki/Discounted_cumulative_gain

    Args:
        k (int): truncation level (the first k document in the list)
        gain_type (str): 'exp2' or 'identity'

    Attributes:
        k (int):
        gain_type (str):
    """

    def __init__(self, k=tf.constant(10.), gain_type: str = 'exp2'):
        super(NdcgMetric, self).__init__(k, gain_type)

    def evaluate(self, rels):
        """
        This method computes normalized discounted cumulative gain (NDCG) for a
        list of relevance scores ranked by a model.
        :param rels:
        :return:
        """

        dcg = super(NdcgMetric, self).evaluate(rels)
        idcg = self.compute_idcg(rels)

        # queries that don't have any relevant documents,
        # the "goodness" of your ranking is always 0
        if idcg == 0.0:
            return 0.0

        return dcg / idcg

    def compute_idcg(self, rels):
        """
        This method computes the ideal discounted cumulative gain (IDCG), also
        called maxDCG. It is DCG for list of relevant documents (ordered by
        their relevance) in corpus up to position k.
        :param rels:
        :return:
        """
        # sort in descending order [3, 2, 1]
        ideal_rels = tf.sort(rels, direction="DESCENDING")
        return super(NdcgMetric, self).evaluate(ideal_rels)
