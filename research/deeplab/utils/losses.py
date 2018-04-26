from __future__ import division
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.losses.losses_impl import Reduction
from tensorflow.python.util.tf_export import tf_export


LARGE = 1e8
MAX_N_SEMANTIC_CLASSES = 21
SMALL = 1e-12

@tf_export("losses.spectral")
def spectral_loss(
        instance_labels,
        embeddings,
        instance_mask,
        scope=None,
        loss_collection=ops.GraphKeys.LOSSES,
        reduction=Reduction.SUM_BY_NONZERO_WEIGHTS,
        subsample_power=12,
        no_semantic_blocking=False,
        normalize=True,
        rebalance_classes=False,
        spherical_packing_radius=1.0):
    """
    Creates a spectral loss. Modified from tf.losses.softmax_cross_entropy.
    :param instance_labels:             `[batch_size, num_pixels]` target instance labels, int16 tensor.
    :param embeddings:                  `[batch_size, num_pixels, embedding_dim]` pixel embedding output of network, float tensor.
    :param instance_mask:               '[batch_size, num_pixels]' binary Tensor.
                                             1 = Semantic class of pixel has instances, learn pixel embedding.
                                            0 = Semantic class of pixel does not have instances, so do not learn embedding.
    :param scope:                       The scope for the operations performed in computing the loss.
    :param loss_collection:             Collection to which the loss will be added.
    :param reduction:                   Type of reduction to apply to loss.
    :param subsample_power:             Uniformly randomly sample 2**subsample_power many pixels per image when computing the loss.
                                        Must be power of 2 to avoid bias in tensorflow random sampling.
                                        If None, do not subsample.
    :param no_semantic_blocking:        If False, compute the loss for each semantic class independently.
    :param normalize:                   Normalize pixel embeddings to have l2 norm of 1.
    :param rebalance_classes:           If True, reweight semantic classes to have equal weight in loss function.
                                        Only applies if no_semantic_blocking=False
    :param spherical_packing_radius:    In (0, 1]. Between cluster embeddings have 0 loss when their inner products are
                                        less than (1 - spherical_packing_radius). Value of 1 corresponds to no packing
                                        (perfect cluster embeddings are orthogonal). Values closer to 0 correspond to
                                        more spherical packing.
    :return loss:                       Tensor of the same type as embeddings. If `reduction` is:
                                            `NONE` = shape [batch_size, subsample, subsample]
                                            Else   = Scalar
    Raises:
        ValueError:               If batch_size and num_pixels of `embeddings` and `instance_labels` do not match
                                  or if the shape of `instance_mask` is invalid or if `instance_mask` is None.  Also if
                                  `instance_labels` or `embeddings` is None.
    """
    assert isinstance(subsample_power, int) or subsample_power is None
    assert isinstance(spherical_packing_radius, float) and 0.0 < spherical_packing_radius <= 1.0
    print('Using spherical packing radius of {}'.format(spherical_packing_radius))
    assert 0 <= subsample_power <= 16 or subsample_power is None  # tradeoff between memory and variance.
    if instance_labels is None:
        raise ValueError("instance_labels must not be None.")
    if embeddings is None:
        raise ValueError("embeddings must not be None.")
    if instance_mask is None:
        raise ValueError("instance_mask must not be None.")  # TODO: Allow None, and set to no mask.
    
    with ops.name_scope(scope, "spectral_loss", (embeddings, instance_labels, instance_mask)) as scope:
        label_assertions = [tf.assert_greater_equal(tf.reduce_max(instance_labels), 0)]
        with tf.control_dependencies(label_assertions):
            embeddings.get_shape()[0:2].assert_is_compatible_with(instance_labels.get_shape())
            instance_labels.get_shape().assert_is_compatible_with(instance_mask.get_shape())

            if normalize:
                embeddings = tf.nn.l2_normalize(embeddings, axis=2)

            # Subsample pixels which are not masked (i.e. belong to a semantic class with instances)
            instance_mask = math_ops.cast(instance_mask, dtypes.float32)  # tf.multinomial only accepts float probabilities

            # Rebalance semantic classes
            # probs should be inverse semantic prevalences. so classes with half as many pixels will have twice the prob

            if subsample_power is not None:
                if rebalance_classes:
                    print('Rebalancing semantic classes')
                    semantic_labels = tf.cast(tf.floor(tf.divide(instance_labels, 1000)), tf.int32)  # See cityscapesscripts/preparation/json2instanceImg.py
                    semantic_onehot = tf.one_hot(semantic_labels, MAX_N_SEMANTIC_CLASSES)  # batch_size x subsample x N_SEMANTIC_CLASSES
                    semantic_counts = tf.reduce_sum(semantic_onehot, axis=1,
                                                    keepdims=True)  # batch_size x 1 x N_SEMANTIC_CLASSES
                    inverse_semantic_prevalence = tf.truediv(1.0, semantic_counts)
                    inverse_semantic_prevalence = tf.where(tf.is_inf(inverse_semantic_prevalence), tf.zeros_like(inverse_semantic_prevalence), inverse_semantic_prevalence)
                    sample_weights = tf.multiply(semantic_onehot, inverse_semantic_prevalence)  # batch_size x subsample x N_SEMANTIC_CLASSES
                    sample_weights = tf.reduce_sum(sample_weights, 2)  # batch_size x subsample
                    sample_weights = tf.multiply(sample_weights, instance_mask)
                    sample_weights = tf.divide(sample_weights, tf.reduce_sum(sample_weights, axis=1, keepdims=True))  # normalize probabilities. may be unnnecessary
                else:
                    sample_weights = tf.add(instance_mask, SMALL)
                subsample = 2 ** subsample_power
                logprob = tf.log(sample_weights)  # instance_mask*LARGE
                with tf.device("/cpu:0"):
                    sample_indices = tf.multinomial(logprob, subsample, output_dtype=tf.int32)  # batch_size x subsample
                # Check none of indices are out of bounds. Bug in tensorflow when values are very large or small:
                # See https://github.com/tensorflow/tensorflow/issues/2774
                index_assertions = [tf.assert_less(tf.reduce_max(sample_indices), instance_labels.get_shape()[1])]
                with tf.control_dependencies(index_assertions):
                    instance_labels = batch_gather(instance_labels, sample_indices)  # batch_size x subsample
                    embeddings = batch_gather(embeddings, sample_indices)  # batch_size x subsample x embedding_dim

            semantic_labels = tf.cast(tf.floor(tf.divide(instance_labels, 1000)),
                                      tf.int32)  # See cityscapesscripts/preparation/json2instanceImg.py
            if not no_semantic_blocking:
                print("Computing spectral loss only within semantic classes.")
                sample_weights = labels_to_adjacency(semantic_labels)  # batch_size x subsample x subsample
            else:
                sample_weights = 1.0

            # Compute A
            A = labels_to_adjacency(instance_labels)  # batch_size x subsample x subsample

            # Compute VV^T
            A_predicted = tf.matmul(embeddings, tf.transpose(embeddings, [0, 2, 1]))  # batch_size x subsample x subsample

            # Compute loss
            between_cluster_mask = tf.cast(~A, A_predicted.dtype)
            A_float32 = tf.cast(A, A_predicted.dtype)
            packed_error_between_cluster = \
                tf.maximum(tf.multiply(between_cluster_mask,
                                       (A_predicted - 1.0 + spherical_packing_radius)/spherical_packing_radius), 0.0)
            error_within_cluster = tf.multiply(A_float32, A_float32 - A_predicted)
            error = packed_error_between_cluster + error_within_cluster
            loss = tf.losses.mean_squared_error(error, tf.zeros_like(error), scope=scope, loss_collection=loss_collection,
                                                reduction=reduction, weights=sample_weights)
            return loss


def batch_gather(params, indices):
    """
    Perform different gather operation on each sample in batch. See tf.gather for more information.
    :param params:                   [nbatch x a x ...] tensor
    :param indices:                  [nbatch x b] tensor. indices[i] specifies the gather operation for params[i].
    :return gathered_params:         [nbatch x b x ...] tensor, where gathered_params[i] are gathered from params[i]
                                     according to indices[i].
    """
    batch_size, num_indices = indices.shape
    row_indices = tf.range(batch_size)  # nbatch
    row_indices = _tile_along_new_axis(row_indices, num_indices, axis=1)  # nbatch x b
    indices_nd = tf.stack([row_indices, indices], axis=2)  # nbatch x b x 2
    gathered_params = tf.gather_nd(params, indices_nd)  # nbatch x b x ...
    return gathered_params


def _tile_along_new_axis(params, multiples, axis=-1):
    """
    Tiles params multiples times along new axis.
    :param params:        Tensor
    :param multiples:     Integer, number of times to repeat along new axis
    :param axis:          Axis to repeat params along
    :return tiled_params:
    """
    params = tf.expand_dims(params, axis=axis)
    multiples_vec = tf.one_hot(axis, tf.rank(params), on_value=multiples, off_value=1)
    tiled_params = tf.tile(params, multiples_vec)
    return tiled_params


def labels_to_adjacency(labels, m=None):
    """
    Perform all pairwise label comparisons to create a binary adjacency matrix, specifying whether two pixels have the
    same label.
    :param labels:       N x M Tensor, where N is the batch size and M is the number of nodes. Each entry is an integer
                         label.
    :return adjacency:   N x M x M ByteTensor Variable.
    """
    if m is None:
        m = tf.convert_to_tensor(labels.shape[1])
    labels = _tile_along_new_axis(labels, m, axis=1)
    adjacency = tf.equal(labels, tf.transpose(labels, [0, 2, 1]))
    return adjacency
