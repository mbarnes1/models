import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.losses.losses_impl import Reduction
from tensorflow.python.util.tf_export import tf_export


LARGE = 1e8


@tf_export("losses.spectral")
def spectral_loss(
        instance_labels,
        embeddings,
        instance_mask,
        scope=None,
        loss_collection=ops.GraphKeys.LOSSES,
        reduction=Reduction.SUM_BY_NONZERO_WEIGHTS,
        subsample_power=13,
        semantic=False,
        normalize=True):
    """
    Creates a spectral loss. Modified from tf.losses.softmax_cross_entropy.
    :param instance_labels:  `[batch_size, num_pixels]` target instance labels.
    :param embeddings:       `[batch_size, num_pixels, embedding_dim]` pixel embedding output of the network.
    :param instance_mask:    '[batch_size, num_pixels]' binary Tensor.
                                1 = Semantic class of pixel has instances, learn pixel embedding.
                                0 = Semantic class of pixel does not have instances, so do not learn embedding.
    :param scope:            The scope for the operations performed in computing the loss.
    :param loss_collection:  Collection to which the loss will be added.
    :param reduction:        Type of reduction to apply to loss.
    :param subsample_power:  Uniformly randomly sample 2**subsample_power many pixels per image when computing the loss.
                             Must be power of 2 to avoid bias in tensorflow random sampling.
    :param semantic:         Computes the loss for each semantic class independently (default : False - Computes for
                             all classes together, implying an error interclasses)
    :param normalize:        Normalize pixel embeddings to have l2 norm of 1.
    :return loss:            Tensor of the same type as embeddings. If `reduction` is:
                                `NONE` = shape [batch_size, subsample, subsample]
                                Else   = Scalar
    Raises:
        ValueError:          If batch_size and num_pixels of `embeddings` and `instance_labels` do not match
                             or if the shape of `instance_mask` is invalid or if `instance_mask` is None.  Also if
                             `instance_labels` or `embeddings` is None.
    """
    assert isinstance(subsample_power, int)
    assert 0 <= subsample_power <= 16  # tradeoff between memory and variance.
    subsample = 2**subsample_power
    if instance_labels is None:
        raise ValueError("instance_labels must not be None.")
    if embeddings is None:
        raise ValueError("embeddings must not be None.")
    if instance_mask is None:
        raise ValueError("instance_mask must not be None.")  # TODO: Allow None, and set to no mask.
    with ops.name_scope(scope, "spectral_loss", (embeddings, instance_labels, instance_mask)) as scope:
        #embeddings = ops.convert_to_tensor(embeddings)
        #instance_labels = ops.convert_to_tensor(instance_labels)
        #instance_labels = math_ops.cast(instance_labels, embeddings.dtype)  # keep these as ints
        embeddings.get_shape()[0:2].assert_is_compatible_with(instance_labels.get_shape())
        instance_labels.get_shape().assert_is_compatible_with(instance_mask.get_shape())

        if normalize:
            embeddings = tf.nn.l2_normalize(embeddings, axis=2)

        # Subsample pixels which are not masked (i.e. belong to a semantic class with instances)
        instance_mask = math_ops.cast(instance_mask, dtypes.float32)  # tf.multinomial only accepts float probabilities
        large_instance_mask = instance_mask*LARGE
        tf.assert_equal(tf.is_finite(large_instance_mask), True)
        with tf.device("/cpu:0"):
            sample_indices = tf.multinomial(large_instance_mask, subsample, output_dtype=tf.int32)  # batch_size x subsample
        instance_labels = batch_gather(instance_labels, sample_indices)  # batch_size x subsample
        embeddings = batch_gather(embeddings, sample_indices)  # batch_size x subsample x embedding_dim

        if semantic:
            # Compute mask for pixels which belong to same semantic class.
            batch_size, _ = instance_labels.shape

            # Find the different class
            semantic_labels = tf.cast(tf.floor(instance_labels / 1000), tf.int32) # See cityscapesscripts/preparation/json2instanceImg.py
            semanticClass, _ = tf.unique(tf.reshape(semantic_labels, [-1]))

            def errorSemanticClass(sc):
                """
                    Computes the error associated to the given semantic class
                    On the different images of the batch
                
                    Arguments:
                        sc {class} -- Semantic Class
                """
                loss = 0.
                for image in range(batch_size):
                    # Select semantic pixels
                    selection = tf.equal(semantic_labels[image], sc)
                    selection_size = tf.reduce_sum(tf.cast(selection, tf.int32))

                    labelsSC = tf.expand_dims(tf.boolean_mask(instance_labels[image], selection), 0) # 1 x selection
                    embeddingSC = tf.expand_dims(tf.boolean_mask(embeddings[image], selection, axis = 0), 0) # 1 x selection x embedding_dim

                    # Constructs adjency
                    ASCImage = labels_to_adjacency(labelsSC, selection_size)
                    A_predictedSCImage = tf.matmul(embeddingSC, tf.transpose(embeddingSC, [0, 2, 1]))

                    loss += tf.losses.mean_squared_error(ASCImage, A_predictedSCImage, scope=scope, loss_collection=loss_collection, reduction=reduction)
                return loss

            def totalLoss():
                lossSemanticClass = tf.map_fn(errorSemanticClass, semanticClass, dtype='float32')
                return tf.reduce_sum(lossSemanticClass)

            def zeroLoss():
                return 0.

            loss = tf.cond(tf.size(semanticClass) > 0, totalLoss, zeroLoss)
        else:
            # Compute A
            A = labels_to_adjacency(instance_labels)# batch_size x subsample x subsample
            
            # Compute VV^T
            A_predicted = tf.matmul(embeddings, tf.transpose(embeddings, [0, 2, 1]))  # batch_size x subsample x subsample

            # Compute loss
            # TODO: pass weights, which is semantic adjacency matrix
            loss = tf.losses.mean_squared_error(A, A_predicted, scope=scope, loss_collection=loss_collection, reduction=reduction)
        
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


def labels_to_adjacency(labels, m = None):
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