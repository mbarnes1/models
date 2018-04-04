#!/usr/bin/python
"""
Round precomputed pixel instance embeddings and evaluate performance.
"""

import argparse
from datasets.cityscapesScripts.cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling import args as eval_args
from datasets.cityscapesScripts.cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling import evaluateImgLists, printResults
from functools import partial
import imageio
from itertools import izip
import numpy as np
import os
from utils.cluster_utils import kwik_cluster, lp_cost


#
EMBEND = '.npy'  # predicted pixel embedding file ending
# For now, assume both these files in same directory, e.g. data/datasets/cityscapes/gtFine/val/
IGNORELABEL = 255
IMGEND = '_gtFine_instanceIds.png'  # ground truth instance labels file ending
SEMEND = '_gtFine_labelIds.png'  # predicted semantic label file ending. For now, use ground truth


def batch_eval(args):
    """
    :param args:
    """
    results_dir = os.path.join(args.log_dir, 'roundings')
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    for dirName, _, fileList in os.walk(args.dataset_dir):
        for file in fileList:
            if file.endswith(IMGEND):
                gt_instance_path = os.path.join(dirName, file)
                image_name = file.rstrip(IMGEND)
                semantic_path = os.path.join(dirName, '{}{}'.format(image_name, SEMEND))
                embedding_path = os.path.join(args.log_dir, '{}{}'.format(image_name, EMBEND))
                print 'Image {}'.format(image_name)
                results_dict = eval_embedding(embedding_path, semantic_path, gt_instance_path, results_dir, image_name)
                printResults(results_dict['averages'], eval_args)


def eval_embedding(embedding_path, semantic_path, gt_path, results_dir, image_name, viz=False):
    """

    :param embedding_path: Path to predicted pixel embedding file.
    :param semantic_path: Path to predicted semantic image label file.
    :param gt_path: Path to ground truth instance labels.
    :param results_dir: Write rounding results to this directory.
    :param image_name: Name of this image.
    :param viz: If true, visualize the instance embeddings and return plot handle.
    :return results_dict: See cityscapeScripts for definition.
    """
    cost_function = partial(lp_cost, p=10)

    embeddings = np.load(embedding_path)
    h, w, d = embeddings.shape
    pred_labels = kwik_cluster(np.reshape(embeddings, [-1, d]), cost_function)
    pred_labels = np.reshape(pred_labels, [h, w])

    semantic_labels = imageio.imread(semantic_path)
    if pred_labels.shape != semantic_labels.shape:
        raise ValueError('Prediction and Semantic label shapes {} and {} do not match'.format(pred_labels.shape, semantic_labels.shape))

    instance_labels, instance_counts = np.unique(pred_labels, return_counts=True)

    results_file_path = os.path.join(results_dir, '{}.txt'.format(image_name))
    if not os.path.exists(os.path.dirname(results_file_path)):
        os.makedirs(os.path.dirname(results_file_path))
    with open(results_file_path, 'wb') as f:
        for instance_label, instance_count in izip(instance_labels, instance_counts):
            instance_mask = pred_labels == instance_label

            # Predict semantic label for this instance
            semantic_labels_this_instance = semantic_labels[instance_mask]
            unique_semantic_labels, semantic_counts = np.unique(semantic_labels_this_instance, return_counts=True)
            ind = np.argmax(semantic_counts)
            majority_vote_semantic_label = unique_semantic_labels[ind]
            if majority_vote_semantic_label != IGNORELABEL:
                # Write mask file
                mask_filename = "{}_{}.png".format(image_name, instance_label)
                # Write mask to file. Set mask value to 255 so visualization is black & white.
                imageio.imwrite(os.path.join(results_dir, mask_filename), 255*instance_mask.astype('uint8'))

                # TODO: Better confidence prediction than the size of the cluster
                f.write('{} {} {}\n'.format(mask_filename, majority_vote_semantic_label, len(semantic_labels_this_instance)))

    eval_args.predictionPath = os.path.abspath(results_dir)
    eval_args.quiet = True
    eval_args.JSONOutput = False
    if os.path.isfile(eval_args.gtInstancesFile):
        os.remove(eval_args.gtInstancesFile)
    results_dict = evaluateImgLists([results_file_path], [gt_path], eval_args)
    if os.path.isfile(eval_args.gtInstancesFile):
        os.remove(eval_args.gtInstancesFile)
    return results_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset_dir",
                        help='Path to directory containing instance and semantic ID PNG files. Files must match pattern'
                             '{imagename}_instanceIds.png, e.g. frankfurt_000001_080091_instanceIds.png')

    parser.add_argument("log_dir",
                        help='Path to directory containing pixel embedding files. '
                             'Files must match pattern {imagename}.npy')

    args = parser.parse_args()

    batch_eval(args)
