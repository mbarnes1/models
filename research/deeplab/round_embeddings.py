"""
Round precomputed pixel instance embeddings and evaluate performance.
"""

from __future__ import division
import argparse
from datasets.cityscapesScripts.cityscapesscripts.helpers.labels import id2hasinstances
from datasets.cityscapesScripts.cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling import args as eval_args
from datasets.cityscapesScripts.cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling import evaluateImgLists, printResults
from functools import partial
import imageio
from itertools import izip
import matplotlib.cm
import matplotlib.colors
from multiprocessing import Pool
import numpy as np
import os
from utils.cluster_utils import kwik_cluster, lp_cost


EMBEND = '.npy'  # predicted pixel embedding file ending
# For now, assume both these files in same directory, e.g. data/datasets/cityscapes/gtFine/val/
IGNORELABEL = 255
IMGEND = '_gtFine_instanceIds.png'  # ground truth instance labels file ending
GTSEMEND = '_gtFine_labelIds.png'  # predicted semantic label file ending. For now, use ground truth


def batch_eval(args):
    """
    :param args:
    """
    if not os.path.exists(args.round_dir):
        os.mkdir(args.round_dir)

    print('Gathering all ground truth instance files...')
    input_list = []
    for dir_name, _, fileList in os.walk(args.dataset_dir):
        for file_name in fileList:
            if file_name.endswith(IMGEND):
                input_list.append((dir_name, file_name, args))
    print('Found {} ground truth images.'.format(len(input_list)))
    input_list = input_list[0:min(len(input_list), args.max_images)]

    if args.num_processes > 1:
        p = Pool(args.num_processes)
        outputs = []
        for i, output in enumerate(p.imap_unordered(single_eval, input_list), 1):
            print('done {:4.2f}%'.format(100*i / len(input_list)))
            outputs.append(output)
    else:
        outputs = map(single_eval, input_list)

    pred_paths = []
    gt_paths = []
    num_instances = 0
    for output in outputs:
        pred_paths.append(output[0])
        gt_paths.append(output[1])
        num_instances += output[2]
    num_instances /= len(outputs)

    # Compute final, dataset wide results
    results_dict = evaluate_img_lists(pred_paths, gt_paths, args.emb_dir)
    print 'Final results:'
    printResults(results_dict['averages'], eval_args)
    print 'Average number of instances per image {}'.format(num_instances)
    return results_dict


def single_eval(inputs):
    dir_name, file_name, args = inputs

    image_name = file_name.rstrip(IMGEND)
    if args.true_semantic:
        semantic_path = os.path.join(args.semantic_dir, '{}{}'.format(image_name, GTSEMEND))
    else:
        semantic_path = os.path.join(args.semantic_dir, image_name)
    embedding_path = os.path.join(args.emb_dir, '{}{}'.format(image_name, EMBEND))
    gt_instance_path = os.path.join(dir_name, file_name)
    pred_path, img_path, num_instances = round_embedding(embedding_path, semantic_path, args.round_dir, image_name)

    # Individual results
    #results_dict = evaluate_img_lists([pred_path], [gt_instance_path], results_dir)
    #print('Rounding results for image {}'.format(image_name))
    #printResults(results_dict['averages'], eval_args)
    return pred_path, gt_instance_path, num_instances


def round_embedding(embedding_path, semantic_path, results_dir, image_name):
    """

    :param embedding_path: Path to predicted pixel embedding file.
    :param semantic_path:  Path to predicted semantic image label file.
    :param results_dir:    Write rounding results to this directory.
    :param image_name:     Name of this image.
    :return pred_path:     Path to prediction TXT image.
    :return img_path:      Path to prediction PNG image.
    """
    semantic_labels = imageio.imread(semantic_path)

    cost_function = partial(lp_cost, p=10)
    embeddings = np.load(embedding_path)
    h, w, d = embeddings.shape

    if semantic_labels.shape != (h, w):
        raise ValueError('Prediction and Semantic label shapes {} and {} do not match'.format((h, w),
                                                                                              semantic_labels.shape))
    pred_labels = kwik_cluster(np.reshape(embeddings, [-1, d]), cost_function, blocks=np.reshape(semantic_labels, [-1]))
    pred_labels = np.reshape(pred_labels, [h, w])

    instance_labels, instance_counts = np.unique(pred_labels, return_counts=True)

    # Write color image of predicted instances
    img_path = os.path.join(results_dir, '{}_pred_instances.png'.format(image_name))
    write_color_img(pred_labels, img_path)

    # Write individual masks and metafile for processing by cityscapeScripts
    pred_path = os.path.join(results_dir, '{}.txt'.format(image_name))
    if not os.path.exists(os.path.dirname(pred_path)):
        os.makedirs(os.path.dirname(pred_path))
    instances_per_semantic = {}
    with open(pred_path, 'wb') as f:
        for instance_label, instance_count in izip(instance_labels, instance_counts):
            instance_mask = pred_labels == instance_label

            # Predict semantic label for this instance
            semantic_labels_this_instance = semantic_labels[instance_mask]
            unique_semantic_labels, semantic_counts = np.unique(semantic_labels_this_instance, return_counts=True)
            ind = np.argmax(semantic_counts)
            majority_vote_semantic_label = unique_semantic_labels[ind]
            if majority_vote_semantic_label != IGNORELABEL and id2hasinstances[majority_vote_semantic_label]:
                # Write mask file
                mask_filename = "{}_{}.png".format(image_name, instance_label)
                # Write mask to file. Set mask value to 255 so visualization is black & white.
                imageio.imwrite(os.path.join(results_dir, mask_filename), 255*instance_mask.astype('uint8'))
                # TODO: Better confidence prediction than the size of the cluster
                f.write('{} {} {}\n'.format(mask_filename, majority_vote_semantic_label, len(semantic_labels_this_instance)))

                if majority_vote_semantic_label not in instances_per_semantic:
                    instances_per_semantic[majority_vote_semantic_label] = instance_label * instance_mask.astype('uint8')
                else:
                    instances_per_semantic[majority_vote_semantic_label] += instance_label * instance_mask.astype('uint8')
    for semantic_label, instance_labels in instances_per_semantic.iteritems():
        output_path = os.path.join(results_dir, '{}_semantic_{}.png'.format(image_name, semantic_label))
        write_color_img(instance_labels, output_path)
    num_instances = len(instance_labels)
    return pred_path, img_path, num_instances


def get_color_img(labels):
    """
    Shuffle labels and create a color image array. Always keep label 0 as black.
    :param labels: h x w numpy array
    :return color_img: h x w x 4 rgba numpy array, dtype=uint8
    """
    instance_labels = np.unique(labels)
    shuffle_map = np.random.permutation(len(instance_labels))
    shuffle_map = {old_label: new_label for old_label, new_label in izip(instance_labels, shuffle_map)}
    shuffled_labels = np.vectorize(shuffle_map.get)(labels)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=(len(instance_labels) - 1))
    cmap = matplotlib.cm.jet
    colormap = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    color_img = colormap.to_rgba(shuffled_labels)
    color_img = (color_img * 255).astype('uint8')
    color_img[labels == 0, :] = [0, 0, 0, 255]  # set label 0 to black
    return color_img


def write_color_img(labels, path):
    """
    Shuffle labels and write to a color image. Always keep label 0 as black.
    :param labels: h x w numpy array
    :param path: Path to write image to
    """
    color_img = get_color_img(labels)
    imageio.imwrite(path, color_img)


def evaluate_img_lists(pred_paths, gt_paths, results_dir):
    """

    :param pred_paths:    List of paths to prediction txt files.
    :param gt_paths:      List of paths to ground truth png files.
    :param results_dir:   Path to results directory.
    :return results_dict: See cityscapeScripts for definition.
    """
    eval_args.predictionPath = os.path.abspath(results_dir)
    eval_args.quiet = True
    eval_args.JSONOutput = False
    if os.path.isfile(eval_args.gtInstancesFile):
        os.remove(eval_args.gtInstancesFile)
    results_dict = evaluateImgLists(pred_paths, gt_paths, eval_args)
    if os.path.isfile(eval_args.gtInstancesFile):
        os.remove(eval_args.gtInstancesFile)
    return results_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset_dir",
                        help='Path to directory containing true instance PNG files.'
                             'Files must match pattern:'
                             '{imagename}_instanceIds.png, e.g. frankfurt_000001_080091_instanceIds.png')
    
    parser.add_argument("semantic_dir",
                        help='Path to directory containing semantic ID PNG files.')
    
    parser.add_argument("emb_dir",
                        help='Path to directory containing pixel embedding files.'
                             'Files must match pattern {imagename}.npy')

    parser.add_argument("round_dir",
                        help='Directory to save rounding results to.')

    parser.add_argument("--true_semantic", action='store_true', default=False,
                        help='Set if semantic_dir points to ground truth semantic files, which have directory'
                             'and file format {city}/{city}_instanceIds.png')

    parser.add_argument("--max_images", default=np.Inf, type=int,
                        help='Evaluate at most this many images')

    parser.add_argument("--num_processes", default=1, type=int,
                        help="Number of parallel processes.")

    args = parser.parse_args()

    batch_eval(args)
