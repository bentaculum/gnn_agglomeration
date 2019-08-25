import torch
import argparse
import os
import datetime
import logging
import json
import pytz
from operator import attrgetter


class SortingHelpFormatter(argparse.HelpFormatter):
    """
    sort arguments when listed on the command line via --help option
    """

    def add_arguments(self, actions):
        actions = sorted(actions, key=attrgetter('option_strings'))
        super(SortingHelpFormatter, self).add_arguments(actions)


class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=SortingHelpFormatter,
            allow_abbrev=False
        )
        self.default = {}

        self.parser.add_argument(
            '--config_from_file',
            type=str,
            help="absolute path to config file to be used (json format)")
        self.default['config_from_file'] = None

        self.parser.add_argument(
            '--nodes',
            type=positive_int,
            help='Number of nodes in the graph')
        self.default['nodes'] = 100

        self.parser.add_argument(
            '--self_loops',
            type=str2bool,
            help='whether all nodes in the RAG will have artificial self-loops')
        self.default['self_loops'] = True

        self.parser.add_argument(
            '--euclidian_dimensionality',
            type=positive_int,
            help='Dimension of the Euclidian space, used in data.pos')
        self.default['euclidian_dimensionality'] = 3

        self.parser.add_argument(
            '--feature_dimensionality',
            type=positive_int,
            help='Dimension of the feature space, used in data.x')
        self.default['feature_dimensionality'] = 6

        self.parser.add_argument(
            '--pseudo_dimensionality',
            type=positive_int,
            help='Dimension of the pseudo coordinates, according to their type')
        self.default['pseudo_dimensionality'] = 4

        self.parser.add_argument(
            '--kernel_size',
            type=positive_int,
            help='Attention: -, Splines: # of control points, GMM: # of mixture components')
        self.default['kernel_size'] = 2

        self.parser.add_argument(
            '--data_transform',
            type=str,
            choices=['Cartesian', 'LocalCartesian',
                     'Distance', 'Polar', 'Spherical'],
            help='define the edge attributes (pseudo coordinates)')
        self.default['data_transform'] = 'Cartesian'

        self.parser.add_argument(
            '--theta_max',
            type=float,
            help='nodes with lower euclidian distance will be connected')
        self.default['theta_max'] = 0.2

        self.parser.add_argument(
            '--theta',
            type=float,
            help='euclidian neighborhood distance')
        self.default['theta'] = 0.1

        self.parser.add_argument(
            '--dataset_type_train',
            type=str,
            choices=[
                'DiameterDataset',
                'CountNeighborsDataset',
                'IterativeDataset',
                'HemibrainDatasetRandom',
                'HemibrainDatasetRandomInMemory',
                'HemibrainDatasetBlockwise',
                'HemibrainDatasetBlockwiseInMemory'
            ],
            help='choose from different types of local datasets')
        self.default['dataset_type_train'] = 'HemibrainDatasetRandomInMemory'

        self.parser.add_argument(
            '--dataset_type_val',
            type=str,
            choices=[
                'HemibrainDatasetRandom',
                'HemibrainDatasetRandomInMemory',
                'HemibrainDatasetBlockwise',
                'HemibrainDatasetBlockwiseInMemory'
            ],
            help='choose from different types of local datasets')
        self.default['dataset_type_val'] = 'HemibrainDatasetBlockwiseInMemory'

        self.parser.add_argument(
            '--dataset_type_test',
            type=str,
            choices=[
                'HemibrainDatasetRandom',
                'HemibrainDatasetRandomInMemory',
                'HemibrainDatasetBlockwise',
                'HemibrainDatasetBlockwiseInMemory'
            ],
            help='choose from different types of local datasets')
        self.default['dataset_type_test'] = 'HemibrainDatasetBlockwiseInMemory'

        self.parser.add_argument(
            '--run_path',
            type=str,
            help='directory to save temporary outputs')
        self.default['run_path'] = 'runs'

        self.parser.add_argument(
            '--summary_dir',
            type=str,
            help='relative directory to save temporary summary')
        self.default['summary_dir'] = 'summary'

        self.parser.add_argument(
            '--outputs_dir',
            type=str,
            help='relative directory to save raw outputs from network')
        self.default['outputs_dir'] = 'outputs'

        self.parser.add_argument(
            '--outputs_interval',
            type=int,
            help='interval to write batch outputs to file')
        self.default['outputs_interval'] = 100

        self.parser.add_argument(
            '--write_summary',
            type=str2bool,
            help='option to use tensorboardx to monitor the training')
        self.default['write_summary'] = True

        self.parser.add_argument(
            '--log_only_gradients',
            type=str2bool,
            help='whether to write gradients to tensorboard')
        self.default['log_only_gradients'] = True

        self.parser.add_argument(
            '--log_histograms',
            type=str2bool,
            help='whether to perform the costly plotting of histograms')
        self.default['log_histograms'] = False

        self.parser.add_argument(
            '--log_per_epoch_only',
            type=str2bool,
            help='minimal logging, only the loss and the metric')
        self.default['log_per_epoch_only'] = True

        self.parser.add_argument(
            '--log_namespaces',
            type=str,
            nargs='+',
            help='If you want to log only specific namespaces (e.g. layers), specify them here')
        self.default['log_namespaces'] = []

        self.parser.add_argument(
            '--model_dir',
            type=str,
            help='relative directory to save temporary model, for both training and inference')
        self.default['model_dir'] = 'model'

        self.parser.add_argument(
            '--plot_error_locations',
            action='store_true',
            help='if passed, the plot errors by location will be created')
        self.default['plot_error_locations'] = False

        self.parser.add_argument(
            '--plot_graphs_testset',
            type=int,
            help='how many graphs to plot from the test set for visual inspection. -1 means all')
        self.default['plot_graphs_testset'] = 50

        self.parser.add_argument(
            '--plot_targets_vs_predictions',
            type=str2bool,
            help='option to plot confusion matrix or similar plot, depending on the model type')
        self.default['plot_targets_vs_predictions'] = False

        self.parser.add_argument(
            '--validation_split',
            type=unit_float,
            help='define size of validation set, 0 <= ... <= 1')
        self.default['validation_split'] = 0.1

        self.parser.add_argument(
            '--test_split',
            type=unit_float,
            help='define size of test set, 0 <= ... <= 1')
        self.default['test_split'] = 0.1

        self.parser.add_argument(
            '--model',
            type=str,
            help='OurConvModel | GcnModel | GmmConvModel | etc.')
        self.default['model'] = 'OurConvModel'

        self.parser.add_argument(
            '--model_type',
            type=str,
            choices=['ClassificationProblem', 'RegressionProblem', 'CosineEmbeddingLossProblem'])
        self.default['model_type'] = 'CosineEmbeddingLossProblem'

        self.parser.add_argument(
            '--training_epochs',
            type=positive_int,
            help='number of training epochs')
        self.default['training_epochs'] = 100

        self.parser.add_argument(
            '--samples',
            type=positive_int,
            help='Number of random graphs to create, if a new dataset is created')
        self.default['samples'] = 4096

        self.parser.add_argument(
            '--epoch_samples_train',
            type=positive_int,
            help='number of training samples drawn by the RandomSampler per epoch')
        self.default['epoch_samples_train'] = 4096

        self.parser.add_argument(
            '--epoch_samples_val',
            type=positive_int,
            help='number of validation samples drawn by the RandomSampler per epoch')
        self.default['epoch_samples_val'] = 20

        self.parser.add_argument(
            '--standardize_targets',
            type=str2bool,
            help='targets to mean 0, std 1, if Regression is performed')
        self.default['standardize_targets'] = False

        self.parser.add_argument(
            '--non_linearity',
            type=str,
            help='Activation function from torch.nn.functional, used for hidden layers, e.g. relu | sigmoid | tanh | leaky_relu')
        self.default['non_linearity'] = 'sigmoid'

        self.parser.add_argument(
            '--batch_size_train',
            type=positive_int,
            help='batch size for training')
        self.default['batch_size_train'] = 1

        self.parser.add_argument(
            '--batch_size_eval',
            type=positive_int,
            help='batch size for evaluation')
        self.default['batch_size_eval'] = 1

        self.parser.add_argument(
            '--hidden_layers',
            type=nonnegative_int,
            help='number of hidden layers')
        self.default['hidden_layers'] = 4

        self.parser.add_argument(
            '--hidden_units',
            type=positive_int,
            nargs='+',
            help='number of units per hidden layer in the GNN')
        self.default['hidden_units'] = [4, 4, 4, 4, 2]

        self.parser.add_argument(
            '--attention_heads',
            nargs='+',
            help='number of attention heads per hidden layer in the GNN')
        self.default['attention_heads'] = [8, 4, 2, 1, 1]

        self.parser.add_argument(
            '--use_bias',
            type=str2bool,
            help='whether to use an additive bias')
        self.default['use_bias'] = True

        self.parser.add_argument(
            '--fc_bias',
            type=str2bool,
            help='whether to use a bias term for the final fully connected layer')
        self.default['fc_bias'] = True

        self.parser.add_argument(
            '--fc_layers',
            type=positive_int,
            help='number of fully connected layers in the end, at least 1')
        self.default['fc_layers'] = 7

        self.parser.add_argument(
            '--fc_layer_dims',
            type=positive_int,
            nargs='*',
            help='list of hidden layer dims for fc layers in the end')
        self.default['fc_layer_dims'] = [32, 32, 16, 16, 8, 8]

        self.parser.add_argument(
            '--fc_dropout_probs',
            type=unit_float,
            nargs='*',
            help='fc network: dropout probs')
        self.default['fc_dropout_probs'] = [0, 0.2, 0, 0, 0, 0]

        self.parser.add_argument(
            '--fc_batch_norm',
            type=str2bool,
            help='whether to use Batch Normalization in the final fully connected layer')
        self.default['fc_batch_norm'] = True

        self.parser.add_argument(
            '--dropout_type',
            type=str,
            choices=['dropout', 'dropout2d'],
            help='dropout | dropout2d')
        self.default['dropout_type'] = 'dropout'

        self.parser.add_argument(
            '--dropout_probs',
            type=unit_float,
            nargs='+',
            help='dropout probabilites during training for the input layer and all the hidden layers')
        self.default['dropout_probs'] = [0.0, 0.0, 0.0, 0.0, 0.0]

        self.parser.add_argument(
            '--adam_lr',
            type=unit_float,
            help='Learning rate for ADAM optimizer')
        self.default['adam_lr'] = 0.0001

        self.parser.add_argument(
            '--adam_weight_decay',
            type=unit_float,
            help='Weight decay for ADAM optimizer')
        self.default['adam_weight_decay'] = 0.0

        self.parser.add_argument(
            '--batch_norm',
            type=str2bool,
            help='whether to use Batch Normalization')
        self.default['batch_norm'] = True

        self.parser.add_argument(
            '--clip_grad',
            type=str2bool,
            help='whether to use gradient clipping')
        self.default['clip_grad'] = False

        self.parser.add_argument(
            '--clip_value',
            type=float,
            help='gradients are clipped at this value')
        self.default['clip_value'] = 1.0

        self.parser.add_argument(
            '--clip_method',
            type=str,
            choices=['value', 'inf', '1', '2'],
            help='gradient clipping per value or L_-norm')
        self.default['clip_method'] = 'inf'

        self.parser.add_argument(
            '--att_heads_concat',
            type=str2bool,
            help='whether to concat or average the outputs of the different attention heads')
        self.default['att_heads_concat'] = True

        self.parser.add_argument(
            '--att_final_dropout',
            type=unit_float,
            help='Dropout probability for the final attention vector')
        self.default['att_final_dropout'] = 0.0

        self.parser.add_argument(
            '--att_layers',
            type=positive_int,
            help='Attention NN: number of layers')
        self.default['att_layers'] = 5

        self.parser.add_argument(
            '--att_layer_dims',
            type=positive_int,
            nargs='+',
            help='Attention NN: list of layer dimensions')
        self.default['att_layer_dims'] = [32, 16, 8, 4, 1]

        self.parser.add_argument(
            '--att_non_linearity',
            type=str,
            help='Attention NN: torch.nn.functional non linearity to use e.g. leaky_relu')
        self.default['att_non_linearity'] = 'sigmoid'

        self.parser.add_argument(
            '--att_batch_norm',
            type=str2bool,
            help='Attention NN: whether to use batch normalization')
        self.default['att_batch_norm'] = True

        self.parser.add_argument(
            '--att_dropout_probs',
            type=unit_float,
            nargs='+',
            help='Attention NN: dropout probabilites during training for the input layer and all the hidden layers')
        self.default['att_dropout_probs'] = [0, 0, 0, 0, 0]

        self.parser.add_argument(
            '--att_bias',
            type=str2bool,
            help='Attention NN: option to use bias')
        self.default['att_bias'] = True

        self.parser.add_argument(
            '--att_normalize',
            type=str2bool,
            help='whether to use a softmax over each neighborhood')
        self.default['att_normalize'] = True

        self.parser.add_argument(
            '--att_nodenet_layers',
            type=positive_int,
            help='number of fc layers before the attention mechanism')
        self.default['att_nodenet_layers'] = 1

        self.parser.add_argument(
            '--att_nodenet_hidden_dims',
            type=positive_int,
            nargs='*',
            help='hidden dimensionality of nodenet')
        self.default['att_nodenet_hidden_dims'] = []

        self.parser.add_argument(
            '--att_use_node_features',
            type=str2bool,
            help='wether the attention function takes the node features as input on top of the pseudo-coordinates')
        self.default['att_use_node_features'] = False

        self.parser.add_argument(
            '--load_model',
            type=str,
            help="Load model from file. 'latest' | relative/path/to/tarfile")
        self.default['load_model'] = None

        self.parser.add_argument(
            '--load_model_version',
            type=str,
            help="which checkpoint to use. 'latest' | checkpoint_name, no extension")
        self.default['load_model_version'] = 'latest'

        self.parser.add_argument(
            '--telegram',
            type=str2bool,
            help='whether to have a Sacred Telegram Observer')
        self.default['telegram'] = True

        self.parser.add_argument(
            '--confusion_matrix_path',
            type=str,
            help='file name of confusion matrix')
        self.default['confusion_matrix_path'] = 'confusion_matrix_test.png'

        self.parser.add_argument(
            '--temp',
            type=str2bool,
            help='If true, save results to temp folder. If false, create timestamped directory.')
        self.default['temp'] = False

        self.parser.add_argument(
            '--checkpoint_interval',
            type=positive_int,
            help='how often to save a checkpoint of the model that can be used for restarting')
        self.default['checkpoint_interval'] = 2

        self.parser.add_argument(
            '--machine',
            type=str,
            choices=[
                'localhost',
                'slowpoke1'],
            help='machine-dependent parameters to be imported, e.g. for connecting to the MongoDB')
        self.default['machine'] = 'slowpoke1'

        self.parser.add_argument(
            '--msts',
            type=positive_int,
            help='How many different classes in an instance of DiameterGraph')
        self.default['msts'] = 5

        self.parser.add_argument(
            '--class_noise',
            type=unit_float,
            help='how much probability mass is spread between the wrong target classes')
        self.default['class_noise'] = 0.2

        self.parser.add_argument(
            '--class_label_feature',
            type=str2bool,
            help='whether to use the noisy class labels as a feature')
        self.default['class_label_feature'] = True

        self.parser.add_argument(
            '--affinity_dist_alpha',
            type=float,
            help='alpha value for the beta dist that generates noisy edge features')
        self.default['affinity_dist_alpha'] = 1

        self.parser.add_argument(
            '--affinity_dist_beta',
            type=float,
            help='beta value for the beta dist that generates noisy edge features')
        self.default['affinity_dist_beta'] = 4

        self.parser.add_argument(
            '--affinities',
            type=str,
            choices=['all_affinities', 'only_gt_affinities',
                     'only_gt_dir_affinities'],
            help='what kind of affinities on the edges')
        self.default['affinities'] = 'only_gt_affinities'

        self.parser.add_argument(
            '--curvature_degree_limit',
            type=positive_int,
            help='Restrict the deviation from the continuing line by this amount of degrees, in both dirs')
        self.default['curvature_degree_limit'] = 45

        self.parser.add_argument(
            '--edge_labels',
            type=str2bool,
            help='whether to use edge labels')
        self.default['edge_labels'] = True

        self.parser.add_argument(
            '--fc_use_edge',
            type=str2bool,
            help='whether to use the edge information for the final fc layer')
        self.default['fc_use_edge'] = True

        #########################

        self.parser.add_argument(
            '--db_name_train',
            type=str,
            help='name of the used mongodb for training')
        self.default['db_name_train'] = 'gnn_agglomeration_hemi_mtlsd_400k_roi_2'
        # self.default['db_name_train'] = 'gnn_agglomeration_hemi_mtlsd_400k_roi_1'

        self.parser.add_argument(
            '--db_name_val',
            type=str,
            help='name of the used mongodb for validation')
        self.default['db_name_val'] = 'gnn_agglomeration_hemi_mtlsd_400k_roi_1'

        self.parser.add_argument(
            '--db_name_test',
            type=str,
            help='name of the used mongodb for test')
        self.default['db_name_test'] = 'gnn_agglomeration_hemi_mtlsd_400k_roi_3'
        # self.default['db_name_test'] = 'gnn_agglomeration_hemi_mtlsd_400k_roi_1'

        self.parser.add_argument(
            '--dataset_path_train',
            type=str,
            help='the directory to read the training dataset from')
        self.default['dataset_path_train'] = 'data/hemi/22_micron_cube/default_train'
        # self.default['dataset_path_train'] = 'data/hemi/12_micron_cube/default_train'

        self.parser.add_argument(
            '--dataset_path_val',
            type=str,
            help='the directory to read the validation dataset from')
        self.default['dataset_path_val'] = 'data/hemi/12_micron_cube/default_val'

        self.parser.add_argument(
            '--dataset_path_test',
            type=str,
            help='the directory to read the test dataset from')
        self.default['dataset_path_test'] = 'data/hemi/17_micron_cube/default_test'
        # self.default['dataset_path_test'] = 'data/hemi/12_micron_cube/debug_test'

        self.parser.add_argument(
            '--train_roi_offset',
            type=positive_int,
            nargs=3,
            help='ROI absolute position of lower vertex')
        self.default['train_roi_offset'] = [150400, 192000, 214400]
        # self.default['train_roi_offset'] = [140800, 205120, 198400]

        self.parser.add_argument(
            '--train_roi_shape',
            type=positive_int,
            nargs=3,
            help='ROI size, starting at roi_offset')
        self.default['train_roi_shape'] = [21800, 21800, 21800]
        # self.default['train_roi_shape'] = [11800, 11800, 11800]

        self.parser.add_argument(
            '--val_roi_offset',
            type=positive_int,
            nargs=3,
            help='ROI absolute position of lower vertex')
        # self.default['val_roi_offset'] = [140800 + 8 * 1180, 205120, 198400]
        self.default['val_roi_offset'] = [140800, 205120, 198400]

        self.parser.add_argument(
            '--val_roi_shape',
            type=positive_int,
            nargs=3,
            help='ROI size, starting at roi_offset')
        self.default['val_roi_shape'] = [11800, 11800, 11800]
        # self.default['val_roi_shape'] = [1180, 11800, 11800]
        # self.default['val_roi_shape'] = [3000, 3000, 3000]

        self.parser.add_argument(
            '--test_roi_offset',
            type=positive_int,
            nargs=3,
            help='ROI absolute position of lower vertex')
        # self.default['test_roi_offset'] = [140800 + 9 * 1180, 205120, 198400]
        self.default['test_roi_offset'] = [153680, 200000, 179200]

        self.parser.add_argument(
            '--test_roi_shape',
            type=positive_int,
            nargs=3,
            help='ROI size, starting at roi_offset')
        # self.default['test_roi_shape'] = [1180, 11800, 11800]
        self.default['test_roi_shape'] = [16800, 16800, 16800]

        #########################

        self.parser.add_argument(
            '--block_size',
            type=positive_int,
            nargs=3,
            help='fixed block size for creating pyg graphs')
        self.default['block_size'] = [1500, 1500, 1500]
        # desired
        # self.default['block_size'] = [3000, 3000, 3000]

        self.parser.add_argument(
            '--block_fit',
            type=str,
            choices=[
                'shrink',
                'overlap'],
            help="For blockwise datasets, how to handle cases where shifting blocks by block_size does not tile the roi_shape")
        self.default['block_fit'] = 'shrink'

        self.parser.add_argument(
            '--block_padding',
            type=positive_int,
            nargs=3,
            help='padding to create an outer mask that guarantees context for all targets the contribute to the loss')
        self.default['block_padding'] = [1500, 1500, 1500]

        self.parser.add_argument(
            '--max_edges',
            type=int,
            help='limit number of edges per graph to avoid out of memory errors on GPU')
        self.default['max_edges'] = 180000

        self.parser.add_argument(
            '--db_host',
            type=str,
            help='path to mongoDB connection file')
        self.default['db_host'] = 'db_host.ini'

        self.parser.add_argument(
            '--nodes_collection',
            type=str,
            help='name of mongodb collection for RAG nodes')
        self.default['nodes_collection'] = 'nodes'

        self.parser.add_argument(
            '--edges_collection',
            type=str,
            help='name of mongodb collection for RAG edges')
        self.default['edges_collection'] = 'edges_hist_quant_50'

        self.parser.add_argument(
            '--embeddings_collection_train',
            type=str,
            help='name of mondogb collection for RAG node embeddings train')
        self.default['embeddings_collection_train'] = 'nodes_embeddings_setup01_300k'

        self.parser.add_argument(
            '--embeddings_collection_val',
            type=str,
            help='name of mondogb collection for RAG node embeddings val')
        self.default['embeddings_collection_val'] = 'nodes_embeddings_setup01_300k'

        self.parser.add_argument(
            '--embeddings_collection_test',
            type=str,
            help='name of mondogb collection for RAG node embeddings test')
        self.default['embeddings_collection_test'] = 'nodes_embeddings_setup01_300k'

        self.parser.add_argument(
            '--graph_type',
            type=str,
            choices=['HemibrainGraphUnmasked', 'HemibrainGraphMasked'],
            help='which type of graph to extract from db')
        self.default['graph_type'] = 'HemibrainGraphMasked'

        self.parser.add_argument(
            '--num_workers',
            type=int,
            help='number of workers for tasks that are split into python subprocesses')
        self.default['num_workers'] = 32

        self.parser.add_argument(
            '--dataloader_pin_memory',
            type=str2bool,
            help='whether to pin memory for pre-fetching')
        self.default['dataloader_pin_memory'] = True

        self.parser.add_argument(
            '--save_processed_train',
            type=str2bool,
            help='whether to store the processed out-of-mem training dataset to file')
        self.default['save_processed_train'] = True

        self.parser.add_argument(
            '--save_processed_val',
            type=str2bool,
            help='whether to store the processed out-of-mem validation dataset to file')
        self.default['save_processed_val'] = True

        self.parser.add_argument(
            '--save_processed_test',
            type=str2bool,
            help='whether to store the processed out-of-mem test dataset to file')
        self.default['save_processed_test'] = True

        self.parser.add_argument(
            '--data_augmentation',
            type=str,
            choices=['AugmentHemibrain'],
            help='either a transform from pytorch geometric or a custom combination of such')
        self.default['data_augmentation'] = 'AugmentHemibrain'

        self.parser.add_argument(
            '--augment_translate_limit',
            type=positive_int,
            nargs=3,
            help='maximal absolute translation for each dimension, in nanometers')
        # should be multiples of 8
        self.default['augment_translate_limit'] = [32, 32, 32]

        self.parser.add_argument(
            '--edge_attr_noise_std',
            type=float,
            help='standard deviation of noise distribution for edge attributes. If 0, no noise is added.')
        self.default['edge_attr_noise_std'] = 0.0

        self.parser.add_argument(
            '--final_training_pass',
            type=str2bool,
            help='whether to do a pass over the training data with network in evaluation mode')
        self.default['final_training_pass'] = False

        self.parser.add_argument(
            '--final_test_pass',
            type=str2bool,
            help='whether to evaluate on the testset')
        self.default['final_test_pass'] = True

        self.parser.add_argument(
            '--write_to_db',
            type=str2bool,
            help='whether to write outputs for the test set back to the database')
        self.default['write_to_db'] = True

        self.parser.add_argument(
            '--gt_merge_score_field',
            type=str,
            help='DB field in edges collection that contains the ground truth score')
        self.default['gt_merge_score_field'] = 'gt_merge_score'

        self.parser.add_argument(
            '--merge_labeled_field',
            type=str,
            help='''DB field in edges collection that contains the gt masking
            which accounts for edges with unknown ground truth''')
        self.default['merge_labeled_field'] = 'merge_labeled'

        # TODO this is not being used right now
        self.parser.add_argument(
            '--out_dimensionality',
            type=positive_int,
            help='number of dimensions for output vector on each node')
        self.default['out_dimensionality'] = 6

        self.parser.add_argument('--summary_per_batch', type=str2bool)
        self.default['summary_per_batch'] = True

    def localhost(self):
        return {
            'mongo_url': 'localhost:27017',
            'mongo_db': 'sacred',
        }

    def slowpoke1(self):
        return {
            'mongo_url': 'slowpoke1.int.janelia.org:27017',
            'mongo_db': 'sacred',
        }

    def overwrite_defaults(self, config_filepath):
        with open(config_filepath) as json_file:
            self.default.update(json.load(json_file))

    def update_defaults_with_cmd_args(self, config):
        # overwrite with possible new config variables, and log a warning
        for k, v in config.items():
            if v is not None:
                logging.warning(
                    '{} = {} from defaults is overwritten with "{}"'.format(
                        k, self.default[k], v))
                self.default[k] = v
        return self.default

    def parse_args(self):
        config_cmd, remaining_args = self.parser.parse_known_args()
        config_cmd = vars(config_cmd)

        # detect root path, one level up from the config file
        self.default['root_dir'] = os.path.dirname(
            os.path.dirname(os.path.realpath(__file__)))

        # load old config and set as default for continuation of training
        if config_cmd['load_model']:
            if config_cmd['load_model'] == 'latest':
                # find latest model in the runs path
                all_runs_dir = os.path.join(
                    self.default['root_dir'], self.default['run_path'])

                # TODO filter for correct format of directory name, instead of
                # '2019'
                runs = sorted([name for name in os.listdir(
                    all_runs_dir) if name.startswith('2019')])

                rel_run_path = runs[-1]
                config_cmd['load_model'] = runs[-1]
            else:
                rel_run_path = config_cmd['load_model']

            config_filepath = os.path.join(
                self.default['root_dir'],
                self.default['run_path'],
                rel_run_path,
                'config.json')
            self.overwrite_defaults(config_filepath)
            config = self.update_defaults_with_cmd_args(config_cmd)

        # for new model, go to a new directory
        else:
            # allow for loading a config from file, set values as default
            if config_cmd['config_from_file']:

                # find latest model in the runs path
                if config_cmd['config_from_file'] == 'latest':
                    all_runs_dir = os.path.join(
                        self.default['root_dir'], self.default['run_path'])
                    # TODO filter for correct format of directory name, instead
                    # of '2019'
                    runs = sorted([name for name in os.listdir(
                        all_runs_dir) if name.startswith('2019')])
                    config_filepath = os.path.join(
                        self.default['root_dir'],
                        self.default['run_path'],
                        runs[-1], 'config.json')
                    config_cmd['config_from_file'] = config_filepath
                    self.overwrite_defaults(config_filepath)
                else:
                    self.overwrite_defaults(config_cmd['config_from_file'])

            config = self.update_defaults_with_cmd_args(config_cmd)

            if config['temp']:
                # save to a temporary directory that will be overwritten
                rel_run_path = 'temp'
            else:
                # create a custom directory for each run
                rel_run_path = datetime.datetime.now(
                    pytz.timezone('US/Eastern')).strftime('%Y%m%dT%H%M%S.%f%z')

        # set the absolute paths in the config file
        config['run_abs_path'] = os.path.join(
            config['root_dir'], config['run_path'], rel_run_path)
        config['dataset_abs_path_train'] = os.path.join(
            config['root_dir'], config['dataset_path_train'])
        config['dataset_abs_path_val'] = os.path.join(
            config['root_dir'], config['dataset_path_val'])
        config['dataset_abs_path_test'] = os.path.join(
            config['root_dir'], config['dataset_path_test'])

        config.update(getattr(self, config['machine'])())

        return config, remaining_args


def unit_float(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]" % (x,))
    return x


def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            "%s is an invalid positive int value" % value)
    return ivalue


def nonnegative_int(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(
            "%s is an invalid positive int value" % value)
    return ivalue


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
