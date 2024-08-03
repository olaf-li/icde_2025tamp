import argparse
import lstm_seq2seq
import learn2learn as l2l
import os
import torch
from task_tree import *
from torch import optim


def _acquire_device(args):
    if args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(
            args.gpu) if not args.use_multi_gpu else args.devices
        device = torch.device('cuda:{}'.format(args.gpu))
        print('Use GPU: cuda:{}'.format(args.gpu))
    else:
        device = torch.device('cpu')
        print('Use CPU')
    return device


def load_the_task_tree_DFS(file_path):

    def dfs_traverse_directory(path, task_tree):
        if os.path.isdir(path):
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    if len(os.listdir(item_path)) == 0:
                        continue
                    if item != "data":
                        new_node = TaskTreeNode(item_path + "\\")
                        task_tree.add_child(new_node)
                        dfs_traverse_directory(item_path, new_node)

    task_tree = TaskTreeNode(file_path)
    dfs_traverse_directory(file_path, task_tree)
    return task_tree


def compute_loss(model, data_x, data_y, args):
    if args.model_name == "lstm_encoder_decoder":
        return model.train_model_meta_adapt(data_x, data_y, target_len=args.seq_out)
    else:
        return model.train_model_meta_adapt(data_x, data_y, target_len=args.seq_out)


def meta_training(model, task_group, args):

    maml = model

    trainable_params = [param for param in maml.parameters() if param.requires_grad]

    trainable_params_detached = [param.clone().detach().requires_grad_(True) for param in trainable_params]
    opt = optim.Adam(trainable_params_detached, args.meta_lr)

    # loss_function = Task_Query_Loss()

    loss = 0.
    torch.backends.cudnn.enabled = False
    for iter in range(args.num_iteration):
        learner = maml.clone()
        evaluation_error = 0.0
        if args.dataset_name == "Porto_Grid":
            x_spt, y_spt, x_qry, y_qry = Porto(task_group.parameter_path, iw=args.seq_in, ow=args.seq_out,
                                                            s=args.stride, features=args.num_features).sample(4, 1, args.meta_batch_size)

        for i in range(args.meta_batch_size):
            x_support = x_spt[i]
            y_support = y_spt[i]
            x_query = x_qry[i]
            y_query = y_qry[i]

            for j in range(args.adapt_steps):
                error = compute_loss(learner, x_support, y_support, args)
                learner.adapt(error)

            evaluation_error += compute_loss(learner, x_query, y_query, args)

        opt.zero_grad()
        evaluation_error = evaluation_error / args.meta_batch_size
        evaluation_error.backward(retain_graph=True)
        opt.step()
        loss += evaluation_error

    torch.save(model.module.state_dict(), task_group.parameter_path + "parameters.pth")
    return loss / args.num_iteration


def train(model, root, args):

    if root is None:
        print("the root node is None")

    print("training: ", root.parameter_path)

    if not root.children:
        loss = meta_training(model.clone(), root, args)
        return loss

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    trainable_params_detached = [param.clone().detach().requires_grad_(True) for param in trainable_params]
    opt = optim.Adam(trainable_params_detached, args.meta_lr)

    loss = 0.
    for child in root.children:
        loss += train(model.clone(), child, args)
    opt.zero_grad()
    loss.backward(retain_graph=True)
    opt.step()
    torch.save(model.module.state_dict(), root.parameter_path + "parameters.pth")
    return loss


def main(args, data_path):

    # acquire the device
    device = _acquire_device(args)
    task_tree = load_the_task_tree_DFS(data_path)
    # init the model and init the parameter
    model = lstm_seq2seq(input_size=args.num_features, hidden_size=args.hidden_size).to(device)

    maml = l2l.algorithms.MAML(model, args.update_lr)

    train(maml, task_tree, args)
    print("training process is finished and the model is saved")


def argparser_return():
    """

    @return:
    """
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    argparser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    argparser.add_argument('--gpu', type=int, default=0, help='gpu')
    argparser.add_argument('--num_features', type=int, default=2, help='the number of features')
    argparser.add_argument('--stride', type=int, default=5, help='the stride of generate sequence to sequence instance')
    argparser.add_argument('--sampling_step', type=int, default=30, help='the number of features')
    argparser.add_argument('--dataset_name', type=str, default="Porto_Grid", help='the name of dataset')


    argparser.add_argument('--model_name', type=str, default="LSTM_Encoder_Decoder",
                           help='the name of the model as the meta learner')
    argparser.add_argument('--meta_batch_size', type=int, default=10, help='the batch size when training the meta learner')

    argparser.add_argument('--seq_in', type=int, default=5, help='input sequence length')
    argparser.add_argument('--seq_out', type=int, default=1, help='prediction sequence length')
    argparser.add_argument('--hidden_size', type=int, default=12, help='hidden_size')

    # set the parameter of meta learning
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.001)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.0015)
    argparser.add_argument('--adapt_steps', type=int, help='the step of adaption', default=4)

    argparser.add_argument('--facter', type=str, help='the facter for task cluster', default="learning_path")

    argparser.add_argument('--num_iteration', type=int, help='the number of iteration for outer meta update', default=32)

    args = argparser.parse_args()
    return args
