import os
import random

import torch
import numpy as np

from utils.parser import parse_args
from utils.data_loader import load_data
from utils.evaluate import test
from utils.helper import early_stopping


def main():
    def get_feed_dict(train_p, type_num, ui_dict, start, end, n_negs=1):

        def sampling(user_item, train_set, n):
            neg_items = []
            for user, _ in user_item.cpu().numpy():
                user = int(user)
                negitems = []
                for i in range(n):  # sample n times
                    while True:
                        negitem = random.choice(range(n_items))
                        if negitem not in train_set[user]:
                            break
                    negitems.append(negitem)
                neg_items.append(negitems)
            return neg_items

        feed_dict = {}
        entity_pairs = train_p[start:end]
        user = entity_pairs[:, 0]
        type_n = torch.tensor(type_num[user]).t()

        feed_dict['users'] = entity_pairs[:, 0]
        feed_dict['type_n'] = type_n
        # print(feed_dict['users'])
        feed_dict['pos_items'] = entity_pairs[:, 1]
        feed_dict['neg_items'] = torch.LongTensor(sampling(entity_pairs,
                                                           ui_dict,
                                                           n_negs * args.K)).to(device)
        return feed_dict

    """fix the random seed"""
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")
    #device =torch.device("cpu")

    """build dataset"""
    #train_p, train_c, train_v, user_dict, n_params, norm_mat_p, norm_mat_c, norm_mat_v
    train_p, user_dict, n_params, norm_mat_p, norm_mat_c, norm_mat_v, type_num = load_data(args)


    train_p = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_p], np.int32)) #转tensor

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_negs = args.n_negs

    """define model"""
    from modules.LightGCN import LightGCN
    #from modules.NGCF import NGCF

    model = LightGCN(n_params, args, norm_mat_p, norm_mat_c, norm_mat_v).to(device)


    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("start training ...")
    best_recall = 0.0
    count = 0
    for epoch in range(args.epoch):
        print("epoch: ", epoch)
        # shuffle training data
        train_p_ = train_p
        index = np.arange(len(train_p_))
        np.random.shuffle(index)
        train_p_ = train_p_[index].to(device)
        type_num_ = type_num.to(device)
        """training"""
        model.train()
        loss, s = 0, 0
        hits = 0

        while s + args.batch_size <= len(train_p):
            #print("batch: ", c)
            if s + 2 * args.batch_size <= len(train_p):
                batch = get_feed_dict(train_p_,
                                      type_num_,
                                      user_dict['train_user_set_p'],
                                      s, s + args.batch_size,
                                      n_negs)

                batch_loss = model(batch)

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                loss += batch_loss
                s += args.batch_size
            else:
                batch = get_feed_dict(train_p_,
                                      type_num_,
                                      user_dict['train_user_set_p'],
                                      s, s + args.batch_size,
                                      n_negs)
                batch_loss = model(batch)

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                loss += batch_loss
                s += args.batch_size

                batch = get_feed_dict(train_p_,
                                      type_num_,
                                      user_dict['train_user_set_p'],
                                      s, len(train_p),
                                      n_negs)

                batch_loss = model(batch)

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                loss += batch_loss
                s += args.batch_size

       # K = [10, 40, 80]
        recall_tem ,ndcg_tem= test(model, user_dict, n_params)
        if recall_tem > best_recall:
            best_recall = recall_tem
            count = 0
        else:
            count+=1
        if count >10:
            break
    print("recall: ",best_recall,"ndcg: ",ndcg_tem )

if __name__ == '__main__':
    main()
