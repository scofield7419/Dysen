import json
import os
from transformers import AutoProcessor, CLIPModel, AutoTokenizer
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
import backoff
import openai
import argparse
import random
from functools import lru_cache
import numpy as np
from tqdm import tqdm
from transformers import logging

# logging.set_verbosity_warning()
logging.set_verbosity_error()
import tensorboard_logger as tb_logger

from dysen.policy_model import PolicyNetwork, Reward
from dysen import utils
# from data import COCO2014, collate_fn
from base_prompt import build_prompt_for_action_planning, build_prompt_for_scene_imagination, build_prompt_for_scene_polishment
from scripts.sample_text2video import sample_text2video
from vdm.utils.common_utils import instantiate_from_config
from scripts.sample_utils import load_model
from vdm.samplers.ddim import DDIMSampler


@backoff.on_exception(backoff.expo, (openai.error.RateLimitError,
                                     openai.error.APIError, openai.error.APIConnectionError,
                                     openai.error.Timeout, openai.error.ServiceUnavailableError))
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(
        model=kwargs['engine'],
        temperature=kwargs['temperature'],
        max_tokens=kwargs['max_tokens'],
        presence_penalty=kwargs['presence_penalty'],
        frequency_penalty=kwargs['frequency_penalty'],
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": kwargs['prompt']},
        ]
    )


@lru_cache(maxsize=10000)
def get_gpt_output(prompt, **kwargs):
    gpt_logger.write(prompt)
    response = completions_with_backoff(prompt=prompt, engine=kwargs['engine'],
                                        temperature=kwargs['temperature'], max_tokens=kwargs['max_tokens'],
                                        presence_penalty=kwargs['presence_penalty'],
                                        frequency_penalty=kwargs['frequency_penalty'])

    response_str = response['choices'][0]['message']['content']
    gpt_logger.write(response_str)
    gpt_logger.write('#' * 55)
    return response_str


def get_batch_reward_loss(reward_model, diff_model_lst, sampler, scores, cand_examples, train_batch, batch_videos_gt_old,
                          args, id_map=None, **kwargs):
    batch_loss = 0
    batch_reward = 0
    batch_videos_pred = []
    batch_captions = []
    batch_log_prob = []
    batch_action_planning_pred = []
    batch_action_planning_gt = []
    batch_dsg_pred = []
    batch_dsg_gt = []

    batch_vidoes_gt = []
    ## loop over the training examples
    for i in tqdm(range(len(scores))):
        # interact with the environment to get rewards, which in our case is to feed the prompt into GPT-3 and evaluate the prediction
        cand_prob = scores[i, :].clone().detach()
        cand_prob = cand_prob.cpu().numpy()
        cand_prob = np.nan_to_num(cand_prob, nan=0.000001)  # replace np.nan with 0
        cand_prob /= cand_prob.sum()  # make probabilities sum to 1
        # print(f"cand_prob: {cand_prob}")

        # sample shot_pids from the cand_prob distribution
        # cand_prob = None # ablation: random selection
        cids = np.random.choice(range(len(cand_prob)), args.shot_number, p=cand_prob, replace=False)

        # reverse shot_pids so more relevant prompt will be put closer to the question
        cids = cids[::-1]
        # print(f"cids: {cids}")

        if id_map is not None:
            pool_ids = cids.copy()
            cids = id_map[i][pool_ids]
        else:
            pool_ids = cids

        shot_cand = [cand_examples[cid] for cid in cids]
        # print(f"shot_pids: {shot_pids}")

        # get the output from GPT
        gpt_args = dict(
            engine=args.engine,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            presence_penalty=args.presence_penalty,
            frequency_penalty=args.frequency_penalty
        )

        # Step 1: perform action planning
        prompt = build_prompt_for_action_planning(shot_cand, train_batch[i], args)

        output = get_gpt_output(prompt, **gpt_args)
        # extract the format from ChatGPT's output
        ap_prediction = utils.extract_prediction_for_action_planning(output)
        batch_action_planning_pred.append(ap_prediction)

        # Step 2: transform action sequence into DSG
        dsg = utils.build_dsg_from_action_planning(**ap_prediction)

        # Step 3: perform scene imagination and polishment over DSG
        prompt = build_prompt_for_scene_imagination(shot_cand, dsg, args)
        output = get_gpt_output(prompt, **gpt_args)
        # extract the format from ChatGPT's output
        si_prediction = utils.extract_prediction_for_scene_imagination(output)

        prompt = build_prompt_for_scene_polishment(shot_cand, si_prediction, args)
        output = get_gpt_output(prompt, **gpt_args)
        # extract the format from ChatGPT's output
        sp_prediction = utils.extract_prediction_for_scene_polishment(output)
        batch_dsg_pred.append(sp_prediction)

        videos_pred = sample_text2video(diff_model_lst, train_batch[i]['captions'], args.n_samples, args.batch_size,
                                      sampler=sampler)
        batch_videos_pred.append(videos_pred)

        batch_captions.append(train_batch[i]['captions'])
        batch_vidoes_gt.append(batch_videos_gt_old[i])

        batch_action_planning_pred.append(train_batch[i]["actions"])
        batch_dsg_pred.append(train_batch[i]["dsg"])

        log_prob = 0
        for pid in pool_ids:
            log_prob += torch.log(scores[i, pid])
        # print(f"log_prob: {log_prob}")
        batch_log_prob.append(log_prob)

    batch_reward, batch_reward_raw = reward_model(batch_videos_pred, batch_vidoes_gt,
                                                  dsg_pred=batch_dsg_pred, dsg_gt=batch_dsg_gt,
                                                  action_pred=batch_action_planning_pred, action_gt=batch_action_planning_gt)

    batch_reward = batch_reward_raw  # remove batch norm
    batch_log_prob = torch.stack(batch_log_prob)
    batch_loss = (-batch_log_prob * batch_reward).sum()
    # batch_reward = batch_reward.sum().item()
    batch_reward = batch_reward_raw.sum().item()

    return cids, batch_reward, batch_loss


def resume(ckpt_dir, model, optimizer, lr_scheduler):
    f_names = os.listdir(ckpt_dir)
    max_old_epo = 0
    for fn in f_names:
        if fn[:5] == 'state':
            s = int(fn.split('.')[0].split('_')[-1])
            if s > max_old_epo:
                max_old_epo = s

    weight_ckpt = torch.load(os.path.join(ckpt_dir, f'ckpt_{max_old_epo}.pt'))
    state_ckpt = torch.load(os.path.join(ckpt_dir, f'state_{max_old_epo}.pt'))
    model.linear.load_state_dict(weight_ckpt)
    optimizer.load_state_dict(state_ckpt['optimizer'])
    # lr_scheduler.load_state_dict(state_ckpt['lr_scheduler'])
    return max_old_epo


def policy_gradient_train(policy_model, reward_model, diff_model_lst, sampler, cand_examples,
                          cand_ids, all_train_feats, args, **kwargs):
    # REINFORCE
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=args.lr)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000000, gamma=0.1)

    # resume
    start_epoch = 0
    if len(args.resume) > 0:
        max_old_epo = resume(args.resume, policy_model, optimizer, lr_scheduler)
        start_epoch = max_old_epo + 1

    # use new learning rate
    for params in optimizer.param_groups:
        params['lr'] = args.lr

    # data
    emb_inp_cand = all_train_feats[cand_ids]
    emb_inp_cand = emb_inp_cand.to(args.device)

    train_dataset = instantiate_from_config(args.data)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)

    reward_history = []
    loss_history = []

    total_reward_history = []  # epoch based
    total_loss_history = []  # epoch based

    STOP_FLAG = False
    Eiters = 0

    for epoch in range(start_epoch, start_epoch + args.epochs):
        logger.write(f"Epoch: {epoch}")

        total_train_reward = 0
        total_train_loss = 0

        # We can simply set the batch_size to len(train_data) in few-shot setting.
        for batch_i, data_batch in enumerate(train_loader):
            logger.write(f"Batch: {batch_i}")

            train_batch, train_batch_emb_inp, train_batch_ids, train_batch_videos = data_batch
            train_batch_emb_inp = train_batch_emb_inp.to(args.device)

            # We need to encode cands again every time we update the network
            embedding_cands = policy_model(emb_inp_cand)  # len(cand_examples) x embedding_size
            embedding_ctxt = policy_model(train_batch_emb_inp)  # len(train_batch) x embedding_size

            scores = torch.mm(embedding_ctxt, embedding_cands.t())  # len(train_batch) x len(cand_examples)
            sort_idx = None
            # # print(f"unnormed scores: {scores}")

            scores = F.softmax(scores / args.policy_temperature, dim=1)  # len(train_batch) x len(cand_examples)

            cids, reward, loss = get_batch_reward_loss(reward_model, diff_model_lst, sampler, scores, cand_examples, train_batch,
                                                       train_batch_videos, args, id_map=sort_idx)

            logger.write(f"cids for sample[-1] in batch: {cids}")
            logger.write(f"Cand prob for sample[-1] in batch: {[round(x, 5) for x in scores[-1, :].tolist()]}")
            logger.write(f"### reward for the batch: {reward}")
            logger.write(f"### loss for the batch: {loss}\n")

            # linear layer has Weight and bias
            # prev_param = list(policy_model.linear.parameters())[0].clone()
            # print(f"prev_param: {prev_param.data}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # for each iteration/batch
            total_train_reward += reward
            total_train_loss += loss.item()

            reward_history.append(reward)
            loss_history.append(loss.item())
            tb_logger.log_value('reward', reward, step=Eiters)
            tb_logger.log_value('loss', loss.item(), step=Eiters)
            tb_logger.log_value('LR', optimizer.param_groups[0]['lr'], step=Eiters)
            Eiters += 1

            if np.isnan(loss.item()):
                STOP_FLAG = True
                break

        # for each epoch
        total_reward_history.append(total_train_reward)
        total_loss_history.append(total_train_loss)

        best_reward = max(total_reward_history)
        best_loss = min(total_loss_history)

        best_reward_epoch = total_reward_history.index(best_reward) + start_epoch
        best_loss_epoch = total_loss_history.index(best_loss) + start_epoch

        logger.write("============================================")
        logger.write(f"### Epoch: {epoch} / {args.epochs}")
        logger.write(f"### Total reward: {total_train_reward}, " + f"Total loss: {round(total_train_loss, 5)}, " +
                     f"Best reward: {best_reward} at epoch {best_reward_epoch}, " +
                     f"Best loss: {round(best_loss, 5)} at epoch {best_loss_epoch}\n")

        # save every epoch
        ckpt_file = os.path.join(args.ckpt_path, f"ckpt_{epoch}.pt")
        torch.save(policy_model.linear.state_dict(), ckpt_file)
        state_file = os.path.join(args.ckpt_path, f"state_{epoch}.pt")
        state = {'optimizer': optimizer.state_dict(), 'lr_scheduler': lr_scheduler.state_dict()}
        torch.save(state, state_file)
        logger.write(f"saved the ckpt to {ckpt_file} and {state_file}")

        # save best epoch
        if epoch == best_reward_epoch:
            ckpt_file = os.path.join(args.ckpt_path, "ckpt_best_reward.pt")
            torch.save(policy_model.linear.state_dict(), ckpt_file)
            logger.write(f"saved the best reward ckpt to {ckpt_file}")

        if epoch == best_loss_epoch:
            ckpt_file = os.path.join(args.ckpt_path, "ckpt_best_loss.pt")
            torch.save(policy_model.linear.state_dict(), ckpt_file)
            logger.write(f"saved the best loss ckpt to {ckpt_file}")

        # save reward and loss history
        history = {
            "reward_history": reward_history,
            "loss_history": loss_history,
            "total_reward_history": total_reward_history,
            "total_loss_history": total_loss_history,
        }
        history_file = os.path.join(args.ckpt_path, "history.json")
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2, separators=(',', ': '))

        # print cache info
        logger.write(get_gpt_output.cache_info())
        logger.write("============================================\n")
        lr_scheduler.step()

        if STOP_FLAG:
            break

    # save in the end
    ckpt_file = os.path.join(args.ckpt_path, "ckpt_final.pt")
    torch.save(policy_model.linear.state_dict(), ckpt_file)


def parse_args():
    parser = argparse.ArgumentParser()

    # User options
    parser.add_argument('--exp', type=str, default='exp0')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--shot_number', type=int, default=8, help='Number of n-shot training examples.')
    parser.add_argument('--seed', type=int, default=53, help='random seed')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument("-b", "--base", nargs="*", metavar="base_config.yaml",
                        help="paths to base configs. Loaded from left-to-right. Parameters can be overwritten or added with command-line options of the form `--key value`.",
                        default=list())

    # Data
    parser.add_argument('--data_root', type=str,
                        default='.')
    parser.add_argument('--sampled_data_dir', type=str, default='./data')

    parser.add_argument('--train_number', type=int, default=640, help='Number of training samples.')
    parser.add_argument('--cand_number', type=int, default=80, help='Number of candidate prompts.')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers to load data.')

    # GPT settings
    parser.add_argument('--engine', type=str, default='gpt-3.5-turbo', choices=['text-davinci-002', 'gpt-3.5-turbo'])
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_tokens',
                        type=int,
                        default=512,
                        help='The maximum number of tokens allowed for the generated answer.')
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)

    # Policy gradient settings
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--model_config',
                        type=str,
                        default='openai/clip-vit-base-patch32')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate of policy network.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs.')
    parser.add_argument('--embedding_size', type=int, default=128, help='Policy network final layer hidden state size.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=20,
                        help='Policy network training batch size. Set to train_number by default.')
    parser.add_argument('--policy_temperature', type=float, default=1.0)
    parser.add_argument('--diff_ckpt', type=str, default='model.ckpt')
    parser.add_argument('--ckpt_root', type=str, default='./models/t2v/')

    args = parser.parse_args()
    from omegaconf import OmegaConf
    config = OmegaConf.load(args.base)
    config.update(args)

    currentDateAndTime = datetime.now()
    currentTime = currentDateAndTime.strftime("_%Y_%m_%d_%H_%M_%S")
    config.exp = config.exp + currentTime

    # print and save the args
    config.ckpt_path = os.path.join(config.ckpt_root, config.exp)
    utils.create_dir(config.ckpt_path)
    _logger = utils.Logger(config.ckpt_path + '/args.txt')

    print('====Input Arguments====')
    _logger.write(json.dumps(vars(config), indent=2, sort_keys=False))
    return config


if __name__ == '__main__':
    args = parse_args()
    utils.set_seed(args.seed)

    cand_examples, cand_ids, all_train_feats = utils.load_data(args)

    ## policy network
    policy_model = PolicyNetwork(model_config=args.model_config,
                                 in_dim=768,
                                 add_linear=True,
                                 embedding_size=args.embedding_size,
                                 freeze_encoder=True)

    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")  # one GPU
    args.device = device
    policy_model = policy_model.to(device)

    reward_model = Reward(args.n_samples, args.batch_size, args, device=device)
    reward_model = reward_model.to(device)

    # get model & sampler
    diff_model_lst, _, _ = load_model(args, args.ckpt_path)
    ddim_sampler = DDIMSampler(diff_model_lst) if args.sample_type == "ddim" else None

    ## TRAINING
    logger = utils.Logger(os.path.join(args.ckpt_path, 'log.txt'))
    gpt_logger = utils.Logger(os.path.join(args.ckpt_path, 'gpt_log.txt'))
    tb_logger.configure(args.ckpt_path, flush_secs=5)
    policy_gradient_train(policy_model, reward_model, diff_model_lst, ddim_sampler, cand_examples,
                          cand_ids, all_train_feats, args)



