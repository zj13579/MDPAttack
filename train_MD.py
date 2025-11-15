import torch
import math
import argparse
from MDPAttack.multinomial_diffusion.diffusion_utils.utils.paths import add_parent_path
from MDPAttack.multinomial_diffusion.diffusion_utils.utils.seeds import set_seeds

# Exp
from MDPAttack.multinomial_diffusion.segmentation_diffusion.experiment import add_exp_args

# Data
add_parent_path(level=1)
from MDPAttack.multinomial_diffusion.data_process import get_data, get_data_id, add_data_args

# Model
from MDPAttack.multinomial_diffusion.segmentation_diffusion.model import get_model, get_model_id, add_model_args
from MDPAttack.multinomial_diffusion.segmentation_diffusion.layers.layers import *
from MDPAttack.multinomial_diffusion.diffusion_utils.diffusion_multinomial import *

# Optim
from MDPAttack.multinomial_diffusion.diffusion_utils.optim.multistep import get_optim, get_optim_id, add_optim_args


class MDPModelTrainer:
    def __init__(self, dataset_name='ml-100k', device_id=1):

        self.dataset_name = dataset_name
        self.device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
        self.epochs = {
            'filmtrust': 50,
            'ml-100k': 50,
            'ml-1m': 100，
            'ml-2m': 200，
        }.get(dataset_name, 50)

        parser = argparse.ArgumentParser()
        add_exp_args(parser)
        add_data_args(parser)
        add_model_args(parser)
        add_optim_args(parser)
        args = parser.parse_known_args()[0]

        set_seeds(args.seed)
        args.device = str(self.device)
        args.dataset = self.dataset_name
        self.args = args

        self.train_loader, self.data_shape, num_classes = get_data(self.args)
        self.args.num_classes = num_classes
        self.data_id = get_data_id(self.args)

    def loglik_bpd(self, model, x):
        return -model.log_prob(x).sum() / (math.log(2) * x.numel())

    def elbo_bpd(self, model, x):
        return self.loglik_bpd(model, x)

    def train_and_save_model(self, save_path=None):
        model_MD = get_model(self.args, self.data_shape)
        optimizer, scheduler_iter, scheduler_epoch = get_optim(self.args, model_MD)

        bpd_list = []
        for i in range(self.epochs):
            print(f'-------The {i + 1}th iteration begin-------')
            model_MD.train()
            loss_sum = 0.0
            loss_count = 0

            for x in self.train_loader:
                x = x.to(torch.int64).to(self.device)
                optimizer.zero_grad()

                loss = self.elbo_bpd(model_MD.to(self.device), x)
                loss.backward()

                if self.args.clip_value:
                    torch.nn.utils.clip_grad_value_(model_MD.parameters(), self.args.clip_value)
                if self.args.clip_norm:
                    torch.nn.utils.clip_grad_norm_(model_MD.parameters(), self.args.clip_norm)

                optimizer.step()
                if scheduler_iter:
                    scheduler_iter.step()

                loss_sum += loss.detach().cpu().item() * len(x)
                loss_count += len(x)

                print(f'Training. Epoch: {i + 1}/{self.epochs}, Datapoint: {loss_count}/{len(self.train_loader.dataset)}, Bits/dim: {loss_sum / loss_count:.3f}', end='\r')

            print('')
            if scheduler_epoch:
                scheduler_epoch.step()
            bpd_list.append(loss_sum / loss_count)

        if save_path is None:
            save_path = f'./MDPAttack/results/{self.dataset_name}_MD_model.pth'

        torch.save({
            'model_state_dict': model_MD.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)

        self.save_path = save_path
        return model_MD

    def load_trained_model(self):
        save_path = f'./MDPAttack/results/{self.dataset_name}_MD_model.pth'

        model_MD = get_model(self.args, data_shape=self.data_shape)
        saved_model = torch.load(save_path)
        model_MD.load_state_dict(saved_model['model_state_dict'])
        return model_MD
