import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
from sklearn.metrics import mean_squared_error
from math import sqrt
import os
from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader as GeometricDataLoader

def split_numsamples(dataset: torch.utils.data.Dataset, numtrain: int, numtest: int):
    total_samps = len(dataset) + 1
    train_prop = numtrain / total_samps
    print(f'Train proportion: {train_prop}')
    train_s, test_s = torch.utils.data.random_split(dataset, [train_prop, 1-train_prop])

    test_prop = numtest / (total_samps*(1-train_prop))
    print(f'Test proportion: {test_prop}')
    # chuck the unused samples away
    test_s, _ = torch.utils.data.random_split(test_s, [test_prop, 1-test_prop])
    return train_s, test_s


@hydra.main(config_path='conf', config_name='config')
def train(cfg: DictConfig):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(OmegaConf.to_yaml(cfg))
    print(f'Using device {device}')
    print("Working directory : {}".format(os.getcwd()))

    model = instantiate(cfg.model_data.model).to(device)
    dataset = instantiate(cfg.model_data.dataset, DEVICE=device)
    train_set, test_set = split_numsamples(dataset, cfg.stl_data.num_train, cfg.stl_data.num_test)

    # workaround for no hydra partial instantiation
    if model.lmax is not None:
        loader = GeometricDataLoader
    else:
        loader = DataLoader
    
    train_loader_yp = loader(train_set, batch_size=cfg.batch_size, shuffle=True)
    test_loader_yp_ = loader(test_set, batch_size=len(test_set), shuffle=True)

    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    losses = []
    test_losses = []
    for j in range(cfg.n_epochs):
        running_loss = 0
        for i, data in enumerate(train_loader_yp):
            inputs, labels = data
            
            optim.zero_grad()
            
            outputs = model(inputs).squeeze()
            
            loss = loss_fn(outputs, labels)
            loss.backward()
            
            optim.step()
            running_loss += loss.item()

        def do_eval():
            with torch.no_grad():
                for i, data in enumerate(test_loader_yp_):

                        inputs, labels = data
                        output = model(inputs)

                return sqrt(mean_squared_error(labels.cpu().numpy(), output.cpu().numpy().squeeze()))
            
        rmse = do_eval()
        running_loss_sample = running_loss / len(train_loader_yp)
        rmse_sample = rmse
        if cfg.verbose:
            print('Epoch {} loss: {}, test RMSE: {}'.format(j + 1, running_loss_sample, rmse_sample))

        losses.append(running_loss_sample)
        test_losses.append(rmse_sample)
        
        if cfg.weights.save_weights:
            if not os.path.exists("weights"):
                os.makedirs("weights")
            
            # hydra saves weights in output folder (is cwd)
            if (j+1) % cfg.weights.save_epoch_modulus == 0:
                save_loc = f'weights/model_epoch_{j}.pt'
                print("Saving weights at:", save_loc)
                torch.save(model.state_dict(), save_loc)


if __name__ == "__main__":
    train()