import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
from math import sqrt
import os
from tqdm import tqdm

def split_numsamples(dataset, numtrain, numtest):
    # (Kept for backwards compatibility with old configs)
    total_samps = len(dataset) + 1
    train_prop = numtrain / total_samps
    train_s, test_s = torch.utils.data.random_split(dataset, [train_prop, 1-train_prop])
    test_prop = numtest / (total_samps*(1-train_prop))
    test_s, _ = torch.utils.data.random_split(test_s, [test_prop, 1-test_prop])
    return train_s, test_s

@hydra.main(config_path='configs', config_name='config')
def train(cfg: DictConfig):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(OmegaConf.to_yaml(cfg))
    print(f'Using device {device}')
    
    # --- DYNAMIC CONFIG ROUTING ---
    if "model_data" in cfg:
        print("Detected OLD config structure. Loading manual dataset...")
        model = instantiate(cfg.model_data.model).to(device)
        dataset = instantiate(cfg.model_data.dataset, DEVICE=device)
        train_set, test_set = split_numsamples(dataset, cfg.stl_data.num_train, cfg.stl_data.num_test)
        
        if hasattr(model, 'lmax') or 'GCN' in cfg.model_data.model._target_:
            from torch_geometric.data import DataLoader as GeometricDataLoader
            loader = GeometricDataLoader
        else:
            from torch.utils.data import DataLoader
            loader = DataLoader
            
        train_loader_yp = loader(train_set, batch_size=cfg.batch_size, shuffle=True)
        test_loader_yp_ = loader(test_set, batch_size=cfg.batch_size, shuffle=True)
        
        lr = cfg.lr
        n_epochs = cfg.n_epochs
        save_weights = cfg.weights.save_weights
        save_modulus = cfg.weights.save_epoch_modulus
        
    else:
        print("Detected NEW Lightning config structure. Running Lightning Trainer...")
		# 1. Instantiate DataModule
        datamodule = instantiate(cfg.dataset)
		
		# 2. Instantiate the FULL Lightning Module (not just the raw PyTorch model)
		# This will contain your model, loss function, and optimizer logic
        lightning_module = instantiate(cfg.model)
		
		# 3. Instantiate the Lightning Trainer
        trainer = instantiate(cfg.trainer)
		
		# 4. Train! (This replaces your entire custom 'for' loop)
        trainer.fit(model=lightning_module, datamodule=datamodule)
		
		# Return here so it doesn't run the manual PyTorch loop below
        return

        print("Detected NEW Lightning config structure. Loading DataModule...")
        # 1. Initialize DataModule directly
        datamodule = instantiate(cfg.dataset)
        datamodule.setup()
        train_loader_yp = datamodule.train_dataloader()
        test_loader_yp_ = datamodule.val_dataloader() # Use val for testing during training
        
        # 2. Extract raw PyTorch model from the Lightning config
        model = instantiate(cfg.model.model_config).to(device)
        
        # 3. Pull training params from Lightning config locations
        lr = cfg.model.optimizer_config.lr
        n_epochs = cfg.trainer.max_epochs
        
        # Default saving behavior for new configs
        save_weights = True
        save_modulus = 5

    # --- STANDARD TRAINING LOOP ---
    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    
    for j in range(n_epochs):
        running_loss = 0
        pbar = tqdm(enumerate(train_loader_yp), total=len(train_loader_yp))
        pbar.set_description(f"loss: 0")
        
        model.train()
        for i, data in pbar:
            optim.zero_grad()
            
            if isinstance(data, (list, tuple)):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
            else:
                data = data.to(device)
                labels = data.y
                outputs = model(data).squeeze()
            
            loss = loss_fn(outputs, labels.squeeze())
            loss.backward()
            pbar.set_description(f"loss: {format(loss.item(), '.5f')}")
            optim.step()
            running_loss += loss.item()

        def do_eval():
            running_test_loss = 0
            model.eval()
            with torch.no_grad():
                for i, data in enumerate(test_loader_yp_):
                    if isinstance(data, (list, tuple)):
                        inputs, labels = data
                        inputs, labels = inputs.to(device), labels.to(device)
                        output = model(inputs).squeeze()
                    else:
                        data = data.to(device)
                        labels = data.y
                        output = model(data).squeeze()

                    loss = loss_fn(output, labels.squeeze())
                    running_test_loss += loss.item()
            return running_test_loss
                
        running_test_loss = do_eval()
        rmse_test = sqrt(running_test_loss / len(test_loader_yp_))
        running_loss_sample = sqrt(running_loss / len(train_loader_yp))
        
        print('Epoch {} loss: {:.5f}, test RMSE: {:.5f}'.format(j + 1, running_loss_sample, rmse_test))
        
        if save_weights:
            if not os.path.exists("weights"):
                os.makedirs("weights")
            if (j+1) % save_modulus == 0:
                save_loc = f'weights/model_epoch_{j}.pt'
                torch.save(model.state_dict(), save_loc)

if __name__ == "__main__":
    train()
