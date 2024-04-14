import cv2
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from Neural_model import VAE, CustomImageDataset, EncoderDecoder
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import wandb
import yaml
import os
import argparse

def main(epochs):

    if wandb_flag == True:

        wandb.init(project=proj_name)
        running_name = wandb.run.name
        config = wandb.config
        config.pre_trained = pre_train_flag

        if config.pre_trained == True:
            # Load the YAML file
            pretrained_model = 'balmy-valley-10'
            with open(f'results/{pretrained_model}/config.yaml', 'r') as yaml_file:
                config_dict = yaml.safe_load(yaml_file)
            config = {k: v for k, v in config_dict.items() if not k.startswith('_')}
            config = argparse.Namespace(**config)
            config.pre_trained = True

            pretrain_model_path = f'results/{pretrained_model}/best_model.pt'
    else:
        config = {}
        config = argparse.Namespace(**config)
        config.pre_trained = pre_train_flag
        running_name = 'zzz_test'


    config.log_pth = f'data/{running_name}/'
    config.patience = 50
    config.loss_d_epoch = 20
    config.dataset_path = dataset_path
    config.num_data = num_data
    config.scheduler_factor = 0.1
    config.conv_hidden = [16, 32, 64, 128, 256]
    config.latent_dim = 4096
    config.latent_enable = True
    config.lr = 0.01
    # config.kl_weight = 0.01
    config.kl_weight = 0.0001

    os.makedirs(config.log_pth, exist_ok=True)

    # Assuming 'config' is your W&B configuration object
    try:
        config_dict = dict(config)  # Convert to dictionary if necessary
    except:
        config_dict = vars(config)
    # Save as YAML
    with open(config.log_pth + 'config.yaml', 'w') as yaml_file:
        yaml.dump(config_dict, yaml_file, default_flow_style=False)
    print(config)

    # Mapping device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Use: ', device)

    # Model instantiation

    if config.pre_trained:
        model = torch.load(pretrain_model_path, map_location=device)
    else:
        # model = VAE(
        #     conv_hiddens=config.conv_hidden, latent_dim=config.latent_dim, img_length_width=128, latent_dim_enable=config.latent_enable,
        #     kl_weight=config.kl_weight
        # ).to(device)
        model = EncoderDecoder(in_channels=3).to(device)
        # model = VAE(
        #     ).to(device)
        # model.load_state_dict(checkpoint)


    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.scheduler_factor,
                                                           patience=config.loss_d_epoch, verbose=True)
    # criterion = nn.BCELoss(reduction='sum')

    min_loss = np.inf
    for epoch in range(epochs):
        model.train()
        train_loss = []
        # train_recon_loss = []
        # train_kl_loss = []
        for batch_idx, (img_rdm, img_neat) in enumerate(train_loader):
            img_rdm = img_rdm.to(device)
            img_neat = img_neat.to(device)
            optimizer.zero_grad()
            img_recon, latent = model(img_rdm)

            # img_check = (img_recon.squeeze().cpu().detach().numpy().transpose([1, 2, 0]) * 255).astype(np.uint8)
            # cv2.namedWindow('zzz', 0)
            # cv2.resizeWindow('zzz', 1280, 960)
            # cv2.imshow("zzz", img_check)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

            loss = model.loss_function(img_recon, img_rdm)
            loss.backward()
            train_loss.append(loss.item())
            # train_recon_loss.append(recon_loss.item())
            # train_kl_loss.append(kl_loss.item())
            optimizer.step()

        train_loss = np.mean(np.asarray(train_loss))
        # train_recon_loss = np.mean(np.asarray(train_recon_loss))
        # train_kl_loss = np.mean(np.asarray(train_kl_loss))

        model.eval()
        val_loss = []
        # val_recon_loss = []
        # val_kl_loss = []
        with torch.no_grad():
            for img_rdm, img_neat in test_loader:
                img_rdm = img_rdm.to(device)
                img_neat = img_neat.to(device)
                img_recon, latent = model(img_rdm)
                loss = model.loss_function(img_recon, img_rdm)
                val_loss.append(loss.item())
                # val_recon_loss.append(recon_loss.item())
                # val_kl_loss.append(kl_loss.item())

            val_loss = np.mean(np.asarray(val_loss))
            # val_recon_loss = np.mean(np.asarray(val_recon_loss))
            # val_kl_loss = np.mean(np.asarray(val_kl_loss))
        scheduler.step(val_loss)


        if val_loss < min_loss:
            print(f'Epoch {epoch}, Train Loss: {train_loss}, Validation Loss: {val_loss}, Lr: {optimizer.param_groups[0]["lr"]}')
            min_loss = val_loss
            PATH = config.log_pth + '/best_model.pt'
            torch.save(model, PATH)
            abort_learning = 0
        if val_loss > 10e8:
            print('outlier!')
            abort_learning = 10000
        else:
            abort_learning += 1

        if epoch % 20 == 0:
            torch.save(model, config.log_pth + '/latest_model.pt')

        if wandb_flag == True:
            wandb.log({"train_loss": train_loss,
                       # "train_recon_loss": train_recon_loss,
                       # "train_kl_loss": train_kl_loss,
                       "valid_loss": val_loss,
                       # "valid_recon_loss": val_recon_loss,
                       # "valid_kl_loss": val_kl_loss,
                       "learning_rate": optimizer.param_groups[0]['lr'],
                       })

        if abort_learning > config.patience:
            print('abort training!')
            break



if __name__ == "__main__":

    num_epochs = 500
    num_data = 1200
    dataset_path = '../../../knolling_dataset/VAE_329_obj4/'
    wandb_flag = False
    pre_train_flag = False
    proj_name = "VAE_knolling"

    train_input = []
    train_output = []
    test_input = []
    test_output = []
    num_train = int(num_data * 0.8)
    num_test = int(num_data - num_train)

    batch_size = 64
    transform = Compose([
                        ToTensor()  # Normalize the image
                        ])
    train_dataset = CustomImageDataset(input_dir=dataset_path + 'images_before/',
                                       output_dir=dataset_path + 'images_before/',
                                       num_img=num_train, num_total=num_data, start_idx=0,
                                       transform=transform)
    # test_dataset = CustomImageDataset(input_dir=dataset_path + 'images_before/',
    #                                    output_dir=dataset_path + 'images_before/',
    #                                    num_img=num_test, num_total=num_data, start_idx=num_train,
    #                                     transform=transform)
    test_dataset = CustomImageDataset(input_dir=dataset_path + 'images_before/',
                                      output_dir=dataset_path + 'images_before/',
                                      num_img=num_train, num_total=num_data, start_idx=0,
                                      transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Training the VAE for a few epochs to see the initial behavior
    main(num_epochs)
