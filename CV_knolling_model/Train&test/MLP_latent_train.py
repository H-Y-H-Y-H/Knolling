import cv2
import torch
from torch import nn
from torchvision import datasets, transforms
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, ToPILImage

from torch.utils.data import DataLoader
from Neural_model import VAE, CustomImageDataset, EncoderDecoder, MLP_latent, EmbeddedImageDataset
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
        running_name = 'zzz_test_MLP'

    config.messy_encoder_path = 'results/scarlet-monkey-66/best_model.pt'
    config.tidy_encoder_path = 'results/peach-water-65/best_model.pt'
    config.log_pth = f'results/{running_name}/'
    config.patience = 50
    config.loss_d_epoch = 20
    config.dataset_path = dataset_path
    config.num_data = num_data
    config.scheduler_factor = 0.1
    config.mlp_hidden = [256]
    config.latent_dim = embedding_dim
    config.lr = 0.0001
    config.messy_kl_weight = 0.00005
    config.tidy_kl_weight = 0.00005
    # config.kl_weight = 0.0001

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

    messy_model = EncoderDecoder(in_channels=3, latent_dim=img_latent_dim, kl_weight=config.messy_kl_weight).to(device)
    tidy_model = EncoderDecoder(in_channels=3, latent_dim=img_latent_dim, kl_weight=config.tidy_kl_weight).to(device)
    messy_model.load_state_dict(torch.load(config.messy_encoder_path, map_location=device))
    tidy_model.load_state_dict(torch.load(config.tidy_encoder_path, map_location=device))

    if config.pre_trained:
        model = MLP_latent(mlp_hidden=config.mlp_hidden, out_layer=img_latent_dim, in_layer=img_latent_dim + embedding_dim).to(device)
        model.load_state_dict(torch.load(pretrain_model_path, map_location=device))
    else:
        model = MLP_latent(mlp_hidden=config.mlp_hidden, out_layer=img_latent_dim, in_layer=img_latent_dim + embedding_dim).to(device)

    # load embedding
    embedding = nn.Embedding(num_sol, embedding_dim).to(device)

    # Training setup
    optimizer = torch.optim.Adam(list(embedding.parameters())+list(model.parameters()), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.scheduler_factor,
                                                           patience=config.loss_d_epoch, verbose=True)

    min_loss = np.inf
    for epoch in range(epochs):
        model.train()
        train_loss = []
        index = 0
        for batch_idx, (messy_img, tidy_img, index_sol_list) in enumerate(train_loader):

            optimizer.zero_grad()

            messy_img = messy_img.to(device)
            tidy_img = tidy_img.to(device)
            index_sol_list = index_sol_list.to(device)
            index_embedding = embedding(index_sol_list)

            messy_latent = messy_model.get_latent_space(messy_img)
            messy_img_test, _, _ = messy_model(messy_img)
            img_messy_gt = messy_img[1].detach().cpu()
            img_messy_recon = messy_img_test[1].detach().cpu()
            combined = torch.cat((img_messy_recon, img_messy_gt), 1)
            img = ToPILImage()(combined)
            img.save(f'eval_{index}.jpg')

            tidy_latent = tidy_model.get_latent_space(tidy_img)
            tidy_img_test, _, _ = tidy_model(tidy_img)
            img_tidy_gt = (tidy_img.cpu().detach().numpy()[0] * 255).astype(np.uint8)
            img_tidy_gt = np.transpose(img_tidy_gt, [1, 2, 0])
            img_tidy_recon = (tidy_img_test.cpu().detach().numpy()[0] * 255).astype(np.uint8)
            img_tidy_recon = np.transpose(img_tidy_recon, [1, 2, 0])
            tidy_compare = np.concatenate((img_tidy_gt, img_tidy_recon), axis=1)
            cv2.namedWindow('zzz', 0)
            # cv2.resizeWindow('zzz', 1280, 960)
            cv2.imshow("zzz", tidy_compare)
            cv2.waitKey()
            cv2.destroyAllWindows()

            messy_latent = torch.flatten(messy_latent, 1)
            tidy_latent = torch.flatten(tidy_latent, 1)

            messy_embedded_latent = torch.cat((messy_latent, index_embedding), 1)

            pred_tidy_latent = model(messy_embedded_latent)

            tra_loss = model.loss_function(pred_tidy_latent, tidy_latent)
            tra_loss.backward()
            train_loss.append(tra_loss.item())
            optimizer.step()

            index += 1

        train_loss = np.mean(np.asarray(train_loss))

        model.eval()
        valid_loss = []
        with torch.no_grad():
            for messy_img, tidy_img, index_sol_list in test_loader:

                messy_img = messy_img.to(device)
                tidy_img = tidy_img.to(device)
                index_sol_list = index_sol_list.to(device)
                index_embedding = embedding(index_sol_list)

                messy_latent = messy_model.get_latent_space(messy_img)
                tidy_latent = tidy_model.get_latent_space(tidy_img)
                messy_latent = torch.flatten(messy_latent, 1)
                tidy_latent = torch.flatten(tidy_latent, 1)

                messy_embedded_latent = torch.cat((messy_latent, index_embedding), 1)

                pred_tidy_latent = model(messy_embedded_latent)

                val_loss = model.loss_function(pred_tidy_latent, tidy_latent)
                valid_loss.append(val_loss.item())

            valid_loss = np.mean(np.asarray(valid_loss))
        scheduler.step(valid_loss)


        if valid_loss < min_loss:
            print(f'Epoch {epoch}, Train Loss: {train_loss}, Validation Loss: {valid_loss}, Lr: {optimizer.param_groups[0]["lr"]}')
            min_loss = valid_loss
            PATH = config.log_pth + '/best_model.pt'
            torch.save(model.state_dict(), PATH)
            abort_learning = 0
        if valid_loss > 10e4:
            print('outlier!')
            abort_learning = 10000
        else:
            abort_learning += 1

        if epoch % 20 == 0:
            torch.save(model.state_dict(), config.log_pth + '/latest_model.pt')

        if wandb_flag == True:
            wandb.log({"train_loss": train_loss,
                       "valid_loss": valid_loss,
                       "learning_rate": optimizer.param_groups[0]['lr'],
                       })

        if abort_learning > config.patience:
            print('abort training!')
            break

if __name__ == '__main__':
    torch.manual_seed(0)

    num_epochs = 500
    num_scenario = 100
    num_sol = 12
    num_data = num_scenario * num_sol

    dataset_path = '../../../knolling_dataset/VAE_329_obj4/'
    wandb_flag = False
    pre_train_flag = False
    proj_name = "MLP_latent"

    train_input = []
    train_output = []
    test_input = []
    test_output = []
    num_train_sce = int(num_scenario * 0.8)
    num_test_sce = int(num_scenario - num_train_sce)

    img_latent_dim = 256
    embedding_dim = img_latent_dim

    batch_size = 64
    transform = Compose([
        ToTensor()  # Normalize the image
    ])

    train_dataset = EmbeddedImageDataset(input_dir=dataset_path + 'images_before/',
                                         output_dir=dataset_path + 'images_after/',
                                         num_img=num_train_sce, sce_start_idx=0,
                                         transform=transform, num_sol=num_sol, num_sce=num_train_sce)
    test_dataset = EmbeddedImageDataset(input_dir=dataset_path + 'images_before/',
                                        output_dir=dataset_path + 'images_after/',
                                        num_img=num_test_sce, sce_start_idx=num_train_sce,
                                        transform=transform, num_sol=num_sol, num_sce=num_test_sce)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    main(num_epochs)