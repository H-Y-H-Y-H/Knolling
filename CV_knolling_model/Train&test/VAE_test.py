import numpy as np
from torch.utils.data import DataLoader
from VAE_model import VAE, CustomImageDataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, ToPILImage
import torch
from VAE_model import VAE, CustomImageDataset, EncoderDecoder
import numpy as np
import wandb
import yaml
import os
import argparse

def reconstruct(device, dataloader, model):
    model.eval()

    eval_loss = []
    eval_recon_loss = []
    eval_kl_loss = []
    index = 0
    for img_rdm, img_neat in dataloader:
        img_rdm = img_rdm.to(device)
        img_recon, mean, logvar = model(img_rdm)
        loss, recon_loss, kl_loss = model.loss_function(img_recon, img_rdm, mean, logvar)
        eval_loss.append(loss.item())
        eval_recon_loss.append(recon_loss.item())
        eval_kl_loss.append(kl_loss.item())
        output = img_recon[0].detach().cpu()
        input = img_rdm[0].detach().cpu()
        combined = torch.cat((output, input), 1)
        img = ToPILImage()(combined)
        img.save(f'results/{running_name}/eval_{index}.jpg')

        index += 1

        if index >= num_sample:
            break

    eval_loss = np.mean(np.asarray(eval_loss))
    eval_recon_loss = np.mean(np.asarray(eval_recon_loss))
    eval_kl_loss = np.mean(np.asarray(eval_kl_loss))
    print('eval loss:', eval_loss)
    print('eval recon loss:', eval_recon_loss)
    print('eval kl loss:', eval_kl_loss)

    with open(f'results/{running_name}/report_128.txt', "w") as f:
        f.write('----------- Dataset -----------\n')

        f.write('----------- Dataset -----------\n')

        f.write('----------- Statistics -----------\n')
        f.write(f'eval loss: {eval_loss}\n')
        f.write(f'eval recon loss: {eval_recon_loss}\n')
        f.write(f'eval kl loss: {eval_kl_loss}\n')
        f.write('----------- Statistics sundry_box_4-----------\n')

    # batch = next(iter(dataloader))
    # x = batch[0:1, ...].to(device)
    # output = model(x)[0]
    # output = output[0].detach().cpu()
    # input = batch[0].detach().cpu()
    # combined = torch.cat((output, input), 1)
    # img = ToPILImage()(combined)
    # img.save(f'results/{running_name}/tmp.jpg')

def gaussian_sample(device, model, batch_size, latent_dim):
    model.eval()

    for i in range(num_sample):
        input_sample = torch.randn(batch_size, latent_dim).to(device)
        img_recon = model.sample(input_sample)
        output = img_recon[12].detach().cpu()
        input = input_sample[12].detach().cpu()
        # combined = torch.cat((output, input), 1)
        img = ToPILImage()(output)
        img.save(f'results/{running_name}/sample_{i}.jpg')

    eval_loss = []
    eval_recon_loss = []
    eval_kl_loss = []
    index = 0
    # for img_rdm, img_neat in dataloader:
    #     img_rdm = img_rdm.to(device)
    #     img_recon, mean, logvar = model(img_rdm)
    #     loss, recon_loss, kl_loss = model.loss_function(img_recon, img_rdm, mean, logvar)
    #     eval_loss.append(loss.item())
    #     eval_recon_loss.append(recon_loss.item())
    #     eval_kl_loss.append(kl_loss.item())
    #     output = img_recon[12].detach().cpu()
    #     input = img_rdm[12].detach().cpu()
    #     combined = torch.cat((output, input), 1)
    #     img = ToPILImage()(combined)
    #     img.save(f'results/{running_name}/tmp_{index}_128.jpg')
    #
    #     index += 1
    #
    # eval_loss = np.mean(np.asarray(eval_loss))
    # eval_recon_loss = np.mean(np.asarray(eval_recon_loss))
    # eval_kl_loss = np.mean(np.asarray(eval_kl_loss))
    # print('eval loss:', eval_loss)
    # print('eval recon loss:', eval_recon_loss)
    # print('eval kl loss:', eval_kl_loss)
    #
    # with open(f'results/{running_name}/report_128.txt', "w") as f:
    #     f.write('----------- Dataset -----------\n')
    #
    #     f.write('----------- Dataset -----------\n')
    #
    #     f.write('----------- Statistics -----------\n')
    #     f.write(f'eval loss: {eval_loss}\n')
    #     f.write(f'eval recon loss: {eval_recon_loss}\n')
    #     f.write(f'eval kl loss: {eval_kl_loss}\n')
    #     f.write('----------- Statistics sundry_box_4-----------\n')

def main():

    if before_after == 'before':
        dataset_path = '../../../knolling_dataset/VAE_329_obj4/images_before/'
    elif before_after == 'after':
        dataset_path = '../../../knolling_dataset/VAE_329_obj4/images_after/'

    wandb_flag = False
    proj_name = "VAE_knolling"

    train_input = []
    train_output = []
    test_input = []
    test_output = []
    num_train = int(num_data * 0.8)
    num_test = int(num_data - num_train)

    batch_size = 64
    with open(f'results/{running_name}/config.yaml', 'r') as yaml_file:
        config_dict = yaml.safe_load(yaml_file)
    config = {k: v for k, v in config_dict.items() if not k.startswith('_')}
    config = argparse.Namespace(**config)
    config.pre_trained = True
    # config.mlp_latent_enable = True

    device = 'cuda:0'
    pretrain_model_path = f'results/{running_name}/best_model.pt'

    # # customized setting
    # config.latent_dim = 256
    # config.kl_weight = 0.00005
    config.mlp_latent_enable = False
    config.mlp_hidden = [256]
    # # customized setting

    model = EncoderDecoder(in_channels=3, latent_dim=config.latent_dim, kl_weight=config.kl_weight,
                           mlp_hidden=config.mlp_hidden, mlp_latent_enable=config.mlp_latent_enable).to(device)

    model.load_state_dict(torch.load(pretrain_model_path, map_location=device))

    if flag == 'sample':
        gaussian_sample(device, model=model, batch_size=batch_size, latent_dim=config.latent_dim)
    else:
        transform = Compose([
            ToTensor()  # Normalize the image
        ])
        train_dataset = CustomImageDataset(input_dir=dataset_path,
                                           output_dir=dataset_path,
                                           num_img=num_train, num_total=num_data, start_idx=0,
                                           transform=transform)
        test_dataset = CustomImageDataset(input_dir=dataset_path,
                                          output_dir=dataset_path,
                                          num_img=num_test, num_total=num_data, start_idx=num_train,
                                          transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # model = torch.load(f'results/{running_name}/best_model.pt', 'cuda:0').to(device)
        reconstruct(device, dataloader=train_loader, model=model)

def show_structure():

    device = 'cuda:0'
    model = torch.load(f'results/{running_name}/best_model.pt', 'cuda:0').to(device)

    print(model)

if __name__ == '__main__':

    before_after = 'before'
    flag = 'eval'
    torch.manual_seed(0)
    running_name = 'zzz_test'

    # running_name = 'peach-water-65'
    running_name = 'scarlet-monkey-66'
    #
    # running_name = 'ethereal-sea-79'
    #
    # running_name = 'stoic-surf-84'
    # running_name = 'pleasant-resonance-87'
    #
    # running_name = 'solar-violet-39'

    num_epochs = 100
    num_data = 1200
    num_sample = 10

    main()
    # show_structure()
