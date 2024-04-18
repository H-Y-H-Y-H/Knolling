import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch.nn.functional as F

class CustomImageDataset(Dataset):
    def __init__(self, input_dir, output_dir, num_img, num_total, start_idx, transform=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.num_img = num_img
        self.num_total=num_total
        self.start_idx = start_idx
        self.transform = transform

        self.img_input = []
        self.img_output = []
        for i in tqdm(range(num_img)):

            img_input = Image.open(input_dir + '%d.png' % (i+ self.start_idx))
            img_input = self.transform(img_input)
            self.img_input.append(img_input)

            # img_check = (img_input.numpy() * 255).astype(np.uint8)
            # img_check = np.transpose(img_check, [1, 2, 0])
            # cv2.namedWindow('zzz', 0)
            # cv2.resizeWindow('zzz', 1280, 960)
            # cv2.imshow("zzz", img_check)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

            img_output = Image.open(output_dir + '%d.png' % (i+ self.start_idx))
            img_output = self.transform(img_output)
            self.img_output.append(img_output)

            # img_check = (img_output.numpy() * 255).astype(np.uint8)
            # img_check = np.transpose(img_check, [1, 2, 0])
            # cv2.namedWindow('zzz', 0)
            # cv2.resizeWindow('zzz', 1280, 960)
            # cv2.imshow("zzz", img_check)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

    def __len__(self):
        return self.num_img

    def __getitem__(self, idx):

        return self.img_input[idx], self.img_output[idx]

class EmbeddedImageDataset(Dataset):
    def __init__(self, input_dir, output_dir, num_img, sce_start_idx, num_sol, num_sce, transform=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.num_img = num_img
        self.sce_start_idx = sce_start_idx
        self.transform = transform

        self.img_input = []
        self.img_output = []
        self.index_sol_list = []

        for i in tqdm(range(num_sce)):

            for j in range(num_sol):

                img_input_path = input_dir + '%d.png' % (num_sol * (i + self.sce_start_idx) + j)
                img_output_path = output_dir + '%d.png' % (num_sol * (i + self.sce_start_idx) + j)
                img_input = Image.open(img_input_path)
                img_input = self.transform(img_input)
                img_output = Image.open(img_output_path)
                img_output = self.transform(img_output)
                for k in range(num_sol):
                    self.img_input.append(img_input)
                    self.img_output.append(img_output)
                    self.index_sol_list.append(k)
        print('here')

        # for i in tqdm(range(num_img)):
        #
        #     index_sol = i % num_sol
        #
        #     img_input = Image.open(input_dir + '%d.png' % (i+ self.start_idx))
        #     img_input = self.transform(img_input)
        #
        #     self.img_input.append(img_input)
        #     self.index_sol_list.append(index_sol)
        #
        #     img_output = Image.open(output_dir + '%d.png' % (i+ self.start_idx))
        #     img_output = self.transform(img_output)
        #
        #     self.img_output.append(img_output)
        self.index_sol_list = torch.LongTensor(self.index_sol_list)
        # self.embedded_index = embedding(self.index_sol_list)

    def __len__(self):
        return self.num_img

    def __getitem__(self, idx):

        return self.img_input[idx], self.img_output[idx], self.index_sol_list[idx]

class VAE(nn.Module):
    """VAE for 64x64 face generation.

    The hidden dimensions can be tuned.
    """

    def __init__(self, conv_hiddens=[16, 32, 64, 128, 256], latent_dim=128, img_length_width=128, latent_dim_enable=True,
                 kl_weight=0.01) -> None:
        super().__init__()

        # encoder
        prev_channels = 3
        modules = []
        img_length = 128
        self.kl_weight = kl_weight

        for cur_channels in conv_hiddens:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(prev_channels,
                              cur_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1), nn.BatchNorm2d(cur_channels),
                    nn.ReLU()))
            prev_channels = cur_channels
            img_length //= 2
        self.encoder = nn.Sequential(*modules)

        if latent_dim_enable == False:
            latent_dim = prev_channels * img_length * img_length

        self.mean_linear = nn.Linear(prev_channels * img_length * img_length,
                                     latent_dim)
        self.var_linear = nn.Linear(prev_channels * img_length * img_length,
                                    latent_dim)
        self.latent_dim = latent_dim
        # decoder
        modules = []
        self.decoder_projection = nn.Linear(
            latent_dim, prev_channels * img_length * img_length)
        self.decoder_input_chw = (prev_channels, img_length, img_length)
        for i in range(len(conv_hiddens) - 1, 0, -1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(conv_hiddens[i],
                                       conv_hiddens[i - 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(conv_hiddens[i - 1]), nn.ReLU()))
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(conv_hiddens[0],
                                   conv_hiddens[0],
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                nn.BatchNorm2d(conv_hiddens[0]), nn.ReLU(),
                nn.Conv2d(conv_hiddens[0], 3, kernel_size=3, stride=1, padding=1),
                nn.ReLU()))
        self.decoder = nn.Sequential(*modules)

    def forward(self, x):
        encoded = self.encoder(x)
        encoded = torch.flatten(encoded, 1)
        mean = self.mean_linear(encoded)
        logvar = self.var_linear(encoded)
        eps = torch.randn_like(logvar)
        std = torch.exp(logvar / 2)
        z = eps * std + mean
        x = self.decoder_projection(z)
        x = torch.reshape(x, (-1, *self.decoder_input_chw))
        decoded = self.decoder(x)

        return decoded, mean, logvar

    def sample(self, device='cuda'):
        z = torch.randn(1, self.latent_dim).to(device)
        x = self.decoder_projection(z)
        x = torch.reshape(x, (-1, *self.decoder_input_chw))
        decoded = self.decoder(x)
        return decoded

    def loss_function(self, y, y_hat, mean, logvar):
        recons_loss = F.mse_loss(y_hat, y)
        kl_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mean ** 2 - torch.exp(logvar), 1), 0)
        loss = recons_loss + kl_loss * self.kl_weight
        return loss, recons_loss, kl_loss * self.kl_weight

class MLP_latent(torch.nn.Module):

    def __init__(self, out_layer, in_layer, mlp_hidden = [4096, 2048, 4096]):
        super(MLP_latent, self).__init__()

        # self.layer1 = nn.Linear(input_size, hidden1_size)
        # self.relu = nn.ReLU()
        # self.layer2 = nn.Linear(hidden1_size, hidden2_size)
        # self.output_layer = nn.Linear(hidden2_size, num_classes)

        modules = []
        prev_layer = in_layer
        for cur_layer in mlp_hidden:
            modules.append(nn.Sequential(
                    nn.Linear(prev_layer, cur_layer),
                    nn.ReLU()))
            prev_layer = cur_layer
        modules.append(nn.Sequential(
                    nn.Linear(mlp_hidden[-1], out_layer)))

        self.MLP_module = nn.Sequential(*modules)

    def forward(self, x):
        output = self.MLP_module(x)
        return output

        # x_shape = x.size()
        # encoded = torch.flatten(x, 1)
        # mean = self.MLP_module(encoded)
        # logvar = self.MLP_module(encoded)
        # # out = out.resize(x_shape[0], x_shape[1], x_shape[2], x_shape[3])
        # eps = torch.randn_like(logvar)
        # std = torch.exp(logvar / 2)
        # z = eps * std + mean
        # out = self.decoder_projection(z)
        # out = torch.reshape(out, (-1, x_shape[1], x_shape[2], x_shape[3]))
        # return out, mean, logvar

    def loss_function(self, pred_latent, latent):

        mse_loss = F.mse_loss(latent, pred_latent)
        return mse_loss

def conv2d_bn_relu(inch, outch, kernel_size, stride=1, padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.Conv2d(inch, outch, kernel_size=kernel_size, stride=stride, padding=padding),
        torch.nn.BatchNorm2d(outch),
        torch.nn.ReLU()
    )
    return convlayer


def conv2d_bn_sigmoid(inch, outch, kernel_size, stride=1, padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.Conv2d(inch, outch, kernel_size=kernel_size, stride=stride, padding=padding),
        torch.nn.BatchNorm2d(outch),
        torch.nn.Sigmoid()
    )
    return convlayer


def deconv_sigmoid(inch, outch, kernel_size, stride=1, padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(inch, outch, kernel_size=kernel_size, stride=stride, padding=padding),
        torch.nn.Sigmoid()
    )
    return convlayer


def deconv_relu(inch, outch, kernel_size, stride=1, padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(inch, outch, kernel_size=kernel_size, stride=stride, padding=padding),
        torch.nn.BatchNorm2d(outch),
        torch.nn.ReLU()
    )
    return convlayer

class EncoderDecoder(torch.nn.Module):
    # def __init__(self, in_channels):
    #     super(EncoderDecoder, self).__init__()
    #
    #     self.conv_stack1 = torch.nn.Sequential(
    #         conv2d_bn_relu(in_channels, 16, 4, stride=2),
    #         conv2d_bn_relu(16, 16, 3)
    #     )
    #     self.conv_stack2 = torch.nn.Sequential(
    #         conv2d_bn_relu(16, 16, 4, stride=2),
    #         conv2d_bn_relu(16, 16, 3)
    #     )
    #     self.conv_stack3 = torch.nn.Sequential(
    #         conv2d_bn_relu(16, 32, 4, stride=2),
    #         conv2d_bn_relu(32, 32, 3)
    #     )
    #     self.conv_stack4 = torch.nn.Sequential(
    #         conv2d_bn_relu(32, 32, 4, stride=2),
    #         conv2d_bn_relu(32, 32, 3),
    #     )
    #     self.conv_stack5 = torch.nn.Sequential(
    #         conv2d_bn_relu(32, 64, 4, stride=2),
    #         conv2d_bn_relu(64, 64, 3),
    #     )
    #
    #     self.conv_stack6 = torch.nn.Sequential(
    #         conv2d_bn_relu(64, 64, (3, 4), stride=(1, 2)),
    #         conv2d_bn_relu(64, 64, 3),
    #     )
    #
    #     self.deconv_6 = deconv_relu(64, 64, (3, 4), stride=(1, 2))
    #     self.deconv_5 = deconv_relu(67, 64, 4, stride=2)
    #     self.deconv_4 = deconv_relu(67, 32, 4, stride=2)
    #     self.deconv_3 = deconv_relu(35, 32, 4, stride=2)
    #     self.deconv_2 = deconv_relu(35, 16, 4, stride=2)
    #     self.deconv_1 = deconv_sigmoid(19, 3, 4, stride=2)
    #
    #     self.predict_6 = torch.nn.Conv2d(64, 3, 3, stride=1, padding=1)
    #     self.predict_5 = torch.nn.Conv2d(67, 3, 3, stride=1, padding=1)
    #     self.predict_4 = torch.nn.Conv2d(67, 3, 3, stride=1, padding=1)
    #     self.predict_3 = torch.nn.Conv2d(35, 3, 3, stride=1, padding=1)
    #     self.predict_2 = torch.nn.Conv2d(35, 3, 3, stride=1, padding=1)
    #
    #     self.up_sample_6 = torch.nn.Sequential(
    #         torch.nn.ConvTranspose2d(3, 3, (3, 4), stride=(1, 2), padding=1, bias=False),
    #         torch.nn.Sigmoid()
    #     )
    #     self.up_sample_5 = torch.nn.Sequential(
    #         torch.nn.ConvTranspose2d(3, 3, 4, stride=2, padding=1, bias=False),
    #         torch.nn.Sigmoid()
    #     )
    #     self.up_sample_4 = torch.nn.Sequential(
    #         torch.nn.ConvTranspose2d(3, 3, 4, stride=2, padding=1, bias=False),
    #         torch.nn.Sigmoid()
    #     )
    #     self.up_sample_3 = torch.nn.Sequential(
    #         torch.nn.ConvTranspose2d(3, 3, 4, stride=2, padding=1, bias=False),
    #         torch.nn.Sigmoid()
    #     )
    #     self.up_sample_2 = torch.nn.Sequential(
    #         torch.nn.ConvTranspose2d(3, 3, 4, stride=2, padding=1, bias=False),
    #         torch.nn.Sigmoid()
    #     )
    #
    # def encoder(self, x):
    #     conv1_out = self.conv_stack1(x)
    #     conv2_out = self.conv_stack2(conv1_out)
    #     conv3_out = self.conv_stack3(conv2_out)
    #     conv4_out = self.conv_stack4(conv3_out)
    #     conv5_out = self.conv_stack5(conv4_out)
    #     conv6_out = self.conv_stack6(conv5_out)
    #     return conv6_out
    #
    # def decoder(self, x):
    #
    #     deconv6_out = self.deconv_6(x)
    #     predict_6_out = self.up_sample_6(self.predict_6(x))
    #
    #     concat_6 = torch.cat([deconv6_out, predict_6_out], dim=1)
    #     deconv5_out = self.deconv_5(concat_6)
    #     predict_5_out = self.up_sample_5(self.predict_5(concat_6))
    #
    #     concat_5 = torch.cat([deconv5_out, predict_5_out], dim=1)
    #     deconv4_out = self.deconv_4(concat_5)
    #     predict_4_out = self.up_sample_4(self.predict_4(concat_5))
    #
    #     concat_4 = torch.cat([deconv4_out, predict_4_out], dim=1)
    #     deconv3_out = self.deconv_3(concat_4)
    #     predict_3_out = self.up_sample_3(self.predict_3(concat_4))
    #
    #     concat2 = torch.cat([deconv3_out, predict_3_out], dim=1)
    #     deconv2_out = self.deconv_2(concat2)
    #     predict_2_out = self.up_sample_2(self.predict_2(concat2))
    #
    #     concat1 = torch.cat([deconv2_out, predict_2_out], dim=1)
    #     predict_out = self.deconv_1(concat1)
    #
    #     return predict_out

    def __init__(self, in_channels, latent_dim=4096, kl_weight=0.0001):
        super(EncoderDecoder, self).__init__()

        self.kl_weight = kl_weight

        self.conv_stack1 = torch.nn.Sequential(
            conv2d_bn_relu(in_channels, 32, 4, stride=2),
            conv2d_bn_relu(32, 32, 3)
        )
        self.conv_stack2 = torch.nn.Sequential(
            conv2d_bn_relu(32, 32, 4, stride=2),
            conv2d_bn_relu(32, 32, 3)
        )
        self.conv_stack3 = torch.nn.Sequential(
            conv2d_bn_relu(32, 64, 4, stride=2),
            conv2d_bn_relu(64, 64, 3)
        )
        self.conv_stack4 = torch.nn.Sequential(
            conv2d_bn_relu(64, 128, 4, stride=2),
            conv2d_bn_relu(128, 128, 3),
        )
        self.conv_stack5 = torch.nn.Sequential(
            conv2d_bn_relu(128, 128, (3, 4), stride=(1, 2)),
            conv2d_bn_relu(128, 128, 3),
        )

        self.deconv_5 = deconv_relu(128, 64, (3, 4), stride=(1, 2))
        self.deconv_4 = deconv_relu(67, 64, 4, stride=2)
        self.deconv_3 = deconv_relu(67, 32, 4, stride=2)
        self.deconv_2 = deconv_relu(35, 16, 4, stride=2)
        self.deconv_1 = deconv_sigmoid(19, 3, 4, stride=2)

        self.predict_5 = torch.nn.Conv2d(128, 3, 3, stride=1, padding=1)
        self.predict_4 = torch.nn.Conv2d(67, 3, 3, stride=1, padding=1)
        self.predict_3 = torch.nn.Conv2d(67, 3, 3, stride=1, padding=1)
        self.predict_2 = torch.nn.Conv2d(35, 3, 3, stride=1, padding=1)

        self.up_sample_5 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3, 3, (3, 4), stride=(1, 2), padding=1, bias=False),
            torch.nn.Sigmoid()
        )
        self.up_sample_4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3, 3, 4, stride=2, padding=1, bias=False),
            torch.nn.Sigmoid()
        )
        self.up_sample_3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3, 3, 4, stride=2, padding=1, bias=False),
            torch.nn.Sigmoid()
        )
        self.up_sample_2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3, 3, 4, stride=2, padding=1, bias=False),
            torch.nn.Sigmoid()
        )

        self.mean_linear = nn.Linear(4096, latent_dim)
        self.var_linear = nn.Linear(4096, latent_dim)
        self.decoder_projection = nn.Linear(latent_dim, 4096)

        self.encoded_shape = [64, 128, 8, 4]

    def encoder(self, x):
        conv1_out = self.conv_stack1(x)
        conv2_out = self.conv_stack2(conv1_out)
        conv3_out = self.conv_stack3(conv2_out)
        conv4_out = self.conv_stack4(conv3_out)
        conv5_out = self.conv_stack5(conv4_out)
        return conv5_out

    def decoder(self, x):
        deconv5_out = self.deconv_5(x)
        predict_5_out = self.up_sample_5(self.predict_5(x))

        concat_5 = torch.cat([deconv5_out, predict_5_out], dim=1)
        deconv4_out = self.deconv_4(concat_5)
        predict_4_out = self.up_sample_4(self.predict_4(concat_5))

        concat_4 = torch.cat([deconv4_out, predict_4_out], dim=1)
        deconv3_out = self.deconv_3(concat_4)
        predict_3_out = self.up_sample_3(self.predict_3(concat_4))

        concat2 = torch.cat([deconv3_out, predict_3_out], dim=1)
        deconv2_out = self.deconv_2(concat2)
        predict_2_out = self.up_sample_2(self.predict_2(concat2))

        concat1 = torch.cat([deconv2_out, predict_2_out], dim=1)
        predict_out = self.deconv_1(concat1)
        return predict_out

    def forward(self, x):

        # latent = self.encoder(x)
        # out = self.decoder(latent)

        encoded = self.encoder(x)
        encoded_shape = encoded.size()
        encoded_flatten = torch.flatten(encoded, 1)
        x = torch.reshape(encoded_flatten, (-1, encoded_shape[1], encoded_shape[2], encoded_shape[3]))
        # x == encoded all!!!!!!

        mean = self.mean_linear(encoded_flatten)
        logvar = self.var_linear(encoded_flatten)
        eps = torch.randn_like(logvar)
        std = torch.exp(logvar / 2)
        z = eps * std + mean
        sampled_x = self.decoder_projection(z)
        sampled_x = torch.reshape(sampled_x, (-1, encoded_shape[1], encoded_shape[2], encoded_shape[3]))

        out = self.decoder(sampled_x)
        # out = self.decoder(x)
        return out, mean, logvar

    def get_latent_space(self, x):
        encoded = self.encoder(x)
        encoded_shape = encoded.size()
        encoded_flatten = torch.flatten(encoded, 1)
        x = torch.reshape(encoded_flatten, (-1, encoded_shape[1], encoded_shape[2], encoded_shape[3]))
        # x == encoded all!!!!!!

        mean = self.mean_linear(encoded_flatten)
        logvar = self.var_linear(encoded_flatten)
        eps = torch.randn_like(logvar)
        std = torch.exp(logvar / 2)
        z = eps * std + mean

        return z

    def loss_function(self, y, y_hat, mean, logvar):

        recons_loss = F.mse_loss(y_hat, y)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean ** 2 - torch.exp(logvar), 1), 0)
        loss = recons_loss + kl_loss * self.kl_weight
        # loss = recons_loss
        return loss, recons_loss, kl_loss * self.kl_weight

    def sample(self, input_sample):
        sampled_x = self.decoder_projection(input_sample)
        sampled_x = torch.reshape(sampled_x, (-1, self.encoded_shape[1], self.encoded_shape[2], self.encoded_shape[3]))

        out = self.decoder(sampled_x)

        return out