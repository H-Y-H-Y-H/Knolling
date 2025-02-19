import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


# # 128x96
# img_size1 = 128
# img_size2 = 96
#
# messy_data_path = '../dataset/VAE_1020_obj4/origin_images_before/'
# tidy_data_path = '../dataset/VAE_1020_obj4/origin_images_after/'
#
# messy_image_files = [f'label_0_{i}.png' for i in range(12)]
# tidy_image_files = [f'label_0_{i}.png' for i in range(12)]
#
#
# fig, axes = plt.subplots(3, 4, figsize=(15, 10))
# fig.suptitle("Original and Resized Images")
#
# for idx, img_file in enumerate(messy_image_files):
#     # Load the image
#     img_path = os.path.join(messy_data_path, img_file)
#     img = plt.imread(img_path)
#
#     # Resize the image
#     resized_image = cv2.resize(img, (img_size1, img_size2), interpolation=cv2.INTER_LINEAR)
#
#     # Display the resized image
#     ax = axes[idx // 4, idx % 4]
#     ax.imshow(resized_image, cmap='gray')
#     ax.set_title(f"Image {img_file} ({img_size1}x{img_size2})")
#     ax.axis('off')
#
# # Adjust layout
# plt.tight_layout()
# plt.subplots_adjust(top=0.9)
# plt.show()
#
# for idx in range(1,1220):
#     input_embed_id_0 = idx // 144
#     input_embed_id_1 = (idx % 144) // 12
#
#     label_embed_id_0 = idx // 144
#     label_embed_id_1 = (idx % 144) % 12
#     print(f'label_{input_embed_id_0}_{input_embed_id_1}.png',f'label_{label_embed_id_0}_{label_embed_id_1}.png')
#     print()


###############################################################
############# Visualize the Gaussian Distribution ##############
###############################################################

from sklearn.decomposition import PCA
from scipy.stats import norm
import pandas as pd

obj_num = 2
messy_data_path = f'C:/Users/yuhan/PycharmProjects/Knolling_data/dataset_white/obj{obj_num}/'
tidy_data_path = f'C:/Users/yuhan/PycharmProjects/Knolling_data/dataset_white/obj{obj_num}/'

messy_latent = []
tidy_latent = []
for i in range(2000):
    latent_data = np.load(messy_data_path+f'latent_after/label_{i}_{0}.npy')
    messy_latent.append(latent_data)



flattened_latents = np.asarray(messy_latent).reshape(len(messy_latent),-1)

# PCA Visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(flattened_latents)

plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
plt.title("PCA of Latent Space")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

# Histogram for Gaussian Distribution
plt.figure(figsize=(8, 6))
for i in range(3):  # Plot first three dimensions
    plt.hist(flattened_latents[:, i], bins=50, alpha=0.5, label=f'Dimension {i+1}')
plt.title("Histogram of Latent Dimensions")
plt.legend()
plt.show()

# Pairplot for Pairwise Relationships
import seaborn as sns
sns.pairplot(pd.DataFrame(flattened_latents[:, :5], columns=[f'Dim {i+1}' for i in range(5)]))
plt.show()

# 2D t-SNE Visualization
from sklearn.manifold import TSNE
tsne_result = TSNE(n_components=2).fit_transform(flattened_latents)

plt.figure(figsize=(8, 6))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.5)
plt.title("t-SNE of Latent Space")
plt.show()
