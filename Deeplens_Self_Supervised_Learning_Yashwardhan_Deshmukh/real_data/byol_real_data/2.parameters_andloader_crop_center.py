# Algorithm hyperparameters
input_shape = (128, 128, 3)
num_epochs = 20
batch_size = 256 
width = 64
lr = 1e-4

def crop_center(img, cropx, cropy):
    y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty+cropy, startx:startx+cropx]

fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))

channel_names = ['g-band', 'r-band', 'i-band', 'Combined RGB']

for i, (folder_path, title) in enumerate([
    ('/kaggle/input/real-dataset-cleaned/dataset/lenses', 'Lenses'),
    ('/kaggle/input/real-dataset-cleaned/dataset/nonlenses', 'Non Lenses')]):
    file_name = os.listdir(folder_path)[7]
    file_path = os.path.join(folder_path, file_name)
    loaded_file = np.load(file_path, allow_pickle=True)
    print(f'Shape of raw {title}: {loaded_file.shape}')

    cropped_images = []

    if loaded_file.ndim == 3:
        for j in range(3): 
            img_channel = loaded_file[j]  # Select channel
            crop_size_x, crop_size_y = 128, 128
            if img_channel.shape[0] < 128 or img_channel.shape[1] < 128:
                crop_size_x = min(img_channel.shape[0], 128)
                crop_size_y = min(img_channel.shape[1], 128)
            cropped_image = crop_center(img_channel, crop_size_x, crop_size_y)
            cropped_images.append(cropped_image)  # Store the cropped image
            print(f'Shape of {title} ({channel_names[j]}): {cropped_image.shape}')
            axs[i, j].imshow(cropped_image, cmap='inferno')
            axs[i, j].set_title(f"{title} ({channel_names[j]})")
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])

        rgb_image = np.dstack(cropped_images)  # Stack along the third dimension to form an RGB image
        print(f'New Shape of {title} (all bands): {rgb_image.shape}')
        axs[i, 3].imshow(rgb_image)
        axs[i, 3].set_title(f"{title} (Combined RGB)")
        axs[i, 3].set_xticks([])
        axs[i, 3].set_yticks([])

    else:
        raise ValueError("Unexpected image data format")

plt.tight_layout()
plt.show()