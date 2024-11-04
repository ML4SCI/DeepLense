class DeepLensDataLoader:
    def __init__(self, data_path='Model_II', val_path = 'Model_II_test', test_size=0.1, random_state=42):
        self.classes = ['axion', 'cdm', 'no_sub']
        self.class_indices = {class_name: idx for idx, class_name in enumerate(self.classes)}

        file_paths = []
        for class_name in self.classes:
            class_dir = os.path.join(data_path, class_name)
            file_paths += [os.path.join(class_dir, file) for file in os.listdir(class_dir)]
            
        self.train_files, self.test_files = train_test_split(file_paths, test_size=test_size, random_state=random_state)
        self.steps_per_epoch = len(self.train_files) // batch_size
        self.test_steps = len(self.test_files) // batch_size
        
        val_paths = []
        for class_name_val in self.classes:
            class_dir_val = os.path.join(val_path, class_name_val)
            val_paths += [os.path.join(class_dir_val, file) for file in os.listdir(class_dir_val)]
        random.shuffle(val_paths) 
        
        self.val_files = val_paths
        self.val_steps = len(self.val_files) // batch_size
    
    def get_data(self):
        return self.steps_per_epoch, self.test_steps, self.val_steps

    def load_image(self, input_path):
        class_name = os.path.basename(os.path.dirname(input_path))
        if class_name == 'axion':
            input = np.load(input_path, allow_pickle=True)[0][..., np.newaxis]
        else:
            input = np.load(input_path, allow_pickle=False)[..., np.newaxis]
        
        img_3d = np.repeat(input, 3, axis=-1) # For RGB channel
        img_3d  = (img_3d - np.min(img_3d)) / (np.max(img_3d) - np.min(img_3d)) # For RGB channel
        return img_3d

    def pretrain_generator(self):
        while True:
            for batch_id in range(self.steps_per_epoch):
                yield self.get_batch_pretraining(batch_id)
                
    def get_batch_pretraining(self, batch_id):
        batch_paths = self.train_files[batch_id*batch_size:(batch_id+1)*batch_size]
        augmented_images_1, augmented_images_2 = [], []
        for input_path in batch_paths:
            img_3d = self.load_image(input_path)
            augmented_images_1.append(augment_image_pretraining(img_3d))
            augmented_images_2.append(augment_image_pretraining(img_3d))
        x_batch_1 = np.array(augmented_images_1)
        x_batch_2 = np.array(augmented_images_2)
        return x_batch_1, x_batch_2  # (bs, 64, 64, 3), (bs, 64, 64, 3)


    def get_batch_finetuning(self, files, batch_id):
        batch_paths = files[batch_id*batch_size:(batch_id+1)*batch_size]
        batch_images = []
        batch_output = []
        for input_path in batch_paths:
            class_name = os.path.basename(os.path.dirname(input_path))
            label = self.class_indices[class_name] 
            img_3d = self.load_image(input_path)
            batch_images.append(img_3d)
            batch_output.append(label)
        x_batch = np.array(batch_images)
        y_batch = to_categorical(batch_output, num_classes=3) # For one hot
#         y_batch = np.array(batch_output) # for label
        return x_batch, y_batch  # (bs, 64, 64, 3), (bs)
        
    def train_generator(self):
        while True:
            random.shuffle(self.train_files)
            for batch_id in range(self.steps_per_epoch):
                yield self.get_batch_finetuning(self.train_files, batch_id)

    def test_generator(self):
        while True:
            for batch_id in range(self.test_steps):
                yield self.get_batch_finetuning(self.test_files, batch_id)
                
    def validation_generator(self):
        while True:
            for batch_id in range(self.val_steps):
                yield self.get_batch_finetuning(self.val_files, batch_id)

    def plot_sample_image(self, image_path):
        img_3d = self.load_image(image_path)
        class_name = os.path.basename(os.path.dirname(image_path))
        label = self.class_indices[class_name]
        # Augment the image for pretraining and finetuning
        pretraining_augmented_img = augment_image_pretraining(img_3d)
        # Plot the original, pretraining augmented, and finetuning augmented images
        fig, axs = plt.subplots(1, 2, figsize=(6, 6))
        
        axs[0].imshow(np.mean(img_3d, axis=2), cmap = 'inferno')
        axs[0].set_title('Original Image - Label: ' + class_name, fontsize = 10)
        
        axs[1].imshow(np.mean(pretraining_augmented_img, axis=2), cmap = 'inferno')
        axs[1].set_title('Pretraining Augmented Image', fontsize = 10)
        
        # Remove the x and y ticks
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.show()
        
data = DeepLensDataLoader()
steps_per_epoch, test_steps, val_steps = data.get_data()

data_loader = DeepLensDataLoader(data_path='Model_II')

# Get the first 5 image paths
first_five_images = data_loader.train_files[:2]

# Plot the images
for image_path in first_five_images:
    data_loader.plot_sample_image(image_path)