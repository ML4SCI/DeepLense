import os
import gdown
import splitfolders
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
from config.data_config import DATASET
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import ToPILImage
import torchvision.transforms as T

import os

import gdown
import matplotlib.pyplot as plt
import numpy as np
import splitfolders
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToPILImage

from utils.augmentation import DefaultTransformations
from utils.util import make_directories


def download_dataset(
    filename: str,
    url: str = "https://drive.google.com/uc?id=1m7QzSzXyE8u_QoYplN9dIe-X2pf1KXxt",
) -> None:
    """Downloads dataset from Google drive

    Args:
        filename (str): output directory name
        url (str, optional): URL to dataset. Defaults to "https://drive.google.com/uc?id=1m7QzSzXyE8u_QoYplN9dIe-X2pf1KXxt".
    """
    if not os.path.isfile(filename):
        try:
            gdown.download(url, filename, quiet=False)
        except Exception as e:
            print(e)
    else:
        print("File exists")


def extract_split_dataset(
    filename: str,
    destination_dir: str = "data",
    dataset_name: str = "Model_I",
    split: bool = False,
) -> None:
    """Extract from .tar file and splits dataset (90:10) into train and validation set

    Args:
        filename (str): tar filename
        destination_dir (str, optional): output directory name. Defaults to "data".
        dataset_name (str, optional): dataset name: Model_I, Model_II, Model_III. Defaults to "Model_I".
        split (bool, optional): whether to split or not. Defaults to False.
    """
    # only extracting folder
    if not split:
        print("Extracting folder ...")
        os.system(f"tar xf {filename} --directory {destination_dir}")
        print("Extraction complete")
        # os.system(f"rm -r {filename}")

    # splitting folder
    else:
        os.system(
            f"tar xf {filename} --directory {destination_dir} ; mv {destination_dir}/{dataset_name} {destination_dir}/{dataset_name}_raw"
        )
        splitfolders.ratio(
            f"{destination_dir}/{dataset_name}_raw",
            output=f"{destination_dir}/{dataset_name}",
            seed=1337,
            ratio=(0.9, 0.1),
        )
        os.system(f"rm -r {destination_dir}/{dataset_name}_raw")


class DeepLenseDataset(Dataset):
    def __init__(
        self,
        destination_dir: str,
        mode: str,
        dataset_name: str,
        transform=None,
        download="False",
        channels=1,
    ):
        """Class for DeepLense dataset

        Args:
            destination_dir (str): directory where dataset is stored \n
            mode (str): type of dataset:  `train` or `val` or `test`
            dataset_name (str): name of dataset e.g. Model_I
            transform (_type_, optional): transformation of images. Defaults to None.
            download (str, optional): whether to download the dataset. Defaults to "False".
            channels (int, optional): # of channels. Defaults to 1.

        Example:
            >>>     trainset = DeepLenseDataset(
            >>>     dataset_dir,
            >>>     "train",
            >>>     dataset_name,
            >>>     transform=get_transform_train(
            >>>     upsample_size=387,
            >>>     final_size=train_config["image_size"],
            >>>     channels=train_config["channels"]),
            >>>     download=True,
            >>>     channels=train_config["channels"])

        """
        assert mode in ["train", "test", "val"]

        if mode == "train":
            filename = f"{destination_dir}/{dataset_name}.tgz"
            foldername = f"{destination_dir}/{dataset_name}"
            # self.root_dir = foldername + "/train"

        elif mode == "val":
            filename = f"{destination_dir}/{dataset_name}.tgz"
            foldername = f"{destination_dir}/{dataset_name}"
            # self.root_dir = foldername + "/val"

        else:
            filename = f"{destination_dir}/{dataset_name}_test.tgz"
            foldername = f"{destination_dir}/{dataset_name}_test"
            # self.root_dir = foldername

        url = DATASET[f"{dataset_name}"][f"{mode}_url"]

        if download and not os.path.isdir(foldername) is True:
            if not os.path.isfile(filename):
                download_dataset(
                    filename,
                    url=url,
                )
            extract_split_dataset(filename, destination_dir)
        else:
            assert (
                os.path.isdir(foldername) is True
            ), "Dataset doesn't exists, set arg download to True!"

            print(f"{dataset_name} dataset already exists")

        self.root_dir = foldername

        self.transform = transform
        classes = os.listdir(
            self.root_dir
        )  # [join(self.root_dir, x).split('/')[3] for x in listdir(self.root_dir)]
        classes.sort()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        self.imagefilename = []
        self.labels = []
        self.channels = channels

        for i in classes:
            for x in os.listdir(os.path.join(self.root_dir, i)):
                self.imagefilename.append(os.path.join(self.root_dir, i, x))
                self.labels.append(self.class_to_idx[i])

    def __getitem__(self, index):
        image, label = self.imagefilename[index], self.labels[index]

        image = np.load(image, allow_pickle=True)
        if label == 0:
            image = image[0]
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = np.expand_dims(image, axis=2)

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
            image = (
                image.float().clone().detach()
            )  # .requires_grad_(True)  # torch.tensor(image, dtype=torch.float32)
        return image, label

    def __len__(self):
        return len(self.labels)


def visualize_samples(
    dataset, labels_map, fig_height=15, fig_width=15, num_cols=5, cols_rows=5
) -> None:
    """Visualize samples from dataset

    Args:
        dataset (torch.utils.data.Dataset): dataset to visualize
        labels_map (dict): dict for mapping labels to number e.g `{0: "axion"}`
        fig_height (int, optional): height of visualized sample. Defaults to 15.
        fig_width (int, optional): width of visualized sample. Defaults to 15.
        num_cols (int, optional): # of columns of images in a window. Defaults to 5.
        cols_rows (int, optional): # of rows of images in a window. Defaults to 5.
    """
    # labels_map = {0: "axion", 1: "cdm", 2: "no_sub"}
    figure = plt.figure(figsize=(fig_height, fig_width))
    cols, rows = num_cols, cols_rows
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(f"{labels_map[label]}")
        plt.axis("off")
        # im = ToPILImage()(img)
        img = img.squeeze()
        plt.imshow(img, cmap="gray")
        # plt.imshow(img)
    plt.show()


def visualize_samples_ssl(
    dataset, labels_map, fig_height=15, fig_width=15, num_cols=5, cols_rows=5
) -> None:
    """Visualize samples from dataset

    Args:
        dataset (torch.utils.data.Dataset): dataset to visualize
        labels_map (dict): dict for mapping labels to number e.g `{0: "axion"}`
        fig_height (int, optional): height of visualized sample. Defaults to 15.
        fig_width (int, optional): width of visualized sample. Defaults to 15.
        num_cols (int, optional): # of columns of images in a window. Defaults to 5.
        cols_rows (int, optional): # of rows of images in a window. Defaults to 5.
    """
    # labels_map = {0: "axion", 1: "cdm", 2: "no_sub"}
    fig = plt.figure(figsize=(fig_height, fig_width))

    # Number of rows and columns for the outer subplots
    num_rows_outer = 2
    num_cols_outer = 2

    # Number of rows and columns for the inner subplots
    num_rows_inner = 1
    num_cols_inner = 2

    # Generate the outer subplots
    outer_subplot_index = 1
    for row in range(num_rows_outer):
        for col in range(num_cols_outer):
            outer_subplot = fig.add_subplot(
                num_rows_outer, num_cols_outer, outer_subplot_index
            )

            outer_subplot_index += 1
            sample_idx = torch.randint(len(dataset), size=(1,)).item()
            img1, img2, label = dataset[sample_idx]
            outer_subplot.set_title(f"{labels_map[label]}")
            img = [img1, img2]

            # Generate the inner subplots within the current outer subplot
            inner_subplot_index = 1
            for inner_row in range(num_rows_inner):
                for inner_col in range(num_cols_inner):
                    inner_subplot = outer_subplot.inset_axes(
                        [
                            inner_col / num_cols_inner,
                            inner_row / num_rows_inner,
                            1 / num_cols_inner,
                            1 / num_rows_inner,
                        ]
                    )

                    plot_img = img[inner_col].squeeze()
                    inner_subplot.imshow(plot_img)  # cmap="gray"
                    inner_subplot_index += 1

    # Add a title to the main figure
    fig.suptitle("Main Figure")

    # Display the figure
    plt.show()


class DeepLenseDataset_dataeff(Dataset):
    # TODO: add val-loader + splitting
    def __init__(
        self,
        destination_dir,
        mode,
        dataset_name,
        transform=None,
        download="False",
        channels=1,
    ):
        assert mode in ["train", "test", "val"]

        if mode == "train":
            filename = f"{destination_dir}/{dataset_name}.tgz"
            foldername = f"{destination_dir}/{dataset_name}"
            # self.root_dir = foldername + "/train"

        elif mode == "val":
            filename = f"{destination_dir}/{dataset_name}.tgz"
            foldername = f"{destination_dir}/{dataset_name}"
            # self.root_dir = foldername + "/val"

        else:
            filename = f"{destination_dir}/{dataset_name}_test.tgz"
            foldername = f"{destination_dir}/{dataset_name}_test"
            # self.root_dir = foldername

        url = DATASET[f"{dataset_name}"][f"{mode}_url"]

        if download and not os.path.isdir(foldername) is True:
            if not os.path.isfile(filename):
                download_dataset(
                    filename,
                    url=url,
                )
            extract_split_dataset(filename, destination_dir)
        else:
            assert (
                os.path.isdir(foldername) is True
            ), "Dataset doesn't exists, set arg download to True!"

            print("Dataset already exists")

        self.root_dir = foldername

        self.transform = transform
        classes = os.listdir(
            self.root_dir
        )  # [join(self.root_dir, x).split('/')[3] for x in listdir(self.root_dir)]
        classes.sort()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        # self.imagefilename = []
        self.labels = []
        self.channels = channels

        # TODO: make dynamic
        if mode == "train":
            num = 87525
        elif mode == "test":
            num = 15000

        # if not os.path.exists(f"images_mmep_{mode}.npy"):

        self.images_mmep = np.memmap(
            f"images_mmep_{mode}.npy",
            dtype="int16",
            mode="w+",
            shape=(num, 150, 150),
        )

        self.labels_mmep = np.memmap(
            f"labels_mmep_{mode}.npy", dtype="float64", mode="w+", shape=(num, 1)
        )

        w_index = 0
        for i in classes:
            for x in os.listdir(os.path.join(self.root_dir, i)):
                self.imagefilename = os.path.join(self.root_dir, i, x)
                image = np.load(self.imagefilename, allow_pickle=True)
                label = self.class_to_idx[i]
                if label == 0:
                    image = image[0]
                self.images_mmep[w_index, :] = image
                self.labels_mmep[w_index] = label
                self.labels.append(self.class_to_idx[i])
                w_index += 1

    def __getitem__(self, index):
        image = np.asarray(self.images_mmep[index])
        label = np.asarray(self.labels_mmep[index], dtype="int64")[0]

        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        if self.channels == 3:
            image = Image.fromarray(image.astype("uint8")).convert("RGB")
        else:
            image = Image.fromarray(image.astype("uint8"))  # .convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.labels)


class CustomDataset(Dataset):
    """Create custom dataset for the given data"""

    def __init__(self, root_dir, mode, transform=None):
        assert mode in ["train", "test", "val"]

        self.root_dir = root_dir

        if mode == "train":
            self.root_dir = self.root_dir + "/train"
        elif mode == "test":
            self.root_dir = self.root_dir + "/test"
        else:
            self.root_dir = self.root_dir + "/val"

        self.transform = transform
        classes = os.listdir(self.root_dir)
        classes.sort()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        self.imagefilename = []
        self.labels = []

        for i in classes:
            for x in os.listdir(os.path.join(self.root_dir, i)):
                self.imagefilename.append(os.path.join(self.root_dir, i, x))
                self.labels.append(self.class_to_idx[i])

    def __getitem__(self, index):
        image, label = self.imagefilename[index], self.labels[index]

        image = Image.open(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.labels)


class CustomDatasetSSL(Dataset):
    """Create custom dataset for the given data"""

    def __init__(self, root_dir, mode, transforms=None):
        assert mode in ["train", "test", "val"]

        self.root_dir = root_dir

        if mode == "train":
            self.root_dir = self.root_dir + "/train"
        elif mode == "test":
            self.root_dir = self.root_dir + "/test"
        else:
            self.root_dir = self.root_dir + "/val"

        self.transforms = transforms
        classes = os.listdir(self.root_dir)
        classes.sort()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        self.imagefilename = []
        self.labels = []

        for i in classes:
            for x in os.listdir(os.path.join(self.root_dir, i)):
                self.imagefilename.append(os.path.join(self.root_dir, i, x))
                self.labels.append(self.class_to_idx[i])

    def __getitem__(self, index):
        image, label = self.imagefilename[index], self.labels[index]

        image = Image.open(image)
        ret = []
        if self.transforms is not None:
            for t in self.transforms:
                ret.append(t(image))
        ret.append(label)
        return ret

    def __len__(self):
        return len(self.labels)


class DeepLenseDatasetSSL(Dataset):
    """Create custom dataset for the given data"""

    def __init__(
        self,
        destination_dir: str,
        mode: str,
        dataset_name: str,
        transforms=None,
        download="False",
        channels=1,
    ):
        assert mode in ["train", "test", "val"]

        if mode == "train":
            filename = f"{destination_dir}/{dataset_name}.tgz"
            foldername = f"{destination_dir}/{dataset_name}"
            # self.root_dir = foldername + "/train"

        elif mode == "val":
            filename = f"{destination_dir}/{dataset_name}.tgz"
            foldername = f"{destination_dir}/{dataset_name}"
            # self.root_dir = foldername + "/val"

        else:
            filename = f"{destination_dir}/{dataset_name}_test.tgz"
            foldername = f"{destination_dir}/{dataset_name}_test"
            # self.root_dir = foldername

        url = DATASET[f"{dataset_name}"][f"{mode}_url"]

        if download and not os.path.isdir(foldername) is True:
            if not os.path.isfile(filename):
                download_dataset(
                    filename,
                    url=url,
                )
            extract_split_dataset(filename, destination_dir)
        else:
            assert (
                os.path.isdir(foldername) is True
            ), "Dataset doesn't exists, set arg download to True!"

            print(f"{dataset_name} dataset already exists")

        self.root_dir = foldername

        self.transforms = transforms
        classes = os.listdir(self.root_dir)
        classes.sort()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        self.imagefilename = []
        self.labels = []
        self.channels = channels

        for i in classes:
            for x in os.listdir(os.path.join(self.root_dir, i)):
                self.imagefilename.append(os.path.join(self.root_dir, i, x))
                self.labels.append(self.class_to_idx[i])

    def __getitem__(self, index):
        image, label = self.imagefilename[index], self.labels[index]

        image = np.load(image, allow_pickle=True)
        if label == 0:
            image = image[0]
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = np.expand_dims(image, axis=2)

        ret = []
        if self.transforms is not None:
            for t in self.transforms:
                image_torch = torch.from_numpy(image)
                image_torch.to(dtype=torch.int32)
                image_torch = image_torch.permute(2, 0, 1)

                transform_to_pil = T.ToPILImage()
                image_torch_int = image_torch.to(dtype=torch.int32)
                image_pil = transform_to_pil(image_torch_int)
                image_t = t(image_pil)
                image_t = image_t.float().clone().detach()
                ret.append(image_t)
        ret.append(label)
        return ret

    def __len__(self):
        return len(self.labels)


class DefaultDatasetSetupSSL:
    def __init__(self) -> None:
        # parent directory
        current_file = os.path.abspath(__file__)
        parent_directory = os.path.dirname(current_file)

        self.data_dir = os.path.join(parent_directory, "../data")
        self.default_transform = DefaultTransformations()
        self.train_transforms = self.default_transform.get_train_transforms_ssl()

        # make data directory if doesnt exists
        make_directories([self.data_dir])

    def setup(self, dataset_name="Model_III"):
        self.default_dataset_cfg = {}
        self.default_dataset_cfg["dataset_name"] = dataset_name
        self.default_dataset_cfg["dataset"] = DATASET[
            self.default_dataset_cfg["dataset_name"]
        ]
        self.default_dataset_cfg["classes"] = self.default_dataset_cfg["dataset"][
            "classes"
        ]  # {0: "axion", 1: "no_sub"}  #
        self.default_dataset_cfg["train_url"] = self.default_dataset_cfg["dataset"][
            "train_url"
        ]

    def get_dataset(self, mode="train"):
        assert mode in ["train", "test", "val"]

        dataset = DeepLenseDatasetSSL(
            destination_dir=self.data_dir,
            dataset_name=self.default_dataset_cfg["dataset_name"],
            mode=mode,
            transforms=self.train_transforms,
            download=True,
            channels=1,
        )

        # get the number of samples in train and test set
        print(f"{mode} data: {len(dataset)}")

        return dataset

    def visualize_dataset(self, dataset):
        visualize_samples_ssl(dataset, labels_map=self.default_dataset_cfg["classes"])


def get_samplers(config, trainset, testset):
    config.defrost()
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == "part":
        indices = np.arange(dist.get_rank(), len(trainset), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            trainset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    indices = np.arange(dist.get_rank(), len(testset), dist.get_world_size())
    sampler_val = SubsetRandomSampler(indices)

    return sampler_train, sampler_val


class SubsetRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.epoch = 0
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = epoch
