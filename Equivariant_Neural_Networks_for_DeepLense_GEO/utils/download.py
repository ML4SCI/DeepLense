import gdown

MODELS = {
    "Model-1": {
        "train": "https://drive.google.com/file/d/1QMVLpqag6S9JWqzmGM_pK4C0F1eBVIfV/view?usp=sharing",
        "test": "https://drive.google.com/file/d/1rUAKLLS3p9jDaL9R9m84JVKvMcUuVsO1/view?usp=sharing",
    },
    "Model-2": {
        "train": "https://drive.google.com/file/d/1HYPkdtVUj9xsoGzFDxT4rhl37KmqDCg4/view?usp=sharing",
        "test": "https://drive.google.com/file/d/1PFdpqk7XOAKtg0Cnav4HTzyJiudx9dZv/view?usp=sharing",
    },
    "Model-3": {
        "train": "https://drive.google.com/file/d/1ynKMJoEeKKJqLfuKRR1Y7rQjeBMM0w94/view?usp=sharing",
        "test": "https://drive.google.com/file/d/18BuCv40t6qmiNnhjJF1y9rqSBhBOfDon/view?usp=sharing",
    },
    "Model-4": {
        "train": "https://drive.google.com/file/d/1vGkfEgEiapZoHUt1E6Tlhi-hKJ7dF2Ke/view?usp=sharing",
        "test": None,
    },
    "Model-5": {
        "train": "https://drive.google.com/file/d/1-eG5Y2ETDFPP_lwjkoxzRGZC7bm0rtCK/view?usp=sharing",
        "test": None
    },
}


def shareable_link_to_url(link):
    base = "https://drive.google.com/uc?id="
    id = link.split("/")[-2]
    return base + id


def download(args):
    model = args["model"]  # [TODO] to arg parser
    if model not in MODELS:
        print("Model not found")
        return

    model = MODELS[model]
    train_link = model["train"]
    train_url = shareable_link_to_url(train_link)
    gdown.download(train_url, output="data/", quiet=False)

    if model["test"] is not None:
        test_link = model["test"]
        test_url = shareable_link_to_url(test_link)
        gdown.download(test_url, output="data/", quiet=False)
