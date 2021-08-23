from torchvision import transforms

def supervised_augmentations():
    """
    Return best augmentations for the Supervised algorithm.

    Returns:
    --------
    train_transform: Torchvision transforms
        Augmentations for the training set.

    test_transform: Torchvision transforms
        Augmentations for the test set.
    """

    train_transform = transforms.Compose([transforms.Resize(150),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          ])

    test_transform = transforms.Compose([transforms.Resize(150),
                                        ])

    return train_transform, test_transform

def adda_augmentations():
    """
    Return best augmentations for the ADDA algorithm.

    Returns:
    --------
    train_transform_source: Torchvision transforms
        Augmentations for the source training set.

    train_transform_target: Torchvision transforms
        Augmentations for the target training set.

    test_transform: Torchvision transforms
        Augmentations for the test set.
    """
    
    train_transform_source = transforms.Compose([transforms.Resize(150),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.RandomVerticalFlip(),
                                                 ])

    train_transform_target = transforms.Compose([transforms.Resize(150),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.RandomVerticalFlip(),
                                                 ])

    test_transform = transforms.Compose([transforms.Resize(150),
                                         ])

    return train_transform_source, train_transform_target, test_transform

def self_ensemble_augmentations():
    """
    Return best augmentations for the Self-Ensemble algorithm.

    Returns:
    --------
    train_transform_source: Torchvision transforms
        Augmentations for the source training set.

    train_transform_target: Torchvision transforms
        Augmentations for the target training set.

    test_transform: Torchvision transforms
        Augmentations for the test set.
    """
    
    train_transform_source = transforms.Compose([transforms.Resize(150),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.RandomVerticalFlip(),
                                                 transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1.)),
                                                 ])

    train_transform_target = transforms.Compose([transforms.Resize(150),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.RandomVerticalFlip(),
                                                 ])

    test_transform = transforms.Compose([transforms.Resize(150),
                                         ])

    return train_transform_source, train_transform_target, test_transform

def cgdm_augmentations():
    """
    Return best augmentations for the CGDM algorithm.

    Returns:
    --------
    train_transform_source: Torchvision transforms
        Augmentations for the source training set.

    train_transform_target: Torchvision transforms
        Augmentations for the target training set.

    test_transform: Torchvision transforms
        Augmentations for the test set.
    """
    
    train_transform_source = transforms.Compose([transforms.Resize(150),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.RandomVerticalFlip(),
                                                 ])

    train_transform_target = transforms.Compose([transforms.Resize(150),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.RandomVerticalFlip(),
                                                 ])

    test_transform = transforms.Compose([transforms.Resize(150),
                                         ])

    return train_transform_source, train_transform_target, test_transform

def adamatch_augmentations():
    """
    Return best augmentations for the AdaMatch algorithm.

    Returns:
    --------
    train_transform_source_weak: Torchvision transforms
        Weak augmentations for the source training set.

    train_transform_source_strong: Torchvision transforms
        Strong augmentations for the source training set.

    train_transform_target_weak: Torchvision transforms
        Weak augmentations for the target training set.

    train_transform_target_strong: Torchvision transforms
        Strong augmentations for the target training set.

    test_transform: Torchvision transforms
        Augmentations for the test set.
    """
    
    train_transform_source_weak = transforms.Compose([transforms.Resize(150),
                                                      transforms.RandomHorizontalFlip(),
                                                      transforms.RandomVerticalFlip(),
                                                      ])

    train_transform_target_weak = transforms.Compose([transforms.Resize(150),
                                                      transforms.RandomHorizontalFlip(),
                                                      transforms.RandomVerticalFlip(),
                                                      ])

    train_transform_source_strong = transforms.Compose([transforms.Resize(150),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.RandomVerticalFlip(),
                                                        transforms.RandomInvert(),
                                                        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1.)),
                                                        transforms.RandomErasing(),
                                                        ])

    train_transform_target_strong = transforms.Compose([transforms.Resize(150),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.RandomVerticalFlip(),
                                                        transforms.RandomInvert(),
                                                        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1.)),
                                                        transforms.RandomErasing(),
                                                        ])

    test_transform = transforms.Compose([transforms.Resize(150),
                                        ])

    return train_transform_source_weak, train_transform_target_weak, train_transform_source_strong, train_transform_target_strong, test_transform