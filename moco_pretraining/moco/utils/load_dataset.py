from torchvision import datasets, transforms
from .dataloader_med import ChestX_ray14
import aihc_utils.image_transform as image_transform

def load_dataset(split, args):

    if args.aug_setting == 'moco_v2':
        # print("data moco_v2")
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        
        train_augmentation = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        test_augmentation = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        data_list = getattr(args, f'{split}_list')

        if split == "train":
            return ChestX_ray14(args.data_path, data_list, augment=train_augmentation, num_class=args.num_classes)
        elif split == "test" or split == 'val':
            return ChestX_ray14(args.data_path, data_list, augment=test_augmentation, num_class=args.num_classes)
        
    else:
        # print("data ch")
        data_list = getattr(args, f'{split}_list')
        train_augmentation = transforms.Compose(image_transform.get_transform(args, training=True))
        test_augmentation = transforms.Compose(image_transform.get_transform(args, training=False))
        if split == "train":
            return ChestX_ray14(args.data_path, data_list, augment=train_augmentation, num_class=args.num_classes)
        elif split == "test" or split == 'val':
            return ChestX_ray14(args.data_path, data_list, augment=test_augmentation, num_class=args.num_classes)


