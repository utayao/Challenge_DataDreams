import sys
sys.path.append('../')
from utils.data_generator import TrainDataGenerator



def main(argv=None):

    train_data = TrainDataGenerator(
        train_dir='../data', cancer_data_augmentation=None,
        non_cancer_data_augmentation=None,
        shuffle=True,
        cv=None,
        image_resize=(224, 224),
        subset=False,
        normalize=False
    )

    train_data.extract_patches(number_of_cancer_images=50000, number_of_non_cancer_images=50000,
                               save_path="../data/patches")


main()
