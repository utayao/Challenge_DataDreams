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
        subset=True,
        normalize=False
    )

    train_data.extract_patches(number_of_each_cancer_images=100, number_of_each_non_cancer_images=100,
                               save_path="../data/patches")


main()
