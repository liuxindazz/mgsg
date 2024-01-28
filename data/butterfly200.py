import torch.utils.data as data
from os.path import join
from PIL import Image
from torchvision import transforms

class Butterfly200(data.Dataset):
    def __init__(self, image_dir, list_path, input_transform = None, level='species'):
        super(Butterfly200, self).__init__()
        print(list_path)

        self.level = level

        name_list = []
        family_label_list = []
        subfamily_label_list = []
        genus_label_list = []
        species_label_list = []
        with open(list_path, 'r') as f:
            for l in f.readlines():
                imagename, species_label, genus_label, subfamily_label, family_label  = l.strip().split(' ')
                name_list.append(imagename)
                family_label_list.append(int(family_label))
                subfamily_label_list.append(int(subfamily_label))
                genus_label_list.append(int(genus_label))
                species_label_list.append(int(species_label))

        self.image_filenames = [join(image_dir, x) for x in name_list]
        self.input_transform = input_transform

        self.family_label_list = family_label_list
        self.subfamily_label_list = subfamily_label_list
        self.genus_label_list = genus_label_list
        self.species_label_list = species_label_list

    def __getitem__(self, index):
        imagename = self.image_filenames[index]
        input = Image.open(self.image_filenames[index]).convert('RGB')
        if self.input_transform:
            input = self.input_transform(input)

        family_label = self.family_label_list[index] - 1
        subfamily_label = self.subfamily_label_list[index] - 1
        genus_label = self.genus_label_list[index] - 1
        species_label = self.species_label_list[index] - 1
        
        if self.level == 'family':
            target = family_label
        if self.level == 'subfamily':
            target = subfamily_label
        if self.level == 'genus':
            target = genus_label
        if self.level == 'species':
            target = species_label    

        return input, target

    def __len__(self):
        return len(self.image_filenames)



# if __name__ == '__main__':
#     test_transforms_list = [
#             transforms.Resize(int(448/0.875)),
#             transforms.CenterCrop(448),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=(0.485, 0.456, 0.406),
#                                  std=(0.229, 0.224, 0.225))
#         ]
#     test_set = Butterfly200('./images_small/', 'Butterfly200_test_release.txt', transforms.Compose(test_transforms_list), 'species')
#     print(test_set.__len__())