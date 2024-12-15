import os
import random
import shutil


source_folder = './PepperSeedlings/original'   # original dataset folder
train_folder = './PlantNet/PlantNet_pytorch/data/FPSprocessed/train'   # trainset_dir
test_folder = './PlantNet/PlantNet_pytorch/data/FPSprocessed/test'   # testset_dir


txt_files = [os.path.join(source_folder, f) for f in os.listdir(source_folder) if f.endswith('.txt')]

if not os.path.exists(train_folder):
    os.makedirs(train_folder)
if not os.path.exists(test_folder):
    os.makedirs(test_folder)

all_files_number = 10
selected_files_number = 2  # test files number  (8 if train)
random_numbers = random.sample(range(all_files_number), selected_files_number)

for number in range(len(txt_files)):
    file_to_copy = txt_files[number]
    file_name = os.path.basename(file_to_copy)
    if number in random_numbers:
        target_path = os.path.join(test_folder, file_name)
    else:
        target_path = os.path.join(train_folder, file_name)
        # move files
    shutil.copy(file_to_copy, target_path)
    print(f"copied {file_name} and pasted to {target_path}")

print("done")
