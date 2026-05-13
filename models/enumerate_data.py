import os

def rename_images_safe(folder_path):
    files = os.listdir(folder_path)

    # Step 1: Rename to temp names
    for i, file in enumerate(files):
        old_path = os.path.join(folder_path, file)
        temp_path = os.path.join(folder_path, f"temp_{i}.jpg")
        os.rename(old_path, temp_path)

    # Step 2: Rename to final names
    temp_files = os.listdir(folder_path)

    for i, file in enumerate(temp_files):
        old_path = os.path.join(folder_path, file)
        new_path = os.path.join(folder_path, f"{i+1}.jpg")
        os.rename(old_path, new_path)

    print(f"Safely renamed images in {folder_path}")


rename_images_safe("dataset/drowsy")
rename_images_safe("dataset/not_drowsy")