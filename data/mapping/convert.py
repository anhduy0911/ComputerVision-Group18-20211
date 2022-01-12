import os
import os.path as path
import shutil

def main():
    source_data_root_dir = "consumer"
    destination_data_root_dir = "converted"

    os.mkdir(destination_data_root_dir)

    mapping_file_path = "sample_images.txt"

    previous_pill_id = ""
    current_image_output_dir = ""
    current_image_index = 0

    with open(path.join(source_data_root_dir, mapping_file_path)) as mapping_file:
        for mapping_entry in mapping_file:
            entry_columns = mapping_entry.strip().split("|")
            current_pill_id = entry_columns[0]

            if current_pill_id != previous_pill_id:
                pill_name = entry_columns[4]
                current_image_output_dir = path.join(destination_data_root_dir, pill_name)
                os.mkdir(current_image_output_dir)

                previous_pill_id = current_pill_id
                current_image_index = 0
            
            shutil.copy2(path.join(source_data_root_dir, entry_columns[2]), current_image_output_dir)
            current_image_index += 1

if __name__ == "__main__":
    main()