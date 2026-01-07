import os

def write_subdirectories_to_file(directory, output_file):
    try:
        # Get all subdirectories in the specified directory
        subdirectories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        
        # Write subdirectory names to the output file
        with open(output_file, 'w') as file:
            for subdir in subdirectories:
                file.write(subdir + '\n')
        
        print(f"Subdirectories written to {output_file} successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Specify the directory and output file path
directory_path = "data/datasets/test_unseen/"  # Replace with the path to your directory
output_file_path = "lists/neurit/test_unseen.txt"   # Replace with your desired output file path

write_subdirectories_to_file(directory_path, output_file_path)
