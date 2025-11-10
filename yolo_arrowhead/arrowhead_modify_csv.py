# import pandas as pd
# import ast

# def scale_landmarks(csv_path, output_path, orig_width=1136, orig_height=852, new_width=800, new_height=600):
#     # Scaling factors
#     scale_x = new_width / orig_width
#     scale_y = new_height / orig_height
    
#     df = pd.read_csv(csv_path)
    
#     def scale_coordinates(coord_str):
#         coords = ast.literal_eval(coord_str)  
#         scaled_coords = [(int(x * scale_x), int(y * scale_y)) for x, y in coords]
#         return str(scaled_coords)  
    
#     # Apply scaling
#     df["Coordinates"] = df["Coordinates"].apply(scale_coordinates)
    
#     # Save modified CSV
#     df.to_csv(output_path, index=False)
#     print(f"Modified CSV saved to: {output_path}")

# if __name__ == "__main__":
    
#     original_csv_path = "dataset/aft_yolo_detect/detections_AC.csv"
#     modified_csv_path = "dataset/to_be_used/detections_AC_modified.csv"

#     scale_landmarks(original_csv_path, modified_csv_path)


##########################################################################################


import os
import pandas as pd
import ast
from PIL import Image

def scale_landmarks(df, orig_width, orig_height, new_width=800, new_height=600):
    # Scaling factors
    scale_x = new_width / orig_width
    scale_y = new_height / orig_height
    
    def scale_coordinates(coord_str):
        coords = ast.literal_eval(coord_str)  
        scaled_coords = [(int(x * scale_x), int(y * scale_y)) for x, y in coords]
        return str(scaled_coords)
    
    # Apply scaling to all coordinates in the dataframe
    df["Coordinates"] = df["Coordinates"].apply(scale_coordinates)
    
    return df

def get_image_dimensions(image_path):
    """Get the dimensions of the image using PIL"""
    with Image.open(image_path) as img:
        return img.size  # returns (width, height)

def process_images_in_folder(csv_path, folder_path, output_path):
    # Read the original CSV file
    df = pd.read_csv(csv_path)
    
    # List to store processed DataFrames for each image
    processed_data = []

    # Iterate over all images in the folder
    for filename in df["Filename"]:
        image_path = os.path.join(folder_path, filename)
        
        # Check if the image file exists in the folder
        if os.path.exists(image_path):
            # Get the dimensions of the current image
            orig_width, orig_height = get_image_dimensions(image_path)
            
            # Filter CSV for the current image's filename
            image_df = df[df["Filename"] == filename]
            
            # Scale landmarks for the current image
            scaled_image_df = scale_landmarks(image_df, orig_width, orig_height)
            
            # Append the processed DataFrame to the list
            processed_data.append(scaled_image_df)
        else:
            print(f"Image {filename} not found in the folder: {image_path}")
    
    # Concatenate all processed DataFrames into one
    final_df = pd.concat(processed_data, ignore_index=True)
    
    # Save the final DataFrame to a single CSV file
    final_df.to_csv(output_path, index=False)
    print(f"Modified CSV saved to: {output_path}")

if __name__ == "__main__":

    original_csv_path = "dataset/aft_yolo_detect/detections_FL.csv"
    folder_path = "dataset/aft_yolo_detect/FL/good"  # folder containing images
    modified_csv_path = "dataset/to_be_used/detections_FL_modified.csv"  # path for the final merged CSV

    process_images_in_folder(original_csv_path, folder_path, modified_csv_path)

