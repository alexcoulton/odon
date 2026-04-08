import argparse
import tifffile
import zarr
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image
from ome_zarr.scale import Scaler

def convert_tif_to_zarr(input_path, output_path):
    print(f"Reading: {input_path}")
    img = tifffile.imread(input_path)
    
    # Updated logic for multi-channel images
    if img.ndim == 2:
        axes = ["y", "x"]
    elif img.ndim == 3:
        # Assuming shape is (Channels, Y, X)
        axes = ["c", "y", "x"]  
    elif img.ndim == 4:
        # Assuming shape is (Channels, Z, Y, X)
        axes = ["c", "z", "y", "x"]
    elif img.ndim == 5:
        # Assuming shape is (Time, Channels, Z, Y, X)
        axes = ["t", "c", "z", "y", "x"]
    else:
        axes = None
        print(f"Warning: Unusual number of dimensions ({img.ndim}). Proceeding without explicit axes.")

    print(f"Image shape: {img.shape}. Assigned Axes: {axes}")
    
    # Initialize the Zarr store
    store = parse_url(output_path, mode="w").store
    root = zarr.group(store=store)
    
    print(f"Writing OME-ZARR to: {output_path}")
    
    # Write the image, automatically generating 4 pyramid levels.
    # OME-Zarr scaler automatically knows NOT to downsample 'c' or 't' axes.
    write_image(image=img, group=root, axes=axes, scaler=Scaler(max_layer=4))
    print("Conversion complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a multi-channel TIF to OME-ZARR.")
    parser.add_argument("input_tif", help="Path to the input .tif file")
    parser.add_argument("output_zarr", help="Path to the output .zarr directory")
    args = parser.parse_args()

    convert_tif_to_zarr(args.input_tif, args.output_zarr)
