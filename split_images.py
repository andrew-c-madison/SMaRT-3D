# Import modules
import os
from image_slicer import slice, save_tiles, get_basename

# Methods
def get_file_names(dir):
    f = [os.path.join(dir, f) for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    return f

def get_subdirs(dir):
    sd = []
    for (dirpath, dirnames, filenames) in os.walk(dir):
        sd.extend(dirnames)
        break
    return sd

def slice_and_save(s_dir, t_dir, v_dir, n, m):
    # t_dir holds unsliced images, s_dir will contain sliced images, n is number of tiles
    for sd in get_subdirs(s_dir):
        print "SUB DIR: {}".format(sd)

        # Create sub-directories if they do not already exist
        if not os.path.exists(os.path.join(v_dir, sd)):
            os.makedirs(os.path.join(v_dir, sd))
        if not os.path.exists(os.path.join(t_dir, sd)):
            os.makedirs(os.path.join(t_dir, sd))

        # Iterate through all images and labels, slice, and save into t_dir, and v_dir
        for i,f in enumerate(get_file_names(os.path.join(s_dir, sd))):
            if i%m==0: # save to v_dir/sd/
                tiles = slice(f, n, save=False)
                save_tiles(tiles, prefix=get_basename(f), directory=os.path.join(v_dir, sd))
                print "{} sliced and saved to {}".format(f, os.path.join(v_dir, sd))
            else: # save to t_dir/sd/
                tiles = slice(f, n, save=False)
                save_tiles(tiles, prefix=get_basename(f), directory=os.path.join(t_dir, sd))
                print "{} sliced and saved to {}".format(f, os.path.join(t_dir, sd))
        print "============================"


# Define source directory
src_path = os.path.join("Full")

# Define target directories
trn_path = os.path.join("Train")
val_path = os.path.join("Validation")

# Define parameters
N = 16 # number of tiles per image
M = 3  # modulus divisor, store every mth image in validation path
slice_and_save(src_path, trn_path, val_path, N, M)
