import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import image
import os
from scipy import ndimage


def get_map(filename):
    im = image.imread(filename).T

    thresh = 0.99
    im[im < thresh] = 0.
    im[im >= thresh] = 1.

    im = 1-im.astype(int)

    
    # # # crop up to the edges of the buildings
    while np.all(im[0, :] == 0):
        im = im[1:, :]
    while np.all(im[-1, :] == 0):
        im = im[:-1, :]
    while np.all(im[:, 0] == 0):
        im = im[:, 1:]
    while np.all(im[:, -1] == 0):
        im = im[:, :-1]

    im = 1-im

    # blacken everything outside the actual map
    for idx in range(im.shape[0]):
        for jdx in range(im.shape[1]):
            if im[idx, jdx] != 0:
                im[idx, jdx] = 0
            else:
                break
        for jdx in range(im.shape[1]-1, 0, -1):
            if im[idx, jdx] != 0:
                im[idx, jdx] = 0
            else:
                break

        # plt.imshow(im, cmap='gray')
        # plt.show()
    return im

def _coord_transform(world_coords_orig, map_name):
    if map_name == "building1":  # or map_name == 'building1_big':
        world_coords_new = world_coords_orig.copy()
        world_coords_new[:, 0] = -world_coords_orig[:, 0]
        world_coords_new[:, 1] = world_coords_orig[:, 1]
    elif map_name == "building3":
        world_coords_new = world_coords_orig.copy()
        world_coords_new[:, 1] = world_coords_orig[:, 0]
        world_coords_new[:, 0] = world_coords_orig[:, 1]
    elif map_name == "building2_f1":
        world_coords_new = world_coords_orig.copy()
        world_coords_new[:, 0] = world_coords_orig[:, 1]
        world_coords_new[:, 1] = world_coords_orig[:, 0]
    elif map_name == "building2_f2":
        world_coords_new = world_coords_orig.copy()
        world_coords_new[:, 0] = world_coords_orig[:, 1]
        world_coords_new[:, 1] = world_coords_orig[:, 0]
    return world_coords_new

def get_map_by_name(map_file):
    return get_map(map_file)

def _world_to_image_coords(world_coords, map_size, world_size, origin):
    true_x_size = world_size[0]  # m
    true_y_size = world_size[1]  # m
    world_origin_offset_x = origin[0]  # m from top left corner of map
    world_origin_offset_y = origin[1]  # m from top left corner of map
    scale_factor_x = true_x_size/map_size[0]  # m/pixel
    scale_factor_y = true_y_size/map_size[1]  # m/pixel

    image_coord_x = np.round(
        (world_coords[:, 0] + world_origin_offset_x)/scale_factor_x).astype(int)
    image_coord_y = np.round(
        (world_coords[:, 1] + world_origin_offset_y)/scale_factor_y).astype(int)
    image_coords = np.hstack([image_coord_x[:, None], image_coord_y[:, None]])
    return image_coords



def world_to_image_coords(world_coords, map_name):
    return _world_to_image_coords(_coord_transform(world_coords, map_name), _map_sizes[map_name], _world_sizes[map_name], _origins[map_name])

# files = os.listdir('./data/datasets/idol/building3/known/')
# map_file = "./data/datasets/idol/building3.png"
# map_name = 'building3'
# _map_sizes = {
#     "building3":  get_map_by_name(map_file).shape,
#     "building2_f1":   get_map_by_name(map_file).shape,
#     "building2_f2":   get_map_by_name(map_file).shape,
#     "building1": get_map_by_name(map_file).shape,
# }

# _world_sizes = {
#     "building3":  (121,    102.5),
#     "building2_f1":   (69., 51),
#      "building2_f2":   (69, 54.),#"building2_f2":   (69., 54.),
#     "building1": (54, 18),
# }

# _origins = {  # world frame
#     "building3":  (60.960, 10.973),
#     "building2_f1":   (26.5,   34.5),
#      "building2_f2":   (22.5,   22.1),#"building2_f2":   (22.6,   22.1),
#     "building1": (14.023, 5.2),
# }
# # theta = 0.2822
# theta = 1.8510

# for file in files:
#     file_name = "./data/datasets/idol/building3/known/" + str(file) 
#     # file_name = "./../data/datasets/idol_dataset/building1/train/10.feather"
    
#     df = pd.read_feather(file_name)
#     # positions = df[['processedPosX', 'processedPosY']].values
#     x = df['processedPosX']*np.cos(theta) - df['processedPosY']*np.sin(theta)
#     y = df['processedPosX']*np.sin(theta) + df['processedPosY']*np.cos(theta)
#     positions = np.stack([x, y], axis=1)
#     # plt.plot(positions)
#     # plt.show()
#     plt.plot(positions[:, 0], positions[:, 1])
#     plt.show()

    # img = get_map_by_name(map_file)
    # map = ndimage.binary_dilation(get_map_by_name(map_file))
    # # plt.imshow(map.T, cmap='gray')
    # # plt.show()
    # # plt.imshow(img, cmap='gray')
    # # plt.show()
    
    # new_coor = world_to_image_coords(positions, map_name)
    # # plt.plot(positions[:, 0], positions[:, 1])
    # # plt.show()
    
    # plt.imshow(img.T, cmap='gray')
    # plt.plot(new_coor[:, 0], new_coor[:, 1])
    # plt.show()

data_list = []
list_file = './lists/idol/known_building2.txt'
data_dir = './data/datasets/idol_dataset/'
with open(list_file) as f:
    lines = [s.strip() for s in f.readlines() if len(s) > 0 and s[0] != '#']
for line in lines:
    data_name = line.split(',')[0]  # get the trajectory file name (without extension)
    data_list.append(data_name)

map_name = 'building2_f2'
map_file = "./data/datasets/idol/building2.png"
_map_sizes = {
    "building3":  get_map_by_name(map_file).shape,
    "building2_f1":   get_map_by_name(map_file).shape,
    "building2_f2":   get_map_by_name(map_file).shape,
    "building1": get_map_by_name(map_file).shape,
}

_world_sizes = {
    "building3":  (121,    102.5),
    "building2_f1":   (69., 51),
     "building2_f2":   (69, 54.),#"building2_f2":   (69., 54.),
    "building1": (54, 18),
}

_origins = {  # world frame
    "building3":  (60.960, 10.973),
    "building2_f1":   (26.5,   34.5),
     "building2_f2":   (22.5,   22.1),#"building2_f2":   (22.6,   22.1),
    "building1": (14.023, 5.2),
}
for file in data_list:
    df = pd.read_feather(data_dir + file + '.feather')
    # theta = 0.2822
    theta = 1.8510
        # positions[:, 0] = positions[:, 0]*np.cos(theta) - positions[:, 1]*np.sin(theta)
    x = df['processedPosX']*np.cos(theta) - df['processedPosY']*np.sin(theta)
    # positions[:, 1] = positions[:, 0]*np.sin(theta) + positions[:, 1]*np.cos(theta)
    y = df['processedPosX']*np.sin(theta) + df['processedPosY']*np.cos(theta)
    
    positions = np.stack([x, y], axis=1)
    plt.plot(positions[:, 0], positions[:, 1])
    plt.show()
    img = get_map_by_name(map_file)
    map = ndimage.binary_dilation(get_map_by_name(map_file))
    # plt.imshow(map.T, cmap='gray')
    # plt.show()
    # plt.imshow(img, cmap='gray')
    # plt.show()
    
    new_coor = world_to_image_coords(positions, map_name)
    # plt.plot(positions[:, 0], positions[:, 1])
    # plt.show()
    
    plt.imshow(img.T, cmap='gray')
    plt.plot(new_coor[:, 0], new_coor[:, 1])
    plt.show()
