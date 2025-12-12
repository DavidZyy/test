import matplotlib.pyplot as plt
import numpy as np

# generate positions
# return list[tuple[int, int]]
# for covinience, picture width = height, window width = height, windo_size is odd
def natten(width, window_size):
    result = []
    for i in range(0, width):
        for j in range(0, width):
            row = i * width + j # row in attention map

            b_i = i - window_size // 2
            b_j = j - window_size // 2
            if b_i < 0:
                b_i = 0
            if b_i + window_size > width:
                b_i = width - window_size

            if b_j < 0:
                b_j = 0
            if b_j + window_size > width:
                b_j = width - window_size

            for x in range(b_i, b_i + window_size):
                for y in range(b_j, b_j + window_size):
                    col = x * width + y
                    result.append((row, col))
    return result

# picture, window, tile, are all squares
def sta(width, window_size, tile_size, pic):
    result = []
    width_tile = width // tile_size
    window_tile = window_size // tile_size
    # loop across tile
    for i in range(0, width_tile):
        for j in range(0, width_tile):
            b_i = i - window_tile // 2
            b_j = j - window_tile // 2

            if b_i < 0:
                b_i = 0
            if b_i + window_tile > width_tile:
                b_i = width_tile - window_tile

            if b_j < 0:
                b_j = 0
            if b_j + window_tile > width_tile:
                b_j = width_tile - window_tile

            # loop in tile
            for p in range(0, tile_size):
                for q in range(0, tile_size):
                    # row = (i*tile_size + p)*width + (j*tile_size + q)
                    row = int(pic[i*tile_size + p][j*tile_size + q])
                    
                    for x in range((b_i*tile_size), (b_i*tile_size)+window_size): 
                        for y in range((b_j*tile_size), (b_j*tile_size)+window_size):
                            # col = x * width + y
                            col = int(pic[x][y])
                            result.append((row, col))

    return result

# plot attention map from given positions
# input : list[tuple[int, int]], width: int
# background is white, attention positions are red, on position one grid cell
def plot(positions, width):

    # 2D attention map (not flattened)
    att = np.zeros((width, width))

    for (row, col) in positions:
        att[row][col] = 1

    plt.figure(figsize=(6, 6))
    plt.imshow(att, cmap='Reds', interpolation='nearest')

    # grid lines
    # plt.xticks(range(width))
    # plt.yticks(range(width))
    # plt.grid(color='black', linewidth=0.5)

    plt.title("Attention Map (red = attended)")
    plt.show()

'''
0   1   2   3            0   1   4   5
4   5   6   7     ->     2   3   6   7
8   9   10  11           8   9   12  13
12  13  14  15           10  11  14  15
'''
def rerange_to_tile(width, tile_size):
    pic = np.zeros((width, width))
    width_tile = width // tile_size

    pixel_id = 0
    for i in range(0, width_tile):
        for j in range(0, width_tile):
            
            for x in range(0, tile_size):
                for y in range(0, tile_size):

                    row = (i*tile_size + x)
                    col = (j*tile_size + y)

                    pic[row][col] = pixel_id
                    pixel_id += 1 
    return pic

     
if __name__ == "__main__":
    # width = 24
    # window_size = 13
    # positions = natten(width, window_size)

    width = 24
    window_size = 12
    tile_size = 4
    pic = rerange_to_tile(width, tile_size)
    positions = sta(width, window_size, tile_size, pic)
    plot(positions, width*width)

    # print(pic)
