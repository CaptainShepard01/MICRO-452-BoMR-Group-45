import numpy as np
import matplotlib.pyplot as plt

ALLOW_DIAG = True

global_map = [['.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.'],
              ['.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.'],
              ['.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.'],
              ['.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.'],
              ['x','x','x','x','x','.','.','.','x','.','.','.','.','.','.','.'],
              ['.','.','.','.','.','.','.','.','x','.','.','.','.','.','.','.'],
              ['.','.','.','.','.','.','.','.','x','.','.','x','x','x','.','.'],
              ['.','.','.','.','.','.','.','.','x','.','.','.','.','x','.','.'],
              ['.','.','.','.','.','.','.','.','x','.','.','g','.','x','.','.'],
              ['.','.','.','.','.','.','.','.','x','.','.','.','.','x','.','.'],
              ['.','.','.','.','.','.','.','.','x','.','.','.','.','x','.','.'],
              ['.','.','x','x','x','x','x','x','x','x','x','x','x','x','.','.'],
              ['.','.','x','.','.','.','.','.','.','.','.','.','.','.','.','.'],
              ['.','.','x','.','.','.','.','.','.','.','.','.','.','.','.','.'],
              ['.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.'],
              ['.','.','.','.','.','.','.','s','.','.','.','.','.','.','.','.']]

def get_cell(code):
    for y in range(len(global_map)):
        for x in range(len(global_map[0])):
            if global_map[y][x] == code:
                return [x, y]
            
def get_code(x, y):
    return global_map[y][x]
            
def get_neighbours_index(grid_map, cell):
    x = cell[0]
    y = cell[1]
    neighbours = []

    if y > 0:
        neighbours.append([x,y-1])
    if x < len(grid_map[0]) - 1:
        neighbours.append([x+1,y])
    if y < len(grid_map) - 1:
        neighbours.append([x,y+1])
    if x > 0:
        neighbours.append([x-1,y])

    if ALLOW_DIAG:
        if y > 0 and x > 0:
            neighbours.append([x-1,y-1])
        if y > 0 and x < len(grid_map[0]) - 1:
            neighbours.append([x+1,y-1])
        if y < len(grid_map) - 1 and x < len(grid_map[0]) - 1:
            neighbours.append([x+1,y+1])
        if y < len(grid_map) - 1 and x > 0:
            neighbours.append([x-1,y+1])

    return neighbours


def has_marked_neighbour(grid_map, cell):
    neighbours = get_neighbours_index(grid_map, cell)
    for c in neighbours:
        if grid_map[c[1]][c[0]] != -1:
            return True
    return False

def lowest_marked_neighbour(grid_map, cell):
    x = cell[0]
    y = cell[1]
    neighbours = get_neighbours_index(grid_map, cell)

    min = grid_map[y][x]
    idx = 0
    for i in range(len(neighbours)):
        c = neighbours[i]
        val = grid_map[c[1]][c[0]]
        if val != -1 and val < min:
            min = val
            idx = i

    return neighbours[idx]

def search_path():
    n = 0
    grid_map = np.zeros((len(global_map), len(global_map[0])))
    grid_map.fill(-1)
    path = []

    S = get_cell('s')
    G = get_cell('g')
    grid_map[S[1]][S[0]] = n

    while grid_map[G[1]][G[0]] == -1:
        marked_list = []
        n += 1
        for y in range(len(grid_map)):
            for x in range(len(grid_map[0])):
                val = grid_map[y][x]
                if get_code(x, y) != 'x' and val == -1 and has_marked_neighbour(grid_map, [x,y]):
                    marked_list.append([x,y])

        for c in marked_list:
            grid_map[c[1]][c[0]] = n

    current = G
    path.append(current)

    i = 0
    while S not in path:
        c = lowest_marked_neighbour(grid_map, current)
        path.append(c)
        current = c
        i += 1
        if i > 1000:
            break

    for c in path:
        global_map[c[1]][c[0]] = 'o'
    
    plot_grid(grid_map, S, G)


def plot_grid(grid_map, S, G):
    fig, ax = plt.subplots()

    dct = {'.': 1, 's': 1, 'g':1, 'x': 10, 'o': 5.}
    n = [[dct[i] for i in j] for j in global_map]
    ax.imshow(n, cmap='Greys')

    for y in range(len(grid_map)):
        for x in range(len(grid_map[0])):
            c = int(grid_map[x][y])
            if c != -1 and ([y,x] != S and [y,x] != G):
                ax.text(y, x, c, va='center',ha='center')

    ax.text(S[0], S[1], 'S', color='g', va='center', ha='center')
    ax.text(G[0], G[1], 'G', color='r', va='center', ha='center')

    plt.show()


search_path()