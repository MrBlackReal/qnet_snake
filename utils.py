from snake_game import BLOCK_SIZE

def manhattan_distance(start, end):
    x, y = start
    x1, y1 = end

    dx = abs(x1 - x)
    dy = abs(y1 - y)

    return dx + dy

def manhattan_distance_blocks(start, end):
    x, y = start
    x1, y1 = end

    dx = abs(x1 - x)
    dy = abs(y1 - y)

    return (dx + dy) / BLOCK_SIZE