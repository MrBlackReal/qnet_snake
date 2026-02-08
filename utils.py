from snake_game import BLOCK_SIZE, Point

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

def get_accessible_area(head, obstacles, w, h, limit=60):
    """
    Performs a BFS flood-fill to count reachable free blocks.
    stops early if limit is reached to save performance.
    """
    queue = [head]
    visited = {head}
    count = 0
    
    while queue:
        curr = queue.pop(0)
        count += 1
        
        if count >= limit:
            return limit
            
        # Check 4 neighbors
        for dx, dy in [(BLOCK_SIZE, 0), (-BLOCK_SIZE, 0), (0, BLOCK_SIZE), (0, -BLOCK_SIZE)]:
            next_pt = Point(curr.x + dx, curr.y + dy)
            
            # Check bounds
            if next_pt.x < 0 or next_pt.x >= w or next_pt.y < 0 or next_pt.y >= h:
                continue
                
            # Check obstacles and visited
            if next_pt not in obstacles and next_pt not in visited:
                visited.add(next_pt)
                queue.append(next_pt)
                
    return count