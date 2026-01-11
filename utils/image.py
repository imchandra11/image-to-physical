def convertToBoundingBox(center_x, center_y, width, height, Value, 
                         x_max, y_max, margin=5):
    """
    Convert from center and height/width to bounding box with max limits 
    and margin.
    """
    xTL = max(center_x - abs(width) / 2 - margin, 0)
    yTL = max(center_y - abs(height) / 2 - margin, 0)
    yBR = min(center_y + abs(height) / 2 + margin, y_max)
    xBR = min(center_x + abs(width) / 2 + margin, x_max)
    
    return [int(xTL), int(yTL), int(xBR), int(yBR), Value]