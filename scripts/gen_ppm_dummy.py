width = height = 256

with open("dummy.ppm", "wb") as f: 
    f.write(b"P6\n%d %d\n255\n" % (width, height))
    for y in range(height):
        for x in range(width):
            r = x % 256
            g = y % 256
            b = (x ^ y) % 256 
            f.write(bytes([r, g, b]))