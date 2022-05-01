import random
import os
import pickle

def get_colors():
    def generate_colors(n):
        rgb_values = []
        hex_values = []
        r = int((random.random() + 0.01) / 1 * 256)
        g = int((random.random() + 0.01) / 1 * 256)
        b = int((random.random() + 0.01) / 1 * 256)
        step = 256 / n
        for _ in range(n):
            r += step
            g += step
            b += step
            r = ((int(r) % 256)) % 255
            g = ((int(g) % 256) + 15) % 255
            b = ((int(b) % 256) + 105) % 255
            r_hex = hex(r)[2:]
            g_hex = hex(g)[2:]
            b_hex = hex(b)[2:]
            hex_values.append('#' + r_hex + g_hex + b_hex)
            rgb_values.append((r / 255, g / 255, b / 255))
        return hex_values, rgb_values

    if os.path.exists("visualization_tool/visualization_utils/colors.pkl"):
        COLORS = pickle.load(open("visualization_tool/visualization_utils/colors.pkl", 'rb'))
    else:
        COLORS = []
        for i in range(50):
            COLORS.append(generate_colors(i + 1))
        pickle.dump(COLORS, open("visualization_tool/visualization_utils/colors.pkl", 'wb'))
    return COLORS
