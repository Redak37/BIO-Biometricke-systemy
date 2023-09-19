"""
    BIO Project

    Authors:    Ivana Saranová
                Radek Duchoň
    File:       main.py
    Date:       12. 12. 2020
"""
from img_analysis import ImageAnalysis


png = 'head/p-10.0_y-10.0_r20.0.png'

try:
    ima = ImageAnalysis(png)
    ima.show_image()
except Exception as ex:
    print(ex)
    exit(0)

roll = round(ima.get_roll_rotation(), 1)
dif1 = abs(roll - ima.expected_roll)

yaw = round(ima.get_yaw_rotation(), 1)
dif2 = abs(yaw - ima.expected_yaw)

pitch = round(ima.get_pitch_rotation(), 1)
dif3 = abs(pitch - ima.expected_pitch)

print(f'Actual:     Roll {roll} Yaw {yaw} Pitch {pitch}')
print(f'Expected:   Roll {ima.expected_roll} Yaw {ima.expected_yaw} Pitch {ima.expected_pitch}')

