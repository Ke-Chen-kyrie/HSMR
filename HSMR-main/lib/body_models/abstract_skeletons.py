# "Skeletons" here means virtual "bones" between joints. It is used to draw the skeleton on the image.

class Skeleton():
    bones = []
    bone_colors = []
    chains = []
    parent = []


class Skeleton_OpenPose25(Skeleton):
    ''' https://www.researchgate.net/figure/Twenty-five-keypoints-of-the-OpenPose-software-model_fig1_374116819 '''
    chain = [
            [ 8, 12, 13, 14, 19, 20],  # left leg
            [14, 21],                  # left heel
            [ 8,  9, 10, 11, 22, 23],  # right leg
            [11, 24],                  # right heel
            [ 8,  1,  0],              # spine & head
            [ 0, 16, 18],              # left face
            [ 0, 15, 17],              # right face
            [ 1,  5,  6,  7],          # left arm
            [ 1,  2,  3,  4],          # right arm
        ]
    bones = [
            [ 8, 12], [12, 13], [13, 14],  # left leg
            [14, 19], [19, 20], [14, 21],  # left foot
            [ 8,  9], [ 9, 10], [10, 11],  # right leg
            [11, 22], [22, 23], [11, 24],  # right foot
            [ 8,  1], [ 1,  0],            # spine & head
            [ 0, 16], [16, 18],            # left face
            [ 0, 15], [15, 17],            # right face
            [ 1,  5], [ 5,  6], [ 6,  7],  # left arm
            [ 1,  2], [ 2,  3], [ 3,  4],  # right arm
        ]
    bone_colors = [
            [ 95,   0, 255], [ 79,   0, 255], [ 83,   0, 255],  # dark blue
            [ 31,   0, 255], [ 15,   0, 255], [  0,   0, 255],  # dark blue
            [127, 205, 255], [127, 205, 255], [ 95, 205, 255],  # light blue
            [ 63, 205, 255], [ 31, 205, 255], [  0, 205, 255],  # light blue
            [255,   0,   0], [255,   0,   0],                   # red
            [191,  63,  63], [191,  63, 191],                   # magenta
            [255,   0, 127], [255,   0, 255],                   # purple
            [127, 255,   0], [ 63, 255,   0], [  0, 255,   0],  # green
            [255, 127,   0], [255, 191,   0], [255, 255,   0],  # yellow

        ]
    parent = [1, 8, 1, 2, 3, 1, 5, 6, -1, 8, 9, 10, 8, 12, 13, 0, 0, 15, 16, 14, 19, 14, 11, 22, 11]
