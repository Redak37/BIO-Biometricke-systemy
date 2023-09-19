"""
    BIO Project

    Authors:    Ivana Saranová
                Radek Duchoň
    File:       img_analysis.py
    Date:       12. 12. 2020
"""
import math
import os.path

from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2

import config

SHAPE_PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'


def get_vector_from_two_points(start, end):
    """
    Create vector from two points
    :param start: Startpoint
    :param end: Endpoint
    :return: Vector from startpoint to endpoint
    """
    return tuple([end[0] - start[0], end[1] - start[1]])


def get_angle_between_vectors(vec1, vec2):
    """
    Returns angle between two vectors
    :param vec1: Vector 1
    :param vec2: Vector 2
    :return: Angle
    """
    unit_vector_1 = vec1 / np.linalg.norm(vec1)
    unit_vector_2 = vec2 / np.linalg.norm(vec2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    angle = np.rad2deg(angle)
    return angle


def get_avg_point(list_of_points):
    x, y = 0, 0
    for p in list_of_points:
        x += p[0]
        y += p[1]

    return tuple([
        round(x / len(list_of_points)),
        round(y / len(list_of_points))
    ])


class ImageAnalysis:
    def __init__(self, img_path):
        """
        :param img_path: Path with PNG Image
        """
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
        self.image = img_path
        self.landmarks = self.get_landmarks()

        self.vertical_vec = get_vector_from_two_points(
            (config.DATASET_IMG_SIZE // 2, config.DATASET_IMG_SIZE // 2),
            (config.DATASET_IMG_SIZE // 2, 0)
        )
        self.horizontal_vec = get_vector_from_two_points(
            (config.DATASET_IMG_SIZE // 2, config.DATASET_IMG_SIZE // 2),
            (config.DATASET_IMG_SIZE, config.DATASET_IMG_SIZE // 2)
        )

        self.expected_pitch = float(os.path.split(self.image)[-1][:-4].split('_')[-3][1:].replace('p', ''))
        self.expected_yaw = float(os.path.split(self.image)[-1][:-4].split('_')[-2][1:])
        self.expected_roll = float(os.path.split(self.image)[-1][:-4].split('_')[-1][1:])

        self.left_eye_center = get_avg_point([self.landmarks[index] for index in config.LEFT_EYE])
        self.right_eye_center = get_avg_point([self.landmarks[index] for index in config.RIGHT_EYE])
        self.left_eye_left_corner = self.landmarks[36]
        self.left_eye_right_corner = self.landmarks[39]
        self.right_eye_left_corner = self.landmarks[42]
        self.right_eye_right_corner = self.landmarks[45]
        self.between_eyes = self.landmarks[27]

    def get_landmarks(self):
        """ Source: http://dlib.net/face_landmark_detection.py.html """
        # load the input image, resize it, and convert it to grayscale
        image = cv2.imread(self.image)
        image = imutils.resize(image, width=config.DATASET_IMG_SIZE)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale image
        rects = self.detector(gray, 1)
        if not rects:
            raise Exception('Face not detected!')

        # loop over the face detections
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            return shape

    def get_roll_rotation(self):
        """
        :return: Approximation of roll of face
        """
        vec = get_vector_from_two_points(self.right_eye_center, self.left_eye_center)
        avg_angle = get_angle_between_vectors(self.horizontal_vec, vec)
        vec = get_vector_from_two_points(self.left_eye_left_corner, self.left_eye_right_corner)
        avg_angle2 = get_angle_between_vectors(self.horizontal_vec, vec)
        vec = get_vector_from_two_points(self.right_eye_left_corner, self.right_eye_right_corner)
        avg_angle3 = get_angle_between_vectors(self.horizontal_vec, vec)

        if vec[1] > self.horizontal_vec[1]:
            avg_angle *= -1
            avg_angle2 *= -1
            avg_angle3 *= -1

        combination = avg_angle
        if abs(avg_angle2) - 1 > abs(avg_angle) > abs(avg_angle3) + 2 \
                or abs(avg_angle2) + 2 < abs(avg_angle) < abs(avg_angle3) - 1:
            combination *= 6
            combination += avg_angle2 + avg_angle3
            combination /= 8

        return combination

    def round_rotation(self, rotation, mod):
        """
        Round rotation to the nearest mod
        :param rotation: type of rotation - roll/yaw/pitch
        :param mod: number
        :return: Rounded rotation
        """
        if mod < 1:
            raise Exception('Incorrect mod for rounding')

        if rotation == 'roll':
            return float(round(self.get_roll_rotation() / mod) * mod)
        if rotation == 'yaw':
            return float(round(self.get_yaw_rotation() / mod) * mod)
        if rotation == 'pitch':
            return float(round(self.get_pitch_rotation() / mod) * mod)

        raise Exception('Incorrect rotation')

    def is_rotation_correct(self, rotation):
        """
        Returns True if calculated rotation of type "rotation" is correct
        :param rotation: type of rotation - roll/yaw/pitch
        :return: True/False
        """
        if rotation == 'roll':
            return self.round_rotation(rotation, 5) == self.expected_roll
        if rotation == 'yaw':
            return self.round_rotation(rotation, 5) == self.expected_yaw
        if rotation == 'pitch':
            return self.round_rotation(rotation, 5) == self.expected_pitch

    def fix_landmarks_to_zero_roll_rotation(self):
        """
        Fix landmarks to cancel roll of face
        :return: new landmarks
        """
        origin = complex(config.MIDDLE_POINT[0], config.MIDDLE_POINT[1])
        points = np.array([complex(landmark[0], landmark[1]) for landmark in self.landmarks])
        angle = self.round_rotation('roll', 5)

        landmarks = (points - origin) * np.exp(complex(0, math.radians(angle))) + origin
        landmarks = [(round(keypoint.real), round(keypoint.imag)) for keypoint in landmarks]

        return landmarks

    def get_yaw_rotation(self):
        """
        :return: Approximation of yaw of face
        """
        self.landmarks = self.fix_landmarks_to_zero_roll_rotation()
        R = self.landmarks[config.EXCA_R]
        L = self.landmarks[config.EXCA_L]
        S = self.landmarks[config.SELION]
        C = (config.DATASET_IMG_SIZE // 2, config.DATASET_IMG_SIZE // 2)

        SCL = math.atan((config.MEAN_BIOCULAR_BREADTH / 2) / (config.MEAN_SELION_TO_BACK_HEAD / 2))
        ratio = (config.MEAN_SELION_TO_BACK_HEAD / 2) / (config.MEAN_BIOCULAR_BREADTH / 2)

        ref_RS = math.dist(R, S)
        ref_LS = math.dist(L, S)
        if (ref_RS == ref_LS) or (ref_LS - 2 <= ref_RS <= ref_LS + 2):
            return 0.0

        elif ref_RS < ref_LS:
            for i in range(C[1] * 10, 10000):
                estimated_Lz = i / 10
                LC = math.dist((L[0], estimated_Lz), C)

                for j in range(C[1] * 10, i):
                    estimated_Sz = j / 10
                    SL = math.dist((S[0], estimated_Sz), (L[0], estimated_Lz))
                    SC = math.dist((S[0], estimated_Sz), C)

                    if ratio - 0.05 <= SC / SL <= ratio + 0.05:
                        LC_x_diff = abs(C[0] - L[0])
                        gamma = math.asin(LC_x_diff / LC)
                        return -math.degrees(SCL + (-gamma if C[0] < L[0] else gamma))

        else:
            for i in range(C[1] * 10, 10000):
                estimated_Rz = i / 10
                RC = math.dist((R[0], estimated_Rz), C)

                for j in range(C[1] * 10, i):
                    estimated_Sz = j / 10
                    SR = math.dist((S[0], estimated_Sz), (R[0], estimated_Rz))
                    SC = math.dist((S[0], estimated_Sz), C)

                    if ratio - 0.05 <= SC / SR <= ratio + 0.05:
                        RC_x_diff = abs(R[0] - C[0])
                        gamma = math.asin(RC_x_diff / RC)
                        return math.degrees(SCL + (-gamma if C[0] > R[0] else gamma))

        return 0.0

    def get_pitch_rotation(self):
        """
        :return: Approximation of pitch of face
        """
        self.landmarks = self.fix_landmarks_to_zero_roll_rotation()
        S = self.landmarks[config.MIDDLE_NOSE]
        M = self.landmarks[config.MENTON]
        C = (config.DATASET_IMG_SIZE // 2, config.DATASET_IMG_SIZE // 2)

        ratio = (config.MEAN_SELION_TO_BACK_HEAD / 2) / config.MEAN_MIDDLE_NOSE_MENTON
        ref_vec = get_vector_from_two_points(C, (C[0] + config.MEAN_SELION_TO_BACK_HEAD / 2, C[1]))

        if (S[1] == C[1]) or (S[1] - 5 <= C[1] <= S[1] + 5):
            return 0.0

        elif S[1] < C[1]:
            for i in range(10000, C[0] * 10, -1):
                estimated_Mz = i / 10
                CM = math.dist(C, (estimated_Mz, M[1]))

                for j in range(i, C[0] * 10, -1):
                    estimated_Sz = j / 10
                    SM = math.dist((estimated_Sz, S[1]), (estimated_Mz, M[1]))

                    if ratio - 0.02 <= CM / SM <= ratio + 0.02:
                        CS_vec = get_vector_from_two_points(C, (estimated_Sz, S[1]))
                        angle = get_angle_between_vectors(CS_vec, ref_vec)
                        return -angle

        else:
            for i in range(10000, C[0] * 10, -1):
                estimated_Sz = i / 10
                SM = math.dist(C, (estimated_Sz, S[1]))

                for j in range(i, C[0] * 10, -1):
                    estimated_Mz = j / 10
                    CM = math.dist(C, (estimated_Mz, M[1]))

                    if ratio - 0.05 <= CM / SM <= ratio + 0.05:
                        CS_vec = get_vector_from_two_points(C, (estimated_Sz, S[1]))
                        angle = get_angle_between_vectors(ref_vec, CS_vec)
                        return angle

        return 0.0

    def show_image(self, landmarks_off=False):
        """
        Show image
        :param landmarks_off: Turn off showing of landmarks on image
        """
        image = cv2.imread(self.image)
        image = imutils.resize(image, width=config.DATASET_IMG_SIZE)

        if not landmarks_off:
            for (x, y) in self.landmarks:
                cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

        cv2.destroyAllWindows()
        cv2.imshow("Output", image)
        cv2.waitKey(0)
