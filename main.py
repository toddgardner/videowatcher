import os

import cv2
import numpy as np
import subprocess as sp
import argparse
import glob

OPENCV_METHODS = {
    "correlation": cv2.HISTCMP_CORREL,
    "chi-squared": cv2.HISTCMP_CHISQR,
    "intersection": cv2.HISTCMP_INTERSECT,
    "hellinger": cv2.HISTCMP_BHATTACHARYYA
}

parser = argparse.ArgumentParser(description='Extract matching frames matching a color histogram from a video.')
parser.add_argument('video_url', metavar='URL', type=str,
                    help='video url to read')
parser.add_argument("-e", "--examples", default="/examples",
                    help="Path to the directory of images")
parser.add_argument("-o", "--output", default="/output",
                    help="Path to the directory of output images")
parser.add_argument("-m", "--method", default="chi-squared", choices=list(OPENCV_METHODS.keys()),
                    help="Histogram comparison method")
parser.add_argument("-c", "--cutoff", type=float, default=2.0, help="Histogram comparison cutoff")
args = vars(parser.parse_args())

# initialize the index dictionary to store the image name
# and corresponding histograms and the images dictionary
# to store the images themselves
reference_hists = []


def calculate_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def within_cutoff(distance):
    if args["method"] in ("correlation", "intersection"):
        if distance > args["cutoff"]:
            return True
    else:
        if distance < args["cutoff"]:
            return True
    return False


# loop over the image paths
for image_path in glob.glob(args["examples"] + "/*.png"):
    image = cv2.imread(image_path)
    # image = cv2.resize(image, (480, 640), interpolation=cv2.INTER_AREA)
    # extract a 3D RGB color histogram from the image,
    # using 8 bins per channel, normalize, and update
    # the index
    hist = calculate_histogram(image)
    reference_hists.append(hist)

if len(reference_hists) == 0:
    print("No references")
    exit(-1)

pipe = sp.Popen([
    "/usr/bin/ffmpeg", "-i", args["video_url"],
    "-loglevel", "quiet", # no text output
    "-an",   # disable audio
    "-f", "image2pipe",
    "-s", "640x480",
    "-pix_fmt", "bgr24",
    "-vcodec", "rawvideo", "-"],
    stdin=sp.PIPE, stdout=sp.PIPE)

currently_in_matching_sequence = False
found_numbers = 0
num_frames = 0
cooldown_frames = 0
back_to_back_matches = 0

os.makedirs(f"{args['output']}/nomatch", exist_ok=True)
os.makedirs(f"{args['output']}/match", exist_ok=True)

while True:
    raw_image = pipe.stdout.read(640*480*3) # read 432*240*3 bytes (= 1 frame)
    image = np.frombuffer(raw_image, dtype='uint8').reshape((480, 640, 3))

    if cooldown_frames > 0:
        cooldown_frames -= 1

    hist = calculate_histogram(image)

    distances = []
    any_within_cutoff = False
    for ref_hist in reference_hists:
        distances.append(cv2.compareHist(ref_hist, hist, OPENCV_METHODS[args["method"]]))

    if any(within_cutoff(d) for d in distances):
        if back_to_back_matches == 5:
            if cooldown_frames == 0:
                if not cv2.imwrite(f"{args['output']}/match/{found_numbers}.png", image):
                    print("Failed save match")
                print(f'Match {found_numbers}: distances {distances}')
                found_numbers += 1
                cooldown_frames = 8
                back_to_back_matches = 0
        else:
            back_to_back_matches += 1
    else:
        back_to_back_matches = 0

        if num_frames % 1000 == 0:
            report_num = int(num_frames / 1000)
            print(f'Failed match {report_num}: distances {distances}')
            if not cv2.imwrite(f"{args['output']}/nomatch/{report_num}.png", image):
                print("Failed write no match")

    num_frames += 1
