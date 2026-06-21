"""
Script credits: https://medium.com/@haydenfaulkner/extracting-frames-fast-from-a-video-using-opencv-and-python-73b9b7dc9661
"""

import cv2
import os
import shutil
import argparse
from decord import VideoReader
from decord import cpu
from pathlib import Path


def extract_frames(video_path, frames_dir, overwrite=False, start=-1, end=-1, every=1):
    """
    Extract frames from a video using decord's VideoReader
    :param video_path: path of the video
    :param frames_dir: the directory to save the frames
    :param overwrite: to overwrite frames that already exist?
    :param start: start frame
    :param end: end frame
    :param every: frame spacing
    :return: count of images saved
    """

    video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible
    frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible

    video_dir, video_filename = os.path.split(video_path)  # get the video path and filename from the path

    assert os.path.exists(video_path)  # assert the video file exists

    # load the VideoReader
    vr = VideoReader(video_path, ctx=cpu(0))  # can set to cpu or gpu .. ctx=gpu(0)

    if start < 0:  # if start isn't specified lets assume 0
        start = 0
    if end < 0:  # if end isn't specified assume the end of the video
        end = len(vr)

    frames_list = list(range(start, end, every))
    saved_count = 0

    if every > 25 and len(frames_list) < 1000:  # this is faster for every > 25 frames and can fit in memory
        frames = vr.get_batch(frames_list).asnumpy()

        for index, frame in zip(frames_list, frames):  # lets loop through the frames until the end
            save_path = os.path.join(frames_dir, video_filename, "{:010d}.jpg".format(index))  # create the save path
            if not os.path.exists(save_path) or overwrite:  # if it doesn't exist or we want to overwrite anyways
                cv2.imwrite(save_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # save the extracted image
                saved_count += 1  # increment our counter by one

    else:  # this is faster for every <25 and consumes small memory
        for index in range(start, end):  # lets loop through the frames until the end
            frame = vr[index]  # read an image from the capture

            if index % every == 0:  # if this is a frame we want to write out based on the 'every' argument
                save_path = os.path.join(frames_dir, video_filename,
                                         "{:010d}.jpg".format(index))  # create the save path
                if not os.path.exists(save_path) or overwrite:  # if it doesn't exist or we want to overwrite anyways
                    cv2.imwrite(save_path, cv2.cvtColor(frame.asnumpy(), cv2.COLOR_RGB2BGR))  # save the extracted image
                    saved_count += 1  # increment our counter by one

    return saved_count  # and return the count of the images we saved


def video_to_frames(video_path, frames_dir, overwrite=False, every=1):
    """
    Extracts the frames from a video
    :param video_path: path to the video
    :param frames_dir: directory to save the frames
    :param overwrite: overwrite frames if they exist?
    :param every: extract every this many frames
    :return: path to the directory where the frames were saved, or None if fails
    """

    video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible
    frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible

    video_dir, video_filename = os.path.split(video_path)  # get the video path and filename from the path

    # make directory to save frames, it's a sub dir in the frames_dir with the video name
    os.makedirs(os.path.join(frames_dir, video_filename), exist_ok=True)

    print("Extracting frames from {}".format(video_filename))

    extract_frames(video_path, frames_dir, overwrite=overwrite, every=every)  # let's now extract the frames

    return os.path.join(frames_dir, video_filename)  # when done return the directory containing the frames


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract frames from MP4 videos and copy them to a combined folder.")
    parser.add_argument("--video_folder", type=str, required=True, help="Path to the folder containing .mp4 videos")
    parser.add_argument("--frames_dir", type=str, required=True, help="Path to the folder where extracted frames will be saved")
    parser.add_argument("--combined_dir", type=str, default=None, help="Path to the folder where combined frames will be saved (defaults to <frames_dir>_combined)")
    parser.add_argument("--frame_rate", type=int, default=5, help="Frame rate spacing: extract every N-th frame (default: 5)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing extracted frames")

    args = parser.parse_args()

    # Resolve paths relative to PROJECT_ROOT if they are relative
    video_folder = Path(args.video_folder)
    if not video_folder.is_absolute():
        video_folder = (PROJECT_ROOT / video_folder).resolve()
    else:
        video_folder = video_folder.resolve()

    frames_dir = Path(args.frames_dir)
    if not frames_dir.is_absolute():
        frames_dir = (PROJECT_ROOT / frames_dir).resolve()
    else:
        frames_dir = frames_dir.resolve()

    if args.combined_dir:
        combined_frames_dir = Path(args.combined_dir)
        if not combined_frames_dir.is_absolute():
            combined_frames_dir = (PROJECT_ROOT / combined_frames_dir).resolve()
        else:
            combined_frames_dir = combined_frames_dir.resolve()
    else:
        combined_frames_dir = (frames_dir.parent / f"{frames_dir.name}_combined").resolve()

    frame_rate = args.frame_rate
    overwrite = args.overwrite

    if not video_folder.exists():
        print(f"Error: Video folder does not exist at '{video_folder}'")
        exit(1)

    print("Running Frame Extraction:")
    print(f"  - Video Folder: {video_folder}")
    print(f"  - Frames Directory: {frames_dir}")
    print(f"  - Combined Directory: {combined_frames_dir}")
    print(f"  - Frame Rate: every {frame_rate} frames")
    print(f"  - Overwrite: {overwrite}")

    video_files = [f for f in os.listdir(video_folder) if f.lower().endswith('.mp4')]
    if not video_files:
        print(f"No .mp4 files found in '{video_folder}'")
        exit(0)

    for video_file in video_files:
        video_path = video_folder / video_file
        video_to_frames(
            video_path=str(video_path),
            frames_dir=str(frames_dir),
            overwrite=overwrite,
            every=frame_rate
        )

    # Combine the frames into a single directory for easier loading later
    os.makedirs(combined_frames_dir, exist_ok=True)

    print(f"Combining all extracted frames into '{combined_frames_dir}'...")
    combined_count = 0
    for subdir in os.listdir(frames_dir):
        subdir_path = frames_dir / subdir
        if subdir_path.is_dir():
            for frame_file in os.listdir(subdir_path):
                frame_path = subdir_path / frame_file
                if frame_path.is_file() and frame_path.suffix.lower() == '.jpg':
                    # Create the new filename and path
                    new_filename = f"{subdir}_{frame_file}"
                    combined_frame_path = combined_frames_dir / new_filename

                    if not combined_frame_path.exists() or overwrite:
                        shutil.copy2(frame_path, combined_frame_path)
                        combined_count += 1
    print(f"Copied {combined_count} frames to '{combined_frames_dir}'.")
