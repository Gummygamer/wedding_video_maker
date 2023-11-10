import cv2
import os

import cv2
import os
import numpy as np

def make_video(image_folder, output_video, extra_images, fps=1, time=3, transition_frames=30):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpeg") or img.endswith(".jpg")]

    if not images:
        print("No images found in the folder.")
        return

    images.sort()  # Sort the images by name
    images.extend(extra_images)

    total_images = len(images)
    print(f"Total images to process: {total_images}")

    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"Failed to read the image: {first_image_path}")
        return

    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for idx, image in enumerate(images):
        print(f"Processing image {idx + 1} of {total_images}...")

        if image in extra_images:
            image_path = image
        else:
            image_path = os.path.join(image_folder, image)

        current_frame = cv2.imread(image_path)
        if current_frame is None:
            print(f"Failed to read the image: {image_path}")
            continue

        current_frame = cv2.resize(current_frame, (width, height), interpolation=cv2.INTER_AREA)

        for _ in range(fps * time):
            video.write(current_frame)

        if idx < len(images) - 1:
            next_image_path = images[idx + 1] if idx + 1 < len(images) else extra_images[0]
            if next_image_path in extra_images:
                next_image_path = next_image_path
            else:
                next_image_path = os.path.join(image_folder, next_image_path)

            next_frame = cv2.imread(next_image_path)
            next_frame = cv2.resize(next_frame, (width, height), interpolation=cv2.INTER_AREA)

            for t in range(transition_frames):
                combined_frame = np.zeros_like(current_frame)
                alpha = t / transition_frames
                for row in range(height):
                    for col in range(width):
                        offset = int(width * alpha)
                        if col + offset < width:
                            combined_frame[row, col] = current_frame[row, col + offset]
                        if col - (width - offset) >= 0:
                            combined_frame[row, col] = next_frame[row, col - (width - offset)]

                video.write(combined_frame.astype(np.uint8))

    cv2.destroyAllWindows()
    video.release()

image_folder = 'groom'
output_video = 'groom_video.mp4'
extra_images = ['proposal.jpeg', 'couple.jpg']
fps = 24
time = 2
transition_frames = 48  # Number of frames for transition

make_video(image_folder, output_video, extra_images, fps, time, transition_frames)