import os
import re
import csv
import warnings
from getpass import getpass
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from IPython.display import display
from PIL import Image
import io
import cv2
import ffmpeg

# Install dependencies
os.system("pip install -qU stability-sdk youtube-transcript-api langchain openai opencv-python yt-dlp ffmpeg-python")

# Set up Google Drive for file storage
try:
    from google.colab import drive
    drive.mount('/content/gdrive')
    outputs_path = "/content/gdrive/MyDrive/Commentator_AI/Transcript_to_Video"
    os.makedirs(outputs_path, exist_ok=True)
except:
    outputs_path = "."
print(f"Files will be saved to {outputs_path}")

# Set YouTube URL
YOUTUBE_URL = "https://www.youtube.com/watch?v=vPKp29Luryc"

def ytIdFromURL(url: str) -> str:
    data = re.findall(r"(?:v=|\\/)([0-9A-Za-z_-]{11}).*", url)
    if data:
        return data[0]
    return None

video_id = ytIdFromURL(YOUTUBE_URL)
if not video_id:
    raise ValueError("video_id isn't set")

out_dir = os.path.join(outputs_path, video_id)
print(f'YouTube ID: {video_id}')

# Get the Audio
audio_file_path = os.path.join(out_dir, 'audio.m4a')
if os.path.exists(audio_file_path):
    print('Audio already downloaded')
else:
    os.system(f'yt-dlp -f "bestaudio[ext=m4a]" -o "{audio_file_path}" "{YOUTUBE_URL}"')

# Get Video Transcript (CSV)
transcript = []
transcript_file_path = os.path.join(out_dir, 'transcript.csv')
fieldnames = ['start', 'duration', 'text']
if os.path.exists(transcript_file_path):
    with open(transcript_file_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        next(reader)  # skip header
        for row in reader:
            transcript.append(row)
        print(f'Read transcript from file: {transcript_file_path}')

if not transcript:
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    print('Got transcript from YouTube API')
    os.makedirs(out_dir, exist_ok=True)
    with open(transcript_file_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        for row in transcript:
            writer.writerow(row)
        print(f'Saved transcript to file: {transcript_file_path}')

# Get OpenAI API key
if 'OPENAI_API_KEY' not in os.environ:
    key = getpass('Enter your OpenAI API key: ')
    if key:
        os.environ['OPENAI_API_KEY'] = key

# ChatGPT selects lyrics to illustrate and generates image descriptions
CHAT_MODEL = 'gpt-4-0613'
TEMPERATURE = 0.5

image_description_csv_text = None
image_description_file_path = os.path.join(out_dir, 'image_descriptions.csv')
fieldnames = ['start', 'duration', 'text', 'description']
if os.path.exists(image_description_file_path):
    with open(image_description_file_path, 'r') as csv_file:
        image_description_csv_text = csv_file.read()
        print(f'Read image descriptions from file: {image_description_file_path}')
        print(image_description_csv_text)

if not image_description_csv_text:
    chat = ChatOpenAI(temperature=TEMPERATURE, model=CHAT_MODEL)
    print('ChatGPT working...')
    prompt_text = '''You're a visual musical artist.
Given the following lyrics choose the phrases that should be illustrated to make a timed music video for this song.
Respond in CSV format with the columns
'start', 'duration', 'text' (for the transcription text), 'description' (for the image description).
For the first row start at time 0 and make an image description that reflects the songs theme.
Keep in mind that each image description will be rendered separately so don't use any references between them.
Also because the image rendering is done in isolation for each description please be sure to include
enough thematic keys in them so the images are holistic related to the song's theme.
=== lyrics ==='''
    with open(transcript_file_path, 'r') as csv_file:
        prompt_text = '\n\n'.join([prompt_text, csv_file.read()])
    response = chat([HumanMessage(content=prompt_text)])
    print('Got image descriptions from ChatGPT')
    image_description_csv_text = response.content
    print(image_description_csv_text)
    os.makedirs(out_dir, exist_ok=True)
    with open(image_description_file_path, 'w', newline='') as csv_file:
        csv_file.write(response.content)
        print(f'Saved image descriptions to file: {image_description_file_path}')

# Connect to the Stability API
IMAGE_MODEL = 'stable-diffusion-xl-beta-v2-2-2'

if 'STABILITY_KEY' not in os.environ:
    key = getpass('Enter your Stability API key: ')
    if key:
        os.environ['STABILITY_KEY'] = key

stability_api = client.StabilityInference(
    key=os.environ['STABILITY_KEY'],
    engine=IMAGE_MODEL, verbose=True,
)

# Generate Images using Stable Diffusion
IMAGE_WIDTH = 896
IMAGE_HEIGHT = 512
IMAGE_GEN_STEPS = 22

def generate_image(prompt: str):
    answers = stability_api.generate(
        prompt=prompt,
        width=IMAGE_WIDTH, height=IMAGE_HEIGHT, steps=IMAGE_GEN_STEPS
    )

    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.finish_reason == generation.FILTER:
                warnings.warn(
                    "Your request activated the API's safety filters and could not be processed."
                    "Please modify the prompt and try again.")
            if artifact.type == generation.ARTIFACT_IMAGE:
                img = Image.open(io.BytesIO(artifact.binary))
                return img

    print(f"No image for '{prompt}'")
    return None

def description_to_filepath(description: str):
    filename = re.sub(r'[^\w\d-]', '_', description).lower()
    return os.path.join(out_dir, filename + '.png')

with open(image_description_file_path, 'r') as csv_file:
    reader = csv.DictReader(csv_file, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        print(row)
        description = row['description']
        image_file_path = description_to_filepath(description)
        if os.path.exists(image_file_path):
            print(f"Skipping description that has an image: {description}")
        else:
            print(f"Generating image for {description}")
            try:
                image = generate_image(description)
                image.save(image_file_path)
                display(image)
            except:
                print("Failed filters, got error")

# Generate Video using OpenCV
video_path = os.path.join(out_dir, 'video.mp4')

video_width = IMAGE_WIDTH
video_height = IMAGE_HEIGHT
print(video_width, ', ', video_height)

fps = 24.0

max_frame_count_limit = int(100 * fps)

total_frame_count = 0

def write_frames(image: Image, limit_seconds: float):
    global total_frame_count
    frame_count_limit = int(limit_seconds * fps)
    print(f"Adding {frame_count_limit - total_frame_count} frames for '{last_row['description']}'")
    if (frame_count_limit - total_frame_count) > max_frame_count_limit:
        frame_count_limit = total_frame_count + max_frame_count_limit
        print('Frame count too high. Reduced to ', frame_count_limit - total_frame_count)
    while total_frame_count < frame_count_limit:
        video.write(image)
        total_frame_count += 1

try:
    fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
    video = cv2.VideoWriter(video_path, fourcc, fps, (video_width, video_height))
    with open(image_description_file_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file, quoting=csv.QUOTE_NONNUMERIC)

        last_row = None
        last_image = None
        for row in reader:
            if last_image is not None:
                write_frames(image=last_image, limit_seconds=row['start'])
            print(row)
            description = row['description']
            image_file_path = description_to_filepath(description)
            if not os.path.exists(image_file_path):
                print(f"Missing image file: {image_file_path}")
                continue
            last_row = row
            last_image = cv2.imread(image_file_path)
        if last_image is not None:
            write_frames(image=last_image, limit_seconds=last_row['start'] + last_row['duration'])

finally:
    print('finalizing')
    cv2.destroyAllWindows()
    video.release()

print(f'Done! Wrote {total_frame_count} frames ({total_frame_count / fps} seconds) to {video_path}')

# Merge Audio with Video using ffmpeg-python
video_with_audio_path = os.path.join(out_dir, 'video_with_audio.mp4')

if os.path.exists(video_with_audio_path):
    print('video with audio already exists: ', video_with_audio_path)
else:
    video_in = ffmpeg.input(video_path)
    audio_in = ffmpeg.input(audio_file_path)
    print('concatenating with ffmpeg')
    try:
        result = ffmpeg.concat(video_in, audio_in, v=1, a=1).output(video_with_audio_path).run()
        print('Done merging')
    except:
        print('Merge failed!')

print(video_with_audio_path)
