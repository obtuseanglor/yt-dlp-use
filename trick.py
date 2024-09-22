import numpy as np
import pandas as pd
import imageio
import os
from multiprocessing import Pool
from itertools import cycle
import warnings
import time
from tqdm import tqdm
from skimage import img_as_ubyte
from skimage.transform import resize
import yt_dlp
from argparse import ArgumentParser
from util import save

warnings.filterwarnings("ignore")


def progress_hook(d):
    if d['status'] == 'downloading':
        print(f"Downloading {d['filename']}: {d['_percent_str']} at {d['_speed_str']} ETA {d['_eta_str']}")
    elif d['status'] == 'finished':
        print(f"Finished downloading {d['filename']}")
    elif d['status'] == 'error':
        print(f"Error downloading {d['filename']}")


def download(video_id, args):
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    video_path = os.path.join(args.video_folder, f"{video_id}.mp4")
    
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'outtmpl': video_path,
        'noplaylist': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        'progress_hooks': [progress_hook],
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    
    return video_path


def run(data):
    video_id, args = data
    video_path = os.path.join(args.video_folder, video_id.split('#')[0] + '.mp4')

    if not os.path.exists(video_path):
        download(video_id.split('#')[0], args)

    if not os.path.exists(video_path):
        print(f'Cannot load video {video_id.split("#")[0]}, broken link')
        return

    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data()['fps']

    df = pd.read_csv(args.metadata)
    df = df[df['video_id'] == video_id]

    all_chunks_dict = [{'start': df['start'].iloc[j], 'end': df['end'].iloc[j],
                        'bbox': list(map(int, df['bbox'].iloc[j].split('-'))), 'frames': []}
                       for j in range(df.shape[0])]

    ref_fps = df['fps'].iloc[0]
    ref_height = df['height'].iloc[0]
    ref_width = df['width'].iloc[0]
    partition = df['partition'].iloc[0]

    try:
        for i, frame in enumerate(reader):
            for entry in all_chunks_dict:
                if (i * ref_fps >= entry['start'] * fps) and (i * ref_fps < entry['end'] * fps):
                    left, top, right, bot = entry['bbox']
                    left = int(left / (ref_width / frame.shape[1]))
                    top = int(top / (ref_height / frame.shape[0]))
                    right = int(right / (ref_width / frame.shape[1]))
                    bot = int(bot / (ref_height / frame.shape[0]))
                    crop = frame[top:bot, left:right]
                    if args.image_shape is not None:
                        crop = img_as_ubyte(resize(crop, args.image_shape, anti_aliasing=True))
                    entry['frames'].append(crop)
    except imageio.core.format.CannotReadFrameError:
        pass

    for entry in all_chunks_dict:
        if 'person_id' in df:
            first_part = df['person_id'].iloc[0] + "#"
        else:
            first_part = ""
        first_part = first_part + '#'.join(video_id.split('#')[::-1])
        path = first_part + '#' + str(entry['start']).zfill(6) + '#' + str(entry['end']).zfill(6) + '.mp4'
        save(os.path.join(args.out_folder, partition, path), entry['frames'], args.format)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video_folder", default='youtube-taichi', help='Path to youtube videos')
    parser.add_argument("--metadata", default='taichi-metadata-new.csv', help='Path to metadata')
    parser.add_argument("--out_folder", default='taichi-png', help='Path to output')
    parser.add_argument("--format", default='.png', help='Storing format')
    parser.add_argument("--workers", default=1, type=int, help='Number of workers')
    parser.add_argument("--image_shape", default=(256, 256), type=lambda x: tuple(map(int, x.split(','))),
                        help="Image shape, None for no resize")
    
    args = parser.parse_args()

    if not os.path.exists(args.video_folder):
        os.makedirs(args.video_folder)
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)
    for partition in ['test', 'train']:
        if not os.path.exists(os.path.join(args.out_folder, partition)):
            os.makedirs(os.path.join(args.out_folder, partition))

    df = pd.read_csv(args.metadata)
    video_ids = set(df['video_id'])

    pool = Pool(processes=args.workers)
    args_list = cycle([args])
    for _ in tqdm(pool.imap_unordered(run, zip(video_ids, args_list))):
        pass
