import os
import os.path as osp
from glob import glob
import time
import sys
from OpenGait.demo.libs.track import writeresult

sys.path.append(os.path.abspath('.') + "/demo/libs/")
from segment import *
from recognise import *

def main():
    output_dir = "./demo/output/OutputVideos/"
    os.makedirs(output_dir, exist_ok=True)
    current_time = time.localtime()
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    video_save_folder = osp.join(output_dir, timestamp)
    
    save_root = './demo/output'
    base_p = '/home/mscherbina/Documents/datasets/simple_mct/'
    video_paths = glob(base_p + '*.mp4')
    video_names = [Path(i).name for i in video_paths]

    track_results = []
    i = 0
    for vid_path, vid_name in zip(video_paths, video_names):
        track_file = video_save_folder + '/' + vid_name
        track_file = track_file.replace(".mp4", "_track.pkl")
        print(f"Tracking video {i}/{len(video_paths)}")
        if not os.path.exists(track_file):
            from track import track, cleanup
            track_obj = track(vid_path, video_save_folder)
            with open(track_file, 'wb') as f:
                pickle.dump(track_obj, f)
        else:
            with open(track_file, 'rb') as f:
                track_obj = pickle.load(f)
        # track_results.append(track_obj)
        i += 1

    exit(0)
    """
    sil_names = [
        save_root + '/GaitSilhouette/' + i.replace('.mp4', '')
        for i in video_names
    ]

    exist = True
    for i in sil_names:
        exist &= os.path.exists(i)
    siluettes = []
    print("Exist: ", exist)
    if exist:
        for i in video_paths:
            siluettes.append(
                getsil(i, save_root + '/GaitSilhouette/')
            )
    else:
        for p, tr in zip(video_paths, track_results):
            siluettes.append(
                seg(p, tr, save_root + '/GaitSilhouette/')
            )

    feats = []
    for i in siluettes:
        feats.append(
            extract_sil(i, save_root+'/GaitFeatures/')
        )
    """

    gallery_feat = feats[0]
    comp_results = []
    for i in feats[1:]:
        comp_results.append(compare(i, gallery_feat))

    for cr, vp in zip(comp_results, video_paths[1:]):
        # write the result back to the video
        writeresult(cr, vp, video_save_folder)


if __name__ == "__main__":
    main()
