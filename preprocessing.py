import os
import conf as cfg



def extract_iframes(videospath, iframespath):
    videofiles = os.listdir(videospath)
    videofiles = cfg.rm_ds(videofiles)
    for i, videofile in enumerate(videofiles):
        videopath = os.path.join(videospath, videofile)
        command = f"ffmpeg -skip_frame nokey -i {videopath} -vsync vfr -frame_pts true -x264opts no-deblock {iframespath}/video{i}iframe%d.bmp"
        os.system(command=command)





def main():
    videofolder = cfg.paths['videos']
    ifraempath = cfg.paths['iframes']
    # extract_iframes(videospath=videofolder, iframespath=ifraempath)



if __name__ == '__main__':
    main()