# Animal-Tracking #

Tracking rats in an open field setting (black rat on white background). Also will segment out tracking field if corners are marked with green tape

## Dependencies ##

Requires the following packages:

* Python (>= 3.3)
* OpenCV (`pip install opencv-python`)
* Numpy
* IPython
* MatPlotLib
* progressbar (`pip install progressbar2`)

## Usage ##
`python rat_tracker.py [-h] [-L LENGTH] <folder>`


```
Positional Arguments:
folder                Folder with videos to track rats in


Optional Arguments:
-h, --help          show this help message and exit

-L LENGTH, --length LENGTH
                    Length of output video, in s

```
## Behavior ##
First, video will be cropped to remove any large black bars, isolating the actual video. The code will look for 4 green shapes to use as the corners, upon which it will perform a perspective transform up to a 600 px*600 px image. Then it will identify the larget black object as the rat, and compute the centroid. Tracked videos will be saved to a `<folder>/tracked/`, and a csv with the same file name will be saved to `<folder>` with coordinates. Summary statistics are saved in `<folder>/Summary.csv`, including:

* Total time of video (in s)
* Average Speed of rat (pixels / s)
* Percentage of time spent in the center (defined as center 1/3 in both dimensions)
* Number of times crossing the center

Script will automatically segment all videos found in the folder, unless there is a .csv found with the same file name.