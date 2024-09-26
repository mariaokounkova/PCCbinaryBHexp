## Setting up a virtual environment 

To create a valid python environment, run:

```
python3 -mvenv virtualenv
source virtualenv/bin/activate
pip install -r requirements.txt
```

To generate visualization data, do

```
python3 Generate_Visualization.py --q 1 --chiA 0 0 0 --chiB 0 0 0 --save_directory gw_strain_data
```

and then to make a movie with ffmpeg, do

```
ffmpeg -framerate 24 -i frame_%03d.png -c:v libx264 -pix_fmt yuv420p -r 30 output_movie.mp4
```
