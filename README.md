# figgear

Involute spur gears generator and animation example written in Python.
This library is for hobby use, and does not aim for correctness under ISO standard.

## Prerequisites

- Numpy
- Pillow
- Scipy

for gear_anim.py demo app
- Pygame

for gear_3d.py demo app
- PyOpenGL
- PyGLM

## Usage

```sh
git clone https://github.com/chromia/figgear
cd figgear
python gear_anim.py
python gear.py outputfile module tooth_number [options]
python gear.py -h  # for more details
```

To run an Animation app

```sh
python gear_anim.py
```

To run 3D demo app(requires a graphics card which supports OpenGL 3.3 or later)

```sh
python gear_3d.py
```

To generate m=8, z=20 (m is module number, z is the number of teeth).

```sh
python gear.py sample.png 8 20
```

To specify the color of gear, add options thus

```sh
python gear.py sample.png 8 20 -r=255 -g=0 -b=255 -a=255  # RGBA=(255,0,0,255)
```
By default, outputted gear is quite rough. To improve quality add option "--ssaa" (antialias).
```sh
python gear.py sample.png 8 20 -r=255 -g=0 -b=255 -a=255 --ssaa=4
```
