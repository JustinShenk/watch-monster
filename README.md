# Watch Monster

Track monsters on a monitor with this Halloween party toy.

## Demo

![Preview GIF](watch-monster.gif)

## Getting started

#### Raspberry Pi

- Install OpenCV on [Raspberry Pi](http://www.pyimagesearch.com/2016/04/18/install-guide-raspberry-pi-3-raspbian-jessie-opencv-3/) or ubuntu:

```
[compiler] sudo apt-get install build-essential
[required] sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev python-dev python-numpy libpng-dev
```

#### Laptop

- Install OpenCV with Conda

```sh
conda install -c menpo opencv3
```

- Clone the repository:

```
git clone https://github.com/JustinShenk/watch-monster.git
cd watch-monster
```

- Install dependencies:

```
pip install -r requirements.txt
```

- Run the program

```
python watch-monitor.py
```

#### Uploading images to Imgur (Optional)

Images can be stored on a PyImgur album.

```sh
mv config.json.example config.json
```

and add your API key to `config.json`.
