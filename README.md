# image statistics examples

Some hacked-together examples for looking at RAW/JPG image statistics with numpy.

## Requirements

You'll need to install the Pillow and rawpy libraries.
rawpy must be installed from the latest version on github:

```
pip install --upgrade git+git://github.com/letmaik/rawpy.git#egg=rawpy
```

## Usage

I recommend going thru the JPG program first; you can run for example like:
```
./example-jpg.py path/to/images/*.jpg
```

Similarly you can run the RAW version by globbing for the (gzipped) raw files:
```
./example-raw.py path/to/images/*.dng.gz
```
