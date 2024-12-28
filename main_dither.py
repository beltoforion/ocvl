import os

from ocvl.processor.dither_processor import *
from ocvl.processor.greyscale_processor import *
from ocvl.processor.scale_processor import *
from ocvl.processor.rgb_split_processor import *
from ocvl.processor.rgb_join_processor import *
from ocvl.processor.median_processor import *
from ocvl.processor.normalize_processor import *
from ocvl.helper.opencv_helper import *
from ocvl.source.file_source import *
from ocvl.source.video_sink import *
from ocvl.source.file_sink import *


def dither_grey(source : FileSource, sink : FileSink | VideoSink, target_size : tuple[float, float], levels, dither_method):
    last = greyscale_processor = GreyscaleProcessor()
    greyscale_processor.connect_input(0, source.output[0])

    last = scale_processor = ScaleProcessor()
    scale_processor.target_size = target_size
    scale_processor.connect_input(0, greyscale_processor.output[0])

    last = dither_processor = DitherProcessor()
    dither_processor.method = dither_method
    dither_processor.levels = levels
    dither_processor.connect_input(0, scale_processor.output[0])

    # connect the sink to the last processors output
    last.output[0].connect(sink.input[0])

    file : str = source.file_path
    file_name, file_ext = os.path.splitext(file)
    sink.output_path = f"{file_name}_{dither_method.name}_monochrome{file_ext}"

    return source


def dither_rgb(source : FileSource, sink : FileSink | VideoSink, target_size : tuple[float, float], levels, dither_method, rgb_method):
    """ Apply dithering to all color channels and combine the image back into an rgb image.
    """
    last = scale_processor = ScaleProcessor()
    scale_processor.target_size = target_size
    scale_processor.interpolation = cv2.INTER_NEAREST
    scale_processor.connect_input(0, source.output[0])

    last = rgb_split_processor = RgbSplitProcessor()
    rgb_split_processor.connect_input(0, scale_processor.output[0])

    last = dither_blue = DitherProcessor()
    dither_blue.method = dither_method
    dither_blue.levels = levels
    dither_blue.connect_input(0, rgb_split_processor.output[0])

    last = dither_green = DitherProcessor()
    dither_green.method = dither_method
    dither_green.levels = levels
    dither_green.connect_input(0, rgb_split_processor.output[1])

    last = dither_red = DitherProcessor()
    dither_red.method = dither_method
    dither_red.levels = levels
    dither_red.connect_input(0, rgb_split_processor.output[2])

    last = rgb_join_processor = RbgJoinProcessor()
    rgb_join_processor.method = rgb_method
    rgb_join_processor.connect_input(0, dither_blue.output[0])
    rgb_join_processor.connect_input(1, dither_green.output[0])
    rgb_join_processor.connect_input(2, dither_red.output[0])

    # connect the sink to the last processors output
    last.output[0].connect(sink.input[0])

    file = source.file_path
    file_name, file_ext = os.path.splitext(file)
    sink.output_path = f"{file_name}_{dither_method.name}_{rgb_method.name}{file_ext}"

    return source


dither_method = DitherMethod.SIERRA
rgb_method = RgbJoinMethod.THREE_COLOR
#output_format = OutputFormat.PNG
output_format = OutputFormat.SAME_AS_INPUT
num_levels = 3
levels = [int(i * 255 / (num_levels - 1)) for i in range(num_levels)]
#input_file = "images/ship.jpg"
input_file = "images/drone_watertower.mp4"
target_size = (960, 540)

#
# Source and Sink setup
#

source = FileSource(input_file)
#source.max_num_frames = 100

sink = VideoSink() if os.path.splitext(input_file)[1].lower()=='.mp4' else FileSink()
sink.output_format = output_format

#
# Job execution
#
#job = dither_rgb(source, sink, (target_size[0], target_size[1] * (0.75 if rgb_method == RgbJoinMethod.THREE_COLOR else 1)), levels, dither_method, rgb_method)
job = dither_grey(source, sink, target_size, levels, dither_method)
job.start()
