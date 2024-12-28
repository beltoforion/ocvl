import os
import cv2
import screeninfo
import imageio

from ocvl.processor.input_output import *
from ocvl.source.source_sink import *


class VideoSink(Sink):

    def __init__(self):
        super(VideoSink, self).__init__()
        self._add_input(Input(self))
        
        self.__video_initialized = False
        self.__output_path = "out.mp4"
        self.__fps = 30
        self.__video_writer = None
        self.__gif_writer = None
        self.__output_format = OutputFormat.SAME_AS_INPUT


    @staticmethod
    def name_from_source(source_name):
        return source_name + "_video_sink"
    

    @property
    def fps(self):
        return self.__fps
    

    @fps.setter
    def fps(self, fps):
        self.__fps = fps


    @property
    def output_path(self):
        if self.output_format == OutputFormat.GIF:
            file_name, file_ext = os.path.splitext(self.__output_path)
            return file_name + ".gif"
        else:
            return self.__output_path
    

    @output_path.setter
    def output_path(self, output_path):
        self.__output_path = output_path


    @property
    def output_format(self):
        return self.__output_format
    

    @output_format.setter
    def output_format(self, output_format):
        self.__output_format = output_format


    def __init_video(self, screen_width, screen_height, is_color):
        self.__video_initialized = True

        if self.output_format == OutputFormat.SAME_AS_INPUT:
            fourcc = cv2.VideoWriter.fourcc('M','P','4','V')
            self.__video_writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (screen_width, screen_height), isColor=is_color)
        elif self.output_format == OutputFormat.GIF:
            self.__gif_writer = imageio.get_writer(self.output_path, mode='I', fps=self.fps)
        else:
            raise(Exception(f"Unsupported output format {self.output_format}!"))


    def process(self):
        screen = screeninfo.get_monitors()[0]
        screen_width, screen_height = int(screen.width*0.8), int(screen.height*0.8)

        resized_image = cv2.resize(self.input[0].data.image, (screen_width, screen_height))

        if self.__video_initialized == False:
            self.__init_video(screen_width, screen_height, is_color = len(resized_image.shape) > 2)

        cv2.imshow("Image", resized_image)
        cv2.waitKey(1)

        if self.output_format == OutputFormat.GIF:
            if self.__gif_writer is None:
                raise(Exception("GIF writer not initialized!"))

            self.__gif_writer.append_data(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
        else:
            if self.__video_writer is None:
                raise(Exception("Video writer not initialized!"))

            self.__video_writer.write(resized_image)


    def end_of_series(self):
        self.__video_initialized = False

        # Release the video capture and video writer objects
        if self.output_format == OutputFormat.GIF:
            if self.__gif_writer is None:
                raise(Exception("GIF writer not initialized!"))
            
            self.__gif_writer.close()
        else:
            if self.__video_writer is None:
                raise(Exception("Video writer not initialized!"))
            
            self.__video_writer.release()

        cv2.waitKey(0)
        cv2.destroyAllWindows()


