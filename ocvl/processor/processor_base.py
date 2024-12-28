from ocvl.source.source_sink import Sink
from ocvl.processor.io_data import IoData


class ProcessorBase(Sink):
    def __init__(self, name):
        super(ProcessorBase, self).__init__()
        self.__name = name
        self._outputs = []

    def connect_input(self, idx, output):
        output.connect(self._inputs[idx])

    def _add_output(self, output):
        self._outputs.append(output)

    def end_of_series(self):
        """
        Forward End of Stream to all outputs.
        """
        for o in self._outputs:
            o.set(IoData(None))

    @property
    def name(self):
        return self.__name

    @property
    def output(self):
        return self._outputs
