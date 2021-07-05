# https://github.com/joerick/pyinstrument/
import pyinstrument


class TreeProfiler(object):
    """Use as a context manager to profile a block of code"""

    def __init__(self, show_all=False):
        self.profiler = pyinstrument.Profiler()
        self.show_all = show_all # verbose output of pyinstrument profiler

    def __enter__(self):
        print("WITH TREE_PROFILER:")
        self.profiler.start()

    def __exit__(self, *args):
        self.profiler.stop()
        print(self.profiler.output_text(unicode=True, color=True, show_all=self.show_all))