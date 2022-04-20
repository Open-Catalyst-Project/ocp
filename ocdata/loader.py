from itertools import cycle
from shutil import get_terminal_size
from threading import Thread
from time import sleep, time


class Loader:
    def __init__(
        self,
        desc="Loading...",
        end="Done!",
        timeout=0.1,
        timer=True,
        erase=False,
    ):
        """
        A loader-like context manager

        Args:
            desc (str, optional): The loader's description. Defaults to "Loading...".
            end (str, optional): Final print. Defaults to "Done!".
            timeout (float, optional): Sleep time between prints. Defaults to 0.1.
        """
        self.desc = desc
        self.end = end
        self.timeout = timeout
        self.timer = timer
        self.erase = erase

        self._thread = Thread(target=self._animate, daemon=True)
        self.steps = ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"]
        self.done = False

    def start(self):
        self._start_time = time()
        self._thread.start()
        return self

    def _animate(self):
        for c in cycle(self.steps):
            if self.done:
                break
            print(f"\r{self.desc} {c}", flush=True, end="")
            sleep(self.timeout)

    def __enter__(self):
        self.start()

    def stop(self):
        self.done = True
        cols = get_terminal_size((80, 20)).columns

        if self.erase:
            end = f"\r{self.end}"
        else:
            end = self.desc + " | " + self.end

        if self.timer:
            end += f" ({time() - self._start_time:.2f}s)"

        print("\r" + " " * cols, end="", flush=True)
        print(end, flush=True)

    def __exit__(self, exc_type, exc_value, tb):
        # handle exceptions with those variables ^
        self.stop()
