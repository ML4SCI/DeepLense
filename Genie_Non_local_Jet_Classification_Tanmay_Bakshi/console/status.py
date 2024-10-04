from rich.console import Console as RichConsole
from rich.logging import RichHandler
from rich.spinner import Spinner
from rich.table import Table
from rich.status import Status
from rich.live import Live
from rich._log_render import LogRender
from time import time
import logging


class LogStatus(Status):
    def __init__(self,
                 status,
                 console: RichConsole,
                 level: int = logging.INFO,
                 enabled: bool = True,
                 speed: float = 1.0,
                 refresh_per_second: float = 12.5,
                 ):
        super().__init__(status,
                         console=console,
                         spinner='simpleDots',
                         speed=speed,
                         refresh_per_second=refresh_per_second
                         )

        self.status = status
        self.level = level
        self.enabled = enabled
        spinner = Spinner('simpleDots', style='status.spinner', speed=speed)
        record = logging.LogRecord(name=None, level=level, pathname=None, lineno=None, msg=None, args=None,
                                   exc_info=None)
        handler = RichHandler(console=console)
        table = Table.grid()
        table.add_row(self.status, spinner)

        self._spinner = LogRender(show_level=True, time_format='[%X]')(
            console=console,
            level=handler.get_level_text(record),
            renderables=[table]
        )
        self._live = Live(
            self.renderable,
            console=console,
            refresh_per_second=refresh_per_second,
            transient=True,
        )

    def __enter__(self):
        if self.enabled:
            self._start_time = time()
            return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            super().__exit__(exc_type, exc_val, exc_tb)
            self._end_time = time()
            self.console.log(f'{self.status}...done in {self._end_time - self._start_time:.2f} s', level=self.level)