from __future__ import annotations

import time

from .config import settings
from .database import db
from .models import new_id
from .review import work_once


def main() -> None:
    chunk_set_id = new_id("chunks")
    while True:
        if not work_once(db, settings, chunk_set_id):
            time.sleep(1)


if __name__ == "__main__":
    main()
