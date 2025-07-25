from dataclasses import dataclass

from graphrag.Core.Storage.BaseStorage import BaseStorage


@dataclass
class BaseBlobStorage(BaseStorage):
    async def get(self):
        raise NotImplementedError

    async def set(self, blob) -> None:
        raise NotImplementedError
