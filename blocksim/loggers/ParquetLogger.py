from typing import Any
import os
import json

from nptyping import NDArray
import pluggy
from singleton3 import Singleton
import pyarrow as pa
import pyarrow.parquet

from .Parameter import Parameter
from .Logger import Logger
from .. import logger
from ..exceptions import *


__all__ = ["Logger"]

hookimpl = pluggy.HookimplMarker("blocksim")


class Logger(object, metaclass=Singleton):
    @hookimpl
    def test_suitable(self, uri: str) -> bool:
        if uri is None:
            return False

        istat = uri.endswith(".parq")
        return istat

    @hookimpl
    def loadLogFile(self, log: "blocksim.loggers.Logger.Logger", uri: str):
        if not self.test_suitable(uri):
            return False

        if not os.path.exists(uri):
            raise FileNotFoundError(uri)

        log.reset()
        table = pa.parquet.read_table(uri)
        data = {}
        param_desc = json.loads(table.schema.metadata[b"param_desc"])

        for pdesc in param_desc:
            name, unit, desc, typ = pdesc
            dtyp = log.typ_map[typ.encode("utf-8")]
            if name.endswith(".re"):
                lname = name[:-3]
                data[lname] = np.array(table.column(name), dtype=dtyp)
            elif name.endswith(".im"):
                lname = name[:-3]
                data[lname] += 1j * np.array(table.column(name), dtype=dtyp)
            else:
                lname = name
                data[lname] = np.array(table.column(name), dtype=dtyp)

            log.createEmptyValue(
                name=lname,
                unit=unit,
                description=desc,
                dtype=dtyp,
            )

        log.setRawData(data, check_params=True)
        return True

    @hookimpl
    def getRawValue(
        self, log: "blocksim.loggers.Logger.Logger", name: str
    ) -> NDArray[Any, Any]:
        return

    @hookimpl
    def export(self, log: "blocksim.loggers.Logger.Logger", uri: str) -> int:
        if not self.test_suitable(uri):
            return -1

        data = {}
        pdesc = []
        for p in log.getParameters():
            styp = p.getTypeDB()

            if styp == "complex":
                pdesc.append(
                    (
                        p.name + ".re",
                        p.unit,
                        p.description,
                        "C",
                    )
                )
                pdesc.append(
                    (
                        p.name + ".im",
                        "",
                        "",
                        "C",
                    )
                )
                z = log.getRawValue(p.name)
                data[p.name + ".re"] = np.real(z)
                data[p.name + ".im"] = np.imag(z)

            else:
                if styp == "integer":
                    typ = "I"
                elif styp == "float":
                    typ = "F"
                elif styp == "boolean":
                    typ = "B"
                pdesc.append(
                    (
                        p.name,
                        p.unit,
                        p.description,
                        typ,
                    )
                )
                z = log.getRawValue(p.name)
                data[p.name] = z

        table = pa.Table.from_pydict(data, metadata={"param_desc": json.dumps(pdesc)})
        pa.parquet.write_table(table, uri, compression="snappy", version="2.6")

        return 0

    @hookimpl
    def log(
        self, log: "blocksim.loggers.Logger.Logger", name: str, val: float, tindex: int
    ) -> int:
        return 1
