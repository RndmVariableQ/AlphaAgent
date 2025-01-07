import qlib

# qlib.init(provider_uri="~/.qlib/qlib_data/baostock_data")
qlib.init(provider_uri="~/.qlib/qlib_data/us_data")
from qlib.data import D

instruments = D.instruments()
fields = ["$open", "$close", "$high", "$low", "$volume"] # , "$amount", "$turn", "$pettm", "$pbmrq"
data = D.features(instruments, fields, freq="day").swaplevel().sort_index().loc["2022-01-01":].sort_index()

data.to_hdf("./daily_pv_all.h5", key="data")


fields = ["$open", "$close", "$high", "$low", "$volume"] # , "$amount", "$turn", "$pettm", "$pbmrq"
data = (
    (
        D.features(instruments, fields, start_time="2022-01-01", end_time="2024-12-31", freq="day")
        .swaplevel()
        .sort_index()
    )
    .swaplevel()
    .loc[data.reset_index()["instrument"].unique()[:100]]
    .swaplevel()
    .sort_index()
)

data.to_hdf("./daily_pv_debug.h5", key="data")
