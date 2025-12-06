#!/usr/bin/env python
import glob

import numpy as np
from obspy.core import read, Stream

from response_spectrum import (
    NigamJennings,
    plot_response_spectra,
    plot_time_series,
)


def run_pgapgv(data_pattern="data/*"):
    """
    Process all strong-motion files matching data_pattern and
    make response-spectrum and time-series plots for each trace.
    """
    files = glob.glob(data_pattern)
    if not files:
        print(f"No files found for pattern: {data_pattern}")
        return

    # Read all files into a single Stream
    st = Stream()
    for curfile in files:
        st += read(curfile)

    # Clean and sort traces
    st._cleanup()
    st.sort(["network", "station", "channel", "starttime"])

    # Periods for response spectrum
    pers = np.linspace(0.01, 20.0, 300)

    for tr in st:
        # Basic preprocessing
        tr.detrend("demean")
        tr.taper(max_percentage=0.05, type="cosine")
        tr.filter("highpass", freq=1.0)

        # Compute response spectrum using Nigam & Jennings
        rs = NigamJennings(
            tr.data,
            1.0 / tr.stats.sampling_rate,
            pers,
            damping=0.5,
            units="m/s/s",
        )
        spec, ts, acc, vel, dis = rs.evaluate()

        # ----- Response spectrum plot (PNG) -----
        spec_name = f"Spec_{tr.stats.station}_{tr.stats.channel}.png"
        plot_response_spectra(spec, filename=spec_name)

        # ----- Time-series plot (PNG) -----
        ts_name = f"TS_{tr.stats.station}_{tr.stats.channel}.png"
        plot_time_series(
            ts["Acceleration"],
            1.0 / tr.stats.sampling_rate,
            velocity=ts["Velocity"],
            displacement=ts["Displacement"],
            filename=ts_name,
        )


if __name__ == "__main__":
    run_pgapgv()
