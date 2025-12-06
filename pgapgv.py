#!/usr/bin/env python
from obspy.core import read, Stream
import glob
import numpy as np
import matplotlib.pyplot as plt
from obspy.signal.trigger import z_detect
from response_spectrum import ResponseSpectrum, NigamJennings, plot_response_spectra, plot_time_series
import sm_utils


def run_pgapgv(data_pattern="data/*"):
    """Process all files matching data_pattern and make spectra + time-series plots."""
    files = glob.glob(data_pattern)

    st = Stream()
    for curfile in files:
        st += read(curfile)

    st._cleanup()
    st.sort(['network', 'station', 'channel', 'starttime'])

    pers = np.linspace(0.01, 20, 300)

    for idx, tr in enumerate(st):
        tr.detrend('demean')
        tr.taper(max_percentage=0.05, type='cosine')
        tr.filter('highpass', freq=1.0)

        rs = NigamJennings(
            tr.data,
            1 / tr.stats.sampling_rate,
            pers,
            damping=0.5,
            units="m/s/s",
        )
        spec, ts, acc, vel, dis = rs.evaluate()

        # Response spectrum plot (saved to PNG)
        plt.figure(figsize=(8, 8))
        plt.loglog(spec["Period"], spec["Pseudo-Acceleration"] * 0.1 / 9.81,
                   'k', linewidth=2)
        plt.xlim([0.1, 10])
        plt.xlabel('Period (s)', fontsize=18)
        plt.ylabel('Acceleration (g)', fontsize=18)
        plt.title('QCN Pseudo-Acceleration Response Spectra', fontsize=18)
        plt.tick_params(labelsize=18)
        spec_name = f"Spec_{tr.stats.station}_{tr.stats.channel}.png"
        plt.savefig(spec_name, dpi=400)
        plt.close()

        # Time-series plot (uses provided helper)
        ts_name = f"TS_{tr.stats.station}_{tr.stats.channel}.png"
        plot_time_series(
            ts["Acceleration"],
            1 / tr.stats.sampling_rate,
            velocity=ts["Velocity"],
            displacement=ts["Displacement"],
            filename=ts_name,
        )


if __name__ == "__main__":
    run_pgapgv()
