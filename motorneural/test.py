from motorneural.datasets.hatsopoulos import HatsoData
from motorneural.analysis import plot_event_triggered_response, plot_trajectories
import matplotlib.pyplot as plt


if __name__ == "__main__":
    data_dir = "~/data/hatsopoulos"
    data = HatsoData.make(data_dir, "CO_01", lag=.1, bin_sz=.01)
    print("Loaded " + str(data))
    plot_event_triggered_response(data, event="max_spd", tradius=[.3, .4], group_by="ang", shuff=False)
    if data.task == "CO":
        plot_trajectories(data, group_by="ang", event_markers=["mv_st", "max_spd"])
    plt.show()
