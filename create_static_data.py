import pandas as pd
import numpy as np

input_file_name = "test_data/imu_test3.csv"

output_file_name = "static_test.csv"

static_window_length_in_s = 9
repetitions = 3


def main():

    try:
        imu_data = pd.read_csv(input_file_name, engine="python")

        start_time = int(imu_data["timestamp [ns]"].iloc[0])
        end_time = start_time + 1_000_000_000 * static_window_length_in_s

        static_beginning = imu_data[imu_data["timestamp [ns]"] <= end_time].copy()

        timestamps = static_beginning["timestamp [ns]"].values
        dt_ns = int(np.mean(np.diff(timestamps)))

        extended_df = pd.concat([static_beginning] * repetitions, ignore_index=True)

        # Generate a perfectly continuous, monotonically increasing timestamp array
        n_rows = len(extended_df)
        new_timestamps = start_time + np.arange(n_rows, dtype=np.int64) * dt_ns

        # Overwrite timestamps
        extended_df["timestamp [ns]"] = new_timestamps

        extended_df.to_csv(output_file_name, index=False)

        # # was ist die letzte row in unserem static time window?
        # last_row = 1
        # while imu_data["timestamp [ns]"].iloc[last_row] < end_time:
        #     last_row += 1

        # print(last_row)
        # # output: 1036
        # # also grob ca 103 Hz über die 10 sekunden hinweg
        # # an sich muss ich also nur auf den unix time stamp 10/1036 Sekunden oben drauf machen
        # # 1 Sekunde sind 0.000 000 001 e18

        # # irgendwie werden die Werte ab 1000 doch etwas anders, vielleicht habe ich die 10 Sekunden falsch abgelesen aus den eigentlichen Daten und die Bewegung beginnt schon früher
        # # last_row = 980

        # # jetzt kann ich einmal gyro und accel nach den ersten 1036 rows loopen

        # # eine neue Tabelle erstellen, darin die ersten 10 Sekunden einfach so kopieren
        # simulated_table = []
        # for index, row in imu_data.iterrows():
        #     if index > last_row:
        #         break

        #     simulated_table.append(row.copy())

        # # dann die darauffolgenden 50 Sekunden simulieren
        # # dafür einfach für gyro und accel nochmal genau das, was in den ersten 10 Sekunden passiert ist loopen
        # # für die Zeit den Mittelwert nehmen aus der Anzahl der Samples in den ersten 10 Sekunden und übersetzen in jeden nächsten time step
        # sample_step_time = 90_000_000_000 * (9 / last_row)
        # for i in range(last_row * 5):
        #     current_row = imu_data.iloc[i % last_row].copy()
        #     current_row["timestamp [ns]"] = end_time + (sample_step_time) * (i + 1)
        #     simulated_table.append(current_row)

        # df = pd.DataFrame(simulated_table)
        # df.to_csv(output_file_name, index=False)

        # optional noch eine Visualisierung, also einfach ein Diagramm was zeigen sollte das alle gyro Daten sich grob um die 0 herum bewegen sollten und alle accel Daten auch bis auf die z-Richtung, also da wo gravity daraus eine 1 macht

    except FileNotFoundError:
        print(f"Error: Could not find '{input_file_name}'.")


if __name__ == "__main__":
    main()
