import pandas as pd
import tqdm
import matplotlib.pyplot as plt
import time
import numpy as np
import statsmodels.formula.api as smf

class ACCREDataProcessor:
    def __init__(self):
        self.jobs = None
        self.servers_9204 = None
        self.jobs_filtered_by_completed = None
        self.jobs_grouped = None
        self.data = None
        self.rolling_range_results = None

    def logistic(self, x):
        return 1/(1+np.exp(-x))

    def read_csv_chunk(self, filename, data_chunk_size):
        """
        Reads a CSV file in chunks and concatenates the chunks into a single DataFrame.

        Args:
            filename (str): The path to the CSV file.
            data_chunk_size (int): Number of rows per chunk.

        Returns:
            pandas.DataFrame: The concatenated DataFrame containing all data from the CSV file.
        """
        print(f"Now reading '{filename}'.")
        total_rows = sum(1 for _ in open(filename)) - 1
        with tqdm.tqdm(total=total_rows, unit="rows", desc="Reading CSV") as pbar:
            data_chunks = []
            for data_chunk in pd.read_csv(filename, chunksize=data_chunk_size):
                data_chunks.append(data_chunk)
                pbar.update(len(data_chunk))
                time.sleep(0.01)
        print(f"Completed reading '{filename}'.")
        return pd.concat(data_chunks, ignore_index=True)

    def setup_accre_data(self, chunk_size=10000):
        """
        Sets up and processes data for jobs and servers.

        Args:
            chunk_size (int, optional): Number of rows to process per chunk. Defaults to 10000.

        Returns:
            None
        """
        if all(var is None for var in [self.jobs, self.servers_9204, self.jobs_filtered_by_completed, self.jobs_grouped, self.data]):
            print("Initializing the data.")
            self.jobs = self.read_csv_chunk('../data/updated_jobs.csv', chunk_size)
            self.servers_9204 = self.read_csv_chunk('../data/servers_9204.csv', chunk_size)

            self.jobs['BEGIN_dt'] = pd.to_datetime(self.jobs['BEGIN_dt'])
            self.jobs['END_dt'] = pd.to_datetime(self.jobs['END_dt'])
            self.servers_9204['datetime'] = pd.to_datetime(self.servers_9204['datetime'])
            self.servers_9204 = self.servers_9204[self.servers_9204['5'].str.contains('sbatch')]
            self.jobs_filtered_by_completed = self.jobs[self.jobs['STATE'] == 'COMPLETED']

            self.jobs_filtered_by_completed['end_dt_min_start'] = self.jobs_filtered_by_completed['END_dt'].dt.floor('min')
            self.jobs_grouped = (
                self.jobs_filtered_by_completed.groupby('end_dt_min_start')['STATE']
                .count()
                .reset_index(name='total_completed_jobs')
            )

            min_floor_9204 = self.servers_9204
            min_floor_9204['datetime'] = self.servers_9204['datetime'].dt.floor('min')
            grouped_servers_9204 = (
                min_floor_9204.groupby('datetime')['slurm_success']
                .min()
                .reset_index(name='slurm_success')
            )
            data_df = (
                pd.merge(right=grouped_servers_9204, left=self.jobs_grouped, right_on='datetime', left_on='end_dt_min_start', how='outer')
                .dropna(subset='end_dt_min_start')
                .drop(columns=['datetime'])
                .rename(columns={'end_dt_min_start':'datetime'})
                .set_index('datetime')
            )
            self.data = data_df[['total_completed_jobs', 'slurm_success']]

    def identify_slurm_periods(self, begin=None, end=None):
        """
        Identifies periods of success and failure in SLURM jobs.

        Args:
            begin (str, optional): Start date in ISO format (e.g., 'YYYY-MM-DD'). Defaults to None.
            end (str, optional): End date in ISO format (e.g., 'YYYY-MM-DD'). Defaults to None.

        Returns:
            tuple: Two lists of tuples representing success and failure periods. Each tuple contains the start and end datetime.
        """
        self.setup_accre_data()
        if begin and end:
            filtered_servers = self.servers_9204[self.servers_9204['datetime'].between(begin, end)]
        else:
            filtered_servers = self.servers_9204

        filtered_servers = filtered_servers.sort_values('datetime').reset_index(drop=True)
        slurm_successes = []
        slurm_failures = []

        current_status = None
        start_time = None

        for _, row in filtered_servers.iterrows():
            status = row['slurm_success']
            time = row['datetime']

            if current_status is None:
                current_status = status
                start_time = time
            elif status != current_status:
                if current_status == 1:
                    slurm_successes.append((start_time, time))
                else:
                    slurm_failures.append((start_time, time))
                current_status = status
                start_time = time

        if start_time is not None:
            if current_status == 1:
                slurm_successes.append((start_time, filtered_servers['datetime'].iloc[-1]))
            else:
                slurm_failures.append((start_time, filtered_servers['datetime'].iloc[-1]))

        return slurm_successes, slurm_failures

    def plot_jobs_and_servers_with_slurm_periods(self, begin_date, end_date, rolling=0):
        """
        Plots total completed jobs and SLURM periods of success and failure.

        Args:
            rolling (int, optional): Size of the rolling average window. Defaults to 0 (no rolling average).
            begin_date (str): Start date for the plot in ISO format (e.g., 'YYYY-MM-DD'). This parameter is required.
            end_date (str): End date for the plot in ISO format (e.g., 'YYYY-MM-DD'). This parameter is required.

        Returns:
            None

        Raises:
            ValueError: If `begin_date` or `end_date` is not provided or is invalid.
        """
        if not begin_date or not end_date:
            raise ValueError("Both 'begin_date' and 'end_date' must be provided and valid.")
        
        self.setup_accre_data()
        
        jobs_filtered = self.jobs_grouped[self.jobs_grouped['end_dt_min_start'].between(begin_date, end_date)]
        success_regions, failure_regions = self.identify_slurm_periods(begin=begin_date, end=end_date)

        x = jobs_filtered['end_dt_min_start']
        y = jobs_filtered['total_completed_jobs']

        plt.figure(figsize=(12, 8))
        plt.plot(x, y, color='red', alpha=0.7, label='Total Completed Jobs')

        for start, end in success_regions:
            plt.axvspan(start, end, color='green', alpha=0.3)

        for start, end in failure_regions:
            plt.axvspan(start, end, color='red', alpha=0.3)

        handles = [
            plt.Line2D([0], [0], color='green', lw=4, alpha=0.3, label='Slurm Success Period'),
            plt.Line2D([0], [0], color='red', lw=4, alpha=0.3, label='Slurm Failure Period'),
            plt.Line2D([0], [0], color='red', label='Total Completed Jobs')
        ]

        if rolling > 0:
            plt.plot(x, y.rolling(rolling).mean(), color='black', alpha=.7, label='Rolling Average')
            handles.append(plt.Line2D([0], [0], color='black', label=f'Total Completed Jobs (Rolling {rolling} min Average)'))

        plt.legend(handles=handles, loc='upper left')
        plt.title(f'Total Jobs Completed and Slurm Periods ({begin_date} to {end_date})', fontsize=14)
        plt.xlabel('Datetime', fontsize=12)
        plt.ylabel('Jobs Completed', fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def test_range_rolling_completed_jobs(self, range_begin=1, range_end=5, step_size=1, closed='left'):
        self.setup_accre_data()
        window_sizes = [str(x)+'min' for x in range(range_begin, range_end, step_size)]
        self.rolling_range_results = {}
        for window in window_sizes:
            rolling_col_name = f'rolling_completed_jobs_{window}'
            self.data[rolling_col_name] = self.data['total_completed_jobs'].rolling(window, closed=closed).mean()
            formula = f'slurm_success ~ total_completed_jobs + {rolling_col_name}'
            model = smf.logit(formula, data=self.data).fit()
            self.rolling_range_results[window] = model
            print(f"Summary for {window} rolling average:")
            print(model.summary())

    def calculate_rolling_probability(self, rolling_jobs_completed: float, total_completed_jobs: int, rolling_period: str):
        if self.rolling_range_results is None:
            raise ValueError("Function 'test_range_rolling_completed_jobs' has not yet been called.")
        if rolling_period not in self.rolling_range_results:
            raise ValueError(f"'{rolling_period}' not found in self.rolling_range_results. Please try a period within the range in the format '1min'.")
        rolling_formula_result = self.rolling_range_results[rolling_period].params['Intercept'] + self.rolling_range_results[rolling_period].params['total_completed_jobs']*total_completed_jobs + self.rolling_range_results[rolling_period].params['rolling_completed_jobs_'+rolling_period]*rolling_jobs_completed
        return self.logistic(rolling_formula_result)
        