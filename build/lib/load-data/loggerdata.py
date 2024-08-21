import pandas as pd

class LoggerData:
    def __init__(self, logger_id, dataframe=None):
        self.logger_id = logger_id
        self.dataframe = dataframe if dataframe is not None else pd.DataFrame()
        self.status = {
            'concatenated': False,
            'processed': False,
            'checked_for_jumps': False
        }

    def update_status(self, status_key, value=True):
        if status_key in self.status:
            self.status[status_key] = value
        else:
            raise KeyError(f"Status key '{status_key}' not found in LoggerData status.")

    def concatenate(self, additional_data):
        self.dataframe = pd.concat([self.dataframe, additional_data], ignore_index=True)
        self.update_status('concatenated')

    def process(self):
        # Implement your data processing logic here
        self.update_status('processed')

    def check_for_jumps(self):
        # Implement your logic to check for jumps in timesteps
        self.update_status('checked_for_jumps')

    def __repr__(self):
        return f"LoggerData(logger_id={self.logger_id}, status={self.status})"