from loggerdata import LoggerData

class DataStreams:
    def __init__(self):
        self.data = {}

    def add_logger_data(self, logger_id, dataframe=None):
        # Standardize the logger ID (e.g., remove hyphens, convert to uppercase)
        standardized_id = logger_id.replace('-', '').upper()
        self.data[standardized_id] = LoggerData(logger_id=standardized_id, dataframe=dataframe)

    def get_logger_data(self, logger_id):
        standardized_id = logger_id.replace('-', '').upper()
        if standardized_id in self.data:
            return self.data[standardized_id]
        else:
            raise KeyError(f"Logger ID '{logger_id}' not found in DataStreams.")

    def __getitem__(self, key):
        return self.get_logger_data(key)

    def __repr__(self):
        return f"DataStreams(data={list(self.data.keys())})"