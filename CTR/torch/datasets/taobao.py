from datetime import date

from .features import FeatureEncoder as BaseFeatureEncoder

class FeatureEncoder(BaseFeatureEncoder):
    def convert_hour(self, df, col_name):
        return df['time_stamp'].apply(lambda ts: ts[11:13])

    def convert_weekday(self, df, col_name):
        def _convert_weekday(timestamp):
            dt = date(int(timestamp[0:4]), int(timestamp[5:7]), int(timestamp[8:10]))
            return dt.strftime('%w')
        return df['time_stamp'].apply(_convert_weekday)

    def convert_weekend(self, df, col_name):
        def _convert_weekend(timestamp):
            dt = date(int(timestamp[0:4]), int(timestamp[5:7]), int(timestamp[8:10]))
            return '1' if dt.strftime('%w') in ['6', '0'] else '0'
        return df['time_stamp'].apply(_convert_weekend)