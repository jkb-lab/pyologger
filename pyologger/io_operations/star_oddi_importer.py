import pandas as pd

from pyologger.io_operations.csv_importer import CSVImporter


class StarOddiImporter(CSVImporter):
    """Star Oddi CSV/Parquet importer with optional ADC->degree pitch/roll calibration."""

    STAR_ODDI_ADC_THRESHOLD = 1000

    @staticmethod
    def _to_numeric(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce")

    def _maybe_calibrate_pitch_roll(self, df: pd.DataFrame) -> pd.DataFrame:
        pitch_col = next((c for c in df.columns if str(c).strip().lower() == "pitch"), None)
        roll_col = next((c for c in df.columns if str(c).strip().lower() == "roll"), None)
        if pitch_col is None and roll_col is None:
            return df

        calibrated_pitch = False
        calibrated_roll = False
        pitch_before = pitch_after = None
        roll_before = roll_after = None

        if pitch_col is not None:
            pitch_vals = self._to_numeric(df[pitch_col])
            if pitch_vals.gt(self.STAR_ODDI_ADC_THRESHOLD).any():
                pitch_before = (float(pitch_vals.min(skipna=True)), float(pitch_vals.max(skipna=True)))
                # Pitch(deg) = (ADC(Pitch) - 1220)*180/(2706-1220)
                pitch_cal = (pitch_vals - 1220.0) * 180.0 / (2706.0 - 1220.0)
                df[pitch_col] = pitch_cal
                pitch_after = (float(pitch_cal.min(skipna=True)), float(pitch_cal.max(skipna=True)))
                calibrated_pitch = True

        if roll_col is not None:
            roll_vals = self._to_numeric(df[roll_col])
            if roll_vals.gt(self.STAR_ODDI_ADC_THRESHOLD).any():
                roll_before = (float(roll_vals.min(skipna=True)), float(roll_vals.max(skipna=True)))
                # Roll(deg) = (ADC(Roll) - 1259)*180/(2746-1259)
                roll_cal = (roll_vals - 1259.0) * 180.0 / (2746.0 - 1259.0)
                df[roll_col] = roll_cal
                roll_after = (float(roll_cal.min(skipna=True)), float(roll_cal.max(skipna=True)))
                calibrated_roll = True

        if calibrated_pitch or calibrated_roll:
            print(
                f"🔧 Applied Star Oddi pitch/roll calibration for logger {self.logger_id} "
                f"(detected values > {self.STAR_ODDI_ADC_THRESHOLD})."
            )
            if calibrated_pitch and pitch_before and pitch_after:
                print(
                    f"   pitch range: {pitch_before[0]:.2f}..{pitch_before[1]:.2f} ADC "
                    f"-> {pitch_after[0]:.2f}..{pitch_after[1]:.2f} deg"
                )
            if calibrated_roll and roll_before and roll_after:
                print(
                    f"   roll range: {roll_before[0]:.2f}..{roll_before[1]:.2f} ADC "
                    f"-> {roll_after[0]:.2f}..{roll_after[1]:.2f} deg"
                )

        return df

    def apply_post_rename_transforms(self, df: pd.DataFrame, channel_metadata: dict) -> pd.DataFrame:
        updated = self._maybe_calibrate_pitch_roll(df)
        # Chain to the base transforms so acceleration unit standardization
        # is not skipped by this override.
        return super().apply_post_rename_transforms(updated, channel_metadata)
