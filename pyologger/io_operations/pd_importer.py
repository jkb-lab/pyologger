import pandas as pd

from pyologger.io_operations.csv_importer import CSVImporter


class PDImporter(CSVImporter):
    """CSV importer with PD-specific channel cleanup rules."""

    def apply_post_rename_transforms(self, df: pd.DataFrame, channel_metadata: dict) -> pd.DataFrame:
        updated_df = df

        depth_cols = self._get_signal_columns(channel_metadata, "depth", updated_df.columns)
        preferred_depth = self._pick_preferred_column(depth_cols, "depth")
        if preferred_depth is not None:
            updated_df = updated_df.copy()
            updated_df[preferred_depth] = pd.to_numeric(updated_df[preferred_depth], errors="coerce") * -1
            print(f"🔁 Applied PD depth inversion to column '{preferred_depth}'.")

        light_cols = self._get_signal_columns(channel_metadata, "light", updated_df.columns)
        velocity_cols = self._get_signal_columns(channel_metadata, "velocity", updated_df.columns)
        preferred_light = self._pick_preferred_column(light_cols, "light")
        preferred_velocity = self._pick_preferred_column(velocity_cols, "velocity")

        if preferred_light is not None and preferred_velocity is not None:
            light_values = pd.to_numeric(updated_df[preferred_light], errors="coerce")
            velocity_values = pd.to_numeric(updated_df[preferred_velocity], errors="coerce")
            mask = (light_values > 60000) & (velocity_values > 2)
            if mask.any():
                if updated_df is df:
                    updated_df = updated_df.copy()
                updated_df.loc[mask, preferred_velocity] = 0
                print(
                    f"🧹 Applied PD velocity cleanup to {int(mask.sum())} row(s) using "
                    f"'{preferred_light}' and '{preferred_velocity}'."
                )

        stroke_cols = self._get_signal_columns(channel_metadata, "stroke_rate", updated_df.columns)
        converted_stroke_cols: list[str] = []
        for stroke_col in stroke_cols:
            meta = channel_metadata.get(stroke_col, {})
            raw_unit = str(meta.get("unit", "")).strip().lower()
            standardized_unit = str(meta.get("standardized_unit", "")).strip().lower()
            if self._is_hz_rate_unit(raw_unit) or self._is_hz_rate_unit(standardized_unit):
                if updated_df is df:
                    updated_df = updated_df.copy()
                updated_df[stroke_col] = pd.to_numeric(updated_df[stroke_col], errors="coerce") * 60.0
                meta["unit"] = "spm"
                meta["standardized_unit"] = "spm"
                channel_metadata[stroke_col] = meta
                converted_stroke_cols.append(stroke_col)

        if converted_stroke_cols:
            print(
                "🔁 Converted PD stroke_rate from Hz to spm for column(s): "
                f"{converted_stroke_cols}"
            )

        # Chain to the base transforms so acceleration unit standardization
        # is not skipped by this override.
        return super().apply_post_rename_transforms(updated_df, channel_metadata)

    @staticmethod
    def _is_hz_rate_unit(unit: str) -> bool:
        normalized = str(unit or "").strip().lower().replace(" ", "")
        return normalized in {
            "hz",
            "1/s",
            "s^-1",
            "s-1",
            "sec^-1",
            "second^-1",
            "persecond",
        }
