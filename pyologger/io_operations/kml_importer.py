import os
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pytz

from pyologger.io_operations.base_importer import BaseImporter
from pyologger.process_data.sampling import calculate_sampling_frequency


class KMLImporter(BaseImporter):
    """
    Handles KML files containing GPS location data with timestamps.
    Parses placemark coordinates and stores in signal_data['location'].
    """

    def process_files(self, files, enforce_frequency: bool = True):
        """
        Process KML files and extract GPS location data.
        
        Args:
            files: List of file paths (relative to data_folder)
            enforce_frequency: Ignored for KML files (location data is typically sparse)
            
        Returns:
            None (stores location data directly in data_reader.signal_data['location'])
        """
        logger_id = self.logger_id
        
        # Filter for KML files
        kml_files = [
            f for f in files
            if f.lower().endswith(('.kml', '.KML'))
        ]
        
        if not kml_files:
            print(f"ℹ️ No KML files found for logger {logger_id}")
            return None
        
        print(f"📍 Processing {len(kml_files)} KML file(s) for GPS location data:")
        for kml_file in kml_files:
            print(f"   • {kml_file}")
        
        # Parse all KML files and combine
        all_location_data = []
        for kml_file in kml_files:
            file_path = os.path.join(self.data_reader.data_folder, kml_file)
            df = self._parse_kml_file(Path(file_path))
            if df is not None and not df.empty:
                all_location_data.append(df)
        
        if not all_location_data:
            print(f"⚠️ No valid GPS data extracted from KML files for {logger_id}")
            return None
        
        # Combine all location data
        location_df = pd.concat(all_location_data, ignore_index=True)
        location_df = location_df.sort_values('datetime').reset_index(drop=True)
        
        # Store in data_reader.signal_data
        self._store_location_data(location_df)
        
        return None  # KML importer stores data directly, doesn't return for further processing
    
    def _parse_kml_file(self, kml_path: Path) -> Optional[pd.DataFrame]:
        """
        Parse a KML file and extract GPS coordinates with timestamps.
        Handles multiple KML formats:
        - Standard KML with TimeStamp/when
        - Vectronics ExtendedData format with UTC_Date/UTC_Time
        - Vectronics GPS Plus X format (Folder/Fixes/Placemark structure)
        
        Args:
            kml_path: Path to the KML file
            
        Returns:
            DataFrame with columns: datetime, latitude, longitude
        """
        try:
            tree = ET.parse(kml_path)
            root = tree.getroot()
            
            # Handle KML namespace
            namespace = {'kml': 'http://www.opengis.net/kml/2.2'}
            earth_namespace = {'kml': 'http://earth.google.com/kml/2.2'}
            
            # Try to determine which namespace to use
            if 'earth.google.com' in root.tag:
                namespace = earth_namespace
            elif 'opengis.net' not in root.tag and not root.tag.endswith('kml'):
                # Try without namespace
                namespace = {}
            
            # Extract placemarks (GPS points)
            if namespace:
                placemarks = root.findall('.//kml:Placemark', namespace)
            else:
                placemarks = root.findall('.//Placemark')
            
            if not placemarks:
                print(f"   ⚠️ No Placemarks found in {kml_path.name}")
                return None
            
            location_records = []
            
            # Get timezone for local time conversion
            tz_name = self.data_reader.deployment_info.get("Time Zone") or "UTC"
            try:
                local_tz = pytz.timezone(tz_name)
            except Exception:
                print(f"   ⚠️ Invalid timezone '{tz_name}'. Using UTC.")
                local_tz = pytz.UTC
            
            for placemark in placemarks:
                # Skip First/Last position marks (not actual fixes)
                name_elem = placemark.find('kml:name', namespace) if namespace else placemark.find('name')
                if name_elem is not None and name_elem.text:
                    name_text = name_elem.text.strip()
                    if name_text in ['First Position Mark', 'Last Position Mark']:
                        continue
                
                dt_local = None
                
                # Method 1: Try standard KML TimeStamp/when (GPS Plus X format)
                timestamp_elem = None
                if namespace:
                    timestamp_elem = placemark.find('.//kml:TimeStamp/kml:when', namespace)
                else:
                    timestamp_elem = placemark.find('.//TimeStamp/when')
                
                if timestamp_elem is not None and timestamp_elem.text:
                    # GPS Plus X format: <when>2021-06-25T15:00:39Z</when>
                    timestamp_str = timestamp_elem.text.strip()
                    try:
                        dt_utc = pd.to_datetime(timestamp_str, utc=True)
                        dt_local = dt_utc.tz_convert(local_tz)
                    except Exception as e:
                        print(f"   ⚠️ Could not parse timestamp '{timestamp_str}': {e}")
                        continue
                
                # Method 2: If no standard timestamp, try Vectronics ExtendedData format
                elif timestamp_elem is None or timestamp_elem.text is None:
                    # Look for ExtendedData with UTC_Date and UTC_Time
                    extended_data = placemark.find('.//kml:ExtendedData', namespace) if namespace else placemark.find('.//ExtendedData')
                    if extended_data is not None:
                        utc_date = None
                        utc_time = None
                        
                        for simple_data in extended_data.findall('.//kml:SimpleData', namespace) if namespace else extended_data.findall('.//SimpleData'):
                            name = simple_data.get('name')
                            if name == 'UTC_Date':
                                utc_date = simple_data.text
                            elif name == 'UTC_Time':
                                utc_time = simple_data.text
                        
                        if utc_date and utc_time:
                            try:
                                # Combine date and time
                                datetime_str = f"{utc_date} {utc_time}"
                                dt_utc = pd.to_datetime(datetime_str, utc=True)
                                dt_local = dt_utc.tz_convert(local_tz)
                            except Exception as e:
                                print(f"   ⚠️ Could not parse Vectronics timestamp '{datetime_str}': {e}")
                                continue
                
                if dt_local is None:
                    continue
                
                # Extract coordinates
                if namespace:
                    coords_elem = placemark.find('.//kml:Point/kml:coordinates', namespace)
                else:
                    coords_elem = placemark.find('.//Point/coordinates')
                
                if coords_elem is None or coords_elem.text is None:
                    continue
                
                coords_str = coords_elem.text.strip()
                # KML coordinates format: longitude,latitude,altitude (altitude optional)
                coords_parts = coords_str.split(',')
                
                if len(coords_parts) < 2:
                    continue
                
                try:
                    longitude = float(coords_parts[0])
                    latitude = float(coords_parts[1])
                    
                    # Skip invalid coordinates (0,0) which indicate no fix
                    if longitude == 0 and latitude == 0:
                        continue
                    
                except ValueError:
                    continue
                
                location_records.append({
                    'datetime': dt_local,
                    'latitude': latitude,
                    'longitude': longitude
                })
            
            if not location_records:
                print(f"   ⚠️ No valid GPS points extracted from {kml_path.name}")
                return None
            
            df = pd.DataFrame(location_records)
            print(f"   ✅ Parsed {len(df)} GPS points from {kml_path.name}")
            return df
            
        except Exception as e:
            print(f"   ❌ Error parsing KML file {kml_path.name}: {e}")
            return None
    
    def _store_location_data(self, location_df: pd.DataFrame):
        """
        Store location data in data_reader.signal_data['location'] and
        update data_reader.signal_info['location'] in the same metadata
        shape as other imported signals.
        
        Args:
            location_df: DataFrame with datetime, latitude, longitude columns
        """
        if 'location' in self.data_reader.signal_data:
            existing_df = self.data_reader.signal_data['location']
            location_df = pd.concat([existing_df, location_df], ignore_index=True)

        location_df = (
            location_df
            .dropna(subset=['datetime', 'latitude', 'longitude'])
            .sort_values('datetime')
            .drop_duplicates(subset=['datetime', 'latitude', 'longitude'])
            .reset_index(drop=True)
        )
        self.data_reader.signal_data['location'] = location_df

        if location_df.empty:
            print("   ⚠️ No valid GPS rows available to store in signal_data['location'].")
            return

        channels = ['latitude', 'longitude']
        start_time = location_df['datetime'].iloc[0]
        end_time = location_df['datetime'].iloc[-1]
        max_value = location_df[channels].max().max()
        min_value = location_df[channels].min().min()
        mean_value = location_df[channels].mean().mean()
        data_type_str = str(location_df[channels].dtypes.iloc[0])

        original_frequency = calculate_sampling_frequency(location_df['datetime'].head())

        time_zone_raw = self.data_reader.deployment_info.get('Time Zone')
        tz_name = str(time_zone_raw).strip() if time_zone_raw is not None else "UTC"
        if not tz_name:
            tz_name = "UTC"
        try:
            tz = pytz.timezone(tz_name)
        except Exception:
            print(f"⚠️ Invalid timezone '{tz_name}'. Falling back to UTC.")
            tz = pytz.UTC

        self.data_reader.signal_info['location'] = {
            'channels': channels,
            'metadata': {
                'latitude': {
                    'original_name': 'latitude',
                    'unit': 'degrees',
                    'standardized_unit': 'degrees',
                    'parent_signal': 'location',
                },
                'longitude': {
                    'original_name': 'longitude',
                    'unit': 'degrees',
                    'standardized_unit': 'degrees',
                    'parent_signal': 'location',
                },
            },
            'signal_start_datetime': start_time,
            'signal_end_datetime': end_time,
            'max_value': float(max_value),
            'min_value': float(min_value),
            'mean_value': float(mean_value),
            'data_type': data_type_str,
            'original_units': 'degrees',
            'units': 'degrees',
            'original_sampling_frequency': original_frequency,
            'sampling_frequency': original_frequency,
            'logger_id': self.logger_id,
            'logger_manufacturer': self.logger_manufacturer,
            'processing_step': 'Raw data uploaded',
            'last_updated': pd.Timestamp(datetime.now().astimezone(tz)),
            'details': (
                'Initial, raw signal-specific data and metadata loaded from KML GPS points. '
                f'Original frequency: {original_frequency} Hz.'
            ),
        }

        print(
            f"   ✅ Stored {len(location_df)} GPS location points in signal_data['location'] "
            "with signal_info metadata"
        )
