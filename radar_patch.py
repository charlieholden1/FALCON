import struct
import time
from radar import IWR6843Driver, MAGIC_WORD, TLV_DETECTED_OBJECTS, RadarPoint, logger

def _reader_loop(self) -> None:
    """Background thread to read and parse TLV data from the data port."""
    header_struct = struct.Struct('<Q8I')
    tlv_struct = struct.Struct('<2I')
    obj_struct = struct.Struct('<4f')
    
    buffer = b''
    while self._running:
        try:
            if self._data_serial is None or not self._data_serial.is_open:
                time.sleep(0.01)
                continue
            chunk = self._data_serial.read(self._data_serial.in_waiting or 1)
            if not chunk:
                continue
            buffer += chunk

            while len(buffer) >= 40:
                idx = buffer.find(MAGIC_WORD)
                if idx == -1:
                    buffer = b''
                    break
                if idx > 0:
                    buffer = buffer[idx:]
                if len(buffer) < 40:
                    break

                header = header_struct.unpack(buffer[:40])
                packet_len, num_tlv = header[2], header[7]
                
                if len(buffer) < packet_len:
                    break

                frame_data = buffer[40:packet_len]
                buffer = buffer[packet_len:]
                
                points = []
                offset = 0
                for _ in range(num_tlv):
                    if offset + 8 > len(frame_data): break
                    tlv_type, tlv_len = tlv_struct.unpack(frame_data[offset:offset+8])
                    offset += 8
                    
                    if tlv_type == TLV_DETECTED_OBJECTS:
                        for _ in range((tlv_len - 8) // 16 if tlv_len > 8 else tlv_len // 16):
                            if offset + 16 > len(frame_data): break
                            x, y, z, v = obj_struct.unpack(frame_data[offset:offset+16])
                            points.append(RadarPoint(x=x, y=y, z=z, velocity=v))
                            offset += 16
                    else:
                        offset += (tlv_len - 8) if tlv_len > 8 else tlv_len
                        
                with self._lock:
                    self._frame_buffer.append(points)
                    self._frame_count += 1
                    self._frame_timestamps.append(time.time())

        except Exception as exc:
            if self._running: logger.warning("Radar read error: %s", exc)
            time.sleep(0.1)

IWR6843Driver._reader_loop = _reader_loop
