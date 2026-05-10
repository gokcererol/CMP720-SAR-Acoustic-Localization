"""
LoRa TX — Builds and transmits 31-byte binary packets.
Simulates the ESP32 side of LoRa packet construction.
"""

import struct
import socket
import time
from typing import Dict, Optional


# CRC-8 lookup table (polynomial 0x07)
CRC8_TABLE = [0] * 256
for _i in range(256):
    _crc = _i
    for _ in range(8):
        if _crc & 0x80:
            _crc = ((_crc << 1) ^ 0x07) & 0xFF
        else:
            _crc = (_crc << 1) & 0xFF
    CRC8_TABLE[_i] = _crc


def crc8(data: bytes) -> int:
    """Compute CRC-8 checksum."""
    crc = 0
    for byte in data:
        crc = CRC8_TABLE[crc ^ byte]
    return crc


class LoRaTX:
    """
    Constructs and transmits 31-byte TDoA binary packets.
    Packet format matches the C struct in ESP32 firmware.
    """

    # struct format: <BQHHBBBBbBiiHBB = 31 bytes
    PACKET_FORMAT = "<BQHHBBBBbBiiHBB"
    PACKET_SIZE = 31

    def __init__(self, node_id: int, lora_channel_port: int = 5020,
                 battery_pct: int = 100):
        self.node_id = node_id
        self.lora_channel_port = lora_channel_port
        self.battery_pct = battery_pct
        self.event_count = 0
        self.temperature_c = 22  # Default ambient temp

        # UDP socket to LoRa channel simulator
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def build_packet(self, ts_micros: int, magnitude: int, peak_freq_hz: int,
                     ml_class: int, ml_confidence: int, snr_db: int,
                     gps_hdop: int = 15, gps_lat: float = 39.867000,
                     gps_lon: float = 32.733000) -> bytes:
        """
        Build a 31-byte binary packet.
        Returns the raw bytes ready for transmission.
        """
        self.event_count += 1
        lat_e7 = int(round(float(gps_lat) * 10_000_000.0))
        lon_e7 = int(round(float(gps_lon) * 10_000_000.0))

        lat_e7 = max(-(2**31), min((2**31) - 1, lat_e7))
        lon_e7 = max(-(2**31), min((2**31) - 1, lon_e7))

        # Pack all fields except CRC and reserved
        data_without_crc = struct.pack(
            "<BQHHBBBBbBiiH",
            int(self.node_id),           # uint8  (1B)
            int(ts_micros),              # uint64 (8B)
            int(min(magnitude, 65535)),  # uint16 (2B)
            int(min(peak_freq_hz, 65535)),  # uint16 (2B)
            int(ml_class),               # uint8  (1B)
            int(min(ml_confidence, 100)),  # uint8  (1B)
            int(min(snr_db, 255)),       # uint8  (1B)
            int(min(self.battery_pct, 100)),  # uint8  (1B)
            int(max(-128, min(127, self.temperature_c))),  # int8 (1B)
            int(min(gps_hdop, 255)),     # uint8  (1B)
            int(lat_e7),                 # int32 (4B)
            int(lon_e7),                 # int32 (4B)
            int(self.event_count & 0xFFFF),  # uint16 (2B)
        )

        # Compute CRC-8
        crc = crc8(data_without_crc)

        # Full packet: data + CRC + reserved byte
        packet = data_without_crc + struct.pack("BB", crc, 0x00)

        assert len(packet) == self.PACKET_SIZE, f"Packet size mismatch: {len(packet)}"
        return packet

    def transmit(self, packet: bytes) -> bool:
        """Send packet to the LoRa channel simulator."""
        try:
            self.sock.sendto(packet, ("127.0.0.1", self.lora_channel_port))
            return True
        except Exception as e:
            print(f"[LoRa TX Node {self.node_id}] Send error: {e}")
            return False

    def send_event(self, ts_micros: int, magnitude: int, peak_freq_hz: int,
                   ml_class: int, ml_confidence: int, snr_db: int,
                   gps_lat: float = 39.867000, gps_lon: float = 32.733000) -> bool:
        """Build and transmit a TDoA event packet."""
        packet = self.build_packet(
            ts_micros, magnitude, peak_freq_hz,
            ml_class, ml_confidence, snr_db,
            gps_lat=gps_lat, gps_lon=gps_lon,
        )
        success = self.transmit(packet)

        if success:
            print(f"🚀 [Node {self.node_id}] 31-Byte packet TX | "
                  f"TS: {ts_micros} | Mag: {magnitude} | "
                  f"Class: {ml_class} ({ml_confidence}%) | SNR: {snr_db}dB | "
                  f"Lat/Lon: {gps_lat:.6f},{gps_lon:.6f}")
        return success

    def drain_battery(self, amount: float = 0.1):
        """Reduce battery by given percentage."""
        self.battery_pct = max(0, self.battery_pct - amount)

    def close(self):
        """Close the UDP socket."""
        self.sock.close()


def unpack_packet(data: bytes) -> Optional[Dict]:
    """
    Unpack a received 31-byte packet.
    Returns None if invalid.
    """
    if len(data) != 31:
        return None

    try:
        fields = struct.unpack("<BQHHBBBBbBiiHBB", data)
        packet = {
            "node_id": fields[0],
            "ts_micros": fields[1],
            "magnitude": fields[2],
            "peak_freq_hz": fields[3],
            "ml_class": fields[4],
            "ml_confidence": fields[5],
            "snr_db": fields[6],
            "battery_pct": fields[7],
            "temperature_c": fields[8],
            "gps_hdop": fields[9],
            "latitude_e7": fields[10],
            "longitude_e7": fields[11],
            "latitude": fields[10] / 10_000_000.0,
            "longitude": fields[11] / 10_000_000.0,
            "event_count": fields[12],
            "crc8": fields[13],
            "reserved": fields[14],
        }

        # Verify CRC
        data_without_crc = data[:29]
        expected_crc = crc8(data_without_crc)
        if packet["crc8"] != expected_crc:
            return None  # CRC failure

        return packet
    except struct.error:
        return None
