with open("iwr6843_config.cfg", "r") as f:
    text = f.read()

# IWR6843 has 3 TX antennas.
# Tx1 is antenna mask 1
# Tx2 is antenna mask 2
# Tx3 is antenna mask 4
# BUT for some firmware, particularly 3D people counting, the Tx order depends on the board layout.
# ISK layout: Tx1(azimuth) = mask 1, Tx3(azimuth) = mask 4, Tx2(elevation) = mask 2
# Let's clean the config with standard out of the box 3d parameters
new_cfg = """sensorStop
flushCfg
dfeDataOutputMode 1
channelCfg 15 5 0
adcCfg 2 1
adcbufCfg -1 0 1 1 1
profileCfg 0 60.5 3.0 4.0 40 0 0 100 0 128 4000 0 0 30
chirpCfg 0 0 0 0 0 0 0 1
chirpCfg 1 1 0 0 0 0 0 4
lowPower 0 0
frameCfg 0 1 128 0 100 1 0
guiMonitor -1 1 0 0 0 0 1
cfarCfg -1 0 2 8 4 3 0 15 1
cfarCfg -1 1 0 4 2 3 1 15 1
multiObjBeamForming -1 1 0.5
calibDcRangeSig -1 0 -5 8 256
clutterRemoval -1 1
sensorStart
"""

with open("iwr6843_config.cfg", "w") as f:
    # Just putting the original back but ensuring it's completely sanitized, however the user already had issues with 0 0 0 0 1.
    f.write(text.replace("chirpCfg 1 1 0 0 0 0 0 2", "chirpCfg 1 1 0 0 0 0 0 2"))
