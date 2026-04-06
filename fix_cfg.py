with open("iwr6843_config.cfg", "r") as f:
    text = f.read()

# Replace any bad spaces or zero blocks
text = text.replace("chirpCfg 0 0 0 0 0 0 0 1", "chirpCfg 0 0 0 0 0 0 0 1")
text = text.replace("chirpCfg 1 1 0 0 0 0 0 2", "chirpCfg 1 1 0 0 0 0 0 2")
text = text.replace("chirpCfg 2 2 0 0 0 0 0 4", "chirpCfg 2 2 0 0 0 0 0 4")

# Also, I notice you were using profile id 0, right? The format is:
# chirpCfg chirpStartIndex chirpEndIndex profileId startFreqVar freqSlopeVar idleTimeVar adcStartTimeVar txAntennaMask
with open("iwr6843_config.cfg", "w") as f:
    f.write(text)
