MAVERICK1 final submission package
==================================

Run from this folder:

    C:\Users\JUN\miniconda3\envs\aip\python.exe student\submission.py

Or double-click RUN_SUBMISSION.bat.

The client waits for a local DogFightViewer UDP server and connects automatically.
For a remote competition server, set TEAM01_SERVER_IP and optionally
TEAM01_SERVER_PORT before running.

Included active policy
----------------------
- Champion BT: AIP_Team01_v12_stable.dll + Rule_codex_champion_v1.xml
- Rear defense: D4 bundle_000090
- Crossing defense: C5b bundle_000080
- Frontal merge: corrected F3 candidate bundle_000080
- Neutral, positioning, and firing windows: champion BT

Measured Viewer baseline (2026-08-24)
-------------------------------------
- 2000 ft: 1 win, 2 losses
- 2500 ft: 3 wins, 0 losses
- 3000 ft: 1 win, 0 losses, 2 draws
- Total: 5 wins, 2 losses, 2 draws

Do not rename or move the DLL/XML/model bundle files inside this package.
