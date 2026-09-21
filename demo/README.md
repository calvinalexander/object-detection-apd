# APD demo

Side-by-side comparison of production's detector and our pipeline, on the
recordings in `data/recording/`.

Runs as a systemd **user** service, so it survives logout and restarts on a crash
(lingering is enabled for this user):

    systemctl --user status  apd-demo       # is it up
    systemctl --user restart apd-demo       # after editing app.py / index.html
    systemctl --user stop    apd-demo       # stop it
    systemctl --user disable apd-demo       # stop it starting at boot
    tail -f demo/demo.log                   # its output

Unit file: `~/.config/systemd/user/apd-demo.service`. Then open http://<host>:8010.
To run it by hand instead: `PORT=8010 python demo/app.py`.

Both panes come from one decode of one frame, composited into a single MJPEG
stream, so they cannot drift apart. Every control is a query parameter, so
changing one restarts the stream.

## What the toggles do

| control | effect |
|---|---|
| Panes | show either pipeline alone, or both |
| Production: person / PPE boxes | which of its 7 classes to draw |
| Production: confidence | its detection threshold (config.yaml uses 0.25) |
| Ours: person boxes, pose confidence | the pose stage; 0.40 is what the device runs |
| Ours: head_cap | the packed trunk on the head crop |
| Ours: mask | face crop (current model) / head crop (previous model) / off |
| Ours: ROI gate | skip regions that are clipped, tiny or occluded |
| Ours: show ROI boxes | draw the crops actually fed to a trunk |
| Ours: gloves | crop detector / hand ROI trunk / off |
| every Nth frame | decode stride, for smoother playback |

## The comparison the demo is for

Switch gloves between "crop detector" and "hand ROI". The trunk depends on wrist
keypoints, which agree with the regions it was trained on only 0.48 median and
are wrong 56% of the time, so its boxes land on trays and worktops. The crop
detector needs no hand region. Measured on human labels, gloves MCC goes from
-0.037 to 0.151 while head_cap and mask are unchanged.

The person counts in the two title bars show the other half of the story: the
pose stage finds fewer people than production's detector, which caps everything
downstream.
