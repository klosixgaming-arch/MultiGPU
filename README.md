# MultiGPU

A Blender add-on that renders one animation on every graphics card of one or more computers at
once. One computer (the host) hands out the frames one at a time over your network, so no frame
is ever rendered twice, a faster card simply takes more frames, and a card that crashes has its
frame handed to another.

**Version 2.4.0** · Blender 4.2 or newer (tested on 5.2 and 5.3) · Cycles · Windows, Linux, macOS (Intel and ARM)

## Install

1. Download `multigpu.py` from the [latest release](../../releases/latest).
2. In Blender: **Edit › Preferences › Add-ons › Install from Disk**, pick `multigpu.py`, and tick it.
3. Install the **same version on every computer**. Computers on different versions refuse to
   work together.

The panel is in **Properties › Output › MultiGPU** and in the 3D Viewport sidebar (**N › MultiGPU**).

## Render on one computer

Tick the cards you want and press **Render on N cards**. Each card renders in its own Blender
process and takes the next free frame.

## Render on several computers

1. **Host** (the computer with the .blend open): set **Network › Host**. The panel shows the
   address the others type in, plus the Tailscale address if Tailscale is running. Allow the port
   (47800 by default) through this computer's firewall.
2. **Other computers**: **Network › Join**, type the host's address, press **Join**, and leave
   Blender open. Joined hosts are remembered, and you can give them names. A computer can join
   while a render is already running; it just starts taking frames.
3. Press **Render** on the host.

The .blend and the output folder must be on shared storage (a NAS, for example). Each computer
finds the .blend by itself and converts the host's paths to its own, so `B:\Projects\...` on
Windows and `/run/user/1000/gvfs/smb-share:.../Projects/...` on Linux both work. An output path
relative to the .blend, such as `//../renders/`, is the simplest.

## What keeps it safe

- **No duplicate frames:** the host hands out each frame once. A card writes into a hidden
  `.multigpu-partial` folder and moves the finished file into place in one step.
- **Placeholders** (on by default) mark frames that are being rendered with a 0 KB file.
- **Pre-render check:** before rendering, every computer checks that it can find the .blend, its
  textures and caches, the output folder, the needed add-ons and the Blender version. Anything
  wrong is listed for you to confirm.
- **Failures:** a failed frame goes to another card, a stuck frame is also given to an idle card,
  and a card that runs out of GPU memory restarts with tiles half the size.
- **Host takeover:** if the host crashes or drops off the network mid-render, another computer
  becomes the host after about a minute. It removes the dead cards' placeholders and carries on.
  When the old host comes back, it joins as a helper by itself. Turn this off with
  **Take Over if the Host Stops** on a joined computer.
- **Sync Changes:** saving the .blend (Ctrl+S) doesn't change what's rendering. **Sync Changes**
  saves the file and has each card reload it after its current frame.

## Advanced options

Output format (OpenEXR, OpenEXR multilayer, PNG or JPEG, each with its own settings; the default
is OpenEXR, RGB, Float (Full), DWAA at 90%), Overwrite, Placeholders, Persistent Data,
Save First, and the scene's Post Processing switches (Compositing and Sequencer).

**Show Render** opens a window per card showing its frame live, and **Arrange Windows** tiles
them with your Blender window in a grid (Windows and Linux).

## License

GPL-3.0-or-later, like Blender itself. See [LICENSE](LICENSE).
