# MultiGPU

A Blender add-on that renders one animation on every graphics card of one or more computers at
once. One computer (the host) hands out the frames one at a time over your network, so no frame
is ever rendered twice, a faster card simply takes more frames, and a card that crashes has its
frame handed to another.

**Version 2.6.0** · Blender 4.2 or newer (tested on 5.2 and 5.3) · Cycles · Windows, Linux, macOS (Intel and ARM)

## Install

**Drag into Blender:** drag this link onto an open Blender window and confirm:

### [⬇ MultiGPU — drag me into Blender](https://raw.githubusercontent.com/klosixgaming-arch/MultiGPU/main/dist/multigpu-2.6.0.zip?repository=.%2Findex.json&blender_version_min=4.2.0)

Blender asks once to add this page as a repository; say yes and it will also offer updates under
**Edit › Preferences › Get Extensions**. This needs **Allow Online Access** switched on
(Preferences › System › Network).

**Or without online access:** click the link instead to download the `.zip`, then drag that file
into Blender (or use **Preferences › Get Extensions › ⌄ › Install from Disk**).

- Install the **same version on every computer**. Computers on different versions refuse to
  work together.
- Coming from the older single-file `multigpu.py`? Just install this one. It switches the old
  add-on off and keeps your settings.

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
- **A card that drops out:** if a graphics card stops working, the others often get a new address
  from the system. MultiGPU follows each card to its new address, and stops a card with a clear
  message rather than letting Cycles quietly render its frames on the CPU.
- **Copy Missing Files** (on by default): any file that only the host has, for example on its
  own disk, is sent to the computers that can't find it before they render. That covers every
  kind of file Blender knows the scene uses: textures, image sequences, tiled (UDIM) images,
  movies, fonts, volumes, caches, IES and OSL files, linked .blend files and the files those
  use in turn. Files with the same name from different folders are kept apart. Nothing over
  4 GB is sent, and more than 2 GB for one computer is asked about first. Packed files travel
  inside the .blend and need nothing. A file that ships with Blender is taken from each
  computer's own Blender.
- **Broken links are named before the render:** a file that the host itself can't find can't be
  sent, so the check before the render lists it by name.
- **Interrupted asset copies resume:** completed chunks stay in the receiving computer's temporary
  folder across retries and worker restarts. Transfers retry connection failures and adjust chunk
  sizes for slower links. The card's status shows the file and megabytes copied. A changed source
  file starts a fresh copy; an incomplete copy never becomes the file used for rendering.
- **Smart Finish** (on by default): the last frames go to the cards that finish them soonest.
  A slow card doesn't start a frame that the faster cards would have finished before it, and a
  fast card with nothing left to do also renders a frame a slow card still needs long for; the
  first one done counts.
- **Host takeover:** if the host crashes or drops off the network mid-render, another computer
  becomes the host after about a minute. It removes the dead cards' placeholders and carries on.
  When the old host comes back, it joins as a helper by itself. Turn this off with
  **Take Over if the Host Stops** on a joined computer.
- **Sync Changes:** saving the .blend (Ctrl+S) doesn't change what's rendering. **Sync Changes**
  saves the file and has each card reload it after its current frame.

## A computer far away (Syncthing, Dropbox ...)

A friend's computer in another city shouldn't read the scene and save its frames through a slow
connection. Give it its own copy of the project folder, kept in step by a sync program, and
MultiGPU works from that copy:

1. On that computer, open **Preferences › Add-ons › MultiGPU** and set **Project Folder on This
   Computer** to its own copy of the project (the folder the sync program fills).
2. Join the host as usual (over Tailscale, for example).

MultiGPU notices by itself that the computer doesn't see the host's folder directly. That computer
then loads the scene from its own copy, renders into its own copy, and the sync program carries
the frames to everyone. The host still hands out every frame exactly once. It shows how many
frames are still on their way, and frames a computer already holds are not rendered again when
you press Render a second time.

- A computer working from its own copy never takes over as host.
- If its copy of the .blend isn't up to date yet, it waits until the sync program has brought it.
- Tip: add `.multigpu*` to the sync program's ignore list, so the hidden working files don't travel.

## Network security

MultiGPU is made for a network you trust (your home or studio LAN, or a private VPN such as
Tailscale). The connection between computers has no password or encryption.

- Don't open or forward the MultiGPU port (47800 by default) to the internet.
- Only join a host you trust: a joined computer opens the host's .blend and runs the Python
  scripts and drivers saved in it.
- Anyone who can reach the host's port can join it, see the job's file paths and, with
  **Copy Missing Files** on, download the files the scene uses.

## Advanced options

Output format (OpenEXR, OpenEXR multilayer, PNG or JPEG, each with its own settings; the default
is OpenEXR, RGB, Float (Full), DWAA at 90%), Overwrite, Placeholders, Persistent Data,
Save First, Copy Missing Files, Smart Finish, and the scene's Post Processing switches (Compositing and Sequencer).

**Show Render** opens a window per card showing its frame live, and **Arrange Windows** tiles
them with your Blender window in a grid (Windows and Linux).

## License

GPL-3.0-or-later, like Blender itself. See [LICENSE](LICENSE).
