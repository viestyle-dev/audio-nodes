# Security Policy

## Reporting a Vulnerability

Please use GitHub's **Private Vulnerability Reporting**
(Security tab → "Report a vulnerability") on this repository.
Do not open public issues for security reports.

## Scope

audio-nodes reads audio files and plays them to audio devices. `file_path`
is resolved against the graph's data directory, but an absolute path is
honoured, so a graph can read any audio file the engine process can
access; files are decoded in-process by libsndfile (keep `soundfile` up to
date). Playback nodes open the PortAudio device selected by `device_id`,
and `AudioQueryDevice` exposes the host's audio device inventory to the
graph. The plugin does not execute code from files or graph definitions.

The host runtime it plugs into (the nonoha engine) may include script
nodes with their own threat model — see the security policy of the host
project for loading untrusted graphs.

## Supported versions

Security fixes target the latest minor release.
