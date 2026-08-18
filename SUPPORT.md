# Support and compatibility policy

iree-metal is a volunteer-maintained developer preview with no production SLA.
The issue tracker is the public support channel; support is best-effort.

The current preview supports only the matrix recorded in its manifest:

- Apple M4 hardware;
- the macOS version recorded in the release manifest (the wheel's macOS 13
  deployment target is a binary-build setting, not a broad validation claim);
- CPython 3.12;
- JAX and JAXLIB 0.10.2 through the latest verified release below 0.12; and
- the two matching iree-metal wheels from one release.

The release compatibility evidence names every exact JAX/JAXLIB pair tested in
that range. The full semantic, model, and performance gates use the manifest's
locked JAX/JAXLIB pair.

Other Apple GPU generations, Python versions, JAX versions outside that range, dynamic shapes,
multiple devices, and distributed execution may work but are unsupported until
they appear in a release's tested matrix.

Preview versions follow PEP 440 and are immutable once published. Only the
newest preview receives fixes. Experimental profiles and package APIs may be
changed or removed in the next preview; material removals will be called out in
release notes. A preview will not be promoted to a stable compatibility promise
without a separate announcement and policy revision.
