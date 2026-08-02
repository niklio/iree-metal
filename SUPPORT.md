# Support and compatibility policy

iree-metal is a volunteer-maintained developer preview with no production SLA.
The issue tracker is the public support channel; support is best-effort.

The first preview supports only the exact matrix recorded in its manifest:

- Apple M4 hardware;
- the macOS version recorded in the release manifest (the wheel's macOS 13
  deployment target is a binary-build setting, not a broad validation claim);
- CPython 3.12;
- JAX and JAXLIB 0.6.1; and
- the two matching iree-metal wheels from one release.

Other Apple GPU generations, Python versions, JAX versions, dynamic shapes,
multiple devices, and distributed execution may work but are unsupported until
they appear in a release's tested matrix.

Preview versions follow PEP 440 and are immutable once published. Only the
newest preview receives fixes. Experimental profiles and package APIs may be
changed or removed in the next preview; material removals will be called out in
release notes. A preview will not be promoted to a stable compatibility promise
without a separate announcement and policy revision.
